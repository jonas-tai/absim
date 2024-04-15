from experiment import main

import random
import optuna
import argparse
import json
import os

# STEPS:
# - input JSON with tuning specifications
# - for each run:
#   - call main each time with arguments for that run
#   - main loads configs from provided input_args
#   - main performs the run and returns final acc?
# - retrun optimized model

def objective(trial, json_obj):
    input_args = []
    for arg_obj in json_obj:
        # the arg_obj is either const or tuned
        if arg_obj["key_type"] == "const":
            if arg_obj["key"] == "exp_prefix":
                input_args.append("--exp_prefix") 
                input_args.append(f"trial_{trial.number}")
            else:
                input_args.append("--" + arg_obj["key"])
                # the second requirement is because progress_bar is a boolean argument
                if not (isinstance(arg_obj["value"], str) and arg_obj["value"] == ""):
                    input_args.append(str(arg_obj["value"]))
        else:
            input_args.append("--" + arg_obj["key"])
            # this determines the values used by the trial
            if arg_obj["value_type"] == "categorical":
                input_args.append(str(trial.suggest_categorical(arg_obj["key"], arg_obj["value"])))
            elif arg_obj["value_type"] == "float":
                input_args.append(str(trial.suggest_float(arg_obj["key"], arg_obj["value"][0], arg_obj["value"][1])))
            elif arg_obj["value_type"] == "int":
                input_args.append(str(trial.suggest_int(arg_obj["key"], arg_obj["value"][0], arg_obj["value"][1])))
            elif arg_obj["value_type"] == "log":
                input_args.append(str(trial.suggest_float(arg_obj["key"], arg_obj["value"][0], arg_obj["value"][1], log=True)))
            else:
                raise RuntimeError("No value_type for argument")
    print(input_args)
    # the output is the value that you want to maximize with your hyperparameter choice
    return main(input_args=input_args)


def save_best(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    with open(f"./autotune/best_params/best_params_{study.study_name}.json", "w") as json_file:
        best_params = study.best_params
        best_params['trial_number'] = study.best_trial.number
        print(f"Saving best trial to: best_params_{study.study_name}.json")
        json.dump(best_params, json_file)


def run_best(config, json_obj, iteration):
    input_args = []
    # set consts
    for arg_obj in json_obj:
        if arg_obj["key_type"] == "const":
            # some args should be ignored:
            # - seed: we want random seeds for these runs
            if arg_obj["key"] == "seed":
                continue
            if arg_obj["key"] == "exp_prefix":
                input_args.append("--exp_prefix") 
                input_args.append(f"run_{iteration}")
            else:
                input_args.append("--" + arg_obj["key"])
                if not (isinstance(arg_obj["value"], str) and arg_obj["value"] == ""):
                    input_args.append(str(arg_obj["value"]))
    input_args.append("--seed") 
    input_args.append(f"{random.randint(0, 100_000)}")
    for arg_obj in config:
        if arg_obj["key"] == "trial_number":
            continue
        input_args.append("--" + arg_obj["key"])
        if not (isinstance(arg_obj["value"], str) and arg_obj["value"] == ""):
            input_args.append(str(arg_obj["value"]))
    print(input_args)
    # run it
    main(input_args=input_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the in_path is with respect to ./autotune/
    parser.add_argument("in_path", help="input JSON for tuning located in ./autotune/", type=str)
    # how many "more" trials you want to run
    parser.add_argument("--n", default=10, help="number of trials to run (not including previous ones)", type=int)
    parser.add_argument("--evaluate", action="store_true", help="whether or not to run best")
    args = parser.parse_args()

    # load json
    with open(f"./autotune/{args.in_path}") as json_file:
        json_obj = json.load(json_file)

    # unique name of the study (if you want to restart you will need to delete the sql data 
    # for the previous version)
    study_name = f"{args.in_path.replace('.json', '')}_study"  
    # your sql files will be located in ./autotune/sql
    storage_name = "sqlite:///./autotune/sql/{}.db".format(study_name)
    # check locations
    if not os.path.exists("./autotune/sql"):
        os.makedirs("./autotune/sql")
    if not os.path.exists("./autotune/best_params"):
        os.makedirs("./autotune/best_params")
    # this will continue your study from where you left off
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)
    print(f"Completed {len(study.trials)} / {len(study.trials)+args.n} trials")
    # this will do the autotuning and will save the best hyperparameters after each trial
    study.optimize(lambda trial : objective(trial, json_obj), n_trials=args.n, callbacks=[save_best])

    # this will run the model with the best hyperparameters 5 times and save run information to WandB
    print(f"Evaluating for best parameters")
    if args.evaluate:
        for i in range(5):
            run_best(study.best_params, json_obj, i)