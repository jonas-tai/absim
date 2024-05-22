import optuna

# Load the study from the database
study_name = "autotune1_study"  # replace with your study name
storage_name = "sqlite:////home/jonas/projects/absim/simulations/autotune/sql/autotune1_study.db"  # replace with your storage URL
study = optuna.load_study(study_name=study_name, storage=storage_name)

# Iterate over trials and print their outcomes
for trial in study.trials:
    print(f"Trial number: {trial.number}")
    print(f"State: {trial.state}")
    print(f"Value: {trial.value}")
    print(f"Params: {trial.params}")
    print(f"User attributes: {trial.user_attrs}")
    print(f"System attributes: {trial.system_attrs}")
    print("="*40)
