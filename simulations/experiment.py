import json
import os
from typing import Any, List

import torch

import random
import numpy as np
from simulation_args import BaseArgs, HeterogeneousRequestsArgs, SimulationArgs, StaticSlowServerArgs, TimeVaryingServerArgs, log_arguments
from pathlib import Path
from model_trainer import Trainer
from simulations.feature_data_collector import FeatureDataCollector
from simulations.monitor import Monitor
from simulations.plotting import ExperimentPlot
from experiment_runner import ExperimentRunner
from simulations.state import StateParser


TRAIN_POLICIES_TO_RUN = [
    # 'round_robin',
    'ARS',
    # 'response_time',
    # 'weighted_response_time',
    # 'random',
    'DQN'
]

DQN_EXPLR_SETTINGS = [f'DQN_EXPLR_{i * 10}' for i in range(11)]

EVAL_POLICIES_TO_RUN = [
    # 'round_robin',
    'ARS',
    'DQN',
    'random',
    # 'DQN_EXPLR',
    'DQN_DUPL'
] + DQN_EXPLR_SETTINGS

DQN_EXPLR_MAPPING = {f'DQN_EXPLR_{i * 10}': (i / 10.0) for i in range(11)}


def print_monitor_time_series_to_file(file_desc, prefix, monitor) -> None:
    for entry in monitor:
        file_desc.write("%s %s %s\n" % (prefix, entry[0], entry[1]))


def create_experiment_folders(simulation_args: SimulationArgs, state_parser: StateParser) -> Path:
    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    experiment_num = 0
    if simulation_args.args.exp_name != "":
        out_path = Path('..', simulation_args.args.output_folder, simulation_args.args.exp_name, str(experiment_num))
        while os.path.isdir(out_path):
            experiment_num += 1
            out_path = Path('..', simulation_args.args.output_folder,
                            simulation_args.args.exp_name, str(experiment_num))
    else:
        out_path = Path('..', simulation_args.args.output_folder, str(experiment_num))
        while os.path.isdir(out_path):
            experiment_num += 1
            out_path = Path('..', simulation_args.args.output_folder, str(experiment_num))

    simulation_args.args.exp_prefix = str(experiment_num)
    os.makedirs(out_path, exist_ok=True)

    return out_path


def rl_experiment_wrapper(simulation_args: SimulationArgs, input_args: Any | None) -> float:
    state_parser = StateParser(num_servers=simulation_args.args.num_servers,
                               long_tasks_fraction=simulation_args.args.long_tasks_fraction,
                               num_request_rates=len(simulation_args.args.rate_intervals),
                               poly_feat_degree=simulation_args.args.poly_feat_degree)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer = Trainer(state_parser=state_parser, model_structure=simulation_args.args.model_structure, n_actions=simulation_args.args.num_servers,
                      gamma=simulation_args.args.gamma,
                      eps_decay=simulation_args.args.eps_decay, eps_start=simulation_args.args.eps_start, eps_end=simulation_args.args.eps_end,
                      tau=simulation_args.args.tau, tau_decay=simulation_args.args.tau_decay,
                      lr=simulation_args.args.lr, batch_size=simulation_args.args.batch_size, lr_scheduler_gamma=simulation_args.args.lr_scheduler_gamma, lr_scheduler_step_size=simulation_args.args.lr_scheduler_step_size)

    out_path = create_experiment_folders(simulation_args=simulation_args, state_parser=state_parser)

    run_rl_training(simulation_args=simulation_args, trainer=trainer, state_parser=state_parser, out_folder=out_path)
    test_result_autotune = run_rl_test(simulation_args=simulation_args, out_folder=out_path,
                                       experiment='test', trainer=trainer, state_parser=state_parser)
    # additional_test_args = [HeterogeneousRequestsArgs(input_args=input_args, long_tasks_fraction=0.4,
    #   long_task_added_service_time=200)]
    additional_test_args = []

    for args in additional_test_args:
        print('Running with args:')
        print(args)
        run_rl_test(simulation_args=args, out_folder=out_path,
                    experiment=args.to_string(), trainer=trainer, state_parser=state_parser)
    return test_result_autotune


def run_rl_training(simulation_args: SimulationArgs, trainer: Trainer, state_parser: StateParser, out_folder: Path):
    NUM_EPSIODES = simulation_args.args.epochs
    LAST_EPOCH = NUM_EPSIODES - 1
    to_print = False
    utilization = simulation_args.args.utilization

    # Init directories
    experiment_folder = out_folder / 'train'
    plot_path = experiment_folder / simulation_args.args.plot_folder
    data_folder = experiment_folder / simulation_args.args.data_folder
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    train_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder, utilization=utilization)
    train_data_analyzer = FeatureDataCollector(out_folder=data_folder, state_parser=state_parser)

    simulation_args.set_print(to_print)

    log_arguments(experiment_folder, simulation_args)

    duplication_rate = 0.0

    print('Starting experiments')
    for policy in TRAIN_POLICIES_TO_RUN:
        simulation_args.set_policy(policy)
        # if policy == 'DQN':
        #     simulation_args.set_print(True)
        for i_episode in range(NUM_EPSIODES):
            print(i_episode)

            random.seed(i_episode)
            np.random.seed(i_episode)
            torch.manual_seed(i_episode)
            simulation_args.set_seed(i_episode)

            data_point_monitor = experiment_runner.run_experiment(
                simulation_args.args, num_requests=simulation_args.args.num_requests, utilization=utilization, trainer=trainer, duplication_rate=duplication_rate)
            train_plotter.add_data(data_point_monitor, policy, i_episode)

            if simulation_args.args.collect_data_points or i_episode == LAST_EPOCH:
                train_data_analyzer.add_data(data_point_monitor, policy=policy, epoch_num=i_episode)

            if policy == 'DQN':
                # Print number of DQN decisions that matched ARS
                experiment_runner.print_dqn_decision_equal_to_ars_ratio()

                # Update LR
                trainer.scheduler.step()

                trainer.reset_episode_counters()
                # trainer.print_weights()

    print('Finished')
    # train_data_analyzer.run_latency_lin_reg(epoch=LAST_EPOCH)
    trainer.save_models(model_folder=data_folder)

    train_plotter.export_data()

    if simulation_args.args.collect_data_points:
        train_data_analyzer.export_epoch_data(epoch=LAST_EPOCH)
        train_data_analyzer.export_training_data()

    trainer.plot_grads_and_losses(plot_path=plot_path)

    plot_collected_data(plotter=train_plotter, epoch_to_plot=LAST_EPOCH, policies_to_plot=TRAIN_POLICIES_TO_RUN)


def run_rl_test(simulation_args: SimulationArgs, out_folder: Path, experiment: str, trainer: Trainer, state_parser: StateParser) -> float:
    BASE_TEST_SEED = 1111111

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    NUM_TEST_EPSIODES = simulation_args.args.test_epochs
    NUM_TEST_REQUESTS = simulation_args.args.num_requests_test
    LAST_EPOCH = NUM_TEST_EPSIODES - 1
    to_print = False

    experiment_folder = out_folder / experiment
    plot_path = experiment_folder / simulation_args.args.plot_folder
    data_folder = experiment_folder / simulation_args.args.data_folder
    model_folder = out_folder / 'train' / simulation_args.args.data_folder

    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    utilization = simulation_args.args.utilization

    test_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder, utilization=utilization)
    test_data_analyzer = FeatureDataCollector(out_folder=data_folder, state_parser=state_parser)

    simulation_args.set_print(to_print)

    log_arguments(experiment_folder, simulation_args)

    for policy in EVAL_POLICIES_TO_RUN:
        simulation_args.set_policy(policy)
        print(f'Starting Test Sequence for {policy}')
        # Reset hyperparameters
        trainer.EPS_START = simulation_args.args.eps_start
        trainer.EPS_END = simulation_args.args.eps_end
        trainer.eval_mode = False

        duplication_rate = 0.0

        utilization = simulation_args.args.utilization
        if policy == 'DQN_EXPLR':
            trainer.eval_mode = True
            print(f'simulation_args.args.dqn_explr: {simulation_args.args.dqn_explr}')
            trainer.EPS_END = simulation_args.args.dqn_explr
            trainer.EPS_START = simulation_args.args.dqn_explr
            # trainer.LR = 1e-6
        elif policy.startswith('DQN_EXPLR_'):
            trainer.eval_mode = True
            print(f'simulation_args.args.dqn_explr: {simulation_args.args.dqn_explr}')
            simulation_args.args.dqn_explr = DQN_EXPLR_MAPPING[policy]
            trainer.EPS_END = simulation_args.args.dqn_explr
            trainer.EPS_START = simulation_args.args.dqn_explr
            # trainer.LR = 1e-6
        elif policy == 'DQN':
            trainer.EPS_END = 0
            trainer.EPS_START = 0
            trainer.eval_mode = True
        elif policy == 'DQN_DUPL':
            duplication_rate = simulation_args.args.duplication_rate
            trainer.EPS_END = 0
            trainer.EPS_START = 0
            trainer.eval_mode = True
            # utilization -= duplication_rate

        for i_episode in range(NUM_TEST_EPSIODES):
            trainer.load_models(model_folder=model_folder)

            seed = BASE_TEST_SEED + i_episode

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(i_episode)
            simulation_args.set_seed(seed)

            test_data_point_monitor = experiment_runner.run_experiment(
                simulation_args.args, num_requests=NUM_TEST_REQUESTS, utilization=utilization, trainer=trainer, eval_mode=trainer.eval_mode, duplication_rate=duplication_rate)
            test_plotter.add_data(test_data_point_monitor, simulation_args.args.selection_strategy, i_episode)

            if simulation_args.args.collect_data_points or i_episode == LAST_EPOCH:
                test_data_analyzer.add_data(test_data_point_monitor, policy=policy, epoch_num=i_episode)

            if policy.startswith('DQN'):
                # Print number of DQN decisions that matched ARS
                experiment_runner.print_dqn_decision_equal_to_ars_ratio()
                print(f'Exlore actions this episode: {trainer.explore_actions_episode}')
                print(f'Exploit actions this episode: {trainer.exploit_actions_episode}')
                trainer.reset_episode_counters()

    print('Finished')
    # test_data_analyzer.run_latency_lin_reg(epoch=LAST_EPOCH)

    # Export data
    test_plotter.export_data()
    if simulation_args.args.collect_data_points:
        test_data_analyzer.export_epoch_data(epoch=LAST_EPOCH)
        test_data_analyzer.export_training_data()

    plot_collected_data(plotter=test_plotter, epoch_to_plot=LAST_EPOCH, policies_to_plot=EVAL_POLICIES_TO_RUN)

    return test_plotter.get_autotuner_objective()


def plot_collected_data(plotter: ExperimentPlot, epoch_to_plot: int, policies_to_plot: List[str]) -> None:
    plotter.generate_plots()
    plotter.save_stats_to_file()

    plotter.plot_episode(epoch=epoch_to_plot)
    for policy in policies_to_plot:
        plotter.plot_policy_episode(epoch=epoch_to_plot, policy=policy)


# TODO: Make scenarios enum and find better way to select args for them
def main(input_args=None, setting="base") -> None:
    if setting == "base":
        args = BaseArgs(input_args=input_args)
    elif setting == "heterogenous_requests_scenario":
        args = HeterogeneousRequestsArgs(input_args=input_args)
    elif setting == "heterogenous_static_service_time_scenario":
        args = StaticSlowServerArgs(input_args=input_args)
    elif setting == "time_varying_service_time_servers":
        args = TimeVaryingServerArgs(input_args=input_args)
    else:
        raise Exception(f'Unknown setting {setting}')

    EXPERIMENT_NAME = 'fixed_random'

    args.set_policy('ARS')
    args.args.exp_name = EXPERIMENT_NAME
    for utilization in [0.2, 0.3, 0.45, 0.7, 0.9]:
        # args.args.duplication_rate = duplication_rate
        args.args.utilization = utilization

        _ = rl_experiment_wrapper(args, input_args=input_args)

    # elif setting =="slow":
    #     args = SlowServerArgs(0.5,0.5, input_args=input_args)
    # elif setting =="uneven":
    #     args = SlowServerArgs(0.5,0.1, input_args=input_args)

    last = 0.0
    # Heterogeneous requests workloads
    args = HeterogeneousRequestsArgs(input_args=input_args)
    args.args.exp_name = EXPERIMENT_NAME
    for long_task_fraction in [0.2, 0.1, 0.4]:
        for duplication_rate in [0.1]:
            for utilization in [0.2, 0.3, 0.45, 0.7]:
                args.args.duplication_rate = duplication_rate
                args.args.long_tasks_fraction = long_task_fraction
                args.args.utilization = utilization
                last = rl_experiment_wrapper(args, input_args=input_args)

    return
    # Static slow server workloads
    # args = StaticSlowServerArgs(input_args=input_args)
    # args.args.exp_name = EXPERIMENT_NAME
    # args.args.epochs = 300
    # args.args.num_requests = 8000
    # args.args.num_requests_test = 60000
    # args.args.eps_decay = 400000
    # args.args.lr_scheduler_step_size = 70
    # for slow_server_slowness in [2.0, 3.0]:
    #     for dqn_explr in [0.1]:
    #         for utilization in [0.45, 0.7]:
    #             args.args.slow_server_slowness = slow_server_slowness
    #             args.args.dqn_explr = dqn_explr
    #             args.args.utilization = utilization
    #             last = rl_experiment_wrapper(args, input_args=input_args)

    args = TimeVaryingServerArgs(input_args=input_args)
    args.args.exp_name = EXPERIMENT_NAME
    args.args.epochs = 300
    args.args.num_requests = 8000
    args.args.num_requests_test = 60000
    args.args.eps_decay = 400000
    args.args.interval_param = 2500
    args.args.lr_scheduler_step_size = 70
    for time_varying_drift in [2.0, 3.0]:
        for interval in [500]:  # 200, 50
            for dqn_explr in [0.1]:
                for utilization in [0.45, 0.7]:
                    for duplication_rate in [0.05, 0.1]:
                        args.args.duplication_rate = duplication_rate
                        args.args.interval_param = interval
                        args.args.time_varying_drift = time_varying_drift
                        args.args.dqn_explr = dqn_explr
                        args.args.utilization = utilization
                        last = rl_experiment_wrapper(args, input_args=input_args)

    return last


if __name__ == '__main__':
    main(setting="base")
