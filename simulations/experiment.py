import json
import os
from typing import Any, List

import torch

import random
import numpy as np
from simulation_args import BaseArgs, HeterogeneousRequestsArgs, SimulationArgs, log_arguments
from pathlib import Path
from model_trainer import Trainer
from simulations.feature_data_collector import FeatureDataCollector
from simulations.monitor import Monitor
from simulations.plotting import ExperimentPlot
from experiment_runner import ExperimentRunner
from simulations.state import StateParser


TRAIN_POLICIES_TO_RUN = [
    'round_robin',
    'ARS',
    # 'response_time',
    # 'weighted_response_time',
    # 'random',
    'DQN'
]


EVAL_POLICIES_TO_RUN = [
    'round_robin',
    'ARS',
    'DQN',
    'DQN_EXPLR'
]


def print_monitor_time_series_to_file(file_desc, prefix, monitor) -> None:
    for entry in monitor:
        file_desc.write("%s %s %s\n" % (prefix, entry[0], entry[1]))


def create_experiment_folders(simulation_args: SimulationArgs, state_parser: StateParser) -> Path:
    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer = Trainer(state_parser=state_parser, model_structure=simulation_args.args.model_structure, n_actions=simulation_args.args.num_servers,
                      gamma=simulation_args.args.gamma,
                      eps_decay=simulation_args.args.eps_decay, eps_start=simulation_args.args.eps_start, eps_end=simulation_args.args.eps_end,
                      tau=simulation_args.args.tau, tau_decay=simulation_args.args.tau_decay,
                      lr=simulation_args.args.lr, batch_size=simulation_args.args.batch_size, lr_scheduler_gamma=simulation_args.args.lr_scheduler_gamma, lr_scheduler_step_size=simulation_args.args.lr_scheduler_step_size)

    experiment_num = 0
    out_path = Path('..', simulation_args.args.output_folder, str(experiment_num))

    while os.path.isdir(out_path):
        experiment_num += 1
        out_path = Path('..', simulation_args.args.output_folder, str(experiment_num))

    simulation_args.args.exp_prefix = str(experiment_num)
    os.makedirs(out_path, exist_ok=True)

    return out_path


def rl_experiment_wrapper(simulation_args: SimulationArgs, input_args: Any | None) -> None:
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
    run_rl_test(simulation_args=simulation_args, out_folder=out_path,
                experiment='test', trainer=trainer, state_parser=state_parser)
    additional_test_args = [HeterogeneousRequestsArgs(input_args=input_args, long_tasks_fraction=0.8,
                                                      long_task_added_service_time=200)]
    for args in additional_test_args:
        print('Running with args:')
        print(args)
        run_rl_test(simulation_args=args, out_folder=out_path,
                    experiment=args.to_string(), trainer=trainer, state_parser=state_parser)


def run_rl_training(simulation_args: SimulationArgs, trainer: Trainer, state_parser: StateParser, out_folder: Path):

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    NUM_EPSIODES = simulation_args.args.epochs
    LAST_EPOCH = NUM_EPSIODES - 1
    to_print = False

    experiment_folder = out_folder / 'train'
    plot_path = experiment_folder / simulation_args.args.plot_folder
    data_folder = experiment_folder / simulation_args.args.data_folder
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    train_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder)
    train_data_analyzer = FeatureDataCollector(out_folder=data_folder, state_parser=state_parser)

    simulation_args.set_print(to_print)

    log_arguments(experiment_folder, simulation_args)

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

            data_point_monitor = experiment_runner.run_experiment(simulation_args.args, trainer)
            train_plotter.add_data(data_point_monitor, policy, i_episode)

            if simulation_args.args.collect_data_points or i_episode == LAST_EPOCH:
                train_data_analyzer.add_data(data_point_monitor, policy=policy, epoch_num=i_episode)

            if policy == 'DQN':
                # Print number of DQN decisions that matched ARS
                experiment_runner.print_dqn_decision_equal_to_ars_ratio()

                # Update LR
                trainer.scheduler.step()
                # trainer.print_weights()

    print('Finished')
    # train_data_analyzer.run_latency_lin_reg(epoch=LAST_EPOCH)

    train_plotter.export_data(file_name='train_data.csv')

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

    simulation_args.set_num_requests(NUM_TEST_REQUESTS)

    experiment_folder = out_folder / experiment
    plot_path = experiment_folder / simulation_args.args.plot_folder
    data_folder = experiment_folder / simulation_args.args.data_folder
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    test_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder)
    test_data_analyzer = FeatureDataCollector(out_folder=data_folder, state_parser=state_parser)

    simulation_args.set_print(to_print)

    log_arguments(experiment_folder, simulation_args)

    print('Starting Test Sequence')
    for policy in EVAL_POLICIES_TO_RUN:
        simulation_args.set_policy(policy)

        if policy == 'DQN_EXPLR':
            trainer.eval_mode = False
            trainer.EPS_END = simulation_args.args.dqn_explr
            trainer.EPS_START = simulation_args.args.dqn_explr
        elif policy == 'DQN':
            trainer.EPS_START = simulation_args.args.eps_start
            trainer.EPS_END = simulation_args.args.eps_end
            trainer.eval_mode = True

        for i_episode in range(NUM_TEST_EPSIODES):
            seed = BASE_TEST_SEED + i_episode

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(i_episode)
            simulation_args.set_seed(seed)

            test_data_point_monitor = experiment_runner.run_experiment(
                simulation_args.args, trainer, eval_mode=trainer.eval_mode)
            test_plotter.add_data(test_data_point_monitor, simulation_args.args.selection_strategy, i_episode)

            if simulation_args.args.collect_data_points or i_episode == LAST_EPOCH:
                test_data_analyzer.add_data(test_data_point_monitor, policy=policy, epoch_num=i_episode)

            if policy == 'DQN':
                # Print number of DQN decisions that matched ARS
                experiment_runner.print_dqn_decision_equal_to_ars_ratio()

    print('Finished')
    # test_data_analyzer.run_latency_lin_reg(epoch=LAST_EPOCH)

    # Export data
    test_plotter.export_data(file_name='test_data.csv')
    if simulation_args.args.collect_data_points:
        test_data_analyzer.export_epoch_data(epoch=LAST_EPOCH)
        test_data_analyzer.export_training_data()

    plot_collected_data(plotter=test_plotter, epoch_to_plot=LAST_EPOCH, policies_to_plot=EVAL_POLICIES_TO_RUN)

    return test_plotter.get_autotuner_objective()


def plot_collected_data(plotter: ExperimentPlot, epoch_to_plot: int, policies_to_plot: List[str]) -> None:
    fig, ax = plotter.plot_latency()
    fig, ax = plotter.plot_quantile(0.90)
    fig, ax = plotter.plot_quantile(0.95)
    fig, ax = plotter.plot_quantile(0.99)

    fig, ax = plotter.plot_episode(epoch=epoch_to_plot)
    for policy in policies_to_plot:
        fig, ax = plotter.plot_policy_episode(epoch=epoch_to_plot, policy=policy)


def main(input_args=None, setting="base"):
    if setting == "base":
        args = BaseArgs(input_args=input_args)
    elif setting == "heterogenous_requests_scenario":
        args = HeterogeneousRequestsArgs(input_args=input_args)
    else:
        raise Exception(f'Unknown setting {setting}')
    # elif setting =="slow":
    #     args = SlowServerArgs(0.5,0.5, input_args=input_args)
    # elif setting =="uneven":
    #     args = SlowServerArgs(0.5,0.1, input_args=input_args)
    args.set_policy('ARS')
    return rl_experiment_wrapper(args, input_args=input_args)


if __name__ == '__main__':
    main(setting="heterogenous_requests_scenario")
