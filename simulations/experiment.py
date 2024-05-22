import json
import os

import torch

import random
import numpy as np
from simulation_args import SimulationArgs, log_arguments
from pathlib import Path
from model_trainer import Trainer
from simulations.feature_data_collector import FeatureDataCollector
from simulations.monitor import Monitor
from simulations.plotting import ExperimentPlot
from experiment_runner import ExperimentRunner
from simulations.state import StateParser


POLICIES_TO_RUN = [
    'round_robin',
    'ARS',
    # 'response_time',
    # 'weighted_response_time',
    # 'random',
    'DQN'
]


def print_monitor_time_series_to_file(file_desc, prefix, monitor) -> None:
    for entry in monitor:
        file_desc.write("%s %s %s\n" % (prefix, entry[0], entry[1]))


def rl_experiment_wrapper(simulation_args: SimulationArgs) -> None:
    state_parser = StateParser(num_servers=simulation_args.args.num_servers,
                               long_requests_ratio=simulation_args.args.long_tasks_fraction,
                               num_request_rates=len(simulation_args.args.rate_intervals),
                               poly_feat_degree=simulation_args.args.poly_feat_degree)

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer = Trainer(state_parser=state_parser, model_structure=simulation_args.args.model_structure, n_actions=simulation_args.args.num_servers,
                      gamma=simulation_args.args.gamma,
                      eps_decay=simulation_args.args.eps_decay, eps_start=simulation_args.args.eps_start, eps_end=simulation_args.args.eps_end,
                      tau=simulation_args.args.tau, tau_decay=simulation_args.args.tau_decay,
                      lr=simulation_args.args.lr, batch_size=simulation_args.args.batch_size, lr_scheduler_gamma=simulation_args.args.lr_scheduler_gamma, lr_scheduler_step_size=simulation_args.args.lr_scheduler_step_size)
    NUM_EPSIODES = simulation_args.args.epochs
    LAST_EPOCH = NUM_EPSIODES - 1
    to_print = False

    experiment_num = 0
    plot_path = Path('..', simulation_args.args.plot_folder, str(experiment_num))

    while os.path.isdir(plot_path):
        experiment_num += 1
        plot_path = Path('..', simulation_args.args.plot_folder, str(experiment_num))

    simulation_args.args.exp_prefix = str(experiment_num)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    train_plotter = ExperimentPlot(out_folder=plot_path, is_train_data=True)
    train_data_analyzer = FeatureDataCollector(out_folder=plot_path, state_parser=state_parser, is_train_data=True)

    simulation_args.set_print(to_print)

    log_arguments(plot_path, simulation_args)

    print('Starting experiments')
    for policy in POLICIES_TO_RUN:
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
    train_data_analyzer.run_latency_lin_reg(epoch=LAST_EPOCH)

    train_plotter.export_data(file_name='train_data.csv')

    if simulation_args.args.collect_data_points:
        train_data_analyzer.export_epoch_data(epoch=LAST_EPOCH)
        train_data_analyzer.export_training_data()

    trainer.plot_grads_and_losses(plot_path=plot_path)

    fig, ax = train_plotter.plot_latency()
    fig, ax = train_plotter.plot_quantile(0.90)
    fig, ax = train_plotter.plot_quantile(0.95)
    fig, ax = train_plotter.plot_quantile(0.99)

    fig, ax = train_plotter.plot_episode(epoch=LAST_EPOCH)
    for policy in POLICIES_TO_RUN:
        fig, ax = train_plotter.plot_policy_episode(epoch=LAST_EPOCH, policy=policy)

    return run_rl_test(simulation_args=simulation_args, experiment_num=experiment_num, trainer=trainer)


def run_rl_test(simulation_args: SimulationArgs, experiment_num: int, trainer: Trainer):
    BASE_TEST_SEED = 1111111

    state_parser = StateParser(num_servers=simulation_args.args.num_servers,
                               long_requests_ratio=simulation_args.args.long_tasks_fraction,
                               num_request_rates=len(simulation_args.args.rate_intervals),
                               poly_feat_degree=simulation_args.args.poly_feat_degree)

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer.eval_mode = True

    NUM_TEST_EPSIODES = simulation_args.args.test_epochs
    NUM_TEST_REQUESTS = simulation_args.args.num_requests_test
    LAST_EPOCH = NUM_TEST_EPSIODES - 1
    to_print = False

    simulation_args.set_num_requests(NUM_TEST_REQUESTS)

    plot_path = Path('..', simulation_args.args.plot_folder, str(experiment_num))

    simulation_args.args.exp_prefix = str(experiment_num)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    test_plotter = ExperimentPlot(out_folder=plot_path, is_train_data=False)
    test_data_analyzer = FeatureDataCollector(out_folder=plot_path, state_parser=state_parser, is_train_data=False)

    simulation_args.set_print(to_print)

    # TODO: Make separate test folder and log test args there separately
    # log_arguments(plot_path, simulation_args)

    print('Starting Test Sequence')
    for policy in POLICIES_TO_RUN:
        simulation_args.set_policy(policy)
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

    test_plotter.export_data(file_name='test_data.csv')

    if simulation_args.args.collect_data_points:
        test_data_analyzer.export_epoch_data(epoch=LAST_EPOCH)
        test_data_analyzer.export_training_data()

    fig, ax = test_plotter.plot_latency()
    fig, ax = test_plotter.plot_quantile(0.90)
    fig, ax = test_plotter.plot_quantile(0.95)
    fig, ax = test_plotter.plot_quantile(0.99)

    fig, ax = test_plotter.plot_episode(epoch=LAST_EPOCH)
    for policy in POLICIES_TO_RUN:
        fig, ax = test_plotter.plot_policy_episode(epoch=LAST_EPOCH, policy=policy)
    return test_plotter.get_autotuner_objective()


def main(input_args=None, setting="sim"):
    if setting == "sim":
        args = SimulationArgs(input_args=input_args)
    # elif setting =="slow":
    #     args = SlowServerArgs(0.5,0.5, input_args=input_args)
    # elif setting =="uneven":
    #     args = SlowServerArgs(0.5,0.1, input_args=input_args)
    args.set_policy('ARS')
    return rl_experiment_wrapper(args)


if __name__ == '__main__':
    main(setting="sim")
