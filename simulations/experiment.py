import json
import os

import torch

import server
import client
from simulator import Simulation
import workload
import random
import constants
import numpy as np
import sys
import muUpdater
from simulations.monitor import Monitor
from simulation_args import SimulationArgs
from pathlib import Path
from model_trainer import Trainer
from simulations.plotting import ExperimentPlot
from experiment_runner import ExperimentRunner
import matplotlib.pyplot as plt


def print_monitor_time_series_to_file(file_desc, prefix, monitor):
    for entry in monitor:
        file_desc.write("%s %s %s\n" % (prefix, entry[0], entry[1]))


def log_arguments(out_folder: Path):
    args_dict = vars(args.args)

    json_file_path = out_folder / 'arguments.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)


def rl_experiment_wrapper(simulation_args: SimulationArgs):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    experiment_runner = ExperimentRunner()

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer = Trainer(simulation_args.args.num_servers, long_requests_ratio=args.args.long_tasks_fraction,
                      gamma=args.args.gamma,
                      eps_decay=args.args.eps_decay, eps_start=args.args.eps_start, eps_end=args.args.eps_end,
                      tau=args.args.tau, tau_decay=args.args.tau_decay,
                      lr=args.args.lr, batch_size=args.args.batch_size)
    NUM_EPSIODES = 300
    train_plotter = ExperimentPlot()
    test_plotter = ExperimentPlot()
    to_print = False

    experiment_num = 0
    plot_path = Path('..', simulation_args.args.plot_folder, str(experiment_num))

    while os.path.isdir(plot_path):
        experiment_num += 1
        plot_path = Path('..', simulation_args.args.plot_folder, str(experiment_num))

    simulation_args.args.exp_prefix = str(experiment_num)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    simulation_args.set_print(to_print)

    log_arguments(plot_path)

    policies_to_run = [
        'ARS',
        # 'response_time',
        # 'weighted_response_time',
        'random',
        'DQN'
    ]

    print('Starting experiments')
    for policy in policies_to_run:
        simulation_args.set_policy(policy)
        # if policy == 'DQN':
        #     simulation_args.set_print(True)
        for i_episode in range(NUM_EPSIODES):
            print(i_episode)
            simulation_args.set_seed(i_episode)
            latencies = experiment_runner.run_experiment(simulation_args.args, trainer)
            train_plotter.add_data(latencies, simulation_args.args.selection_strategy, i_episode)
            if policy == 'DQN':
                # Print number of DQN decisions that matched ARS
                experiment_runner.print_dqn_decision_equal_to_ars_ratio()

                # Update LR
                trainer.scheduler.step()

                # trainer.print_weights()

                trainer.eval_mode = True
                test_latencies = experiment_runner.run_experiment(
                    simulation_args.args, trainer, eval_mode=trainer.eval_mode)
                test_plotter.add_data(test_latencies, simulation_args.args.selection_strategy, i_episode)
                trainer.eval_mode = False
            else:
                # Note that this uses the same seed at the moment
                test_plotter.add_data(latencies, simulation_args.args.selection_strategy, i_episode)

    print('Finished')
    train_plotter.export_data(out_folder=plot_path, file_name='train_data.csv')
    test_plotter.export_data(out_folder=plot_path, file_name='test_data.csv')

    trainer.plot_grads_and_losses(plot_path=plot_path)

    fig, ax = train_plotter.plot()
    plt.savefig(plot_path / 'pdfs/output_train.pdf')
    plt.savefig(plot_path / 'output_train.jpg')

    fig, ax = train_plotter.plot_quantile(0.90)
    plt.savefig(plot_path / 'pdfs/output_train_p_90.pdf')
    plt.savefig(plot_path / 'output_train_p_90.jpg')

    fig, ax = train_plotter.plot_quantile(0.95)
    plt.savefig(plot_path / 'pdfs/output_train_p_95.pdf')
    plt.savefig(plot_path / 'output_train_p_95.jpg')

    fig, ax = train_plotter.plot_quantile(0.99)
    plt.savefig(plot_path / 'pdfs/output_train_p_99.pdf')
    plt.savefig(plot_path / 'output_train_p_99.jpg')

    plt_episode = NUM_EPSIODES - 1
    fig, ax = train_plotter.plot_episode(epoch=plt_episode)
    plt.savefig(plot_path / f'pdfs/output_train_{plt_episode}_epoch.pdf')
    plt.savefig(plot_path / f'output_train_{plt_episode}_epoch.jpg')

    fig, ax = test_plotter.plot()
    plt.savefig(plot_path / 'pdfs/output.pdf')
    plt.savefig(plot_path / 'output.jpg')

    fig, ax = test_plotter.plot_quantile(0.90)
    plt.savefig(plot_path / 'pdfs/output_p_90.pdf')
    plt.savefig(plot_path / 'output_p_90.jpg')

    fig, ax = test_plotter.plot_quantile(0.95)
    plt.savefig(plot_path / 'pdfs/output_p_95.pdf')
    plt.savefig(plot_path / 'output_p_95.jpg')

    fig, ax = test_plotter.plot_quantile(0.99)
    plt.savefig(plot_path / 'pdfs/output_p_99.pdf')
    plt.savefig(plot_path / 'output_p_99.jpg')


if __name__ == '__main__':
    args = SimulationArgs()
    # args = TimeVaryingArgs(0.1,5)
    # args = SlowServerArgs(0.5,0.5)
    args.set_policy('ARS')
    rl_experiment_wrapper(args)
