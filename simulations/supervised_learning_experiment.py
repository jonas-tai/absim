import json
import os

import torch

import random
import numpy as np
from simulation_args import SimulationArgs, log_arguments
from pathlib import Path
from model_trainer import Trainer
from simulations.feature_data_collector import FeatureDataCollector
from simulations.plotting import ExperimentPlot
from experiment_runner import ExperimentRunner
from simulations.state import StateParser
from simulations.supervised_model_trainer import SupervisedModelTrainer


def supervised_learning_wrapper(simulation_args: SimulationArgs):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state_parser = StateParser(num_servers=simulation_args.args.num_servers,
                               long_tasks_fraction=simulation_args.args.long_tasks_fraction,
                               num_request_rates=len(simulation_args.args.rate_intervals),
                               poly_feat_degree=simulation_args.args.poly_feat_degree)

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    NUM_EPSIODES = 10
    LAST_EPOCH = NUM_EPSIODES - 1
    SEED = 1
    to_print = False

    experiment_num = 0
    plot_path = Path('..', simulation_args.args.plot_folder, 'supervised_learning', str(experiment_num))

    while os.path.isdir(plot_path):
        experiment_num += 1
        plot_path = Path('..', simulation_args.args.plot_folder, str(experiment_num))

    simulation_args.args.exp_prefix = str(experiment_num)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    data_path = Path('/home/jonas/projects/absim/plots/112/data/train_data_points.csv')

    trainer = SupervisedModelTrainer(state_parser=state_parser, lr=args.args.lr, batch_size=args.args.batch_size,
                                     n_labels=simulation_args.args.num_servers, out_folder=plot_path, data_path=data_path, seed=1)

    simulation_args.set_print(to_print)
    log_arguments(plot_path, simulation_args)

    print('Starting experiments')
    trainer.train_model(epochs=NUM_EPSIODES)
    print('Finished')

    trainer.test_model()

    trainer.plot_grads_and_losses(plot_path=plot_path)


if __name__ == '__main__':
    args = SimulationArgs()
    args.set_policy('ARS')
    supervised_learning_wrapper(args)
