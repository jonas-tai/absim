import json
import os
import sys
from typing import List

import torch

import random
import numpy as np
from simulation_args import HeterogeneousRequestsArgs, SimulationArgs, log_arguments
from pathlib import Path
from simulations.training.model_trainer import Trainer
from simulations.plotting import ExperimentPlot
from experiment_runner import ExperimentRunner
from simulations.state import StateParser
from simulations.training.offline_model_trainer import OfflineTrainer
from simulations.training.training_data_collector import TrainingDataCollector
from simulations.workload.workload import BaseWorkload
from simulations.workload.workload_builder import WorkloadBuilder
import constants as const

DQN_EXPLR_MAPPING = {f'DQN_EXPLR_{i}': i / 100.0 for i in range(101)}
DQN_EXPLR_MAPPING.update({f'DQN_EXPLR_{i}_TRAIN': i / 100 for i in range(101)})
DQN_EXPLR_MAPPING.update({f'OFFLINE_DQN_EXPLR_{i}': i / 100.0 for i in range(101)})
DQN_EXPLR_MAPPING.update({f'OFFLINE_DQN_EXPLR_{i}_TRAIN': i / 100 for i in range(101)})

DQN_DUPL_MAPPING = {f'DQN_DUPL_{i}': i / 100.0 for i in range(101)}
DQN_DUPL_MAPPING.update({f'DQN_DUPL_{i}_TRAIN': i / 100 for i in range(101)})
DQN_DUPL_MAPPING.update({f'OFFLINE_DQN_DUPL_{i}': i / 100.0 for i in range(101)})
DQN_DUPL_MAPPING.update({f'OFFLINE_DQN_DUPL_{i}_TRAIN': i / 100 for i in range(101)})


ARS_DUPL_MAPPING = {f'ARS_DUPL_{i}': i / 100.0 for i in range(101)}

BASE_TEST_SEED = 111111
BASE_TEST_EXPLR_SEED = 222222


def print_monitor_time_series_to_file(file_desc, prefix, monitor) -> None:
    for entry in monitor:
        file_desc.write("%s %s %s\n" % (prefix, entry[0], entry[1]))


def create_experiment_folders(simulation_args: SimulationArgs, state_parser: StateParser) -> Path:
    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    experiment_num = 0
    if simulation_args.args.exp_name != "":
        out_path = Path(simulation_args.args.output_folder, simulation_args.args.exp_name, str(experiment_num))
        while os.path.isdir(out_path):
            experiment_num += 1
            out_path = Path(simulation_args.args.output_folder,
                            simulation_args.args.exp_name, str(experiment_num))
    else:
        out_path = Path(simulation_args.args.output_folder, str(experiment_num))
        while os.path.isdir(out_path):
            experiment_num += 1
            out_path = Path(simulation_args.args.output_folder, str(experiment_num))

    print(out_path)
    os.makedirs(out_path, exist_ok=True)

    return out_path


def rl_experiment_wrapper(simulation_args: SimulationArgs, train_workloads: List[BaseWorkload], test_workloads: List[BaseWorkload]) -> float:
    state_parser = StateParser(num_servers=simulation_args.args.num_servers,
                               num_request_rates=len(simulation_args.args.rate_intervals),
                               poly_feat_degree=simulation_args.args.poly_feat_degree)

    random.seed(simulation_args.args.seed)
    np.random.seed(simulation_args.args.seed)
    torch.manual_seed(simulation_args.args.seed)

    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    # print('Warning: pytorch uses non-deterministic algorithms!')
    torch.use_deterministic_algorithms(True)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer = Trainer(state_parser=state_parser, model_structure=simulation_args.args.model_structure, n_actions=simulation_args.args.num_servers, replay_always_use_newest=simulation_args.args.replay_always_use_newest, replay_memory_size=simulation_args.args.replay_memory_size,
                      summary_stats_max_size=simulation_args.args.summary_stats_max_size,
                      gamma=simulation_args.args.gamma,
                      eps_decay=simulation_args.args.eps_decay, eps_start=simulation_args.args.eps_start, eps_end=simulation_args.args.eps_end,
                      tau=simulation_args.args.tau, tau_decay=simulation_args.args.tau_decay,
                      lr=simulation_args.args.lr, batch_size=simulation_args.args.batch_size, lr_scheduler_gamma=simulation_args.args.lr_scheduler_gamma, lr_scheduler_step_size=simulation_args.args.lr_scheduler_step_size,
                      clipping_value=simulation_args.args.clipping_value)

    offline_trainer = OfflineTrainer(state_parser=state_parser,
                                     model_structure=simulation_args.args.model_structure,
                                     n_actions=simulation_args.args.num_servers,
                                     replay_always_use_newest=simulation_args.args.replay_always_use_newest,
                                     offline_train_epoch_len=simulation_args.args.offline_train_epoch_len,
                                     gamma=simulation_args.args.gamma,
                                     eps_decay=simulation_args.args.eps_decay,
                                     eps_start=simulation_args.args.eps_start,
                                     eps_end=simulation_args.args.eps_end,
                                     use_sliding_retrain_memory=simulation_args.args.use_sliding_retrain_memory,
                                     sliding_window_mem_size=simulation_args.args.replay_memory_size,
                                     tau=simulation_args.args.tau,
                                     add_retrain_to_expert_buffer=simulation_args.args.add_retrain_to_expert_buffer,
                                     replay_mem_retrain_expert_fraction=simulation_args.args.replay_mem_retrain_expert_fraction,
                                     tau_decay=simulation_args.args.tau_decay,
                                     lr=simulation_args.args.lr,
                                     batch_size=simulation_args.args.batch_size, clipping_value=simulation_args.args.clipping_value)

    out_path = create_experiment_folders(simulation_args=simulation_args, state_parser=state_parser)

    training_data_folder = out_path / 'collected_training_data'

    training_data_collector = TrainingDataCollector(
        state_parser=state_parser, n_actions=simulation_args.args.num_servers, summary_stats_max_size=simulation_args.args.summary_stats_max_size, offline_trainer=offline_trainer, offline_train_batch_size=simulation_args.args.offline_train_batch_size, data_folder=training_data_folder, num_permutations=simulation_args.args.num_permutations)

    assert simulation_args.args.offline_expert_data == '' or simulation_args.args.offline_model == ''

    if len(const.TRAIN_POLICIES_TO_RUN) > 0:
        run_rl_training(simulation_args=simulation_args, workloads=train_workloads, offline_trainer=offline_trainer,
                        trainer=trainer, state_parser=state_parser, out_folder=out_path, training_data_collector=training_data_collector)

    if simulation_args.args.model_folder == '':
        model_folder = out_path / 'train' / simulation_args.args.data_folder
    else:
        model_folder = Path(simulation_args.args.model_folder)

    expert_data_folder = Path(simulation_args.args.offline_expert_data)

    if simulation_args.args.train_from_expert_data:
        # TODO: Change to None / empty path?
        if expert_data_folder == Path(''):
            print('Using collected training data from this run')
            expert_data_folder = training_data_folder
        print(expert_data_folder)
        offline_model_folder = train_model_from_expert_data(
            simulation_args=simulation_args, offline_trainer=offline_trainer, out_path=out_path, expert_data_folder=expert_data_folder, train_data_collector=training_data_collector)
    else:
        offline_model_folder = Path(simulation_args.args.offline_model)

    simulation_args.args.collect_train_data = False
    trainer.set_model_folder(model_folder=model_folder)
    offline_trainer.set_model_folder(model_folder=offline_model_folder)

    run_rl_tests(simulation_args=simulation_args, workloads=test_workloads, out_folder=out_path, trainer=trainer,
                 offline_trainer=offline_trainer, state_parser=state_parser, training_data_collector=training_data_collector)

    return 0


def train_model_from_expert_data(simulation_args: SimulationArgs, offline_trainer: OfflineTrainer, out_path: Path, expert_data_folder: Path, train_data_collector: TrainingDataCollector) -> None:
    offline_train_out_folder = out_path / 'offline_train' / simulation_args.args.data_folder
    offline_plot_folder = out_path / 'offline_train' / simulation_args.args.plot_folder

    os.makedirs(offline_train_out_folder, exist_ok=True)
    os.makedirs(offline_plot_folder, exist_ok=True)

    # Offline training data given, load data and train model on it
    # TODO: Make this parameter?
    offline_trainer.init_expert_data_from_csv(expert_data_folder=expert_data_folder)
    # train_data_collector.init_expert_data_from_data_collector()

    for _ in range(simulation_args.args.epochs):
        offline_trainer.train_model_from_expert_data_epoch(train_steps=simulation_args.args.offline_train_epoch_len)
        # TODO: Add eval of Offline_DQN as test here to monitor performance
    offline_trainer.save_models_and_stats(offline_train_out_folder)
    offline_trainer.plot_grads_and_losses(plot_path=offline_plot_folder, file_prefix='offline_train')
    offline_model_folder = offline_train_out_folder
    return offline_model_folder


def run_rl_training(simulation_args: SimulationArgs, workloads: List[BaseWorkload], trainer: Trainer, offline_trainer: OfflineTrainer, state_parser: StateParser, training_data_collector: TrainingDataCollector, out_folder: Path):

    if len(workloads) == 0:
        return
    NUM_EPSIODES = simulation_args.args.epochs
    LAST_EPOCH = NUM_EPSIODES - 1

    # Init directories
    experiment_folder = out_folder / 'train'
    plot_path = experiment_folder / simulation_args.args.plot_folder
    data_folder = experiment_folder / simulation_args.args.data_folder
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    # TODO: Fix / workaround
    utilization = workloads[0].utilization
    long_tasks_fraction = workloads[0].long_tasks_fraction

    experiment_runner = ExperimentRunner(state_parser=state_parser, trainer=trainer, offline_trainer=offline_trainer)

    train_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder, state_parser=state_parser,
                                   utilization=utilization, long_tasks_fraction=long_tasks_fraction, retrain_interval=simulation_args.args.offline_train_batch_size)

    # Log arguments and workload config
    log_arguments(experiment_folder, simulation_args)
    for i, workload in enumerate(workloads):
        workload.to_json_file(out_folder=experiment_folder, prefix=f'{i}_')

    duplication_rate = 0.0

    print('Starting experiments')
    print(const.TRAIN_POLICIES_TO_RUN)
    for policy in const.TRAIN_POLICIES_TO_RUN:
        simulation_args.set_policy(policy)
        for i_episode in range(NUM_EPSIODES):
            print(i_episode)
            random.seed(i_episode)
            np.random.seed(i_episode)
            torch.manual_seed(i_episode)
            simulation_args.set_seed(i_episode)
            # randomly select one of the workload configs
            workload = random.choice(workloads)
            # TODO: Log sampled workload configs used somewhere

            data_point_monitor = experiment_runner.run_experiment(
                simulation_args.args, service_time_model=simulation_args.args.service_time_model, workload=workload, duplication_rate=duplication_rate, training_data_collector=training_data_collector)
            train_plotter.add_data(data_point_monitor, policy, i_episode)

            if simulation_args.args.collect_train_data:
                training_data_collector.end_train_episode()

            if policy == 'DQN':
                # Print number of DQN decisions that matched ARS
                experiment_runner.print_dqn_decision_equal_to_ars_ratio()

                # Update LR
                trainer.scheduler.step()

                trainer.reset_episode_counters()
                # trainer.print_weights()

    print('Finished')
    # train_data_analyzer.run_latency_lin_reg(epoch=LAST_EPOCH)
    trainer.save_models_and_stats(model_folder=data_folder)

    train_plotter.export_data()

    if simulation_args.args.collect_train_data:
        # TODO: Make this flag?
        training_data_collector.save_training_data()
        training_data_collector.save_training_data_collector_stats()

    trainer.plot_grads_and_losses(plot_path=plot_path, file_prefix='train')

    plot_collected_data(plotter=train_plotter, epoch_to_plot=LAST_EPOCH, policies_to_plot=const.TRAIN_POLICIES_TO_RUN)


def run_rl_tests(simulation_args: SimulationArgs, workloads: List[BaseWorkload], out_folder: Path, trainer: Trainer, offline_trainer: OfflineTrainer, state_parser: StateParser, training_data_collector: TrainingDataCollector) -> None:
    const.NUM_TEST_EPSIODES = simulation_args.args.test_epochs

    for test_workload in workloads:
        # Start the models and etc.
        # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        LAST_EPOCH = const.NUM_TEST_EPSIODES - 1

        EXPERIMENT = test_workload.to_file_name()

        experiment_folder = out_folder / EXPERIMENT
        plot_path = experiment_folder / simulation_args.args.plot_folder
        data_folder = experiment_folder / simulation_args.args.data_folder
        # model_folder = out_folder / 'train' / simulation_args.args.data_folder

        os.makedirs(plot_path, exist_ok=True)
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(plot_path / 'pdfs', exist_ok=True)

        utilization = test_workload.utilization

        test_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder, state_parser=state_parser,
                                      utilization=utilization, long_tasks_fraction=test_workload.long_tasks_fraction,
                                      retrain_interval=simulation_args.args.offline_train_batch_size)

        experiment_runner = ExperimentRunner(
            state_parser=state_parser, trainer=trainer, offline_trainer=offline_trainer)

        log_arguments(experiment_folder, simulation_args)
        test_workload.to_json_file(out_folder=experiment_folder)

        for policy in const.EVAL_POLICIES_TO_RUN:
            simulation_args.set_policy(policy)
            print(f'Starting Test Sequence for {policy}')

            if policy.startswith('OFFLINE_'):

                run_rl_offline_test(simulation_args=simulation_args, workload=test_workload, policy=policy,
                                    plot_folder=plot_path, data_folder=data_folder, offline_trainer=offline_trainer, experiment_runner=experiment_runner, training_data_collector=training_data_collector, test_plotter=test_plotter)
            elif policy.startswith('DQN'):
                run_rl_dqn_test(simulation_args=simulation_args, workload=test_workload, policy=policy,
                                plot_folder=plot_path, data_folder=data_folder, trainer=trainer, experiment_runner=experiment_runner, training_data_collector=training_data_collector, test_plotter=test_plotter)
            else:
                for i_episode in range(const.NUM_TEST_EPSIODES):
                    duplication_rate = 0
                    seed = BASE_TEST_SEED + i_episode

                    if policy.startswith('ARS_DUPL_'):
                        duplication_rate = ARS_DUPL_MAPPING[policy]
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    simulation_args.set_seed(seed)

                    test_data_point_monitor = experiment_runner.run_experiment(
                        simulation_args.args, service_time_model=simulation_args.args.test_service_time_model, workload=test_workload, duplication_rate=duplication_rate, training_data_collector=training_data_collector)
                    print(f'{i_episode}, {policy}')
                    test_plotter.add_data(test_data_point_monitor, policy=policy, epoch_num=i_episode)

        # Export data
        test_plotter.export_data()

        # TOOD: Remove hacky solution
        plot_collected_data(plotter=test_plotter, epoch_to_plot=LAST_EPOCH,
                            policies_to_plot=test_plotter.df['Policy'].unique())

        if simulation_args.args.collect_train_data:
            training_data_collector.save_training_data()
            training_data_collector.save_training_data_collector_stats()
        print('Finished workload')


def run_rl_offline_test(simulation_args: SimulationArgs, workload: BaseWorkload, plot_folder: Path, data_folder: Path, policy: str, offline_trainer: OfflineTrainer, experiment_runner: ExperimentRunner, training_data_collector: TrainingDataCollector, test_plotter: ExperimentPlot) -> float:
    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    duplication_rate = 0.0

    # Reset hyperparameters
    offline_trainer.EPS_START = 0
    offline_trainer.EPS_END = 0
    offline_trainer.do_active_retraining = True

    if not policy.endswith('_TRAIN'):
        offline_trainer.do_active_retraining = False

    if policy == 'OFFLINE_DQN':
        offline_trainer.EPS_END = 0
        offline_trainer.EPS_START = 0
    elif policy.startswith('OFFLINE_DQN_EXPLR_'):
        print(f'simulation_args.args.dqn_explr: {simulation_args.args.dqn_explr}')
        print(f'OFFLINE_DQN_EXPLR_MAPPING: {DQN_EXPLR_MAPPING[policy]}')
        offline_trainer.EPS_END = DQN_EXPLR_MAPPING[policy]
        offline_trainer.EPS_START = DQN_EXPLR_MAPPING[policy]
    elif policy.startswith('OFFLINE_DQN_DUPL_'):
        print(f'DQN_DUPL_MAPPING: {DQN_DUPL_MAPPING[policy]}')
        duplication_rate = DQN_DUPL_MAPPING[policy]
        offline_trainer.EPS_END = 0
        offline_trainer.EPS_START = 0
    else:
        raise Exception(f'Invalid policy for offline RL adapting: {policy}')

    for i_episode in range(const.NUM_TEST_EPSIODES):
        seed = BASE_TEST_SEED + i_episode

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        simulation_args.set_seed(seed)

        offline_trainer.load_models()
        # Also reset model steps and stats
        offline_trainer.reset_training_stats()
        print(i_episode)

        test_data_point_monitor = experiment_runner.run_experiment(
            simulation_args.args, service_time_model=simulation_args.args.test_service_time_model, workload=workload, duplication_rate=duplication_rate, training_data_collector=training_data_collector)

        if policy.endswith('_TRAIN'):
            offline_model_folder = data_folder / f'{policy}_{i_episode}'
            os.makedirs(offline_model_folder, exist_ok=True)
            offline_trainer.save_models_and_stats(model_folder=offline_model_folder)
            file_prefix = f'{policy}_{i_episode}'
            offline_trainer.plot_grads_and_losses(plot_path=plot_folder, file_prefix=file_prefix)

        print(f'Adding offline data: {policy}')
        test_plotter.add_data(test_data_point_monitor, policy=policy, epoch_num=i_episode)

        # Print number of DQN decisions that matched ARS
        experiment_runner.print_dqn_decision_equal_to_ars_ratio()
        print(f'Exlore actions this episode: {offline_trainer.explore_actions_episode}')
        print(f'Exploit actions this episode: {offline_trainer.exploit_actions_episode}')
        offline_trainer.reset_episode_counters()
        training_data_collector.end_test_episode()


def run_rl_dqn_test(simulation_args: SimulationArgs, workload: BaseWorkload, policy: str, plot_folder: Path,
                    data_folder: Path, trainer: Trainer, experiment_runner: ExperimentRunner,
                    training_data_collector: TrainingDataCollector, test_plotter: ExperimentPlot) -> float:
    trainer.eval_mode = False
    trainer.LR = simulation_args.args.dqn_explr_lr
    duplication_rate = 0.0

    if not policy.endswith('_TRAIN'):
        trainer.eval_mode = True

    if policy == 'DQN':
        trainer.EPS_END = 0
        trainer.EPS_START = 0
    elif policy.startswith('DQN_EXPLR_'):
        print(f'simulation_args.args.dqn_explr: {simulation_args.args.dqn_explr}')
        print(f'DQN_EXPLR_MAPPING: {DQN_EXPLR_MAPPING[policy]}')
        trainer.EPS_END = DQN_EXPLR_MAPPING[policy]
        trainer.EPS_START = DQN_EXPLR_MAPPING[policy]
    elif policy.startswith('DQN_DUPL_'):
        print(f'DQN_DUPL_MAPPING: {DQN_DUPL_MAPPING[policy]}')
        duplication_rate = DQN_DUPL_MAPPING[policy]
        trainer.EPS_END = 0
        trainer.EPS_START = 0
    else:
        raise Exception(f'Invalid policy for adapting: {policy}')

    for i_episode in range(const.NUM_TEST_EPSIODES):
        seed = BASE_TEST_SEED + i_episode

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        simulation_args.set_seed(seed)

        trainer.load_models()
        # Also reset model steps and stats
        trainer.reset_model_training_stats()
        print(i_episode)

        test_data_point_monitor = experiment_runner.run_experiment(
            simulation_args.args, service_time_model=simulation_args.args.test_service_time_model, workload=workload, duplication_rate=duplication_rate, training_data_collector=training_data_collector)

        policy_str = policy
        if policy != 'DQN':
            policy_str = f'{policy_str}_TRAIN'

            model_folder = data_folder / f'{policy}_{i_episode}'
            os.makedirs(model_folder, exist_ok=True)
            trainer.save_models_and_stats(model_folder=model_folder)

            file_prefix = f'{policy}_{i_episode}'
            trainer.plot_grads_and_losses(plot_path=plot_folder, file_prefix=file_prefix)

        test_plotter.add_data(test_data_point_monitor, policy=policy_str, epoch_num=i_episode)

        # Print number of DQN decisions that matched ARS
        experiment_runner.print_dqn_decision_equal_to_ars_ratio()
        print(f'Exlore actions this episode: {trainer.explore_actions_episode}')
        print(f'Exploit actions this episode: {trainer.exploit_actions_episode}')
        trainer.reset_episode_counters()

    # Reset hyperparameters
    trainer.EPS_START = simulation_args.args.eps_start
    trainer.EPS_END = simulation_args.args.eps_end
    trainer.eval_mode = False


def plot_collected_data(plotter: ExperimentPlot, epoch_to_plot: int, policies_to_plot: List[str]) -> None:
    plotter.generate_plots()
    plotter.save_stats_to_file()

    plotter.plot_episode(epoch=epoch_to_plot)
    for policy in policies_to_plot:
        plotter.plot_policy_episode(epoch=epoch_to_plot, policy=policy)


def main(input_args=None) -> None:

    config_folder = Path('./', 'configs')
    workload_builder = WorkloadBuilder(config_folder=config_folder)

    EXPERIMENT_NAME = 'sliding_window_experiment'

    train_workloads = []
    # workload_builder.create_train_base_workloads(
    # long_tasks_fractions=[0.3, 0.35, 0.4], utilizations=[0.7], num_requests=48000)

    lhs_workloads = workload_builder.create_test_base_workloads(
        long_tasks_fractions=[0.3], utilizations=[0.7], num_requests=10000)
    rhs_workloads = workload_builder.create_test_base_workloads(
        long_tasks_fractions=[0.0, 0.2, 0.5, 0.8], utilizations=[0.5, 0.7], num_requests=64000)
    test_workloads = workload_builder.create_chained_workloads(
        first_workloads=lhs_workloads, second_workloads=rhs_workloads)
    # workload_builder.create_test_var_long_tasks_workloads(num_requests=350000)
    # workload_builder.create_test_base_workloads(
    #     utilizations=[0.45, 0.7],
    #     long_tasks_fractions=[0.0, 0.1, 0.3, 0.5, 0.7],
    #     num_requests=16000)

    # test_workloads = workload_builder.create_test_base_workloads(utilizations=[0.6, 0.7, 0.8], long_tasks_fractions=[
    #     0.0, 0.1, 0.3, 0.6, 0.7], num_requests=8000)

    const.EVAL_POLICIES_TO_RUN = [
        'ARS', 'OFFLINE_DQN',
        'OFFLINE_DQN_EXPLR_0_TRAIN',
        'OFFLINE_DQN_EXPLR_10_TRAIN', 'OFFLINE_DQN_EXPLR_15_TRAIN', 'OFFLINE_DQN_EXPLR_20_TRAIN',
        'OFFLINE_DQN_DUPL_10_TRAIN', 'OFFLINE_DQN_DUPL_15_TRAIN',
        'OFFLINE_DQN_DUPL_20_TRAIN']
    # 'ARS', 'OFFLINE_DQN',
    # 'OFFLINE_DQN_EXPLR_20_TRAIN', 'OFFLINE_DQN_EXPLR_30_TRAIN',
    # 'OFFLINE_DQN_DUPL_20_TRAIN', 'OFFLINE_DQN_DUPL_30_TRAIN'
    args = HeterogeneousRequestsArgs(input_args=input_args)
    SEED = args.args.seed
    if args.args.train_policy is not None:
        print(args.args.train_policy)
        const.TRAIN_POLICIES_TO_RUN = [args.args.train_policy]
    else:
        const.TRAIN_POLICIES_TO_RUN = []

    args.args.exp_name = EXPERIMENT_NAME
    args.args.eps_decay = 180000
    args.args.lr_scheduler_step_size = 30
    args.args.replay_always_use_newest = False
    args.args.collect_train_data = False
    args.args.train_from_expert_data = False
    args.args.model_folder = '/home/jonas/projects/absim/outputs/fixed_memory_not_use_latest/0/train/data'
    # args.args.offline_model = ''  # '/home/jonas/projects/absim/outputs/fixed_memory_not_use_latest/0/train/data'
    args.args.lr = 1e-6
    # args.args.replay_memory_size = 5000

    # '/home/jonas/projects/absim/outputs/collect_offline_data/0/collected_training_data'
    # '/home/jonas/projects/absim/outputs/offline_parameter_search/0/offline_train/data', '/home/jonas/projects/absim/outputs/offline_parameter_search/15/offline_train/data', '/home/jonas/projects/absim/outputs/offline_parameter_search/9/offline_train/data'
    args.args.seed = SEED
    last = rl_experiment_wrapper(args,
                                 train_workloads=train_workloads, test_workloads=test_workloads)

    return


if __name__ == '__main__':
    main(sys.argv[1:])
