import json
import os
from typing import Any, Dict, List

import torch

import random
import numpy as np
from simulation_args import BaseArgs, HeterogeneousRequestsArgs, SimulationArgs, StaticSlowServerArgs, TimeVaryingServerArgs, log_arguments
from pathlib import Path
from simulations.training.model_trainer import Trainer
from simulations.feature_data_collector import FeatureDataCollector
from simulations.monitor import Monitor
from simulations.plotting import ExperimentPlot
from experiment_runner import ExperimentRunner
from simulations.state import StateParser
from simulations.workload.workload import BaseWorkload
from simulations.workload.workload_builder import WorkloadBuilder
import constants as const

DQN_EXPLR_MAPPING = {f'DQN_EXPLR_{i}': (i / 100.0) for i in range(101)}
DQN_DUPL_MAPPING = {f'DQN_DUPL_{i}': (i / 100.0) for i in range(101)}


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


def rl_experiment_wrapper(simulation_args: SimulationArgs, train_workloads: List[BaseWorkload], test_workloads: List[BaseWorkload]) -> float:
    state_parser = StateParser(num_servers=simulation_args.args.num_servers,
                               num_request_rates=len(simulation_args.args.rate_intervals),
                               poly_feat_degree=simulation_args.args.poly_feat_degree)

    random.seed(simulation_args.args.seed)
    np.random.seed(simulation_args.args.seed)
    torch.manual_seed(simulation_args.args.seed)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer = Trainer(state_parser=state_parser, model_structure=simulation_args.args.model_structure, n_actions=simulation_args.args.num_servers, replay_always_use_newest=simulation_args.args.replay_always_use_newest, replay_memory_size=simulation_args.args.replay_memory_size,
                      summary_stats_max_size=simulation_args.args.summary_stats_max_size,
                      gamma=simulation_args.args.gamma,
                      eps_decay=simulation_args.args.eps_decay, eps_start=simulation_args.args.eps_start, eps_end=simulation_args.args.eps_end,
                      tau=simulation_args.args.tau, tau_decay=simulation_args.args.tau_decay,
                      lr=simulation_args.args.lr, batch_size=simulation_args.args.batch_size, lr_scheduler_gamma=simulation_args.args.lr_scheduler_gamma, lr_scheduler_step_size=simulation_args.args.lr_scheduler_step_size)

    out_path = create_experiment_folders(simulation_args=simulation_args, state_parser=state_parser)

    if simulation_args.args.model_folder == '':
        run_rl_training(simulation_args=simulation_args, workloads=train_workloads,
                        trainer=trainer, state_parser=state_parser, out_folder=out_path)
        model_folder = out_path / 'train' / simulation_args.args.data_folder
    else:
        model_folder = Path(simulation_args.args.model_folder)

    test_result = 0
    for test_workload in test_workloads:
        print('Running with args:')
        # TODO: Export workload config
        test_result += run_rl_test(simulation_args=simulation_args, workload=test_workload,
                                   out_folder=out_path, model_folder=model_folder, trainer=trainer, state_parser=state_parser)
    return test_result / len(test_workloads)


def run_rl_training(simulation_args: SimulationArgs, workloads: List[BaseWorkload], trainer: Trainer, state_parser: StateParser, out_folder: Path):
    NUM_EPSIODES = simulation_args.args.epochs
    LAST_EPOCH = NUM_EPSIODES - 1
    to_print = False

    # TODO: Fix / workaround
    utilization = workloads[0].utilization
    long_tasks_fraction = workloads[0].long_tasks_fraction

    # Init directories
    experiment_folder = out_folder / 'train'
    plot_path = experiment_folder / simulation_args.args.plot_folder
    data_folder = experiment_folder / simulation_args.args.data_folder
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    train_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder,
                                   utilization=utilization, long_tasks_fraction=long_tasks_fraction)
    train_data_analyzer = FeatureDataCollector(out_folder=data_folder, state_parser=state_parser)

    simulation_args.set_print(to_print)

    # Log arguments and workload config
    log_arguments(experiment_folder, simulation_args)
    for i, workload in enumerate(workloads):
        workload.to_json_file(out_folder=experiment_folder, prefix=f'{i}_')

    duplication_rate = 0.0

    print('Starting experiments')
    for policy in const.TRAIN_POLICIES_TO_RUN:
        simulation_args.set_policy(policy)
        # if policy == 'DQN':
        #     simulation_args.set_print(True)
        for i_episode in range(NUM_EPSIODES):
            print(i_episode)
            random.seed(i_episode)
            np.random.seed(i_episode)
            torch.manual_seed(i_episode)
            simulation_args.set_seed(i_episode)
            # randomly select one of the workload configs
            workload = random.choice(workloads)
            # TODO: Log workload configs used somewhere

            data_point_monitor = experiment_runner.run_experiment(
                simulation_args.args, service_time_model=simulation_args.args.service_time_model, workload=workload, trainer=trainer, duplication_rate=duplication_rate)
            train_plotter.add_data(data_point_monitor, policy, i_episode)

            # TODO: recomment
            if i_episode == LAST_EPOCH:  # simulation_args.args.collect_data_points
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
    trainer.save_models_and_stats(model_folder=data_folder)

    train_plotter.export_data()

    if simulation_args.args.collect_data_points:
        train_data_analyzer.export_epoch_data(epoch=LAST_EPOCH)
        # train_data_analyzer.export_training_data()

    trainer.plot_grads_and_losses(plot_path=plot_path, file_prefix='train')

    plot_collected_data(plotter=train_plotter, epoch_to_plot=LAST_EPOCH, policies_to_plot=const.TRAIN_POLICIES_TO_RUN)


def run_rl_test(simulation_args: SimulationArgs, workload: BaseWorkload, out_folder: Path, model_folder: Path, trainer: Trainer, state_parser: StateParser) -> float:
    BASE_TEST_SEED = 111111
    BASE_TEST_EXPLR_SEED = 222222

    experiment_runner = ExperimentRunner(state_parser=state_parser)

    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    NUM_TEST_EPSIODES = simulation_args.args.test_epochs
    LAST_EPOCH = NUM_TEST_EPSIODES - 1
    to_print = False

    EXPERIMENT = workload.to_file_name()

    experiment_folder = out_folder / EXPERIMENT
    plot_path = experiment_folder / simulation_args.args.plot_folder
    data_folder = experiment_folder / simulation_args.args.data_folder
    # model_folder = out_folder / 'train' / simulation_args.args.data_folder

    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plot_path / 'pdfs', exist_ok=True)

    utilization = workload.utilization

    test_plotter = ExperimentPlot(plot_folder=plot_path, data_folder=data_folder,
                                  utilization=utilization, long_tasks_fraction=workload.long_tasks_fraction)
    test_data_analyzer = FeatureDataCollector(out_folder=data_folder, state_parser=state_parser)

    simulation_args.set_print(to_print)

    log_arguments(experiment_folder, simulation_args)
    workload.to_json_file(out_folder=experiment_folder)

    for policy in const.EVAL_POLICIES_TO_RUN:
        simulation_args.set_policy(policy)
        print(f'Starting Test Sequence for {policy}')
        # Reset hyperparameters
        trainer.EPS_START = simulation_args.args.eps_start
        trainer.EPS_END = simulation_args.args.eps_end
        trainer.eval_mode = True

        duplication_rate = 0.0

        if policy == 'DQN':
            trainer.EPS_END = 0
            trainer.EPS_START = 0
            trainer.eval_mode = True

        for i_episode in range(NUM_TEST_EPSIODES):
            # seed = BASE_TEST_EXPLR_SEED + i_episode
            seed = BASE_TEST_SEED + i_episode

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            simulation_args.set_seed(seed)

            trainer.load_models(model_folder=model_folder)
            # Also reset model steps and stats
            trainer.reset_model_training_stats()
            print(i_episode)

            if policy.startswith('DQN_'):
                trainer.eval_mode = False
                trainer.LR = simulation_args.args.dqn_explr_lr

                if policy == 'DQN_EXPLR':
                    print(f'simulation_args.args.dqn_explr: {simulation_args.args.dqn_explr}')
                    trainer.EPS_END = simulation_args.args.dqn_explr
                    trainer.EPS_START = simulation_args.args.dqn_explr
                elif policy == 'DQN_DUPL':
                    duplication_rate = simulation_args.args.duplication_rate
                    print(f'Duplicating with rate of {duplication_rate}')
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

                test_data_point_monitor = experiment_runner.run_experiment(
                    simulation_args.args, service_time_model=simulation_args.args.test_service_time_model, workload=workload, trainer=trainer, duplication_rate=duplication_rate)

                print('After training:')
                print(trainer.task_id_to_next_state)
                test_plotter.add_data(test_data_point_monitor, policy=f'{policy}_TRAIN', epoch_num=i_episode)

                model_folder = data_folder / f'{policy}_{i_episode}'
                os.makedirs(model_folder, exist_ok=True)
                trainer.save_models_and_stats(model_folder=model_folder)

                # Print number of DQN decisions that matched ARS
                experiment_runner.print_dqn_decision_equal_to_ars_ratio()
                print(f'Exlore actions this episode: {trainer.explore_actions_episode}')
                print(f'Exploit actions this episode: {trainer.exploit_actions_episode}')
                trainer.reset_episode_counters()

                file_prefix = f'{policy}_{i_episode}'
                trainer.plot_grads_and_losses(plot_path=plot_path, file_prefix=file_prefix)

                # Turn off exploration and enter eval mode
                trainer.EPS_END = 0
                trainer.EPS_START = 0
                trainer.eval_mode = True

            seed = BASE_TEST_SEED + i_episode

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            simulation_args.set_seed(seed)

            test_data_point_monitor = experiment_runner.run_experiment(
                simulation_args.args, service_time_model=simulation_args.args.test_service_time_model, workload=workload, trainer=trainer, duplication_rate=duplication_rate)

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
        # test_data_analyzer.export_training_data()

    plot_collected_data(plotter=test_plotter, epoch_to_plot=LAST_EPOCH, policies_to_plot=const.EVAL_POLICIES_TO_RUN)

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

    EXPERIMENT_NAME = 'fixed_rate_intervals'
    SEED = args.args.seed

    config_folder = Path('..', 'configs')
    workload_builder = WorkloadBuilder(config_folder=config_folder)

    args.set_policy('ARS')
    args.args.exp_name = EXPERIMENT_NAME

    last = 0.0

    # train_workloads = workload_builder.create_train_base_workloads(
    #     long_tasks_fractions=[0.3, 0.35, 0.4],
    #     utilizations=[0.45])
    # test_workloads = workload_builder.create_test_base_workloads(
    #     long_tasks_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # args = HeterogeneousRequestsArgs(input_args=input_args)
    # args.args.exp_name = EXPERIMENT_NAME
    # args.args.epochs = 100
    # args.args.num_requests = 8000
    # args.args.num_requests_test = 8000
    # args.args.eps_decay = 180000
    # args.args.lr_scheduler_step_size = 30
    # args.args.model_folder = '/home/jonas/projects/absim/outputs/fixed_rate_intervals/0/train/data'

    # for service_time_model in ['random.expovariate']:  # 'pareto'
    #     for test_service_time_model in ['random.expovariate']:  # 'random.expovariate',
    #         for dqn_explr_lr in [1e-5, 1e-6]:
    #             args.args.seed = SEED
    #             args.args.test_service_time_model = test_service_time_model
    #             args.args.service_time_model = service_time_model
    #             args.args.dqn_explr_lr = dqn_explr_lr
    #             last = rl_experiment_wrapper(args,
    #                                          train_workloads=train_workloads, test_workloads=test_workloads)

    # test_workloads = workload_builder.create_test_base_workloads(long_tasks_fractions=[0.0, 0.2, 0.4, 0.6, 0.8])

    # args.args.model_folder = ''

    # for service_time_model in ['pareto']:  # 'pareto'
    #     for test_service_time_model in ['pareto', 'random.expovariate']:  # 'random.expovariate',
    #         for dqn_explr_lr in [1e-6]:
    #             args.args.seed = SEED
    #             args.args.test_service_time_model = test_service_time_model
    #             args.args.service_time_model = service_time_model
    #             args.args.dqn_explr_lr = dqn_explr_lr
    #             last = rl_experiment_wrapper(args,
    #                                          train_workloads=train_workloads, test_workloads=test_workloads)
    #             args.args.model_folder = '/home/jonas/projects/absim/outputs/fixed_rate_intervals/4/train/data'

    # EXPERIMENT_NAME = 'different_utilization'

    # train_workloads = workload_builder.create_train_base_workloads(
    #     long_tasks_fractions=[0.3, 0.35, 0.4], utilizations=[0.45])
    # test_workloads = workload_builder.create_test_base_workloads(
    #     long_tasks_fractions=[0.0, 0.2, 0.3, 0.4, 0.6, 0.8], utilizations=[0.7])

    # args = HeterogeneousRequestsArgs(input_args=input_args)
    # args.args.exp_name = EXPERIMENT_NAME
    # args.args.epochs = 100
    # args.args.num_requests = 8000
    # args.args.num_requests_test = 8000
    # args.args.eps_decay = 180000
    # args.args.lr_scheduler_step_size = 30
    # args.args.model_folder = '/home/jonas/projects/absim/outputs/fixed_rate_intervals/0/train/data'

    # for service_time_model in ['random.expovariate']:  # 'pareto'
    #     for test_service_time_model in ['random.expovariate']:  # 'random.expovariate',
    #         for dqn_explr_lr in [1e-5, 1e-6]:
    #             args.args.seed = SEED
    #             args.args.test_service_time_model = test_service_time_model
    #             args.args.service_time_model = service_time_model
    #             args.args.dqn_explr_lr = dqn_explr_lr
    #             last = rl_experiment_wrapper(args,
    #                                          train_workloads=train_workloads, test_workloads=test_workloads)

    # EXPERIMENT_NAME = 'new_benchmarks'

    # args = HeterogeneousRequestsArgs(input_args=input_args)
    # args.args.exp_name = EXPERIMENT_NAME
    # args.args.epochs = 50
    # args.args.num_requests = 8000
    # args.args.num_requests_test = 8000
    # args.args.eps_decay = 90000
    # args.args.lr_scheduler_step_size = 20
    # args.args.model_folder = ''

    # for service_time_model in ['random.expovariate', 'pareto']:  # 'pareto'
    #     for test_service_time_model in ['pareto', 'random.expovariate']:  # 'random.expovariate',
    #         for long_tasks_fraction in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    #             train_workloads = workload_builder.create_train_base_workloads(
    #                 long_tasks_fractions=[long_tasks_fraction],
    #                 utilizations=[0.45])
    #             test_workloads = workload_builder.create_test_base_workloads(long_tasks_fractions=[long_tasks_fraction])
    #             args.args.seed = SEED
    #             args.args.test_service_time_model = test_service_time_model
    #             args.args.service_time_model = service_time_model
    #             args.args.dqn_explr_lr = dqn_explr_lr
    #             last = rl_experiment_wrapper(args,
    #                                          train_workloads=train_workloads, test_workloads=test_workloads)

    EXPERIMENT_NAME = 'dupl_experiments'

    train_workloads = workload_builder.create_train_base_workloads(
        long_tasks_fractions=[0.3, 0.35, 0.4], utilizations=[0.45])

    test_workloads = workload_builder.create_test_var_long_tasks_workloads(
        num_requests=128000)

    const.EVAL_POLICIES_TO_RUN = ['DQN_DUPL_30', 'DQN_DUPL_35', 'DQN_DUPL_40', 'DQN_DUPL_45']

    args = HeterogeneousRequestsArgs(input_args=input_args)
    args.args.exp_name = EXPERIMENT_NAME
    args.args.epochs = 100
    args.args.eps_decay = 180000
    args.args.lr_scheduler_step_size = 30
    args.args.model_folder = '/home/jonas/projects/absim/outputs/fixed_rate_intervals/0/train/data'

    for service_time_model in ['random.expovariate']:  # 'pareto'
        for test_service_time_model in ['random.expovariate', 'pareto']:  # 'random.expovariate',
            for dqn_explr_lr in [1e-5, 1e-6]:
                if test_service_time_model == 'random.expovariate' and dqn_explr_lr == 1e-5:
                    continue
                args.args.seed = SEED
                args.args.test_service_time_model = test_service_time_model
                args.args.service_time_model = service_time_model
                args.args.dqn_explr_lr = dqn_explr_lr
                last = rl_experiment_wrapper(args,
                                             train_workloads=train_workloads, test_workloads=test_workloads)

    EXPERIMENT_NAME = 'replay_use_newest'

    test_workloads = workload_builder.create_test_var_long_tasks_workloads(
        num_requests=128000)

    const.EVAL_POLICIES_TO_RUN = [
        'round_robin',
        'ARS',
        'DQN',
        'random',
        # 'DQN_EXPLR',
        # 'DQN_DUPL'
    ] + ['DQN_DUPL_10', 'DQN_DUPL_15', 'DQN_DUPL_20', 'DQN_DUPL_25', 'DQN_DUPL_30', 'DQN_DUPL_35', 'DQN_DUPL_40', 'DQN_DUPL_45'] + ['DQN_EXPLR_0', 'DQN_EXPLR_10', 'DQN_EXPLR_15', 'DQN_EXPLR_20', 'DQN_EXPLR_25']

    args = HeterogeneousRequestsArgs(input_args=input_args)
    args.args.exp_name = EXPERIMENT_NAME
    args.args.epochs = 100
    args.args.eps_decay = 180000
    args.args.lr_scheduler_step_size = 30
    args.args.model_folder = '/home/jonas/projects/absim/outputs/fixed_rate_intervals/0/train/data'
    args.args.replay_always_use_newest = True

    for service_time_model in ['random.expovariate']:  # 'pareto'
        for test_service_time_model in ['random.expovariate', 'pareto']:  # 'random.expovariate',
            for dqn_explr_lr in [1e-5, 1e-6]:
                if test_service_time_model == 'random.expovariate' and dqn_explr_lr == 1e-5:
                    continue
                args.args.seed = SEED
                args.args.test_service_time_model = test_service_time_model
                args.args.service_time_model = service_time_model
                args.args.dqn_explr_lr = dqn_explr_lr
                last = rl_experiment_wrapper(args,
                                             train_workloads=train_workloads, test_workloads=test_workloads)

    # EXPERIMENT_NAME = 'var_workload_32000_with_util_less_trained_model'

    # const.EVAL_POLICIES_TO_RUN = [
    #     # 'round_robin',
    #     'ARS',
    #     'DQN',
    #     'random',
    #     'DQN_DUPL'
    # ] + ['DQN_EXPLR_0', 'DQN_EXPLR_10', 'DQN_EXPLR_15', 'DQN_EXPLR_20']

    # train_workloads = workload_builder.create_train_base_workloads(
    #     long_tasks_fractions=[0.3], utilizations=[0.45])

    # test_workloads = workload_builder.create_test_var_long_tasks_workloads(
    #     num_requests=128000)

    # args = HeterogeneousRequestsArgs(input_args=input_args)
    # args.args.exp_name = EXPERIMENT_NAME
    # args.args.epochs = 100
    # args.args.eps_decay = 180000
    # args.args.lr_scheduler_step_size = 30
    # args.args.model_folder = ''

    # for service_time_model in ['random.expovariate']:  # 'pareto'
    #     for test_service_time_model in ['random.expovariate', 'pareto']:  # 'random.expovariate',
    #         for dqn_explr_lr in [1e-5, 1e-6]:
    #             args.args.seed = SEED
    #             args.args.test_service_time_model = test_service_time_model
    #             args.args.service_time_model = service_time_model
    #             args.args.dqn_explr_lr = dqn_explr_lr
    #             last = rl_experiment_wrapper(args,
    #                                          train_workloads=train_workloads, test_workloads=test_workloads)
    #             if args.args.model_folder == '':
    #                 args.args.model_folder = '/home/jonas/projects/absim/outputs/var_workload_32000_with_util_less_trained_model/0/train/data'

    # EXPERIMENT_NAME = 'var_workload_32000_with_util_alternate_start'

    # const.EVAL_POLICIES_TO_RUN = [
    #     # 'round_robin',
    #     'ARS',
    #     'DQN',
    #     'random',
    #     'DQN_DUPL'
    # ] + ['DQN_EXPLR_0', 'DQN_EXPLR_10', 'DQN_EXPLR_15', 'DQN_EXPLR_20']

    # train_workloads = workload_builder.create_train_base_workloads(
    #     long_tasks_fractions=[0.5, 0.55], utilizations=[0.55])

    # test_workloads = workload_builder.create_test_var_long_tasks_workloads(
    #     num_requests=128000)

    # args = HeterogeneousRequestsArgs(input_args=input_args)
    # args.args.exp_name = EXPERIMENT_NAME
    # args.args.epochs = 100
    # args.args.eps_decay = 180000
    # args.args.lr_scheduler_step_size = 30
    # args.args.model_folder = ''

    # for service_time_model in ['random.expovariate']:  # 'pareto'
    #     for test_service_time_model in ['random.expovariate', 'pareto']:  # 'random.expovariate',
    #         for dqn_explr_lr in [1e-5, 1e-6]:
    #             args.args.seed = SEED
    #             args.args.test_service_time_model = test_service_time_model
    #             args.args.service_time_model = service_time_model
    #             args.args.dqn_explr_lr = dqn_explr_lr
    #             last = rl_experiment_wrapper(args,
    #                                          train_workloads=train_workloads, test_workloads=test_workloads)
    #             if args.args.model_folder == '':
    #                 args.args.model_folder = '/home/jonas/projects/absim/outputs/var_workload_32000_with_util_alternate_start/0/train/data'

    return

    EXPERIMENT_NAME = 'shorter_train'

    train_workloads = workload_builder.create_train_base_workloads(
        long_tasks_fractions=[0.3, 0.35, 0.4],
        utilizations=[0.45])
    test_workloads = workload_builder.create_test_base_workloads(
        long_tasks_fractions=[0.0, 0.2, 0.4, 0.6, 0.8])

    args = HeterogeneousRequestsArgs(input_args=input_args)
    args.args.exp_name = EXPERIMENT_NAME
    args.args.epochs = 50
    args.args.num_requests = 8000
    args.args.num_requests_test = 8000
    args.args.eps_decay = 90000
    args.args.lr_scheduler_step_size = 20
    args.args.model_folder = ''

    for service_time_model in ['random.expovariate']:  # 'pareto'
        for test_service_time_model in ['pareto', 'random.expovariate']:  # 'random.expovariate',
            for dqn_explr_lr in [1e-6]:
                # for _ in [1, 2, 3]:
                args.args.seed = SEED
                args.args.test_service_time_model = test_service_time_model
                args.args.service_time_model = service_time_model
                args.args.dqn_explr_lr = dqn_explr_lr
                last = rl_experiment_wrapper(args,
                                             train_workloads=train_workloads, test_workloads=test_workloads)

    train_workloads = workload_builder.create_train_base_workloads(
        long_tasks_fractions=[0.3, 0.35, 0.4],
        utilizations=[0.45, 0.6, 0.7])
    test_workloads = workload_builder.create_test_base_workloads(
        long_tasks_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8])

    EXPERIMENT_NAME = 'slightly_varied_training_shorter_test'

    args = HeterogeneousRequestsArgs(input_args=input_args)
    args.args.exp_name = EXPERIMENT_NAME
    args.args.epochs = 700
    args.args.num_requests = 2000
    args.args.num_requests_test = 8000
    args.args.eps_decay = 300000
    args.args.lr_scheduler_step_size = 150
    args.args.model_folder = ''

    for service_time_model in ['random.expovariate']:  # 'pareto'
        for test_service_time_model in ['random.expovariate', 'pareto']:
            for dqn_explr_lr in [1e-6]:
                # for _ in [1, 2, 3]:
                args.args.seed = SEED
                args.args.test_service_time_model = test_service_time_model
                args.args.service_time_model = service_time_model
                args.args.dqn_explr_lr = dqn_explr_lr
                last = rl_experiment_wrapper(args,
                                             train_workloads=train_workloads, test_workloads=test_workloads)

    train_workloads = workload_builder.create_train_base_workloads(
        long_tasks_fractions=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        utilizations=[0.45, 0.7, 1.2])
    test_workloads = workload_builder.create_test_base_workloads(
        long_tasks_fractions=[0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9])

    EXPERIMENT_NAME = 'pareto_repeat'

    # TODO: Run same experiments for pareto
    args = HeterogeneousRequestsArgs(input_args=input_args)
    args.args.exp_name = EXPERIMENT_NAME
    args.args.epochs = 700
    args.args.num_requests = 2000
    args.args.eps_decay = 300000
    args.args.model_folder = ''

    for service_time_model in ['pareto']:  # 'pareto'
        for test_service_time_model in ['pareto']:
            for dqn_explr_lr in [1e-6]:
                # for _ in [1, 2, 3]:
                args.args.seed = SEED
                args.args.test_service_time_model = test_service_time_model
                args.args.service_time_model = service_time_model
                args.args.dqn_explr_lr = dqn_explr_lr
                last = rl_experiment_wrapper(args,
                                             train_workloads=train_workloads, test_workloads=test_workloads)

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
    #            args.args.seed = SEED
    #             args.args.slow_server_slowness = slow_server_slowness
    #             args.args.dqn_explr = dqn_explr
    #             args.args.utilization = utilization
    #             last = rl_experiment_wrapper(args)

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
                        args.args.seed = SEED
                        args.args.duplication_rate = duplication_rate
                        args.args.interval_param = interval
                        args.args.time_varying_drift = time_varying_drift
                        args.args.dqn_explr = dqn_explr
                        last = rl_experiment_wrapper(args, input_args=input_args)

    return last


if __name__ == '__main__':
    main(setting="base")
