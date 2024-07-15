import argparse
import json
from pathlib import Path
from typing import Any, List


class SimulationArgs:
    def __init__(self, input_args=None) -> None:
        parser = argparse.ArgumentParser(description='Absinthe sim.')
        parser.add_argument('--num_clients', nargs='?',
                            type=int, default=1, help='Number of clients')
        parser.add_argument('--num_servers', nargs='?',
                            type=int, default=5, help='Number of servers')
        parser.add_argument('--num_workload', nargs='?',
                            type=int, default=1, help='Number of workload generators. Seems to distribute the '
                                                      'tasks out to different clients.')

        parser.add_argument('--replication_factor', nargs='?',
                            type=int, default=5, help='Replication factor (# of choices)')
        parser.add_argument('--selection_strategy', nargs='?',
                            type=str, default="ARS", help='Policy to use for replica selection')
        parser.add_argument('--shadow_read_ratio', nargs='?',
                            type=float, default=0.0, help='Controls the probability of sending a shadow read '
                                                          '(idk exactly what this is, it seems to be a function '
                                                          'that sends requests out to non-chosen servers to force '
                                                          'an update.')
        parser.add_argument('--rate_interval', nargs='?',
                            type=int, default=10, help='Unclear what this one does')
        parser.add_argument('--cubic_c', nargs='?',
                            type=float, default=0.000004,
                            help='Controls sending rate of client (Called gamma in paper)')
        parser.add_argument('--cubic_smax', nargs='?',
                            type=float, default=10, help='Controls sending rate of client. ')
        parser.add_argument('--cubic_beta', nargs='?',
                            type=float, default=0.2, help='Controls sending rate of client')
        parser.add_argument('--hysterisis_factor', nargs='?',
                            type=float, default=2, help='Hysteresis period before another rate change')
        parser.add_argument('--backpressure', action='store_true',
                            default=False, help='Adds backpressure mode which waits once rate limits are reached')
        parser.add_argument('--access_pattern', nargs='?',
                            type=str, default="uniform",
                            help='Key access pattern of requests, e.g., zipfian will cause '
                                 'requests to desire a subset of replica sets')

        parser.add_argument('--exp_name', nargs='?', type=str, default="Name of a set of experiments")

        parser.add_argument('--seed', nargs='?',
                            type=int, default=25072014)
        parser.add_argument('--simulation_duration', nargs='?',
                            type=int, default=10000000, help='Time that experiment takes, '
                                                           'note that if this is too low and numRequests is too high, '
                                                           'it will error')

        # Folders
        parser.add_argument('--data_folder', nargs='?', type=str, default="data")
        parser.add_argument('--plot_folder', nargs='?', type=str, default="plots")
        parser.add_argument('--output_folder',  nargs='?', type=str, default="/data1/outputs")
        parser.add_argument('--model_folder', nargs='?', type=str, default="")

        parser.add_argument('--demand_skew', nargs='?',
                            type=float, default=0, help='Skews clients such that some clients send many'
                                                        ' more requests than others')
        parser.add_argument('--high_demand_fraction', nargs='?',
                            type=float, default=0, help='Fraction of the high demand clients')

        # Slow server setting parameters
        parser.add_argument('--slow_server_fraction', nargs='?',
                            type=float, default=0.0, help='Fraction of slow servers '
                            '(expScenario=heterogenousStaticServiceTimeScenario)')
        parser.add_argument('--slow_server_slowness', nargs='?',
                            type=float, default=2.0, help='Service time change (> 1 make server slower, < 1 makes servers faster)'
                            '(expScenario=heterogenousStaticServiceTimeScenario)')

        # Time varying slow server
        parser.add_argument('--interval_param', nargs='?',
                            type=float, default=5000.0, help='Interval between which server service times change '
                            '(expScenario=timeVaryingServiceTimeServers)')
        parser.add_argument('--time_varying_drift', nargs='?',
                            type=float, default=2.0, help='Service time change (> 1 make server slower, < 1 makes servers faster) '
                                                          '(expScenario=timeVaryingServiceTimeServers)')

        # General Feature parameters
        parser.add_argument('--rate_intervals', nargs='+', default=[100, 50, 10])
        parser.add_argument('--print', action='store_true',
                            default=False, help='Prints latency at the end of the experiment')
        parser.add_argument('--poly_feat_degree', nargs='?',
                            type=float, default=2, help='Degree of created polynomial and interaction features')
        parser.add_argument('--server_concurrency', nargs='?',
                            type=int, default=2, help='Amount of resources per server.')
        parser.add_argument('--service_time', nargs='?',
                            type=float, default=4, help='Mean? service time per server')
        parser.add_argument('--service_time_model', nargs='?',
                            type=str, default="random.expovariate", help='Distribution of service time on server (random.expovariate | constant | math.sin | pareto')
        parser.add_argument('--test_service_time_model', nargs='?',
                            type=str, default="random.expovariate", help='Distribution of service time on server (random.expovariate | constant | math.sin | pareto')

        parser.add_argument('--exp_scenario', nargs='?',
                            type=str, default="base",
                            help='Defines some scenarios for experiments such as \n'
                                 '[base] - default setting\n'
                                 '[heterogenous_requests_scenario] - fraction of requests is long\n'
                                 '[heterogenous_static_nw_delay] - fraction of servers have slow nw\n'
                                 '[multipleServiceTimeServers] - increasing mean service time '
                                 'based on server index\n'
                                 '[heterogenousStaticServiceTimeScenario] - '
                                 'fraction of servers are slower\n'
                                 '[time_varying_service_time_servers] - servers change service times')

        # Slow network setting parameters
        parser.add_argument('--nw_latency_base', nargs='?',
                            type=float, default=0.25, help='Seems to be the time it takes to deliver requests?')
        parser.add_argument('--nw_latency_mu', nargs='?',
                            type=float, default=0.02, help='Seems to be the time it takes to deliver requests?')
        parser.add_argument('--nw_latency_sigma', nargs='?',
                            type=float, default=0.01, help='Seems to be the time it takes to deliver requests?')
        parser.add_argument('--slow_nw_server_fraction', nargs='?',
                            type=float, default=0.4, help='Fraction of servers with slow network'
                                                          '(expScenario=heterogenous_static_nw_delay)')
        parser.add_argument('--slow_nw_server_slowness', nargs='?',
                            type=float, default=10,
                            help='How many times slower than normal the networks of slowed servers are '
                                 '(expScenario=heterogenous_static_nw_delay)')

        # Heterogeneous Tasks workload
        parser.add_argument('--long_task_added_service_time', nargs='?',
                            type=float, default=35,
                            help='How many times slower than short tasks the long tasks take ')

        # RL Training
        parser.add_argument('--epochs', nargs='?',
                            type=int, default=10, help='Number of training epochs')

        # RL Model evaluation
        parser.add_argument('--test_epochs', nargs='?',
                            type=int, default=3, help='Number of test epochs')
        parser.add_argument('--dqn_explr', nargs='?',
                            type=float, default=0.1, help='Exploration used by DQN_EXPLR')
        parser.add_argument('--dqn_explr_lr', nargs='?',
                            type=float, default=1e-6, help='LR used by DQN_EXPLR')

        # RL Model parameters
        parser.add_argument('--model_structure', nargs='?',
                            type=str, default="linear", help='Model layers (linear | three_layers)')
        parser.add_argument('--gamma', nargs='?',
                            type=float, default=0.99, help='Model trainer argument')
        parser.add_argument('--lr', nargs='?',
                            type=float, default=1e-5, help='Model trainer argument')
        parser.add_argument('--tau', nargs='?',
                            type=float, default=0.005, help='Model trainer argument')
        parser.add_argument('--eps_decay', nargs='?',
                            type=int, default=40000, help='Model trainer argument')
        parser.add_argument('--batch_size', nargs='?',
                            type=int, default=32, help='Model trainer argument')
        parser.add_argument('--tau_decay', nargs='?',
                            type=int, default=10000000, help='Model trainer argument')
        parser.add_argument('--eps_start', nargs='?',
                            type=float, default=0.9, help='Model trainer argument')
        parser.add_argument('--eps_end', nargs='?',
                            type=float, default=0.01, help='Model trainer argument')
        parser.add_argument('--lr_scheduler_step_size', nargs='?',
                            type=int, default=70, help='Model trainer argument')
        parser.add_argument('--lr_scheduler_gamma', nargs='?',
                            type=float, default=0.5, help='Model trainer argument')
        parser.add_argument('--summary_stats_max_size', nargs='?',
                            type=int, default=1000, help='Number of stats collected for normalizing')

        # Offline training parameters
        parser.add_argument('--offline_train_batch_size', nargs='?',
                            type=int, default=2000, help='Number of requests after which the model is retrained')
        parser.add_argument('--offline_expert_data', nargs='?', type=str, default="",
                            help='Location of the expert data for offline training.')
        parser.add_argument('--offline_model', nargs='?', type=str, default="",
                            help='Location of the offline model.')
        parser.add_argument('--collect_train_data', action='store_true',
                            default=False, help='If true, log and save all data collected for offline training later')
        parser.add_argument('--train_from_expert_data', action='store_true',
                            default=False, help='If true, log and save all data collected for offline training later')
        parser.add_argument('--offline_train_epoch_len', nargs='?',
                            type=int, default=4000, help='Number of training steps done per offline training / retraining')
        parser.add_argument('--train_policy', nargs='?',
                            type=str, default=None, help='Policy used during training')

        # ReplayMemory parametes
        # TODO: Remove or reactive
        parser.add_argument('--replay_memory_size', nargs='?',
                            type=int, default=10000, help='Replay memory size for online training, offline training uses all collected data!')
        parser.add_argument('--replay_always_use_newest', action='store_true',
                            default=False, help='if true, always add newest transition to sample (see https://arxiv.org/pdf/1712.01275)')

        parser.add_argument('--duplication_rate', nargs='?',
                            type=float, default=0.1, help='Number of requests to duplicate')

        parser.add_argument('--num_permutations', nargs='?',
                            type=int, default=1, help='Number of permutations added per request')

        parser.add_argument('--clipping_value', nargs='?',
                            type=int, default=1, help='Gradient clipping value')
        self.parser = parser
        print(input_args)
        self.args = parser.parse_args(args=input_args)
        print('After')
        print(self.args)

        assert self.args.replication_factor == self.args.num_servers, ('Replication factor is not equal to number of'
                                                                       ' servers, i.e., #actions != #servers')

    def set_policy(self, policy):
        self.args.selection_strategy = policy

    def set_print(self, to_print):
        self.args.print = to_print

    def set_seed(self, seed):
        self.args.seed = seed

    def set_num_requests(self, num_requests):
        self.args.num_requests = num_requests


class TimeVaryingArgs(SimulationArgs):
    def __init__(self, interval_param: float = 1.0, time_varying_drift: float = 1.0, input_args=None) -> None:
        super().__init__(input_args=input_args)
        self.args.exp_scenario = 'time_varying_service_time_servers'
        self.args.interval_param = interval_param
        self.args.time_varying_drift = time_varying_drift

    def to_string(self) -> str:
        return f'time_varying_service_time_servers_{self.args.interval_param}_{self.args.time_varying_drift}'


class StaticSlowServerArgs(SimulationArgs):
    def __init__(self, slow_server_fraction: float = 0.2, slow_server_slowness: float = 3.0, input_args=None) -> None:
        super().__init__(input_args=input_args)
        self.args.exp_scenario = 'heterogenous_static_service_time_scenario'
        self.args.slow_server_fraction = slow_server_fraction
        self.args.slow_server_slowness = slow_server_slowness

    def to_string(self) -> str:
        return f'heterogenous_static_service_time_scenario_{self.args.slow_server_fraction}_{self.args.slow_server_slowness}'


class TimeVaryingServerArgs(SimulationArgs):
    def __init__(self, interval_param: float = 5000.0, time_varying_drift: float = 2.0, input_args=None) -> None:
        super().__init__(input_args=input_args)
        print('Initialized')
        self.args.exp_scenario = 'time_varying_service_time_servers'
        self.args.interval_param = interval_param
        self.args.time_varying_drift = time_varying_drift

    def to_string(self) -> str:
        return f'time_varying_service_time_servers_{self.args.interval_param}_{self.args.time_varying_drift}'


class HeterogeneousRequestsArgs(SimulationArgs):
    def __init__(self, long_task_added_service_time: int = 35, input_args=None) -> None:
        super().__init__(input_args=input_args)
        print('Initialized')
        self.args.exp_scenario = 'heterogenous_requests_scenario'
        self.args.long_task_added_service_time = long_task_added_service_time

    def to_string(self) -> str:
        return f'heterogenous_requests_scenario_{self.args.long_task_added_service_time}'


class BaseArgs(SimulationArgs):
    def __init__(self, input_args=None) -> None:
        super().__init__(input_args=input_args)
        self.args.exp_scenario = 'base'

    def to_string(self) -> str:
        return 'base'


def log_arguments(out_folder: Path, args):
    args_dict = vars(args.args)

    json_file_path = out_folder / 'arguments.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
