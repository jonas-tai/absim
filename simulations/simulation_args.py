import argparse


class SimulationArgs:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Absinthe sim.')
        parser.add_argument('--num_clients', nargs='?',
                            type=int, default=1, help='Number of clients')
        parser.add_argument('--num_servers', nargs='?',
                            type=int, default=5, help='Number of servers')
        parser.add_argument('--num_workload', nargs='?',
                            type=int, default=1, help='Number of workload generators. Seems to distribute the '
                                                      'tasks out to different clients.')
        parser.add_argument('--server_concurrency', nargs='?',
                            type=int, default=1, help='Amount of resources per server.')
        parser.add_argument('--service_time', nargs='?',
                            type=float, default=2, help='Mean? service time per server')
        parser.add_argument('--workload_model', nargs='?',
                            type=str, default="poisson", help='Arrival model of requests from client')
        parser.add_argument('--utilization', nargs='?',
                            type=float, default=1.0, help='Arrival rate of requests')
        parser.add_argument('--service_time_model', nargs='?',
                            type=str, default="random.expovariate", help='Distribution of service time on server')
        parser.add_argument('--replication_factor', nargs='?',
                            type=int, default=5, help='Replication factor (# of choices)')
        parser.add_argument('--selection_strategy', nargs='?',
                            type=str, default="expDelay", help='Policy to use for replica selection')
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
        parser.add_argument('--nw_latency_base', nargs='?',
                            type=float, default=0.0, help='Seems to be the time it takes to deliver requests?')
        parser.add_argument('--nw_latency_mu', nargs='?',
                            type=float, default=0.0, help='Seems to be the time it takes to deliver requests?')
        parser.add_argument('--nw_latency_sigma', nargs='?',
                            type=float, default=0.0, help='Seems to be the time it takes to deliver requests?')
        parser.add_argument('--exp_prefix', nargs='?',
                            type=str, default="7")
        parser.add_argument('--seed', nargs='?',
                            type=int, default=25072014)
        parser.add_argument('--simulation_duration', nargs='?',
                            type=int, default=10000, help='Time that experiment takes, '
                                                         'note that if this is too low and numRequests is too high, '
                                                         'it will error')
        parser.add_argument('--num_requests', nargs='?',
                            type=int, default=400, help='Number of requests')
        parser.add_argument('--log_folder', nargs='?', type=str, default="logs")
        parser.add_argument('--plot_folder', nargs='?', type=str, default="plots")

        parser.add_argument('--exp_scenario', nargs='?',
                            type=str, default="base", help='Defines some scenarios for experiments such as \n'
                                                           '[base] - default setting\n'
                                                           '[multipleServiceTimeServers] - increasing mean service time '
                                                           'based on server index\n'
                                                           '[heterogenousStaticServiceTimeScenario] - '
                                                           'fraction of servers are slower\n'
                                                           '[timeVaryingServiceTimeServers] - servers change service times')
        parser.add_argument('--demand_skew', nargs='?',
                            type=float, default=0, help='Skews clients such that some clients send many'
                                                        ' more requests than others')
        parser.add_argument('--high_demand_fraction', nargs='?',
                            type=float, default=0, help='Fraction of the high demand clients')
        parser.add_argument('--slow_server_fraction', nargs='?',
                            type=float, default=0, help='Fraction of slow servers '
                                                        '(expScenario=heterogenousStaticServiceTimeScenario)')
        parser.add_argument('--slow_server_slowness', nargs='?',
                            type=float, default=0, help='How slow those slowed servers are '
                                                        '(expScenario=heterogenousStaticServiceTimeScenario)')
        parser.add_argument('--interval_param', nargs='?',
                            type=float, default=0.0, help='Interval between which server service times change '
                                                          '(expScenario=timeVaryingServiceTimeServers)')
        parser.add_argument('--time_varying_drift', nargs='?',
                            type=float, default=0.0, help='How much service times change '
                                                          '(expScenario=timeVaryingServiceTimeServers)')
        parser.add_argument('--rate_intervals', nargs='+', default=[1000, 500, 100])
        parser.add_argument('--print', action='store_true',
                            default=True, help='Prints latency at the end of the experiment')

        self.parser = parser
        self.args = parser.parse_args()

        assert self.args.replication_factor == self.args.num_servers, ('Replication factor is not equal to number of'
                                                                     ' servers, i.e., #actions != #servers')

    def set_policy(self, policy):
        self.args.selection_strategy = policy

    def set_print(self, to_print):
        self.args.print = to_print

    def set_seed(self, seed):
        self.args.seed = seed


class TimeVaryingArgs(SimulationArgs):
    def __init__(self, interval_param=1.0, time_varying_drift=1.0):
        super().__init__()
        self.args.exp_scenario = 'time_varying_service_time_servers'
        self.args.interval_param = interval_param
        self.args.time_varying_drift = time_varying_drift


class SlowServerArgs(SimulationArgs):
    def __init__(self, slow_server_fraction=0.5, slow_server_slowness=0.5):
        super().__init__()
        self.args.exp_scenario = 'heterogenous_static_service_time_scenario'
        self.args.slow_server_fraction = slow_server_fraction
        self.args.slow_server_slowness = slow_server_slowness
