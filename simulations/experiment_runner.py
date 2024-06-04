from typing import List
import server
import client
from simulations.state import StateParser
from simulator import Simulation
import workload
import constants
import sys
import simulations.mu_updater as mu_updater
from simulations.monitor import Monitor
from pathlib import Path
from model_trainer import Trainer


class ExperimentRunner:
    def __init__(self, state_parser: StateParser) -> None:
        self.servers: List[server.Server] = []
        self.clients: List[client.Client] = []
        self.workload_gens: List[workload.Workload] = []
        self.state_parser = state_parser

    def reset_stats(self) -> None:
        self.servers = []
        self.clients = []
        self.workload_gens = []

    def print_dqn_decision_equal_to_ars_ratio(self, client_index: int = 0) -> None:
        # Print how often DQN matched ARS in the last run
        ratio = self.clients[client_index].dqn_decision_equal_to_ars / self.clients[client_index].requests_handled
        print(f'DQN matched ARS for {ratio * 100}% of decisions')

    def run_experiment(self, args, num_requests: int, utilization: float, trainer: Trainer = None, duplication_rate: float = 0.0, eval_mode=False) -> Monitor:
        self.reset_stats()

        # Set the random seed
        simulation = Simulation()
        simulation.set_seed(args.seed)

        constants.NW_LATENCY_BASE = args.nw_latency_base
        constants.NW_LATENCY_MU = args.nw_latency_mu
        constants.NW_LATENCY_SIGMA = args.nw_latency_sigma
        constants.NUMBER_OF_CLIENTS = args.num_clients

        assert args.exp_scenario != ""
        assert utilization > 0

        service_rate_per_server = []
        if args.exp_scenario == "base" or args.exp_scenario == 'heterogenous_requests_scenario':
            # Start the servers
            for i in range(args.num_servers):
                serv = server.Server(i,
                                     resource_capacity=args.server_concurrency,
                                     service_time=(args.service_time),
                                     service_time_model=args.service_time_model,
                                     simulation=simulation,
                                     long_task_added_service_time=args.long_task_added_service_time)
                self.servers.append(serv)
        elif args.exp_scenario == "multipleServiceTimeServers":
            # Start the servers
            for i in range(args.num_servers):
                serv = server.Server(i,
                                     resource_capacity=args.server_concurrency,
                                     service_time=((i + 1) * args.service_time),
                                     service_time_model=args.service_time_model,
                                     simulation=simulation,
                                     long_task_added_service_time=args.long_task_added_service_time)
                self.servers.append(serv)
        elif args.exp_scenario == "heterogenous_static_service_time_scenario":
            base_service_time = args.service_time

            assert 0 <= args.slow_server_fraction < 1.0
            assert 0 <= args.slow_server_slowness
            assert not (args.slow_server_slowness == 0 and args.slow_server_fraction != 0)
            assert not (args.slow_server_slowness != 0 and args.slow_server_fraction == 0)
            # TODO: Fix this for heterogeneous requests
            assert args.long_tasks_fraction == 0

            if args.slow_server_fraction > 0.0:
                '''
                Note, this is different from the initial implementation, which sets the fast server rate such that
                the average service time is the service time specified
                '''
                slow_server_rate = 1 / float(base_service_time) * (1 / args.slow_server_slowness)

                num_slow_servers = int(args.slow_server_fraction * args.num_servers)
                num_fast_servers = args.num_servers - num_slow_servers

                slow_server_rates = [slow_server_rate] * num_slow_servers

                fast_server_rate = (1 / float(args.service_time))
                fast_server_rates = [fast_server_rate] * num_fast_servers

                service_rate_per_server = slow_server_rates + fast_server_rates

                server_slow_assignment = [True] * num_slow_servers + [False] * num_fast_servers
            else:
                service_rate_per_server = [1 / float(args.service_time)] * args.num_servers

            simulation.random.shuffle(server_slow_assignment)
            # print(sum(serviceRatePerServer), (1/float(baseServiceTime)) * args.num_servers)

            # We dont scale the rate to the average rate
            # assert sum(service_rate_per_server) > 0.99 * \
            #     (1 / float(base_service_time)) * args.num_servers
            # assert sum(service_rate_per_server) <= \
            #     (1 / float(base_service_time)) * args.num_servers

            # Start the servers
            for i in range(args.num_servers):
                # Service time changes modified via service_time_factor
                service_time = base_service_time

                serv = server.Server(i,
                                     resource_capacity=args.server_concurrency,
                                     service_time=service_time,
                                     service_time_model=args.service_time_model,
                                     simulation=simulation,
                                     long_task_added_service_time=args.long_task_added_service_time)
                if server_slow_assignment[i]:
                    print(f'Slow server with factor: {args.slow_server_slowness}')
                    serv.SERVICE_TIME_FACTOR = args.slow_server_slowness
                self.servers.append(serv)

        elif args.exp_scenario == "time_varying_service_time_servers":
            assert args.interval_param != 0.0
            assert args.time_varying_drift != 0.0

            # Start the servers
            for i in range(args.num_servers):
                serv = server.Server(i,
                                     resource_capacity=args.server_concurrency,
                                     service_time=args.service_time,
                                     service_time_model=args.service_time_model,
                                     simulation=simulation,
                                     long_task_added_service_time=args.long_task_added_service_time)
                mup = mu_updater.MuUpdater(serv,
                                           args.interval_param,
                                           args.service_time,
                                           args.time_varying_drift,
                                           simulation)
                simulation.process(mup.run())
                self.servers.append(serv)
        elif args.exp_scenario == "heterogenous_static_nw_delay":
            # print('heterogenous_static_nw_delay scenario')
            assert 0 < args.slow_nw_server_fraction < 1.0
            assert not (args.slow_nw_server_slowness == 0 and args.slow_nw_server_fraction != 0)
            assert not (args.slow_nw_server_slowness != 0 and args.slow_nw_server_fraction == 0)

            slow_nw_latency_base = constants.NW_LATENCY_BASE * args.slow_nw_server_slowness
            num_slow_nw_servers = int(args.slow_nw_server_fraction * args.num_servers)
            slow_nw_server_latency_bases = [slow_nw_latency_base] * num_slow_nw_servers

            num_fast_nw_servers = args.num_servers - num_slow_nw_servers
            fast_server_nw_rates = [constants.NW_LATENCY_BASE] * num_fast_nw_servers
            nw_latency_bases = slow_nw_server_latency_bases + fast_server_nw_rates

            # Start the servers
            for i in range(args.num_servers):
                serv = server.Server(i,
                                     resource_capacity=args.server_concurrency,
                                     service_time=args.service_time,
                                     service_time_model=args.service_time_model,
                                     simulation=simulation,
                                     nw_latency_base=nw_latency_bases[i],
                                     long_task_added_service_time=args.long_task_added_service_time)
                self.servers.append(serv)
        else:
            print("Unknown experiment scenario")
            sys.exit(-1)

        base_demand_weight = 1.0
        client_weights = []
        assert 0 <= args.high_demand_fraction < 1.0
        assert 0 <= args.demand_skew < 1.0
        assert not (args.demand_skew == 0 and args.high_demand_fraction != 0)
        assert not (args.demand_skew != 0 and args.high_demand_fraction == 0)

        if args.high_demand_fraction > 0.0 and args.demand_skew >= 0:
            heavy_client_weight = base_demand_weight * \
                args.demand_skew / args.high_demand_fraction
            num_heavy_clients = int(args.high_demand_fraction * args.num_clients)
            heavy_client_weights = [heavy_client_weight] * num_heavy_clients

            light_client_weight = base_demand_weight * \
                (1 - args.demand_skew) / (1 - args.high_demand_fraction)
            num_light_clients = args.num_clients - num_heavy_clients
            light_client_weights = [light_client_weight] * num_light_clients
            client_weights = heavy_client_weights + light_client_weights
        else:
            client_weights = [base_demand_weight] * args.num_clients

        assert sum(client_weights) > 0.99 * args.num_clients
        assert sum(client_weights) <= args.num_clients

        # Start workload generators (analogous to YCSB)
        data_point_monitor = Monitor(name="Latency", simulation=simulation)

        # Start the clients
        for i in range(args.num_clients):
            c = client.Client(id_="Client%s" % (i),
                              server_list=self.servers,
                              data_point_monitor=data_point_monitor,
                              state_parser=self.state_parser,
                              replica_selection_strategy=args.selection_strategy,
                              access_pattern=args.access_pattern,
                              replication_factor=args.replication_factor,
                              backpressure=args.backpressure,
                              shadow_read_ratio=args.shadow_read_ratio,
                              rate_interval=args.rate_interval,
                              cubic_c=args.cubic_c,
                              cubic_smax=args.cubic_smax,
                              cubic_beta=args.cubic_beta,
                              hysterisis_factor=args.hysterisis_factor,
                              demand_weight=client_weights[i],
                              rate_intervals=args.rate_intervals,
                              trainer=trainer,
                              simulation=simulation,
                              duplication_rate=duplication_rate)
            self.clients.append(c)

        # This is where we set the inter-arrival times based on
        # the required utilization level and the service time
        # of the overall server pool.

        arrival_rate = args.utilization * \
            sum([server.get_service_rate(long_task_fraction=args.long_tasks_fraction) for server in self.servers])
        inter_arrival_time = 1 / float(arrival_rate)

        updated_arrival_rate = args.utilization * \
            sum([server.get_service_rate(long_task_fraction=args.long_tasks_fraction + 0.4) for server in self.servers])
        updated_inter_arrival_time = 1 / float(updated_arrival_rate)

        # TODO: Use multiple workloads to simulate smoother shift to new workload?
        # More than 1 workload currently not supported
        assert args.num_workload == 1
        for i in range(args.num_workload):
            # w = workload.Workload(i, data_point_monitor,
            #                       self.clients,
            #                       args.workload_model,
            #                       inter_arrival_time * args.num_workload,
            #                       num_requests / args.num_workload,
            #                       simulation,
            #                       long_tasks_fraction=args.long_tasks_fraction
            #                       )
            # TODO: Change!
            w = workload.VariableLongTaskFractionWorkload(i, 30000, updated_inter_arrival_time, data_point_monitor,
                                                          self.clients,
                                                          args.workload_model,
                                                          inter_arrival_time * args.num_workload,
                                                          num_requests / args.num_workload,
                                                          simulation,
                                                          long_tasks_fraction=args.long_tasks_fraction
                                                          )
            simulation.process(w.run())
            self.workload_gens.append(w)

        # Begin simulation
        simulation.run(until=args.simulation_duration)

        #
        # print(a bunch of timeseries)
        #

        # exp_prefix = f'{args.exp_prefix}_test' if eval_mode else args.exp_prefix
        # exp_path = Path('..', args.log_folder, exp_prefix)

        # if not exp_path.exists():
        #     exp_path.mkdir(parents=True, exist_ok=True)

        # pending_requests_fd = open("../%s/%s_PendingRequests" %
        #                            (args.log_folder,
        #                             exp_prefix), 'w')
        # wait_mon_fd = open("../%s/%s_WaitMon" % (args.log_folder,
        #                                          exp_prefix), 'w')
        # act_mon_fd = open("../%s/%s_ActMon" % (args.log_folder,
        #                                        exp_prefix), 'w')
        # latency_fd = open("../%s/%s_Latency" % (args.log_folder,
        #                                         exp_prefix), 'w')
        # latency_tracker_fd = open("../%s/%s_LatencyTracker" %
        #                           (args.log_folder, exp_prefix), 'w')
        # rate_fd = open("../%s/%s_Rate" % (args.log_folder,
        #                                   exp_prefix), 'w')
        # token_fd = open("../%s/%s_Tokens" % (args.log_folder,
        #                                      exp_prefix), 'w')
        # receive_rate_fd = open("../%s/%s_ReceiveRate" % (args.log_folder,
        #                                                  exp_prefix), 'w')
        # ed_score_fd = open("../%s/%s_EdScore" % (args.log_folder,
        #                                          exp_prefix), 'w')
        # server_rrfd = open("../%s/%s_serverRR" % (args.log_folder,
        #                                           exp_prefix), 'w')

        # for clientNode in clients:
        #     print_monitor_time_series_to_file(pending_requests_fd,
        #                                       clientNode.id,
        #                                       clientNode.pendingRequestsMonitor)
        #     print_monitor_time_series_to_file(latency_tracker_fd,
        #                                       clientNode.id,
        #                                       clientNode.latencyTrackerMonitor)
        #     print_monitor_time_series_to_file(rate_fd,
        #                                       clientNode.id,
        #                                       clientNode.rateMonitor)
        #     print_monitor_time_series_to_file(token_fd,
        #                                       clientNode.id,
        #                                       clientNode.tokenMonitor)
        #     print_monitor_time_series_to_file(receive_rate_fd,
        #                                       clientNode.id,
        #                                       clientNode.receiveRateMonitor)
        #     print_monitor_time_series_to_file(ed_score_fd,
        #                                       clientNode.id,
        #                                       clientNode.edScoreMonitor)

        if args.print:
            for serv in self.servers:
                #     print_monitor_time_series_to_file(wait_mon_fd,
                #                                       serv.id,
                #                                       serv.wait_monitor)
                #     print_monitor_time_series_to_file(act_mon_fd,
                #                                       serv.id,
                #                                       serv.act_monitor)
                #     print_monitor_time_series_to_file(server_rrfd,
                #                                       serv.id,
                #                                       serv.server_RR_monitor)
                print("------- Server:%s %s ------" % (serv.id, "WaitMon"))
                print("Mean:", serv.wait_monitor.mean())

                print("------- Server:%s %s ------" % (serv.id, "ActMon"))
                print("Mean:", serv.act_monitor.mean())

            print("------- Latency ------")
            print("Mean Latency:", data_point_monitor.mean())
            for p in [50, 95, 99]:
                print(f"p{p} Latency: {data_point_monitor.percentile(p)}")

            # print_monitor_time_series_to_file(latency_fd, "0",
            #                                   data_point_monitor)
            assert num_requests == len(data_point_monitor)

        return data_point_monitor
