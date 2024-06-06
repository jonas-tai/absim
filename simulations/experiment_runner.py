from typing import Any, Dict, List
import server
import client
from simulations.state import StateParser
from simulations.workload.workload import BaseWorkload, VariableLongTaskFractionWorkload
from simulator import Simulation
import constants
import sys
import simulations.workload.mu_updater as mu_updater
from simulations.monitor import Monitor
from pathlib import Path
from model_trainer import Trainer


class ExperimentRunner:
    def __init__(self, state_parser: StateParser) -> None:
        self.servers: List[server.Server] = []
        self.clients: List[client.Client] = []
        self.workload_gens: List[BaseWorkload] = []
        self.state_parser = state_parser

    def reset_stats(self) -> None:
        self.servers = []
        self.clients = []
        self.workload_gens = []

    def print_dqn_decision_equal_to_ars_ratio(self, client_index: int = 0) -> None:
        # Print how often DQN matched ARS in the last run
        ratio = self.clients[client_index].dqn_decision_equal_to_ars / self.clients[client_index].requests_handled
        print(f'DQN matched ARS for {ratio * 100}% of decisions')

    def run_experiment(self, args, workload: BaseWorkload, service_time_model: str, trainer: Trainer = None, duplication_rate: float = 0.0) -> Monitor:
        self.reset_stats()

        # Set the random seed
        simulation = Simulation()
        simulation.set_seed(args.seed)

        constants.NW_LATENCY_BASE = args.nw_latency_base
        constants.NW_LATENCY_MU = args.nw_latency_mu
        constants.NW_LATENCY_SIGMA = args.nw_latency_sigma
        constants.NUMBER_OF_CLIENTS = args.num_clients

        assert args.exp_scenario != ""

        service_rate_per_server = []
        if args.exp_scenario == "base" or args.exp_scenario == 'heterogenous_requests_scenario':
            # Start the servers
            for i in range(args.num_servers):
                serv = server.Server(i,
                                     resource_capacity=args.server_concurrency,
                                     service_time=(args.service_time),
                                     service_time_model=service_time_model,
                                     simulation=simulation,
                                     long_task_added_service_time=args.long_task_added_service_time)
                self.servers.append(serv)
        elif args.exp_scenario == "multipleServiceTimeServers":
            # Start the servers
            for i in range(args.num_servers):
                serv = server.Server(i,
                                     resource_capacity=args.server_concurrency,
                                     service_time=((i + 1) * args.service_time),
                                     service_time_model=service_time_model,
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
            assert workload.workload_type == 'base'

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
                                     service_time_model=service_time_model,
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
                                     service_time_model=service_time_model,
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
                                     service_time_model=service_time_model,
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

        # TODO: Use multiple workloads to simulate smoother shift to new workload?
        # More than 1 workload currently not supported
        assert args.num_workload == 1
        simulation.process(workload.run(servers=self.servers, clients=self.clients, simulation=simulation))
        self.workload_gens.append(workload)

        # Begin simulation
        simulation.run(until=args.simulation_duration)

        if args.print:
            for serv in self.servers:
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
            assert workload.num_requests == len(data_point_monitor)

        return data_point_monitor
