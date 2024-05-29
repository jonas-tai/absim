from typing import List

from monitor import Monitor
import random
import numpy
import constants
from simulations.model_trainer import Trainer
from task import Task
import math
import threading

from yunomi.stats.exp_decay_sample import ExponentiallyDecayingSample

from simulations.server import Server
from simulations.state import NodeState, State, StateParser
from collections import defaultdict, namedtuple

DataPoint = namedtuple('DataPoint', ('state', 'latency', 'replica_id', 'is_duplicate', 'is_faster_response'))


class Client:
    def __init__(self, id_, server_list: List[Server], data_point_monitor: Monitor, state_parser: StateParser, replica_selection_strategy,
                 access_pattern, replication_factor, backpressure,
                 shadow_read_ratio, rate_interval,
                 cubic_c, cubic_smax, cubic_beta, hysterisis_factor,
                 demand_weight, simulation, duplication_rate: float = 0.0, rate_intervals=None, trainer: Trainer = None):
        self.lock = threading.Lock()

        if rate_intervals is None:
            rate_intervals = [1000, 500, 100]
        self.id = id_
        self.state_parser = state_parser
        self.data_point_monitor = data_point_monitor
        self.server_list = server_list
        self.accessPattern = access_pattern
        self.replication_factor = replication_factor
        self.REPLICA_SELECTION_STRATEGY = replica_selection_strategy
        self.pendingRequestsMonitor = Monitor(name="PendingRequests", simulation=simulation)
        self.latencyTrackerMonitor = Monitor(name="ResponseHandler", simulation=simulation)
        self.rateMonitor = Monitor(name="AlphaMonitor", simulation=simulation)
        self.receiveRateMonitor = Monitor(name="ReceiveRateMonitor", simulation=simulation)
        self.tokenMonitor = Monitor(name="TokenMonitor", simulation=simulation)
        self.edScoreMonitor = Monitor(name="edScoreMonitor", simulation=simulation)
        self.backpressure = backpressure  # True / False
        self.shadow_read_ratio = shadow_read_ratio
        self.demandWeight = demand_weight
        self.simulation = simulation
        self.last_req_start_time = self.simulation.now
        self.time_since_last_req = 0
        self.trainer = trainer
        self.request_rate_monitor = RequestRateMonitor(simulation, rate_intervals)
        # Tracks number of requests handled (requests that arrived, excludes duplicate requests)
        self.requests_handled = 0
        self.dqn_decision_equal_to_ars = 0
        self.duplication_rate = duplication_rate

        # Keep track of which replica round robin serves next
        self.next_RR_replica = 0

        # Book-keeping and metrics to be recorded follow...

        # Keep track of score that ARS assigns to nodes
        self.arsScoresMap = {node: 0.0 for node in server_list}

        # CANNOT USE DEFAULTDICT BECAUSE OF DICT.GET USAGE LATER
        # Number of outstanding requests at the client
        self.pendingRequestsMap = {node: 0 for node in server_list}
        self.pending_long_requests = {node: 0 for node in server_list}
        self.pending_short_requests = {node: 0 for node in server_list}

        # Number of outstanding requests times oracle-service time of replica
        self.pendingXserviceMap = {node: 0 for node in server_list}

        # Last-received response time of server
        self.responseTimesMap = {node: 0.0 for node in server_list}

        # Used to track response time from the perspective of the client
        self.taskSentTimeTracker = {}
        self.taskArrivalTimeTracker = {}
        self.duplicated_tasks_latency_tracker = {}

        # Record waiting and service times as relayed by the server
        self.expected_delay_map = {node: {} for node in server_list}
        self.lastSeen = defaultdict(int)

        # Rate limiters per replica
        self.rateLimiters = {node: RateLimiter("RL-%s" % node.id,
                                               self, 50, rate_interval, simulation)
                             for node in server_list}
        self.lastRateDecrease = defaultdict(int)
        self.valueOfLastDecrease = defaultdict(lambda: 10)
        self.receiveRate = {node: ReceiveRate("RL-%s" % node.id, rate_interval, simulation)
                            for node in server_list}
        self.lastRateIncrease = defaultdict(int)
        self.rateInterval = rate_interval

        # Parameters for congestion control
        self.cubicC = cubic_c
        self.cubicSmax = cubic_smax
        self.cubicBeta = cubic_beta
        self.hysterisis_factor = hysterisis_factor
        self.handled_requests = 0

        # Backpressure related initialization
        if backpressure:
            self.backpressureSchedulers = \
                {node: BackpressureScheduler("BP-%s" % node.id, self, simulation)
                 for node in server_list}
            for node in server_list:
                self.simulation.process(self.backpressureSchedulers[node].run())

        # ds-metrics
        if replica_selection_strategy == "ds":
            self.latencyEdma = {node: ExponentiallyDecayingSample(100, 0.75, self.clock)
                                for node in server_list}
            self.dsScores = {node: 0 for node in server_list}
            for node, rateLimiter in self.rateLimiters.items():
                ds = DynamicSnitch(self, 100, simulation)
                self.simulation.process(ds.run())

    def clock(self):
        """
            Convert to seconds because that's what the
            ExponentiallyDecayingSample
            assumes. Else, the internal Math.exp overflows.
        """
        return self.simulation.now / 1000.0

    def schedule(self, task: Task, replica_set: List[Server] = None):
        first_replica_index = None

        # Pick a random node and it's next RF - 1 number of neighbours
        if self.accessPattern == "uniform":
            first_replica_index = self.simulation.random.randint(0, len(self.server_list) - 1)
        elif self.accessPattern == "zipfian":
            first_replica_index = self.simulation.np_random.zipf(1.5) % len(self.server_list)

        if replica_set is None:
            replica_set = [self.server_list[i % len(self.server_list)]
                           for i in range(first_replica_index,
                                          first_replica_index +
                                          self.replication_factor)]
        start_time = self.simulation.now
        self.request_rate_monitor.add_request(start_time=start_time)
        self.time_since_last_req = start_time - self.last_req_start_time
        self.last_req_start_time = start_time
        self.taskArrivalTimeTracker[task] = start_time

        # make sure replicas are sorted by replica id
        replica_set.sort(key=lambda x: x.id)
        if self.backpressure is False:
            replica_to_serve = self.sort(task, replica_set)
            self.send_request(task, replica_to_serve)
            # TODO: Make this copies to avoid potential race conditions?
            self.maybe_send_duplicate_request(task=task, replica_to_serve=replica_to_serve, replica_set=replica_set)

            self.maybe_send_shadow_reads(replica_to_serve, replica_set)
        else:
            self.backpressureSchedulers[replica_set[0]].enqueue(task, replica_set)

    def send_request(self, task: Task, replica_to_serve: Server):
        nw_delay = replica_to_serve.get_server_nw_latency()
        self.handled_requests += 1

        # Immediately send out request
        message_delivery_process = DeliverMessageWithDelay(simulation=self.simulation)
        self.simulation.process(message_delivery_process.run(task,
                                                             nw_delay,
                                                             replica_to_serve))
        response_handler = ResponseHandler(self.simulation)
        self.simulation.process(response_handler.run(self, task, replica_to_serve))

        # Book-keeping for metrics
        self.pendingRequestsMap[replica_to_serve] += 1
        if task.is_long_task():
            self.pending_long_requests[replica_to_serve] += 1
        else:
            self.pending_short_requests[replica_to_serve] += 1

        self.pendingXserviceMap[replica_to_serve] = \
            (1 + self.pendingRequestsMap[replica_to_serve]) \
            * replica_to_serve.service_time
        self.pendingRequestsMonitor.observe(
            "%s %s" % (replica_to_serve.id,
                       self.pendingRequestsMap[replica_to_serve]))
        self.taskSentTimeTracker[task] = self.simulation.now

    def sort(self, task: Task, original_replica_set: List[Server]) -> List[Server]:
        replica_set = original_replica_set[0:].copy()

        ars_replica_ranking = self.get_ars_ranking(original_replica_set)
        # print('Calculating node state')
        node_states = [self.get_node_state(replica) for replica in replica_set]

        request_rates = self.request_rate_monitor.get_rates()
        state = State(time_since_last_req=self.time_since_last_req, request_trend=request_rates,
                      node_states=node_states, is_long_request=task.is_long_task())

        task.set_state(state=state)

        if self.REPLICA_SELECTION_STRATEGY == "random":
            # Pick a random node for the request.
            # Represents SimpleSnitch + uniform request access.
            # Ignore scores and everything else.
            self.simulation.random.shuffle(replica_set)
        elif self.REPLICA_SELECTION_STRATEGY == "round_robin":
            replica_set[0] = replica_set[self.next_RR_replica]
            # Increase round robin counter
            self.next_RR_replica = (self.next_RR_replica + 1) % len(self.server_list)
        elif self.REPLICA_SELECTION_STRATEGY == "pending":
            # Sort by number of pending requests
            replica_set.sort(key=self.pendingRequestsMap.get)
        elif self.REPLICA_SELECTION_STRATEGY == "response_time":
            # Sort by response times
            replica_set.sort(key=self.responseTimesMap.get)
        elif self.REPLICA_SELECTION_STRATEGY == "weighted_response_time":
            # Weighted random proportional to response times
            m = {}
            for each in replica_set:
                if ("serviceTime" not in self.expected_delay_map[each]):
                    m[each] = 0.0
                else:
                    m[each] = self.expected_delay_map[each]["serviceTime"]
            replica_set.sort(key=m.get)
            total = sum(map(lambda x: self.responseTimesMap[x], replica_set))
            selection = self.simulation.random.uniform(0, total)
            cum_sum = 0
            node_to_select = None
            i = 0
            if total != 0:
                for entry in replica_set:
                    cum_sum += self.responseTimesMap[entry]
                    if selection < cum_sum:
                        node_to_select = entry
                        break
                    i += 1
                assert node_to_select is not None

                replica_set[0], replica_set[i] = replica_set[i], replica_set[0]
        elif self.REPLICA_SELECTION_STRATEGY == "primary":
            pass
        elif self.REPLICA_SELECTION_STRATEGY == "pendingXserviceTime":
            # Sort by response times * client-local-pending-requests
            replica_set.sort(key=self.pendingXserviceMap.get)
        elif self.REPLICA_SELECTION_STRATEGY == "clairvoyant":
            # Sort by response times * pending-requests
            oracle_map = {replica: (1 + len(replica.queueResource.queue))
                          * replica.serviceTime
                          for replica in original_replica_set}
            replica_set.sort(key=oracle_map.get)
        elif self.REPLICA_SELECTION_STRATEGY == "ARS":
            replica_set = ars_replica_ranking
        elif self.REPLICA_SELECTION_STRATEGY == "ds":
            first_node = replica_set[0]
            first_node_score = self.dsScores[first_node]
            badness_threshold = 0.0

            if first_node_score != 0.0:
                for node in replica_set[1:]:
                    new_node_score = self.dsScores[node]
                    if ((first_node_score - new_node_score) / first_node_score
                            > badness_threshold):
                        replica_set.sort(key=self.dsScores.get)
        elif self.REPLICA_SELECTION_STRATEGY in ['DQN', 'DQN_EXPLR']:
            action = self.trainer.select_action(state)

            if self.REPLICA_SELECTION_STRATEGY == 'DQN':
                self.trainer.record_state_and_action(task_id=task.id, state=state, action=action)

            # Map action back to server id
            replica = next(server for server in replica_set if server.get_server_id() == action)

            if ars_replica_ranking[0] == replica:
                self.dqn_decision_equal_to_ars += 1
            # set the first replica to be the "action"
            replica_set[0] = replica
        else:
            print(self.REPLICA_SELECTION_STRATEGY)
            assert False, "REPLICA_SELECTION_STRATEGY isn't set or is invalid"
        self.requests_handled += 1
        return replica_set[0]

    def get_ars_ranking(self, replica_list: List[Server]) -> List[Server]:
        for replica in replica_list:
            self.arsScoresMap[replica] = self.compute_expected_delay(replica)

        replica_set = replica_list.copy()
        replica_set.sort(key=self.arsScoresMap.get)
        return replica_set

    def get_node_state(self, replica: Server) -> NodeState:
        outstanding_requests = self.pendingRequestsMap[replica]
        response_time = self.responseTimesMap[replica]
        long_requests = self.pending_long_requests[replica]
        short_requests = self.pending_short_requests[replica]

        if len(self.expected_delay_map[replica]) != 0:
            metric_map = self.expected_delay_map[replica]
            twice_network_latency = metric_map["nw"]
            service_time = metric_map["serviceTime"]
            wait_time = metric_map["waitingTime"]
            queue_size = metric_map["queueSizeAfter"]
            ars_score = self.arsScoresMap[replica]
            # print(
            #     f'NodeState: twice_network_latency: {twice_network_latency}, service_time: {service_time}, '
            #     f'wait_time: {wait_time}, queue_size: {queue_size}, outstanding_requests: {outstanding_requests}, long_requests: {long_requests}, short_requests: {short_requests}')
            return NodeState(queue_size=queue_size, service_time=service_time, wait_time=wait_time,
                             outstanding_requests=outstanding_requests, response_time=response_time,
                             twice_network_latency=twice_network_latency, outstanding_long_requests=long_requests,
                             outstanding_short_requests=short_requests, ars_score=ars_score)
        else:
            # TOOD: Should we init an empty node state?
            return NodeState(outstanding_requests=outstanding_requests, response_time=response_time, ars_score=0.0)

    def metric_decay(self, replica):
        return math.exp(-(self.simulation.now - self.lastSeen[replica])(2 * self.rateInterval))

    def compute_expected_delay(self, replica):
        total = 0
        if len(self.expected_delay_map[replica]) != 0:
            metric_map = self.expected_delay_map[replica]
            twice_network_latency = metric_map["nw"]
            queue_size_est = (1 + self.pendingRequestsMap[replica]
                              * constants.NUMBER_OF_CLIENTS
                              + metric_map["queueSizeAfter"])
            total += (twice_network_latency +
                      ((queue_size_est ** 3) * (metric_map["serviceTime"])))
            self.edScoreMonitor.observe("%s %s %s %s %s" %
                                        (replica.id,
                                         metric_map["queueSizeAfter"],
                                         metric_map["serviceTime"],
                                         queue_size_est,
                                         total))
        else:
            return 0
        return total

    def maybe_send_duplicate_request(self, task: Task, replica_to_serve: Server, replica_set: List[Server]):
        # Potentially send duplicate request
        if self.simulation.random.random() < self.duplication_rate:
            # Send duplicate request to other replica than exploit request
            replica_set_duplicate_req = [
                replica for replica in replica_set if replica.get_server_id() != replica_to_serve.get_server_id()]
            duplicate_replica = self.simulation.random.shuffle(replica_set_duplicate_req)
            duplicate_task = task.create_duplicate_task()
            self.taskArrivalTimeTracker[duplicate_task] = self.taskArrivalTimeTracker[task]

            self.send_request(task=duplicate_task, replica_to_serve=duplicate_replica)

    def maybe_send_shadow_reads(self, replica_to_serve, replica_set):
        if self.simulation.random.uniform(0, 1.0) < self.shadow_read_ratio:
            for replica in replica_set:
                if replica is not replica_to_serve:
                    shadow_read_task = Task("ShadowRead", None, self.simulation)
                    self.taskArrivalTimeTracker[shadow_read_task] = self.simulation.now
                    self.send_request(shadow_read_task, replica)
                    self.rateLimiters[replica].forceUpdates()

    def update_ema(self, replica, metric_map):
        alpha = 0.9
        if len(self.expected_delay_map[replica]) == 0:
            self.expected_delay_map[replica] = metric_map
            return

        for metric in metric_map:
            self.expected_delay_map[replica][metric] \
                = alpha * metric_map[metric] + (1 - alpha) \
                * self.expected_delay_map[replica][metric]

    def update_rates(self, replica, metricMap, task):
        # Cubic Parameters go here
        # beta = 0.2
        # C = 0.000004
        # Smax = 10
        beta = self.cubicBeta
        C = self.cubicC
        Smax = self.cubicSmax
        hysterisisFactor = self.hysterisis_factor
        currentSendingRate = self.rateLimiters[replica].rate
        currentReceiveRate = self.receiveRate[replica].getRate()

        if currentSendingRate < currentReceiveRate:
            # This means that we need to bump up our own rate.
            # For this, increase the rate according to a cubic
            # window. Rmax is the sending-rate at which we last
            # observed a congestion event. We grow aggressively
            # towards this point, and then slow down, stabilise,
            # and then advance further up. Every rate
            # increase is capped by Smax.
            T = self.simulation.now - self.lastRateDecrease[replica]
            self.lastRateIncrease[replica] = self.simulation.now
            Rmax = self.valueOfLastDecrease[replica]

            newSendingRate = C * (T - (Rmax * beta / C) ** (1.0 / 3.0)) ** 3 + Rmax

            if newSendingRate - currentSendingRate > Smax:
                self.rateLimiters[replica].rate += Smax
            else:
                self.rateLimiters[replica].rate = newSendingRate
        elif (currentSendingRate > currentReceiveRate
              and self.simulation.now - self.lastRateIncrease[replica]
              > self.rateInterval * hysterisisFactor):
            # The hysterisis factor in the condition is to ensure
            # that the receive-rate measurements have enough time
            # to adapt to the updated rate.

            # So we're in here now, which means we need to back down.
            # Multiplicatively decrease the rate by a factor of beta.
            self.valueOfLastDecrease[replica] = currentSendingRate
            self.rateLimiters[replica].rate *= beta
            self.rateLimiters[replica].rate = \
                max(self.rateLimiters[replica].rate, 0.0001)
            self.lastRateDecrease[replica] = self.simulation.now

        assert (self.rateLimiters[replica].rate > 0)
        alphaObservation = (replica.id,
                            self.rateLimiters[replica].rate)
        receiveRateObs = (replica.id,
                          self.receiveRate[replica].getRate())
        self.rateMonitor.observe("%s %s" % alphaObservation)
        self.receiveRateMonitor.observe("%s %s" % receiveRateObs)


class DeliverMessageWithDelay:
    def __init__(self, simulation) -> None:
        # self.simulation.process(self.run())
        # self.simulation.Process.__init__(self, name='DeliverMessageWithDelay')
        self.simulation = simulation

    def run(self, task, delay, replica_to_serve):
        yield self.simulation.timeout(delay)
        replica_to_serve.enqueue_task(task)


class ResponseHandler:
    def __init__(self, simulation) -> None:
        self.simulation = simulation

    def run(self, client: Client, task: Task, replica_that_served):
        yield self.simulation.timeout(0)
        yield task.completion_event

        nw_delay = replica_that_served.get_server_nw_latency()

        yield self.simulation.timeout(nw_delay)

        # OMG request completed. Time for some book-keeping
        client.pendingRequestsMap[replica_that_served] -= 1
        if task.is_long_task():
            client.pending_long_requests[replica_that_served] -= 1
        else:
            client.pending_short_requests[replica_that_served] -= 1

        client.pendingXserviceMap[replica_that_served] = (1 + client.pendingRequestsMap[
            replica_that_served]) * replica_that_served.service_time

        client.pendingRequestsMonitor.observe(
            "%s %s" % (replica_that_served.id, client.pendingRequestsMap[replica_that_served]))

        task_finished = self.simulation.now

        client.responseTimesMap[replica_that_served] = task_finished - client.taskSentTimeTracker[task]
        client.latencyTrackerMonitor.observe("%s %s" % (replica_that_served.id,
                                                        task_finished - client.taskSentTimeTracker[task]))
        metric_map = task.completion_event.value
        metric_map["responseTime"] = client.responseTimesMap[replica_that_served]
        # TODO: Fix naming, not really NW latency but nw latency + wait time
        metric_map["nw"] = metric_map["responseTime"] - metric_map["serviceTime"]  # - metric_map["waitingTime"]

        # TODO: Validate that correct for duplication if using dynamic snitch with duplication
        client.update_ema(replica_that_served, metric_map)
        client.receiveRate[replica_that_served].add(1)

        # Backpressure related book-keeping
        if client.backpressure:
            client.update_rates(replica_that_served, metric_map, task)

        client.lastSeen[replica_that_served] = task_finished

        if client.REPLICA_SELECTION_STRATEGY == "ds":
            client.latencyEdma[replica_that_served].update(metric_map["responseTime"])

        del client.taskSentTimeTracker[task]
        del client.taskArrivalTimeTracker[task]

        # task.start is created at Task creation time in workload.py
        latency = task_finished - task.start

        is_faster_response = True
        if task.has_duplicate or task.is_duplicate:
            with client.lock:
                if task.original_id not in client.duplicated_tasks_latency_tracker:
                    client.duplicated_tasks_latency_tracker[task.original_id] = latency
                else:
                    is_faster_response = False
                    del client.duplicated_tasks_latency_tracker[task.original_id]

        # Does not make sense to record shadow read latencies
        # as a latency measurement
        if task.id != 'ShadowRead':
            # Task completed, we call the trainer to see if we can do a step
            if client.REPLICA_SELECTION_STRATEGY == 'DQN' and not task.is_duplicate:
                client.trainer.execute_step_if_state_present(task_id=task.id, latency=latency)

            state = task.get_state()

            replica_id = replica_that_served.id
            client.data_point_monitor.observe(DataPoint(state=state, latency=latency,
                                              replica_id=replica_id, is_duplicate=task.is_duplicate, is_faster_response=is_faster_response))


class RequestRateMonitor:
    def __init__(self, simulation, rate_intervals: List[int]) -> None:
        # TODO: What time unit does the simulation assume?
        self.simulation = simulation
        self.rate_intervals = rate_intervals
        self.request_times = []

    def add_request(self, start_time: int) -> None:
        self.request_times.append(start_time)

    def get_rates(self) -> List[int]:
        now = self.simulation.now
        rates = defaultdict(int)
        last_index = len(self.request_times)
        max_interval_boundary_reached = False
        # iterate through requests, new requests are at the back of the list
        for index, start_time in enumerate(reversed(self.request_times)):
            if max_interval_boundary_reached:
                break
            max_interval_boundary_reached = True
            for interval in self.rate_intervals:
                if start_time >= (now - interval):
                    last_index = index
                    max_interval_boundary_reached = True
                    rates[interval] += 1
        # Remove requests that are not part of any interval anymore
        self.request_times = self.request_times[(len(self.request_times) - last_index):]

        return [rates[interval] for interval in self.rate_intervals]


class BackpressureScheduler:
    def __init__(self, id_, client, simulation):
        raise NotImplementedError('BackpressureScheduler not used')
        self.simulation = simulation
        self.id = id_
        self.backlogQueue = []
        self.client = client
        self.count = 0
        self.backlogReadyEvent = self.simulation.event()
        self.simulation.Process.__init__(self)

    def run(self):
        while (1):
            yield self.simulation.timeout(0)

            if (len(self.backlogQueue) != 0):
                task, replicaSet = self.backlogQueue[0]
                sortedReplicaSet = self.client.sort(replicaSet)
                sent = False
                minDurationToWait = 1e10  # arbitrary large value
                minReplica = None
                for replica in sortedReplicaSet:
                    currentTokens = self.client.rateLimiters[replica].tokens
                    self.client.tokenMonitor.observe("%s %s"
                                                     % (replica.id,
                                                        currentTokens))
                    durationToWait = \
                        self.client.rateLimiters[replica].tryAcquire()
                    if (durationToWait == 0):
                        assert self.client.rateLimiters[replica].tokens >= 1
                        self.backlogQueue.pop(0)
                        self.client.send_request(task, replica)
                        self.client.maybe_send_shadow_reads(replica, replicaSet)
                        sent = True
                        self.client.rateLimiters[replica].update()
                        break
                    else:
                        if durationToWait < minDurationToWait:
                            minDurationToWait = durationToWait
                            minReplica = replica
                        assert self.client.rateLimiters[replica].tokens < 1

                if (not sent):
                    # Backpressure mode. Wait for the least amount of time
                    # necessary until at least one rate limiter is expected
                    # to be available
                    yield self.simulation.timeout(minDurationToWait)
                    # NOTE: In principle, these 2 lines below would not be
                    # necessary because the rate limiter would have exactly 1
                    # token after minDurationWait. However, due to
                    # floating-point arithmetic precision we might not have 1
                    # token and this would cause the simulation to enter an
                    # almost infinite loop. These 2 lines by-pass this problem.
                    self.client.rateLimiters[minReplica].tokens = 1
                    self.client.rateLimiters[minReplica].lastSent = self.simulation.now
                    minReplica = None
            else:
                yield self.simulation.waitevent, self, self.backlogReadyEvent
                self.backlogReadyEvent = self.simulation.SimEvent()

    def enqueue(self, task, replicaSet):
        self.backlogQueue.append((task, replicaSet))
        self.backlogReadyEvent.signal()


class RateLimiter:
    def __init__(self, id_, client, maxTokens, rateInterval, simulation):
        self.id = id_
        self.rate = 5
        self.lastSent = 0
        self.client = client
        self.tokens = 0
        self.rateInterval = rateInterval
        self.maxTokens = maxTokens
        self.simulation = simulation

    # These updates can be forced due to shadowReads
    def update(self):
        self.lastSent = self.simulation.now
        self.tokens -= 1

    def tryAcquire(self):
        tokens = min(self.maxTokens, self.tokens
                     + self.rate / float(self.rateInterval)
                     * (self.simulation.now - self.lastSent))
        if (tokens >= 1):
            self.tokens = tokens
            return 0
        else:
            assert self.tokens < 1
            timetowait = (1 - tokens) * self.rateInterval / self.rate
            return timetowait

    def forceUpdates(self):
        self.tokens -= 1

    def getTokens(self):
        return min(self.maxTokens, self.tokens
                   + self.rate / float(self.rateInterval)
                   * (self.simulation.now - self.lastSent))


class ReceiveRate:
    def __init__(self, id, interval, simulation):
        self.rate = 10
        self.id = id
        self.interval = int(interval)
        self.last = 0
        self.count = 0
        self.simulation = simulation

    def getRate(self):
        self.add(0)
        return self.rate

    def add(self, requests):
        now = int(self.simulation.now / self.interval)
        if now - self.last < self.interval:
            self.count += requests
            if now > self.last:
                # alpha = (now - self.last)/float(self.interval)
                alpha = 0.9
                self.rate = alpha * self.count + (1 - alpha) * self.rate
                self.last = now
                self.count = 0
        else:
            self.rate = self.count
            self.last = now
            self.count = 0


class DynamicSnitch:
    '''
    Model for Cassandra's native dynamic snitching approach
    '''
    # TODO: Validate that correct for duplication if using dynamic snitch with duplication

    def __init__(self, client, snitchUpdateInterval, simulation):
        self.SNITCHING_INTERVAL = snitchUpdateInterval
        self.client = client
        self.simulation = simulation

    def run(self):
        # Start each process with a minor delay

        while 1:
            yield self.simulation.timeout(self.SNITCHING_INTERVAL)

            # Adaptation of DynamicEndpointSnitch algorithm
            maxLatency = 1.0
            maxPenalty = 1.0
            latencies = [entry.get_snapshot().get_median()
                         for entry in self.client.latencyEdma.values()]
            latenciesGtOne = [latency for latency in latencies if latency > 1.0]
            if (len(latencies) == 0):  # nothing to see here
                continue
            maxLatency = max(latenciesGtOne) if len(latenciesGtOne) > 0 else 1.0
            penalties = {}
            for peer in self.client.server_list:
                penalties[peer] = self.client.lastSeen[peer]
                penalties[peer] = self.simulation.now - penalties[peer]
                if (penalties[peer] > self.SNITCHING_INTERVAL):
                    penalties[peer] = self.SNITCHING_INTERVAL

            penaltiesGtOne = [penalty for penalty in penalties.values()
                              if penalty > 1.0]
            maxPenalty = max(penalties.values()) \
                if len(penaltiesGtOne) > 0 else 1.0

            for peer in self.client.latencyEdma:
                score = self.client.latencyEdma[peer] \
                            .get_snapshot() \
                            .get_median() / float(maxLatency)

                if (peer in penalties):
                    score += penalties[peer] / float(maxPenalty)
                else:
                    score += 1
                assert score >= 0 and score <= 2.0
                self.client.dsScores[peer] = score
