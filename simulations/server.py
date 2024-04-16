import simpy
import math
import random
import sys
from monitor import Monitor
from simulations import constants


class Server:
    """A representation of a physical server that holds resources"""

    def __init__(self, id_, resource_capacity,
                 service_time, service_time_model, simulation, nw_latency_base: float = constants.NW_LATENCY_MU,
                 nw_latency_mu: float = constants.NW_LATENCY_MU,
                 nw_latency_sigma: float = constants.NW_LATENCY_SIGMA,
                 long_task_added_service_time: float = 0):
        self.id = id_
        self.service_time = service_time
        self.service_time_model = service_time_model
        self.queue_resource = simpy.Resource(capacity=resource_capacity, env=simulation)
        self.simulation = simulation
        self.NW_LATENCY_BASE = nw_latency_base
        self.NW_LATENCY_MU = nw_latency_mu
        self.NW_LATENCY_SIGMA = nw_latency_sigma
        self.long_task_added_service_time = long_task_added_service_time
        self.server_RR_monitor = Monitor(simulation)
        self.wait_monitor = Monitor(simulation)
        self.act_monitor = Monitor(simulation)

    def get_server_nw_latency(self):
        return self.NW_LATENCY_BASE + self.simulation.random.normalvariate(self.NW_LATENCY_MU, self.NW_LATENCY_SIGMA)

    def get_server_id(self):
        return self.id

    def enqueue_task(self, task):
        executor = Executor(self, task, self.simulation)
        self.server_RR_monitor.observe(1)
        self.simulation.process(executor.run())
        # self.simulation.activate(executor, executor.run(), self.simulation.now)

    def get_service_time(self, is_long_task=False):
        if self.service_time_model == "random.expovariate":
            service_time = self.simulation.random.expovariate(1.0 / self.service_time)
        elif self.service_time_model == "constant":
            service_time = self.service_time
        elif self.service_time_model == "math.sin":
            service_time = self.service_time + self.service_time * math.sin(1 + self.simulation.now / 100)
        else:
            print("Unknown service time model")
            sys.exit(-1)
        # Add service time if long task
        if is_long_task:
            service_time += self.long_task_added_service_time
        return service_time


class Executor:

    def __init__(self, server, task, simulation):
        self.server = server
        self.task = task
        self.simulation = simulation
        # self.simulation.process(self.run(), 'Executor')

    def run(self):
        start = self.simulation.now
        queue_size_before = len(self.server.queue_resource.queue)
        yield self.simulation.timeout(0)
        request = self.server.queue_resource.request()
        yield request
        wait_time = self.simulation.now - start  # W_i
        service_time = self.server.get_service_time(is_long_task=self.task.is_long_task())  # Mu_i

        yield self.simulation.timeout(service_time)
        self.server.queue_resource.release(request)

        self.server.wait_monitor.observe(wait_time)
        self.server.act_monitor.observe(service_time)

        queue_size_after = len(self.server.queue_resource.queue)
        self.task.signal_task_complete({"waitingTime": wait_time,
                                        "serviceTime": service_time,
                                        "queueSizeBefore": queue_size_before,
                                        "queueSizeAfter": queue_size_after})
