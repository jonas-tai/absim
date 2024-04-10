from global_sim import Simulation
import simpy
import math
import random
import sys
from monitor import Monitor


class Server:
    """A representation of a physical server that holds resources"""

    def __init__(self, id_, resource_capacity,
                 service_time, service_time_model):
        self.id = id_
        self.service_time = service_time
        self.service_time_model = service_time_model
        self.queue_resource = simpy.Resource(capacity=resource_capacity, env=Simulation)
        self.server_RR_monitor = Monitor()
        self.wait_monitor = Monitor()
        self.act_monitor = Monitor()

    def get_server_id(self):
        return self.id

    def enqueue_task(self, task):
        executor = Executor(self, task)
        self.server_RR_monitor.observe(1)
        Simulation.process(executor.run())
        # Simulation.activate(executor, executor.run(), Simulation.now)

    def get_service_time(self):
        if self.service_time_model == "random.expovariate":
            service_time = Simulation.random.expovariate(1.0 / self.service_time)
        elif self.service_time_model == "constant":
            service_time = self.service_time
        elif self.service_time_model == "math.sin":
            service_time = self.service_time + self.service_time * math.sin(1 + Simulation.now / 100)
        else:
            print("Unknown service time model")
            sys.exit(-1)

        return service_time


class Executor:

    def __init__(self, server, task):
        self.server = server
        self.task = task
        # Simulation.process(self.run(), 'Executor')

    def run(self):
        start = Simulation.now
        queue_size_before = len(self.server.queue_resource.queue)
        yield Simulation.timeout(0)
        request = self.server.queue_resource.request()
        yield request
        wait_time = Simulation.now - start  # W_i
        service_time = self.server.get_service_time()  # Mu_i
        yield Simulation.timeout(service_time)
        self.server.queue_resource.release(request)

        self.server.wait_monitor.observe(wait_time)
        self.server.act_monitor.observe(service_time)

        queue_size_after = len(self.server.queue_resource.queue)
        self.task.signal_task_complete({"waitingTime": wait_time,
                                        "serviceTime": service_time,
                                        "queueSizeBefore": queue_size_before,
                                        "queueSizeAfter": queue_size_after})
