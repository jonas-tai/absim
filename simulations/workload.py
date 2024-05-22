import random
from typing import List
from simulations.client import Client
from simulations.monitor import Monitor
import task
import numpy


class Workload:
    def __init__(self, id_, data_point_monitor: Monitor, client_list: List[Client],
                 model, model_param, num_requests, simulation, long_tasks_fraction: float = 0):
        self.data_point_monitor = data_point_monitor
        self.client_list = client_list
        self.model = model
        self.model_param = model_param
        self.num_requests = num_requests
        self.simulation = simulation
        self.total = sum(client.demandWeight for client in self.client_list)
        self.long_tasks_fraction = long_tasks_fraction
        # self.proc = self.simulation.process(self.run(), 'Workload' + str(id_))

    # TODO: also need non-uniform client access
    # Need to pin workload to a client
    def run(self):
        # print(self.model_param)

        task_counter = 0

        while self.num_requests != 0:
            yield self.simulation.timeout(0)

            is_long_task = False if self.simulation.random.random() >= self.long_tasks_fraction else True

            task_to_schedule = task.Task("Task" + str(task_counter),
                                         simulation=self.simulation, is_long_task=is_long_task)
            task_counter += 1

            # Push out a task...
            client_node = self.weighted_choice()

            # print(f'Scheduling Task {task_to_schedule.id}')
            client_node.schedule(task_to_schedule)
            # Simulate client delay
            if self.model == "poisson":
                yield self.simulation.timeout(self.simulation.np_random.poisson(self.model_param))

            # If model is gaussian, add gaussian delay
            # If model is constant, add fixed delay
            if self.model == "constant":
                yield self.simulation.timeout(self.model_param)

            self.num_requests -= 1

    def weighted_choice(self):
        r = self.simulation.random.uniform(0, self.total)
        upto = 0
        for client in self.client_list:
            if upto + client.demandWeight > r:
                return client
            upto += client.demandWeight
        assert False, "Shouldn't get here"
