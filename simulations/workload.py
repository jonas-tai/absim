import json
import random
from typing import Callable, List, Optional
from simulations.client import Client
from simulations.monitor import Monitor
import task
import numpy


class Workload:
    def __init__(self, id_, data_point_monitor: Monitor, client_list: List[Client],
                 model, mean_client_delay, num_requests, simulation, long_tasks_fraction: float = 0, callback: Optional[Callable] = None):
        self.data_point_monitor = data_point_monitor
        self.client_list = client_list
        self.model = model
        self.client_delay_mean = mean_client_delay
        self.num_requests = num_requests
        self.simulation = simulation
        self.total = sum(client.demandWeight for client in self.client_list)
        self.long_tasks_fraction = long_tasks_fraction
        self.callback = callback
        self.workload_type: str = 'base'
        # self.proc = self.simulation.process(self.run(), 'Workload' + str(id_))

    def to_json(self) -> str:
        data = {
            'client_delay_mean': self.client_delay_mean,
            'num_requests': self.num_requests,
            'long_tasks_fraction': self.long_tasks_fraction,
        }
        return json.dumps(data, indent=4)

    def to_json_file(self, filename: str):
        json_data = self.to_json()
        with open(filename, 'w') as file:
            file.write(json_data)

    # TODO: also need non-uniform client access
    # Need to pin workload to a client
    def run(self):
        # print(self.model_param)

        task_counter = 0

        while self.num_requests != 0:
            yield self.simulation.timeout(0)

            self.before_task_creation()
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
                yield self.simulation.timeout(self.simulation.np_random.poisson(self.client_delay_mean))

            # If model is gaussian, add gaussian delay
            # If model is constant, add fixed delay
            if self.model == "constant":
                yield self.simulation.timeout(self.client_delay_mean)

            self.num_requests -= 1
            if self.callback:
                self.callback(self)

    def weighted_choice(self) -> Client:
        r = self.simulation.random.uniform(0, self.total)
        upto = 0
        for client in self.client_list:
            if upto + client.demandWeight > r:
                return client
            upto += client.demandWeight
        assert False, "Shouldn't get here"

    def before_task_creation(self):
        """Hook method to be called before creating a task."""
        pass


# Workload that changes the fraction of long tasks after some threshold
class VariableLongTaskFractionWorkload(Workload):
    def __init__(self, id_, trigger_threshold: int, updated_inter_arrival_time: float, data_point_monitor, client_list: List[Client], model, model_param, num_requests, simulation, long_tasks_fraction: float = 0, ):
        super().__init__(id_, data_point_monitor, client_list, model, model_param, num_requests, simulation, long_tasks_fraction)
        self.trigger_threshold = trigger_threshold
        self.updated_inter_arrival_time = updated_inter_arrival_time
        self.executed_requests = 0
        self.workload_type: str = 'variable_long_task_fraction'

    def before_task_creation(self):
        self.executed_requests += 1
        if self.executed_requests == self.trigger_threshold:
            print(f'Trigger activated, changing long task fraction')
            print(self.executed_requests)
            print(f'Changing client delay from {self.client_delay_mean} to {self.updated_inter_arrival_time}')
            self.long_tasks_fraction += 0.4
            self.client_delay_mean = self.updated_inter_arrival_time

    def to_json(self) -> str:
        base_data = json.loads(super().to_json())
        additional_data = {
            'trigger_threshold': self.trigger_threshold,
            'updated_inter_arrival_time': self.updated_inter_arrival_time,
            'executed_requests': self.executed_requests
        }
        base_data.update(additional_data)
        return json.dumps(base_data, indent=4)

    def to_json_file(self, filename: str):
        json_data = self.to_json()
        with open(filename, 'w') as file:
            file.write(json_data)
