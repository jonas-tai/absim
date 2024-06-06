import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional

import yaml
from simulations.client import Client
from simulations.server import Server
from simulations.monitor import Monitor
import task
import numpy


def calculate_client_delay_mean(servers: List[Server], utilization: float, long_tasks_fraction: float):
    arrival_rate = utilization * \
        sum([server.get_service_rate(long_task_fraction=long_tasks_fraction) for server in servers])
    return 1 / float(arrival_rate)


class BaseWorkload:
    def __init__(self, id_, data_point_monitor: Monitor, clients: List[Client], servers: List[Server],
                 utilization: float, client_model, num_requests: int, simulation, long_tasks_fraction: float = 0):
        assert utilization > 0
        assert 0 <= long_tasks_fraction <= 1.0

        self.id = id_
        self.utilization = utilization
        self.data_point_monitor = data_point_monitor
        self.clients = clients
        self.client_model = client_model

        self.client_delay_mean = calculate_client_delay_mean(
            servers=servers, utilization=utilization, long_tasks_fraction=long_tasks_fraction)
        self.servers = servers
        self.num_requests = num_requests
        self.executed_requests = 0
        self.simulation = simulation
        self.total = sum(client.demandWeight for client in self.clients)
        self.long_tasks_fraction = long_tasks_fraction
        self.workload_type: str = 'base'
        # self.proc = self.simulation.process(self.run(), 'Workload' + str(id_))

    def to_file_name(self) -> str:
        return f'{self.workload_type}_{self.utilization * 100:.2f}_util_{self.long_tasks_fraction * 100:.2f}_long_tasks'

    @classmethod
    def from_dict(cls, config: Dict[str, Any], id_, simulation, data_point_monitor: Monitor, servers: List[Server], clients: List[Client]) -> 'BaseWorkload':
        num_requests = config['num_requests']
        long_tasks_fraction = config['long_tasks_fraction']
        client_model = config['client_model']
        utilization = config['utilization']

        # Create a Workload instance
        return cls(
            id_=id_,
            utilization=utilization,
            data_point_monitor=data_point_monitor,
            clients=clients,
            servers=servers,
            client_model=client_model,
            num_requests=num_requests,
            simulation=simulation,
            long_tasks_fraction=long_tasks_fraction,
        )

    def to_json(self) -> str:
        data = {
            'client_delay_mean': self.client_delay_mean,
            'num_requests': self.num_requests,
            'long_tasks_fraction': self.long_tasks_fraction,
            'client_model': self.client_model,
            'utilization': self.utilization
        }
        return json.dumps(data, indent=4)

    def to_json_file(self, file_path: Path):
        json_data = self.to_json()
        with open(file_path, 'w') as file:
            file.write(json_data)

    # Need to pin workload to a client
    def run(self):
        # print(self.model_param)

        task_counter = 0

        while self.executed_requests < self.num_requests:
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
            if self.client_model == "poisson":
                yield self.simulation.timeout(self.simulation.np_random.poisson(self.client_delay_mean))

            # If model is gaussian, add gaussian delay
            # If model is constant, add fixed delay
            if self.client_model == "constant":
                yield self.simulation.timeout(self.client_delay_mean)

            self.executed_requests += 1

    def weighted_choice(self) -> Client:
        r = self.simulation.random.uniform(0, self.total)
        upto = 0
        for client in self.clients:
            if upto + client.demandWeight > r:
                return client
            upto += client.demandWeight
        assert False, "Shouldn't get here"

    def before_task_creation(self):
        """Hook method to be called before creating a task."""
        pass


# Workload that changes the fraction of long tasks after some threshold
class VariableLongTaskFractionWorkload(BaseWorkload):
    def __init__(self, id_, trigger_threshold: int, updated_long_tasks_fraction: float, data_point_monitor, clients: List[Client], servers: List[Server], client_model: str, utilization: float, num_requests: int, simulation, long_tasks_fraction: float = 0, ):
        super().__init__(id_=id_, data_point_monitor=data_point_monitor, clients=clients, client_model=client_model,
                         num_requests=num_requests, simulation=simulation, long_tasks_fraction=long_tasks_fraction, utilization=utilization, servers=servers)
        self.trigger_threshold = trigger_threshold
        self.updated_long_tasks_fraction = updated_long_tasks_fraction
        self.workload_type: str = 'variable_long_task_fraction'
        self.updated_client_delay_mean = calculate_client_delay_mean(
            servers=self.servers, utilization=self.utilization, long_tasks_fraction=self.updated_long_tasks_fraction)

    def to_file_name(self) -> str:
        return f'{self.workload_type}_{self.updated_long_tasks_fraction * 100:.2f}_updated_long_tasks_{self.utilization * 100:.2f}_util_{self.long_tasks_fraction * 100:.2f}_long_tasks'

    def before_task_creation(self):
        if self.executed_requests == self.trigger_threshold:
            print(f'Trigger activated, changing long task fraction')
            print(self.executed_requests)
            print(f'Changing client delay from {self.client_delay_mean} to {self.updated_client_delay_mean}')
            self.long_tasks_fraction = self.updated_long_tasks_fraction
            self.client_delay_mean = self.updated_client_delay_mean

    @classmethod
    def from_dict(cls, id_: int, config: Dict[str, Any], simulation, data_point_monitor: Monitor, clients: List[Client], servers: List[Server]) -> 'VariableLongTaskFractionWorkload':
        num_requests = config['num_requests']
        long_tasks_fraction = config['long_tasks_fraction']
        trigger_threshold = config['trigger_threshold']
        updated_long_tasks_fraction = config['updated_long_tasks_fraction']
        client_model = config['model']

        # Create a VariableLongTaskFractionWorkload instance
        return cls(
            id_=id_,  # You can set a proper id here
            trigger_threshold=trigger_threshold,
            updated_long_tasks_fraction=updated_long_tasks_fraction,
            data_point_monitor=data_point_monitor,
            clients=clients,
            servers=servers,
            client_model=client_model,
            num_requests=num_requests,
            simulation=simulation,
            long_tasks_fraction=long_tasks_fraction,
        )

    def to_json(self) -> str:
        base_data = json.loads(super().to_json())
        additional_data = {
            'trigger_threshold': self.trigger_threshold,
            'updated_client_delay_mean': self.updated_client_delay_mean,
        }
        base_data.update(additional_data)
        return json.dumps(base_data, indent=4)

    def to_json_file(self, file_path: Path):
        json_data = self.to_json()
        with open(file_path, 'w') as file:
            file.write(json_data)
