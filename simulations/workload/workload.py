import json
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np

from simulations.client import Client
from simulations.constants import ALPHA
from simulations.server import Server
from scipy.stats import pareto
import task

WORKLOAD_CONFIG_FILE_NAME = 'workload_config.json'


def calculate_client_delay_mean(servers: List[Server], utilization: float, long_tasks_fraction: float):
    arrival_rate = utilization * \
        sum([server.get_service_rate(long_task_fraction=long_tasks_fraction) for server in servers])
    return 1 / float(arrival_rate)


class BaseWorkload:
    def __init__(self, id_, utilization: float, arrival_model: str, num_requests: int, long_tasks_fraction: float = 0):
        assert utilization > 0
        assert 0 <= long_tasks_fraction <= 1.0

        self.id = id_
        self.utilization = utilization
        self.arrival_model = arrival_model

        self.client_delay_mean = -1
        self.num_requests = num_requests
        self.executed_requests = 0
        self.long_tasks_fraction = long_tasks_fraction
        self.workload_type: str = 'base'
        # self.proc = self.simulation.process(self.run(), 'Workload' + str(id_))
        self.random = random.Random()
        self.np_random = np.random.default_rng()

    def reset_workload(self):
        self.executed_requests = 0
        self.client_delay_mean = -1

    def to_file_name(self) -> str:
        return f'{self.workload_type}_{self.utilization * 100:.2f}_util_{self.long_tasks_fraction * 100:.2f}_long_tasks'

    @classmethod
    def from_dict(cls, config: Dict[str, Any], id_) -> 'BaseWorkload':
        num_requests = config['num_requests']
        long_tasks_fraction = config['long_tasks_fraction']
        arrival_model = config['arrival_model']
        utilization = config['utilization']

        # Create a Workload instance
        return cls(
            id_=id_,
            utilization=utilization,
            arrival_model=arrival_model,
            num_requests=num_requests,
            long_tasks_fraction=long_tasks_fraction,
        )

    def to_json(self) -> str:
        data = {
            'client_delay_mean': self.client_delay_mean,
            'num_requests': self.num_requests,
            'long_tasks_fraction': self.long_tasks_fraction,
            'arrival_model': self.arrival_model,
            'utilization': self.utilization
        }
        return json.dumps(data, indent=4)

    def to_json_file(self, out_folder: Path, prefix: str = ''):
        json_data = self.to_json()
        with open(out_folder / f'{prefix}{WORKLOAD_CONFIG_FILE_NAME}', 'w') as file:
            file.write(json_data)

    # Need to pin workload to a client
    def run(self, clients: List[Client], servers: List[Server], seed: int, simulation):
        # print(self.model_param)

        task_counter = 0
        self.client_delay_mean = calculate_client_delay_mean(
            servers=servers, utilization=self.utilization, long_tasks_fraction=self.long_tasks_fraction)

        # TODO: Make this use separate randomState generators instead of reseeding the library
        self.random = random.Random(seed)
        self.np_random = np.random.default_rng(seed)

        while self.executed_requests < self.num_requests:
            yield simulation.timeout(0)
            assert self.client_delay_mean > 0

            self.before_task_creation(servers=servers)
            is_long_task = False if self.random.random() >= self.long_tasks_fraction else True
            task_to_schedule = task.Task("Task" + str(task_counter),
                                         simulation=simulation, is_long_task=is_long_task, utilization=self.utilization, long_tasks_fraction=self.long_tasks_fraction)
            task_counter += 1

            # Push out a task...
            client_node = self.weighted_choice(simulation=simulation, clients=clients)

            # print(f'Scheduling Task {task_to_schedule.id}')
            client_node.schedule(task_to_schedule)
            # Simulate client delay
            if self.arrival_model == "poisson":
                yield simulation.timeout(self.np_random.poisson(self.client_delay_mean))

            # If model is gaussian, add gaussian delay
            # If model is constant, add fixed delay
            if self.arrival_model == "constant":
                yield simulation.timeout(self.client_delay_mean)

            if self.arrival_model == "pareto":
                scale = (self.client_delay_mean * (ALPHA - 1)) / ALPHA
                yield simulation.timeout(self.np_random.pareto(ALPHA) * scale)

            self.executed_requests += 1
        self.reset_workload()

    def weighted_choice(self, simulation, clients: List[Client]) -> Client:
        total = sum(client.demandWeight for client in clients)
        r = self.random.uniform(0, total)
        upto = 0
        for client in clients:
            if upto + client.demandWeight > r:
                return client
            upto += client.demandWeight
        assert False, "Shouldn't get here"

    def before_task_creation(self, servers: List[Server]):
        """Hook method to be called before creating a task."""
        pass


# Workload that changes the fraction of long tasks after some threshold
class VariableLongTaskFractionWorkload(BaseWorkload):
    def __init__(self, id_, trigger_threshold: int, updated_long_tasks_fractions: List[float], arrival_model: str, utilization: float, num_requests: int, long_tasks_fraction: float = 0, ):
        super().__init__(id_=id_, arrival_model=arrival_model,
                         num_requests=num_requests, long_tasks_fraction=long_tasks_fraction, utilization=utilization)
        self.trigger_threshold = trigger_threshold
        self.updated_long_tasks_fractions = updated_long_tasks_fractions
        self.original_long_tasks_fraction = self.long_tasks_fraction
        self.workload_type: str = 'variable_long_task_fraction'

    def reset_workload(self):
        super().reset_workload()
        self.long_tasks_fraction = self.original_long_tasks_fraction

    def to_file_name(self) -> str:
        return f'{self.workload_type}_updated_long_tasks_{self.utilization * 100:.2f}_util_{self.long_tasks_fraction * 100:.2f}_long_tasks'

    def before_task_creation(self, servers: List[Server]):
        if self.executed_requests > 0 and self.executed_requests % self.trigger_threshold == 0:

            new_long_task_fraction = self.random.choice(self.updated_long_tasks_fractions)
            print(
                f'Trigger activated, changing long task fraction from {self.long_tasks_fraction} to {new_long_task_fraction}')
            # TODO: Parameterize if keep
            new_utilization = self.random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            print(
                f'Changing utilization from {self.utilization} to {new_utilization}')
            print(self.executed_requests)
            updated_client_delay_mean = calculate_client_delay_mean(
                servers=servers, utilization=self.utilization, long_tasks_fraction=new_long_task_fraction)

            print(f'Changing client delay from {self.client_delay_mean} to {updated_client_delay_mean}')
            self.long_tasks_fraction = new_long_task_fraction
            self.utilization = new_utilization
            self.client_delay_mean = updated_client_delay_mean

    @classmethod
    def from_dict(cls, id_: int, config: Dict[str, Any]) -> 'VariableLongTaskFractionWorkload':
        num_requests = config['num_requests']
        long_tasks_fraction = config['long_tasks_fraction']
        trigger_threshold = config['trigger_threshold']
        updated_long_tasks_fractions = config['updated_long_tasks_fractions']
        arrival_model = config['arrival_model']
        utilization = config['utilization']

        # Create a VariableLongTaskFractionWorkload instance
        return cls(
            id_=id_,  # You can set a proper id here
            trigger_threshold=trigger_threshold,
            updated_long_tasks_fractions=updated_long_tasks_fractions,
            arrival_model=arrival_model,
            num_requests=num_requests,
            long_tasks_fraction=long_tasks_fraction,
            utilization=utilization
        )

    def to_json(self) -> str:
        base_data = json.loads(super().to_json())
        additional_data = {
            'trigger_threshold': self.trigger_threshold,
            'updated_long_tasks_fractions': self.updated_long_tasks_fractions,
        }
        base_data.update(additional_data)
        return json.dumps(base_data, indent=4)

    def to_json_file(self, out_folder: Path, prefix: str = ''):
        json_data = self.to_json()
        with open(out_folder / f'{prefix}{WORKLOAD_CONFIG_FILE_NAME}', 'w') as file:
            file.write(json_data)
