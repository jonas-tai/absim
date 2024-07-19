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
    def __init__(self, id_, workload_seed: int, utilization: float, arrival_model: str, num_requests: int, long_tasks_fraction: float = 0):
        assert utilization > 0
        assert 0 <= long_tasks_fraction <= 1.0

        self.id = id_
        self.utilization = utilization
        self.arrival_model = arrival_model
        self.workload_seed = workload_seed

        self.client_delay_mean = -1
        self.num_requests = num_requests
        self.executed_requests = 0

        self.long_tasks_fraction = long_tasks_fraction
        self.workload_type: str = 'base'
        # self.proc = self.simulation.process(self.run(), 'Workload' + str(id_))
        self.random = random.Random(workload_seed)
        self.np_random = np.random.default_rng(workload_seed)

    def reset_workload(self):
        self.executed_requests = 0
        self.client_delay_mean = -1
        # TODO: Make this use separate randomState generators instead of reseeding the library
        self.random = random.Random(self.workload_seed)
        self.np_random = np.random.default_rng(self.workload_seed)

    def to_file_name(self) -> str:
        return f'{self.workload_type}_{self.utilization * 100:.2f}_util_{self.long_tasks_fraction * 100:.2f}_long_tasks_{self.workload_seed}'

    @classmethod
    def from_dict(cls, config: Dict[str, Any], id_) -> 'BaseWorkload':
        num_requests = config['num_requests']
        long_tasks_fraction = config['long_tasks_fraction']
        arrival_model = config['arrival_model']
        utilization = config['utilization']
        workload_seed = config['workload_seed']

        # Create a Workload instance
        return cls(
            id_=id_,
            workload_seed=workload_seed,
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
            'utilization': self.utilization,
            'workload_seed': self.workload_seed
        }
        return json.dumps(data, indent=4)

    def to_json_file(self, out_folder: Path, prefix: str = ''):
        json_data = self.to_json()
        with open(out_folder / f'{prefix}{WORKLOAD_CONFIG_FILE_NAME}', 'w') as file:
            file.write(json_data)

    # Need to pin workload to a client
    def run(self, clients: List[Client], servers: List[Server], simulation):
        # print(self.model_param)

        task_counter = 0
        self.client_delay_mean = calculate_client_delay_mean(
            servers=servers, utilization=self.utilization, long_tasks_fraction=self.long_tasks_fraction)

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
    def __init__(self, id_, workload_seed: int, trigger_threshold_mean: int, updated_long_tasks_fractions: List[float], updated_utilizations: List[float], arrival_model: str, utilization: float, num_requests: int, long_tasks_fraction: float = 0, ):
        super().__init__(id_=id_, workload_seed=workload_seed, arrival_model=arrival_model,
                         num_requests=num_requests, long_tasks_fraction=long_tasks_fraction, utilization=utilization)
        self.trigger_threshold_mean = trigger_threshold_mean
        self.trigger_threshold_sigma = int(trigger_threshold_mean / 8)
        self.next_trigger_threshold = int(self.random.normalvariate(self.trigger_threshold_mean, 8000))
        self.updated_long_tasks_fractions = updated_long_tasks_fractions
        self.updated_utilizations = updated_utilizations

        self.long_tasks_fraction = self.random.choice(self.updated_long_tasks_fractions)
        self.utilization = self.random.choice(self.updated_utilizations)

        self.req_since_last_trigger = 0

        self.workload_type: str = 'variable_long_task_fraction'

    def reset_workload(self):
        super().reset_workload()
        self.next_trigger_threshold = int(self.random.normalvariate(self.trigger_threshold_mean, 8000))
        self.long_tasks_fraction = self.random.choice(self.updated_long_tasks_fractions)
        self.utilization = self.random.choice(self.updated_utilizations)
        self.req_since_last_trigger = 0

    def to_file_name(self) -> str:
        return f'{self.workload_type}_{self.workload_seed}'

    def before_task_creation(self, servers: List[Server]):
        if self.req_since_last_trigger > 0 and self.req_since_last_trigger % self.next_trigger_threshold == 0:

            new_long_task_fraction = self.random.choice(self.updated_long_tasks_fractions)
            print(
                f'Trigger activated, changing long task fraction from {self.long_tasks_fraction} to {new_long_task_fraction}')
            # TODO: Parameterize if keep
            new_utilization = self.random.choice(self.updated_utilizations)
            print(
                f'Changing utilization from {self.utilization} to {new_utilization}')
            print(self.executed_requests)
            updated_client_delay_mean = calculate_client_delay_mean(
                servers=servers, utilization=self.utilization, long_tasks_fraction=new_long_task_fraction)

            print(f'Changing client delay from {self.client_delay_mean} to {updated_client_delay_mean}')
            self.long_tasks_fraction = new_long_task_fraction
            self.utilization = new_utilization
            self.client_delay_mean = updated_client_delay_mean

            self.req_since_last_trigger = 0
            self.next_trigger_threshold = int(self.random.normalvariate(self.trigger_threshold_mean, sigma=8000))
        self.req_since_last_trigger += 1

    @classmethod
    def from_dict(cls, id_: int, config: Dict[str, Any]) -> 'VariableLongTaskFractionWorkload':
        num_requests = config['num_requests']
        long_tasks_fraction = config['long_tasks_fraction']
        trigger_threshold_mean = config['trigger_threshold_mean']
        updated_long_tasks_fractions = config['updated_long_tasks_fractions']
        updated_utilizations = config['updated_utilizations']
        arrival_model = config['arrival_model']
        utilization = config['utilization']
        workload_seed = config['workload_seed']

        # Create a VariableLongTaskFractionWorkload instance
        return cls(
            id_=id_,  # You can set a proper id here
            workload_seed=workload_seed,
            trigger_threshold_mean=trigger_threshold_mean,
            updated_long_tasks_fractions=updated_long_tasks_fractions,
            updated_utilizations=updated_utilizations,
            arrival_model=arrival_model,
            num_requests=num_requests,
            long_tasks_fraction=long_tasks_fraction,
            utilization=utilization
        )

    def to_json(self) -> str:
        base_data = json.loads(super().to_json())
        additional_data = {
            'trigger_threshold_mean': self.trigger_threshold_mean,
            'updated_long_tasks_fractions': self.updated_long_tasks_fractions,
            'updated_utilizations': self.updated_utilizations,
        }
        base_data.update(additional_data)
        return json.dumps(base_data, indent=4)

    def to_json_file(self, out_folder: Path, prefix: str = ''):
        json_data = self.to_json()
        with open(out_folder / f'{prefix}{WORKLOAD_CONFIG_FILE_NAME}', 'w') as file:
            file.write(json_data)


class ChainedWorkload(BaseWorkload):
    def __init__(self, id_, workload_seed: int, workloads: List[BaseWorkload]):
        num_requests = sum([workload.num_requests for workload in workloads])
        assert len(workloads) > 0

        self.workload_iter = iter(workloads)
        self.current_workload = next(self.workload_iter)
        super().__init__(id_=id_, workload_seed=workload_seed, arrival_model=self.current_workload.arrival_model,
                         num_requests=num_requests, long_tasks_fraction=self.current_workload.long_tasks_fraction,
                         utilization=self.current_workload.utilization)

        self.workloads = workloads
        self.next_workload_change = self.current_workload.num_requests
        self.req_since_last_workload_change = 0

        # TODO make file name unique!
        self.workload_type: str = 'chained_workload'

    def switch_to_next_workload(self) -> None:
        try:
            self.current_workload = next(self.workload_iter)
            self.arrival_model = self.current_workload.arrival_model
            self.long_tasks_fraction = self.current_workload.long_tasks_fraction
            self.utilization = self.current_workload.utilization
            self.next_workload_change = self.current_workload.num_requests
            self.req_since_last_workload_change = 0
        except StopIteration:
            print('Reached end of chained workloads')

    def reset_workload(self) -> None:
        super().reset_workload()
        self.workload_iter = iter(self.workloads)
        self.switch_to_next_workload()
        for workload in self.workloads:
            workload.reset_workload()

    def to_file_name(self) -> str:
        return f'{self.workload_type}_{"_".join([wl.to_file_name() for wl in self.workloads])}'

    def before_task_creation(self, servers: List[Server]):
        if self.req_since_last_workload_change > 0 and self.req_since_last_workload_change % self.next_workload_change == 0:
            self.switch_to_next_workload()

            updated_client_delay_mean = calculate_client_delay_mean(
                servers=servers, utilization=self.utilization, long_tasks_fraction=self.long_tasks_fraction)
            print(f'Changing client delay from {self.client_delay_mean} to {updated_client_delay_mean}')
            self.client_delay_mean = updated_client_delay_mean

        self.req_since_last_workload_change += 1
        self.current_workload.before_task_creation(servers=servers)

    def to_json(self) -> str:
        workload_json_list = [json.loads(workload.to_json()) for workload in self.workloads]

        base_data = json.loads(super().to_json())
        additional_data = {
            'workloads': workload_json_list,
        }
        base_data.update(additional_data)
        return json.dumps(base_data, indent=4)

    def to_json_file(self, out_folder: Path, prefix: str = ''):
        json_data = self.to_json()
        with open(out_folder / f'{prefix}{WORKLOAD_CONFIG_FILE_NAME}', 'w') as file:
            file.write(json_data)
