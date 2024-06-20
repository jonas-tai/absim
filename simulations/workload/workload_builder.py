from pathlib import Path
from typing import Any, Dict, List

import yaml

from simulations.workload.workload import BaseWorkload, VariableLongTaskFractionWorkload


class WorkloadBuilder:
    def __init__(self, config_folder: Path) -> None:
        self.train_config_file = config_folder / 'train' / 'workload_config.yaml'
        self.test_config_file = config_folder / 'test' / 'workload_config.yaml'

        # Load the base configurations from YAML file
        with open(self.train_config_file, 'r') as file:
            self.train_config = yaml.safe_load(file)

        with open(self.test_config_file, 'r') as file:
            self.test_config = yaml.safe_load(file)

    def create_train_base_workloads(self, utilizations: List[float] = [], long_tasks_fractions: List[float] = [], num_requests: int | None = None) -> List[Dict[str, Any]]:
        return self.create_base_workloads(base_config=self.train_config, utilizations=utilizations, long_tasks_fractions=long_tasks_fractions, num_requests=num_requests)

    def create_test_base_workloads(self, utilizations: List[float] = [], long_tasks_fractions: List[float] = [], num_requests: int | None = None) -> List[Dict[str, Any]]:
        return self.create_base_workloads(base_config=self.test_config, utilizations=utilizations, long_tasks_fractions=long_tasks_fractions, num_requests=num_requests)

    def create_base_workloads(self, base_config: Dict[str, Any], utilizations: List[float], long_tasks_fractions: List[float], num_requests: int | None = None) -> List[BaseWorkload]:
        workloads = []
        base_config = base_config['workload']

        if len(utilizations) == 0:
            utilizations = [base_config['utilization']]

        if len(long_tasks_fractions) == 0:
            long_tasks_fractions = [base_config['long_tasks_fraction']]

        if num_requests is None:
            num_requests = base_config['num_requests']

        for utilization in utilizations:
            for long_tasks_fraction in long_tasks_fractions:
                config = base_config.copy()
                config['utilization'] = utilization
                config['long_tasks_fraction'] = long_tasks_fraction
                config['num_requests'] = num_requests
                workload = BaseWorkload.from_dict(id_=1, config=config)
                workloads.append(workload)

        return workloads

    def create_train_var_long_tasks_workloads(self, utilizations: List[float] = [], num_requests: int | None = None) -> List[Dict[str, Any]]:
        return self.create_variable_workloads(base_config=self.train_config, utilizations=utilizations, num_requests=num_requests)

    def create_test_var_long_tasks_workloads(self, utilizations: List[float] = [], num_requests: int | None = None) -> List[Dict[str, Any]]:
        return self.create_variable_workloads(base_config=self.test_config, utilizations=utilizations, num_requests=num_requests)

    def create_variable_workloads(self, base_config: Dict[str, Any], utilizations: List[float] = [], num_requests: int | None = None) -> List[VariableLongTaskFractionWorkload]:
        workloads = []
        base_config = base_config['workload'] | base_config['variable_long_task_fraction_workload']

        if len(utilizations) == 0:
            utilizations = [base_config['utilization']]

        if num_requests is None:
            num_requests = base_config['num_requests']

        for utilization in utilizations:
            config = base_config.copy()
            config['utilization'] = utilization
            config['num_requests'] = num_requests
            workload = VariableLongTaskFractionWorkload.from_dict(id_=1, config=config)

            workloads.append(workload)

        return workloads
