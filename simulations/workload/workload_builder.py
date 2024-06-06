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

    def create_train_base_workloads(self, utilizations: List[float] = [], long_tasks_fractions: List[float] = []) -> List[Dict[str, Any]]:
        return self.create_base_workloads(base_config=self.train_config, utilizations=utilizations, long_tasks_fractions=long_tasks_fractions)

    def create_test_base_workloads(self, utilizations: List[float] = [], long_tasks_fractions: List[float] = []) -> List[Dict[str, Any]]:
        return self.create_base_workloads(base_config=self.test_config, utilizations=utilizations, long_tasks_fractions=long_tasks_fractions)

    def create_base_workloads(self, base_config: Dict[str, Any], utilizations: List[float], long_tasks_fractions: List[float]) -> List[BaseWorkload]:
        workloads = []
        base_config = base_config['workload']

        if len(utilizations) == 0:
            utilizations = [base_config['utilization']]

        if len(long_tasks_fractions) == 0:
            long_tasks_fractions = [base_config['long_tasks_fraction']]

        for utilization in utilizations:
            for long_tasks_fraction in long_tasks_fractions:
                config = base_config.copy()
                config['utilization'] = utilization
                config['long_tasks_fraction'] = long_tasks_fraction
                workload = BaseWorkload.from_dict(id_=1, config=config)
                workloads.append(workload)

        return workloads

    def create_train_var_long_tasks_workloads(self, utilizations: List[float] = [], long_tasks_fractions: List[float] = []) -> List[Dict[str, Any]]:
        return self.create_variable_workloads(base_config=self.train_config, utilizations=utilizations, long_tasks_fractions=long_tasks_fractions)

    def create_test_var_long_tasks_workloads(self, utilizations: List[float] = [], long_tasks_fractions: List[float] = []) -> List[Dict[str, Any]]:
        return self.create_variable_workloads(base_config=self.test_config, utilizations=utilizations, long_tasks_fractions=long_tasks_fractions)

    def create_variable_workloads(self, base_config: Dict[str, Any], utilizations: List[float] = [], long_tasks_fractions: List[float] = [], updated_long_tasks_fractions: List[float] = []) -> List[VariableLongTaskFractionWorkload]:
        workloads = []
        base_config = base_config['workload'] | base_config['variable_long_task_fraction_workload']

        if len(utilizations) == 0:
            utilizations = [base_config['utilization']]

        if len(long_tasks_fractions) == 0:
            long_tasks_fractions = [base_config['long_tasks_fraction']]

        if len(updated_long_tasks_fractions) == 0:
            updated_long_tasks_fractions = [base_config['updated_long_tasks_fraction']]

        for updated_long_tasks_fraction in updated_long_tasks_fractions:
            for utilization in utilizations:
                for long_tasks_fraction in long_tasks_fractions:
                    config = base_config.copy()
                    config['updated_long_tasks_fraction'] = updated_long_tasks_fraction
                    config['utilization'] = utilization
                    config['long_tasks_fraction'] = long_tasks_fraction
                    workload = VariableLongTaskFractionWorkload.from_dict(id_=1, config=config)

                    workloads.append(workload)

        return workloads
