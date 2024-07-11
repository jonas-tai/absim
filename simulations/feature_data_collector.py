from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from simulations.monitor import Monitor
from simulations.client import DataPoint
from simulations.state import StateParser
import simulations.constants as const


@dataclass
class AnalysisData:
    policy: str
    epoch: int
    # Track the number of requests in the last 1s, 0.5s, 0.1s,...
    reward_data: pd.DataFrame
    feature_data: pd.DataFrame


class FeatureDataCollector:
    def __init__(self, out_folder: Path, state_parser: StateParser) -> None:
        self.data: List[AnalysisData] = []
        self.policy_colors = const.POLICY_COLORS
        self.out_folder = out_folder
        self.state_parser = state_parser

    def add_data(self, monitor: Monitor, policy: str, epoch_num: int) -> None:
        data_point_time_tuples: List[Tuple[DataPoint, float]] = monitor.get_data()

        reward_data = pd.DataFrame([{
            "Latency": data_point.latency,
            "Replica": data_point.replica_id,
            "Time": time,
            "Task_time_sent": data_point.task_time_sent,
            "Q_values": data_point.q_values
        } for (data_point, time) in data_point_time_tuples]
        )

        feature_data = pd.DataFrame(np.concatenate([self.state_parser.state_to_tensor(state=data_point.state).cpu().numpy()
                                                    for (data_point, time) in data_point_time_tuples]))

        self.data.append(AnalysisData(policy=policy, epoch=epoch_num,
                         reward_data=reward_data, feature_data=feature_data))

    def export_epoch_data(self, epoch: int) -> None:
        for analysis_data in self.data:
            if analysis_data.epoch != epoch:
                continue
            out_path = self.out_folder / f'{analysis_data.epoch}_{analysis_data.policy}_data.csv.gz'
            combined_df = pd.concat(objs=[analysis_data.reward_data, analysis_data.feature_data], axis=1)
            combined_df.reset_index(drop=True, inplace=True)
            combined_df.to_csv(out_path, compression='gzip')

    def export_training_data(self) -> None:
        train_data_df = pd.DataFrame()
        for analysis_data in self.data:
            combined_df = pd.concat(objs=[analysis_data.reward_data['Replica'], analysis_data.feature_data], axis=1)
            combined_df.reset_index(drop=True, inplace=True)
            train_data_df = pd.concat([train_data_df, combined_df], ignore_index=True)

        train_data_df.to_csv(self.out_folder / 'feature_data.csv.gz', index=False, compression='gzip')

    def run_latency_lin_reg(self, epoch: int) -> None:
        prefix = 'train' if self.is_train_data else 'test'

        for analysis_data in self.data:
            if analysis_data.epoch != epoch:
                continue
            print(f'Executing {prefix} {analysis_data.epoch} {analysis_data.policy} Latency linear regression')
            X = analysis_data.feature_data
            Y = analysis_data.reward_data['Latency']
            reg = LinearRegression().fit(X, Y)
            print(f'Linear regression score: {reg.score(X, Y)}')
            print(f'Coefficients: {reg.coef_}')
