from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from simulations.monitor import Monitor
from simulations.client import LatencyReplica, StateLatency


@dataclass
class AnalysisData:
    policy: str
    epoch: int
    # Track the number of requests in the last 1s, 0.5s, 0.1s,...
    reward_data: pd.DataFrame
    feature_data: pd.DataFrame


class FeatureDataAnalyzer:
    def __init__(self, out_folder: Path, is_train_data: bool = True) -> None:
        self.data: List[AnalysisData] = []
        self.policy_colors = {
            "ARS": "C0",
            "random": "C1",
            "DQN": "C2"
        }
        self.out_folder = out_folder
        self.is_train_data = is_train_data

    def add_data(self, monitor: Monitor, policy: str, epoch_num: int) -> None:
        state_latency_time_tuples: List[Tuple[StateLatency, float]] = monitor.get_data()

        reward_data = pd.DataFrame([{
            "Latency": state_latency.latency,
            "Time": time,
        } for (state_latency, time) in state_latency_time_tuples]
        )

        feature_data = pd.DataFrame(np.concatenate([state_latency.state.numpy()
                                                    for (state_latency, time) in state_latency_time_tuples]))

        self.data.append(AnalysisData(policy=policy, epoch=epoch_num,
                         reward_data=reward_data, feature_data=feature_data))

    def export_data(self) -> None:
        prefix = 'train' if self.is_train_data else 'test'
        data_out_folder = self.out_folder / 'data'
        os.makedirs(data_out_folder, exist_ok=True)

        for analysis_data in self.data:
            out_path = data_out_folder / f'{prefix}_{analysis_data.epoch}_{analysis_data.policy}_data.csv'
            combined_df = pd.concat(objs=[analysis_data.reward_data, analysis_data.feature_data], axis=1)
            combined_df.reset_index(drop=True, inplace=True)
            combined_df.to_csv(out_path)

    def run_latency_lin_reg(self) -> None:
        prefix = 'train' if self.is_train_data else 'test'

        for analysis_data in self.data:
            print(f'Executing {prefix} {analysis_data.epoch} {analysis_data.policy} Latency linear regression')
            X = analysis_data.feature_data
            Y = analysis_data.reward_data['Latency']
            print(list(analysis_data.reward_data['Latency']))
            reg = LinearRegression().fit(X, Y)
            print(f'Linear regression score: {reg.score(X, Y)}')
            print(f'Coefficients: {reg.coef_}')
