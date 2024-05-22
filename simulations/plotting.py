from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from simulations.monitor import Monitor
from simulations.client import DataPoint


class ExperimentPlot:
    def __init__(self, plot_folder: Path, data_folder: Path):
        self.df = None
        self.policy_colors = {
            "ARS": "C0",
            "random": "C1",
            "DQN": "C2",
            "round_robin": "C3",
            'DQN_EXPLR': "C4",
        }
        self.plot_folder = plot_folder
        self.data_folder = data_folder

    def add_data(self, monitor: Monitor, policy: str, epoch_num: int):
        latency_replica_time_tuples: List[Tuple[DataPoint, float]] = monitor.get_data()
        df_entries = [{
            "Time": time,
            "Latency": latency_replica.latency,
            "Replica": latency_replica.replica_id,
            "Epoch": epoch_num,
            "Policy": policy
        } for (latency_replica, time) in latency_replica_time_tuples]

        df = pd.DataFrame(df_entries)

        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat((self.df, df), axis=0)

    def get_autotuner_objective(self):
        return - self.df[self.df['Policy'] == 'DQN']['Latency'].quantile(0.99)

    def export_data(self, file_name: str = 'train_data.csv'):
        out_path = self.data_folder / file_name
        self.df.to_csv(out_path)

    def export_plots(self, file_name: str) -> None:
        plt.savefig(self.plot_folder / f'pdfs/{file_name}.pdf')
        plt.savefig(self.plot_folder / f'{file_name}.jpg')

    def plot_latency(self):
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.lineplot(self.df, x="Epoch", y="Latency", hue="Policy", ax=axes, palette=self.policy_colors)
        plt.yscale('log')
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)

        self.export_plots(file_name=f'latency')

        return fig, axes

    # Plots the last episode of all policies
    def plot_episode(self, epoch: int):
        plt.rcParams.update({'font.size': 14})

        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.scatterplot(self.df[self.df['Epoch'] == epoch], x="Time", y="Latency",
                        hue="Policy", ax=axes, palette=self.policy_colors)
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)

        self.export_plots(file_name=f'output_{epoch}_epoch')
        return fig, axes

    def plot_policy_episode(self, epoch: int, policy: str):
        plt.rcParams.update({'font.size': 14})

        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.scatterplot(self.df[(self.df['Epoch'] == epoch) & (self.df['Policy'] == policy)], x="Time", y="Latency",
                        hue="Replica", ax=axes)
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
        self.export_plots(file_name=f'output_{policy}_{epoch}_epoch')
        return fig, axes

    def plot_quantile(self, quantile: float):
        plt.rcParams.update({'font.size': 14})

        fig, axes = plt.subplots(figsize=(16, 8), dpi=200, nrows=1, ncols=1, sharex='all')
        quantiles = self.df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

        sns.lineplot(data=quantiles, x="Epoch", y=f'Latency', hue="Policy", ax=axes, palette=self.policy_colors)
        plt.title(f'{quantile}th quantile')
        plt.yscale('log')
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)

        self.export_plots(file_name=f'output_p_{int(quantile * 100)}')
        return fig, axes
