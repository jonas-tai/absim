from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from simulations.monitor import Monitor
from simulations.client import DataPoint


POLICY_ORDER = ["DQN", "DQN_EXPLR", "ARS", "round_robin", "random"]

POLICY_COLORS = {
    "ARS": "C0",
    "random": "C1",
    "DQN": "C2",
    "round_robin": "C3",
    'DQN_EXPLR': "C4",
}


class ExperimentPlot:
    def __init__(self, plot_folder: Path, data_folder: Path, use_log_scale: bool = False) -> None:
        self.df = None
        self.plot_folder = plot_folder
        self.data_folder = data_folder
        self.use_log_scale = use_log_scale
        self.policy_order = POLICY_ORDER

    def from_csv(self) -> None:
        self.df = pd.read_csv(self.data_folder / f'data.csv')
        self.policy_order = [policy for policy in POLICY_ORDER if policy in self.df['Policy'].unique()]

    def add_data(self, monitor: Monitor, policy: str, epoch_num: int):
        data_point_time_tuples: List[Tuple[DataPoint, float]] = monitor.get_data()
        df_entries = [{
            "Time": time,
            "Latency": data_point.latency,
            "Replica": data_point.replica_id,
            "Epoch": epoch_num,
            "Policy": policy
        } for (data_point, time) in data_point_time_tuples]

        df = pd.DataFrame(df_entries)

        if self.df is None:
            self.df = df
            self.policy_order = [policy for policy in POLICY_ORDER if policy in self.df['Policy'].unique()]
        else:
            self.df = pd.concat((self.df, df), axis=0)
            self.policy_order = [policy for policy in POLICY_ORDER if policy in self.df['Policy'].unique()]

    def get_autotuner_objective(self):
        return - self.df[self.df['Policy'] == 'DQN']['Latency'].quantile(0.99)

    def export_data(self) -> None:
        out_path = self.data_folder / 'data.csv'
        self.df.to_csv(out_path)

    def export_plots(self, file_name: str) -> None:
        plt.savefig(self.plot_folder / f'pdfs/{file_name}.pdf')
        plt.savefig(self.plot_folder / f'{file_name}.jpg')

    def save_stats_to_file(self) -> None:
        out_file = self.data_folder / 'latency_statistics.txt'

        with open(out_file, 'w') as file:
            file.write('Mean and median latency\n')

            # Mean latency
            mean_latency = self.df.groupby(['Policy'])['Latency'].mean()
            file.write('Mean latency:\n')
            file.write(mean_latency.to_string() + '\n\n')

            # Median latency
            median_latency = self.df.groupby(['Policy'])['Latency'].median()
            file.write('Median latency:\n')
            file.write(median_latency.to_string() + '\n\n')

            # Quantiles
            for quantile in [0.9, 0.95, 0.99]:
                file.write(f'Quantile: {quantile}\n')
                # Calculate the quantile latency for each policy and epoch
                quantile_latency = self.df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

                # Calculate the mean of the quantile latency over all epochs for each policy
                mean_quantile_latency = quantile_latency.groupby('Policy')['Latency'].mean()
                file.write(mean_quantile_latency.to_string() + '\n\n')

    def plot_latency(self):
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')

        sns.lineplot(self.df, x="Epoch", y="Latency", hue="Policy", ax=axes, palette=POLICY_COLORS)

        if self.use_log_scale:
            plt.yscale('log')
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'latency_line')
        plt.close()

    def boxplot_latency(self) -> None:
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')

        # Plotting boxplot
        sns.boxplot(x="Policy", y="Latency", data=self.df, ax=axes,
                    hue="Policy", palette=POLICY_COLORS, order=self.policy_order)

        if self.use_log_scale:
            plt.yscale('log')
        axes.set_title('Latency Distribution by Policy')

        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'latency_boxplot')
        plt.close()

    def plot_episode(self, epoch: int):
        # Plots the specified epoch of all policies
        plt.rcParams.update({'font.size': 14})

        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')

        sns.scatterplot(self.df[self.df['Epoch'] == epoch], x="Time", y="Latency",
                        hue="Policy", ax=axes, palette=POLICY_COLORS)

        axes.get_legend().remove()
        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'output_{epoch}_epoch')
        plt.close()

    def plot_policy_episode(self, epoch: int, policy: str):
        plt.rcParams.update({'font.size': 14})

        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.scatterplot(self.df[(self.df['Epoch'] == epoch) & (self.df['Policy'] == policy)], x="Time", y="Latency",
                        hue="Replica", ax=axes)
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
        self.export_plots(file_name=f'request_distribution_{policy}_{epoch}_epoch')

        plt.close()

    def plot_quantile(self, quantile: float):
        plt.rcParams.update({'font.size': 14})

        fig, axes = plt.subplots(figsize=(16, 8), dpi=200, nrows=1, ncols=1, sharex='all')
        quantiles = self.df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

        sns.lineplot(data=quantiles, x="Epoch", y=f'Latency', hue="Policy", ax=axes, palette=POLICY_COLORS)
        plt.title(f'{quantile}th quantile')

        if self.use_log_scale:
            plt.yscale('log')
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'p_{int(quantile * 100)}')
        plt.close()

    def plot_average_quantile_bar(self, quantile: float) -> None:
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(figsize=(10, 6), dpi=200)

        # Calculate the quantile latency for each policy and epoch
        quantile_latency = self.df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

        # Calculate the mean of the quantile latency over all epochs for each policy
        mean_quantile_latency = quantile_latency.groupby('Policy')['Latency'].mean().reset_index()

        # Create bar plot
        sns.barplot(data=mean_quantile_latency, x='Policy', hue='Policy', y='Latency',
                    palette=POLICY_COLORS, ax=axes, order=self.policy_order)
        axes.set_title(f'Average {quantile*100:.0f}th Quantile Latency by Policy over all epochs')

        plt.tight_layout()

        self.export_plots(file_name=f'bar_p_{int(quantile * 100)}')
        plt.close()

    def plot_average_latency_bar(self):
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(figsize=(10, 6), dpi=200)

        # Calculate the mean of the quantile latency over all epochs for each policy
        mean_quantile_latency = self.df.groupby('Policy')['Latency'].mean().reset_index()

        # Create bar plot
        sns.barplot(data=mean_quantile_latency, x='Policy', hue='Policy', y='Latency',
                    palette=POLICY_COLORS, ax=axes, order=self.policy_order)
        axes.set_title(f'Average Latency by Policy over all epochs')

        plt.tight_layout()

        self.export_plots(file_name=f'bar_mean.pdf')
        plt.close()

    def generate_plots(self):
        self.plot_latency
        self.boxplot_latency()
        self.plot_average_latency_bar()

        self.plot_quantile(0.90)
        self.plot_quantile(0.95)
        self.plot_quantile(0.99)
        self.plot_average_quantile_bar(0.90)
        self.plot_average_quantile_bar(0.95)
        self.plot_average_quantile_bar(0.99)
