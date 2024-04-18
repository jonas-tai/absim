from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from simulations.monitor import Monitor


class ExperimentPlot:
    def __init__(self):
        self.df = None
        self.policy_colors = {
            "ARS": "C0",
            "random": "C1",
            "DQN": "C2"
        }

    def add_data(self, monitor: Monitor, policy: str, epoch_num: int):
        latency_time_tuples = monitor.get_data()
        df_entries = [{
            "Time": time,
            "Latency": latency,
            "Epoch": epoch_num,
            "Policy": policy
        } for (latency, time) in latency_time_tuples]

        df = pd.DataFrame(df_entries)

        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat((self.df, df), axis=0)

    def export_data(self, out_folder: Path, file_name: str = 'train_data.csv'):
        out_path = out_folder / file_name
        self.df.to_csv(out_path)

    def plot(self):
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.lineplot(self.df, x="Epoch", y="Latency", hue="Policy", ax=axes, palette=self.policy_colors)
        plt.yscale('log')
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)

        return fig, axes

    def plot_episode(self, epoch: int):
        plt.rcParams.update({'font.size': 14})

        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.scatterplot(self.df[self.df['Epoch'] == epoch], x="Time", y="Latency",
                        hue="Policy", ax=axes, palette=self.policy_colors)
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
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

        return fig, axes
