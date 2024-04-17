import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from simulations.monitor import Monitor


class ExperimentPlot:
    def __init__(self):
        self.df = None

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

    def plot(self):
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.lineplot(self.df, x="Epoch", y="Latency", hue="Policy", ax=axes)
        plt.yscale('log')
        return fig, axes

    def plot_episode(self, epoch: int):
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.scatterplot(self.df[self.df['Epoch'] == epoch], x="Time", y="Latency", hue="Policy", ax=axes)
        return fig, axes

    def plot_quantile(self, quantile: float):
        fig, axes = plt.subplots(figsize=(16, 8), dpi=200, nrows=1, ncols=1, sharex='all')
        quantiles = self.df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

        sns.lineplot(data=quantiles, x="Epoch", y=f'Latency', hue="Policy", ax=axes)
        plt.title(f'{quantile}th quantile')
        plt.yscale('log')
        return fig, axes
