import seaborn as sns
import pandas as pd

from simulations.monitor import Monitor


class ExperimentPlot:
    def __init__(self):
        self.df = None

    def add_data(self, monitor: Monitor, policy: str, epoch_num: int):
        latencies = monitor.get_data()
        df_entries = [{
            "Latency": l,
            "Epoch": epoch_num,
            "Policy": policy
        } for l in latencies]

        df = pd.DataFrame(df_entries)

        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat((self.df, df), axis=0)

    def plot(self):
        sns.lineplot(self.df, x="Epoch", y="Latency", hue="Policy")
