import json
import os
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import simulations.constants as const
from simulations.monitor import Monitor
from simulations.client import DataPoint
import numpy as np
from matplotlib.patches import Patch


FONT_SIZE = 10


class ExperimentPlot:
    def __init__(self, plot_folder: Path, data_folder: Path, long_tasks_fraction: float = None, utilization: float | None = None, use_log_scale: bool = False) -> None:
        self.df = None
        self.plot_folder: Path = plot_folder
        self.data_folder: Path = data_folder
        self.use_log_scale: bool = use_log_scale
        self.policy_order: List[str] = const.POLICY_ORDER
        self.utilization = utilization
        self.long_tasks_fraction = long_tasks_fraction

        os.makedirs(plot_folder / 'cdf', exist_ok=True)
        os.makedirs(plot_folder / 'pdfs/cdf', exist_ok=True)

        os.makedirs(plot_folder / 'episode', exist_ok=True)
        os.makedirs(plot_folder / 'pdfs/episode', exist_ok=True)

    def from_csv(self) -> None:
        self.df = pd.read_csv(self.data_folder / f'data.csv')
        self.policy_order = [policy for policy in const.POLICY_ORDER if policy in self.df['Policy'].unique()]

        # Get the value for utilization
        base_folder = self.data_folder.parent
        args_file = base_folder / 'workload_config.json'
        with open(args_file, 'r') as file:
            args_data = json.load(file)

        self.utilization = args_data['utilization']
        self.long_tasks_fraction = args_data['long_tasks_fraction']

    def add_data_from_df(self, additional_data: pd.DataFrame) -> None:
        self.df = pd.concat((self.df, additional_data), axis=0)
        self.policy_order = [policy for policy in const.POLICY_ORDER if policy in self.df['Policy'].unique()]

    def add_data(self, monitor: Monitor, policy: str, epoch_num: int):
        data_point_time_tuples: List[Tuple[DataPoint, float]] = monitor.get_data()
        df_entries = [{
            "Time": time,
            "Latency": data_point.latency,
            "Replica": data_point.replica_id,
            "Is_long_request": data_point.state.is_long_request,
            "Is_faster_response": data_point.is_faster_response,
            "Is_duplicate": data_point.is_duplicate,
            "Task_time_sent": data_point.task_time_sent,
            "Epoch": epoch_num,
            "Policy": policy,
            'Utilization': data_point.utilization,
            'Long_tasks_fraction': data_point.long_tasks_fraction
        } for (data_point, time) in data_point_time_tuples]

        df = pd.DataFrame(df_entries)

        if self.df is None:
            self.df = df.reset_index(drop=True)
            self.policy_order = [policy for policy in const.POLICY_ORDER if policy in self.df['Policy'].unique()]
        else:
            df = df.reset_index(drop=True)  # Reset the index of the new dataframe before concatenation
            self.df = pd.concat((self.df, df), axis=0).reset_index(drop=True)
            self.policy_order = [policy for policy in const.POLICY_ORDER if policy in self.df['Policy'].unique()]

    def get_autotuner_objective(self):
        if len(self.df) == 0:
            print('Empty DF, no result for autotuner')
            return 0
        return - self.df[self.df['Policy'] == 'DQN']['Latency'].quantile(0.99)

    def export_data(self) -> None:
        out_path = self.data_folder / 'data.csv'
        self.df.to_csv(out_path)

    def export_plots(self, file_name: str) -> None:
        plt.savefig(self.plot_folder / f'pdfs/{file_name}.pdf')
        plt.savefig(self.plot_folder / f'{file_name}.jpg')
        plt.close()

    def write_df_stats(self, df, file) -> None:
        file.write('Mean and median latency\n')

        # Mean latency
        mean_latency = df.groupby(['Policy'])['Latency'].mean()
        file.write('Mean latency:\n')
        file.write(mean_latency.to_string() + '\n\n')

        # Median latency
        median_latency = df.groupby(['Policy'])['Latency'].median()
        file.write('Median latency:\n')
        file.write(median_latency.to_string() + '\n\n')

        # Quantiles
        for quantile in [0.9, 0.95, 0.99, 0.999]:
            file.write(f'Quantile: {quantile}\n')
            # Calculate the quantile latency for each policy and epoch
            quantile_latency = df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

            # Calculate the mean of the quantile latency over all epochs for each policy
            mean_quantile_latency = quantile_latency.groupby('Policy')['Latency'].mean()
            file.write(mean_quantile_latency.to_string() + '\n\n')

    def save_stats_to_file(self) -> None:
        out_file = self.data_folder / 'latency_statistics.txt'

        with open(out_file, 'w') as file:
            file.write('Overall stats\n')
            self.write_df_stats(df=self.df, file=file)

            file.write('Long requests stats\n')
            self.write_df_stats(df=self.df[self.df['Is_long_request']], file=file)

            file.write('Short requests stats\n')
            self.write_df_stats(df=self.df[self.df['Is_long_request'] == False], file=file)

    # TODO: Write decorator for before and after plotting settings
    def plot_latency(self):
        plt.rcParams.update({'font.size': FONT_SIZE})
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')

        sns.lineplot(self.df, x="Epoch", y="Latency", hue="Policy", ax=axes, palette=const.POLICY_COLORS)

        if self.use_log_scale:
            plt.yscale('log')
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'latency_line')

    def boxplot_latency(self) -> None:
        plt.rcParams.update({'font.size': FONT_SIZE})
        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        # Plotting boxplot
        sns.boxplot(x="Policy", y="Latency", data=self.df, ax=axes,
                    hue="Policy", palette=const.POLICY_COLORS, order=self.policy_order)

        if self.use_log_scale:
            plt.yscale('log')
        axes.set_title(
            f'Latency Distribution {self.utilization * 100}% utilization with {self.long_tasks_fraction * 100}% long tasks')

        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'latency_boxplot')

    def plot_episode(self, epoch: int):
        # Plots the specified epoch of all policies
        plt.rcParams.update({'font.size': FONT_SIZE})

        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')

        sns.scatterplot(self.df[self.df['Epoch'] == epoch], x="Time", y="Latency",
                        hue="Policy", ax=axes, palette=const.POLICY_COLORS)

        axes.get_legend().remove()
        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'episode/output_{epoch}_epoch')

    def plot_policy_episode(self, epoch: int, policy: str):
        plt.rcParams.update({'font.size': FONT_SIZE})

        fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        sns.scatterplot(self.df[(self.df['Epoch'] == epoch) & (self.df['Policy'] == policy)], x="Time", y="Latency",
                        hue="Replica", ax=axes)
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
        self.export_plots(file_name=f'episode/request_distribution_{policy}_{epoch}_epoch')

    def plot_quantile(self, quantile: float):
        plt.rcParams.update({'font.size': FONT_SIZE})

        fig, axes = plt.subplots(figsize=(16, 8), dpi=200, nrows=1, ncols=1, sharex='all')
        quantiles = self.df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

        sns.lineplot(data=quantiles, x="Epoch", y=f'Latency', hue="Policy", ax=axes, palette=const.POLICY_COLORS)
        plt.title(f'{quantile}th quantile')

        if self.use_log_scale:
            plt.yscale('log')
        axes.get_legend().remove()

        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'p_{int(quantile * 100)}')

    def plot_average_quantile_bar_short_long_requests(self, quantile: float, policies: List[str]) -> None:
        df = self.df[self.df['Policy'].isin(policies) & self.df['Is_long_request']]
        self.plot_average_quantile_bar_generic(
            df, quantile=quantile, title_request_types='long requests', file_prefix=f'long_req_', order=policies)

        df = self.df[self.df['Policy'].isin(policies) & (self.df['Is_long_request'] == False)]
        self.plot_average_quantile_bar_generic(
            df, quantile=quantile, title_request_types='short requests', file_prefix=f'short_req_', order=policies)

    def plot_average_quantile_bar(self, quantile: float) -> None:
        self.plot_average_quantile_bar_generic(
            self.df, quantile=quantile, title_request_types='all requests', file_prefix='all_')

    def plot_average_quantile_bar_generic(self, df: pd.DataFrame, quantile: float, title_request_types: str, file_prefix: str = '', order: List[str] | None = None) -> None:
        if order is None:
            order = self.policy_order

        plt.rcParams.update({'font.size': FONT_SIZE})
        fig, axes = plt.subplots(figsize=(10, 6), dpi=200)

        # Calculate the quantile latency for each policy and epoch
        quantile_latency = df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

        # Calculate the mean of the quantile latency over all epochs for each policy
        mean_quantile_latency = quantile_latency.groupby('Policy')['Latency'].mean().reset_index()

        # Create bar plot
        sns.barplot(data=mean_quantile_latency, x='Policy', hue='Policy', y='Latency',
                    palette=const.POLICY_COLORS, ax=axes, order=order)
        axes.set_title(
            f'Average {quantile*100:.1f}th Quantile Latency {title_request_types} at {self.utilization * 100}% utilization with {self.long_tasks_fraction * 100}% long tasks')
        plt.setp(axes.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        self.export_plots(file_name=f'{file_prefix}bar_p_{int(quantile * 1000)}')

    def plot_cdf_quantile(self, df: pd.DataFrame, quantile: float, title_request_types: str, file_prefix: str = ''):
        if len(df) == 0:
            print(f'Empty df, not continuing in plot_cdf_quantile()')
            return

        plt.rcParams.update({'font.size': FONT_SIZE})

        fig, axes = plt.subplots(figsize=(16, 8), dpi=200, nrows=1, ncols=1, sharex='all')

        # Filter the DataFrame to only include latencies within the specified quantile
        # quantiles = df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()
        filtered_df = df[df.groupby(['Policy', 'Epoch'])[
            'Latency'].transform(lambda x: x > x.quantile(quantile))]

        if len(filtered_df) == 0:
            print(f'Empty df, not continuing in plot_cdf_quantile()')
            return

        # Calculate the CDF for each policy
        cdfs = []
        for policy in filtered_df['Policy'].unique():
            policy_data = filtered_df[filtered_df['Policy'] == policy].sort_values(by='Latency')
            policy_data['CDF'] = policy_data['Latency'].rank(method='first') / len(policy_data)
            cdfs.append(policy_data)

        cdf_df = pd.concat(cdfs)

        # Plot the CDFs
        sns.lineplot(data=cdf_df, x='Latency', y='CDF', hue='Policy', ax=axes)
        plt.title(
            f'Cumulative Distribution Function of {(100.0 - quantile*100):.2f}% slowest {title_request_types} at {self.utilization * 100}% utilization with {self.long_tasks_fraction * 100}% long tasks')

        # if self.use_log_scale:
        plt.xscale('log')

        # Adjust legend position and layout
        axes.get_legend().remove()
        fig.legend(loc='lower center', ncols=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        self.export_plots(file_name=f'cdf/{file_prefix}p_{int(quantile * 100)}')

    def plot_average_latency_bar_short_long_request(self, policies: List[str]) -> None:
        df = self.df[self.df['Policy'].isin(policies) & self.df['Is_long_request']]
        self.plot_average_latency_bar_generic(df, title_request_types='long requests',
                                              file_prefix=f'long_req_', order=policies)

        df = self.df[self.df['Policy'].isin(policies) & (self.df['Is_long_request'] == False)]
        self.plot_average_latency_bar_generic(df, title_request_types='short requests',
                                              file_prefix=f'short_req_', order=policies)

    def plot_average_latency_bar(self):
        self.plot_average_latency_bar_generic(df=self.df, title_request_types='all requests', file_prefix='all_')

    def plot_average_latency_bar_generic(self, df: pd.DataFrame, title_request_types: str, file_prefix: str = '', order: List[str] | None = None) -> None:
        if order is None:
            order = self.policy_order

        plt.rcParams.update({'font.size': FONT_SIZE})
        fig, axes = plt.subplots(figsize=(10, 6), dpi=200)

        # Calculate the mean of the quantile latency over all epochs for each policy
        mean_quantile_latency = df.groupby('Policy')['Latency'].mean().reset_index()

        # Create bar plot
        sns.barplot(data=mean_quantile_latency, x='Policy', hue='Policy', y='Latency',
                    palette=const.POLICY_COLORS, ax=axes, order=order)
        axes.set_title(
            f'Average Latency {title_request_types} at {self.utilization * 100}% utilization with {self.long_tasks_fraction * 100}% long tasks')
        plt.setp(axes.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        self.export_plots(file_name=f'{file_prefix}bar_mean')

    def plot_latency_over_time_short_long_request(self, policies: List[str]) -> None:
        df = self.df[self.df['Policy'].isin(policies) & self.df['Is_long_request']]
        self.plot_latency_over_time(df, title_request_types='long requests',
                                    file_prefix=f'long_req_', order=policies)

        df = self.df[self.df['Policy'].isin(policies) & (self.df['Is_long_request'] == False)]
        self.plot_latency_over_time(df, title_request_types='short requests',
                                    file_prefix=f'short_req_', order=policies)

    def plot_latency_over_time(self, df: pd.DataFrame, order: List[str], title_request_types: staticmethod = 'all requests', file_prefix='all_', quantile=0.99):
        percentile = quantile * 100
        df = df[df['Is_faster_response']]
        for epoch in df['Epoch'].unique():
            print(epoch)
            epoch_df = df[df['Epoch'] == epoch]
            AGGREGATION_FACTOR = 4000  # Define your aggregation factor

            fig, axes = plt.subplots(figsize=(14, 8), dpi=200)

            epoch_df = epoch_df.sort_values(by=['Policy', 'Task_time_sent']).reset_index(drop=True)
            epoch_df['Num_request'] = epoch_df.groupby('Policy').cumcount()
            epoch_df['Aggregated_num_requests'] = (epoch_df['Num_request'] // AGGREGATION_FACTOR) * AGGREGATION_FACTOR
            epoch_df['Aggregated_num_requests'] += AGGREGATION_FACTOR

            # Adjust the last group to reflect the actual number of requests
            max_num_request = epoch_df.groupby('Policy')['Num_request'].transform('max')
            epoch_df.loc[epoch_df['Aggregated_num_requests'] == (AGGREGATION_FACTOR + (
                max_num_request // AGGREGATION_FACTOR) * AGGREGATION_FACTOR), 'Aggregated_num_requests'] = max_num_request

            # Combine Utilization and Long_tasks_fraction into a single feature
            epoch_df['Workload_key'] = list(zip(epoch_df['Utilization'], epoch_df['Long_tasks_fraction']))

            # Collect the workload changes
            workload_change_df = epoch_df.groupby(['Policy', 'Workload_key']).agg(
                Start_req=('Num_request', 'min'),
                End_req=('Num_request', 'max')
            ).reset_index()

            workload_changes = list(zip(workload_change_df['Workload_key'],
                                    workload_change_df['Start_req'], workload_change_df['End_req']))

            # Aggregate latency over groups of data points per policy
            aggregated = epoch_df.groupby(['Policy', 'Aggregated_num_requests']).agg(
                Start_Time=('Task_time_sent', 'min'),
                End_Time=('Task_time_sent', 'max'),
                Latency=('Latency', lambda x: np.percentile(x, percentile)),
            ).reset_index()

            # Plot latency over time for each policy
            sns.lineplot(data=aggregated, x='Aggregated_num_requests', y='Latency',
                         hue='Policy', style='Policy')  # palette=const.POLICY_COLORS,

            # Adding color indicators for long_tasks_fraction
            workload_keys = workload_change_df['Workload_key'].values

            # Create a colormap to represent the combined Utilization and Long_tasks_fraction
            unique_combinations = list(set(workload_keys))
            color_map = plt.cm.get_cmap('crest', len(unique_combinations))
            norm = plt.Normalize(vmin=0, vmax=len(unique_combinations) - 1)

            # Map each unique combination to a color
            combination_to_color = {combination: color_map(norm(i))
                                    for i, combination in enumerate(unique_combinations)}

            previous_utl_ltf = workload_keys[0]
            for (workload_key, start, end) in workload_changes:
                # Add a vertical line at the change point
                plt.axvline(x=end, color='grey', linestyle='--', alpha=0.7)
                color = combination_to_color[workload_key]
                plt.axvspan(start, end, color=color, ymin=0.95, ymax=1.0, alpha=0.8)

            # Adding a color bar for the long_tasks_fractions
            # sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            # cbar = plt.colorbar(sm, orientation='horizontal', pad=0.1, ax=axes)
            # cbar.set_label('Utilization and Long Tasks Fraction')

            plt.xlabel('Number of requests')
            plt.ylabel('Latency (ms)')
            plt.yscale('log')

            axes.set_title(f'P{percentile} latency for {title_request_types}')
            handles, labels = axes.get_legend_handles_labels()
            # Add the color bar as a patch in the legend
            # cbar_patch = Patch(facecolor=color_map(norm(np.mean(long_tasks_fractions))), edgecolor='black')

            # Create a patch for each unique combination of Utilization and Long_tasks_fraction
            for combination, color in combination_to_color.items():
                utilization, long_tasks_fraction = combination
                label = f'Utilization: {utilization:.2f}, Long Tasks Fraction: {long_tasks_fraction:.2f}'
                patch = Patch(facecolor=color, edgecolor='black', label=label)
                handles.append(patch)

            # Place the legend below the graph
            axes.legend(handles=handles, title='Policy and Settings',
                        loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

            plt.tight_layout()

            self.export_plots(file_name=f'{epoch}_{file_prefix}latency_over_time')

    def generate_plots(self) -> None:
        # reduced_policies = ['ARS', 'DQN', 'DQN_DUPL'] + ['DQN_EXPLR_0',
        #                                                 'DQN_EXPLR_10', 'DQN_EXPLR_15', 'DQN_EXPLR_20', 'DQN_EXPLR_25']
        reduced_policies = [policy for policy in self.policy_order if policy not in ['random', 'DQN_DUPL']]
        print(reduced_policies)

        print('Before')
        print(len(self.df))
        self.df = self.df[self.df['Is_faster_response']]
        print('After')
        print(len(self.df))

        self.plot_latency()
        self.boxplot_latency()
        self.plot_average_latency_bar()
        self.plot_average_latency_bar_short_long_request(policies=reduced_policies)

        self.plot_quantile(0.90)
        self.plot_quantile(0.95)
        self.plot_quantile(0.99)
        self.plot_average_quantile_bar(0.90)
        self.plot_average_quantile_bar(0.95)
        self.plot_average_quantile_bar(0.99)
        self.plot_average_quantile_bar(0.999)
        self.plot_average_quantile_bar_short_long_requests(quantile=0.9, policies=reduced_policies)
        self.plot_average_quantile_bar_short_long_requests(quantile=0.95, policies=reduced_policies)
        self.plot_average_quantile_bar_short_long_requests(quantile=0.99, policies=reduced_policies)
        self.plot_average_quantile_bar_short_long_requests(quantile=0.999, policies=reduced_policies)

        cdf_policies = ['ARS', 'DQN', 'DQN_EXPLR_10_TRAIN']
        df = self.df[self.df['Policy'].isin(cdf_policies)]
        self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='all requests', file_prefix=f'all_req_')

        df = self.df[self.df['Policy'].isin(cdf_policies) & self.df['Is_long_request']]
        self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='long requests', file_prefix=f'long_req_')

        df = self.df[self.df['Policy'].isin(cdf_policies) & (self.df['Is_long_request'] == False)]
        self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='short requests',
                               file_prefix=f'short_req_')

        # cdf_policies = ['ARS', 'DQN', 'DQN_DUPL', 'DQN_EXPLR_10_TRAIN']
        # df = self.df[self.df['Policy'].isin(cdf_policies)]
        # self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='all requests', file_prefix=f'dupl_all_req_')

        # df = self.df[self.df['Policy'].isin(cdf_policies) & self.df['Is_long_request']]
        # self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='long requests',
        #                        file_prefix=f'dupl_long_req_')

        # df = self.df[self.df['Policy'].isin(cdf_policies) & (self.df['Is_long_request'] == False)]
        # self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='short requests',
        #                        file_prefix=f'dupl_short_req_')

        cdf_policies = reduced_policies
        df = self.df[self.df['Policy'].isin(cdf_policies)]
        self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='all requests', file_prefix=f'explr_all_req_')

        df = self.df[self.df['Policy'].isin(cdf_policies) & self.df['Is_long_request']]
        self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='long requests',
                               file_prefix=f'explr_long_req_')

        df = self.df[self.df['Policy'].isin(cdf_policies) & (self.df['Is_long_request'] == False)]
        self.plot_cdf_quantile(df=df, quantile=0.99, title_request_types='short requests',
                               file_prefix=f'explr_short_req_')
