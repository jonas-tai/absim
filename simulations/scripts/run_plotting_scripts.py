import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from simulations.plotting import ExperimentPlot
from simulations.state import StateParser


MODE = 'recalculate_norm'
EXPERIMENT = 'end_to_end_test_2'  # 'short_long_reward_norm_model_and_adapt'

state_parser = StateParser(num_servers=5,
                           num_request_rates=3,
                           poly_feat_degree=2)


def get_directory_paths(folder_path):
    directory_paths = [Path(os.path.join(folder_path, d)) for d in os.listdir(folder_path)
                       if os.path.isdir(os.path.join(folder_path, d)) and (d.startswith('variable_long_task_fraction') or d.startswith('chained') or d.startswith('base'))]
    return directory_paths
# /data1/outputs/short_long_reward_norm_model_and_adapt/1/base_50.00_util_0.00_long_tasks_532508


def run_plots():
    for i in range(27, 36):
        directories = get_directory_paths(f'/data1/outputs/{EXPERIMENT}/{i}/')

        # base_path = Path(
        # f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{i}/variable_long_task_fraction_updated_long_tasks_20.00_util_20.00_long_tasks')
        for base_path in directories:
            data_folder = base_path / 'data'
            plot_folder = base_path / 'plots'

            plotter = ExperimentPlot(plot_folder=plot_folder, data_folder=data_folder,
                                     state_parser=state_parser, retrain_interval=None)

            try:
                plotter.from_csv()

                plotter.plot_latency_episode_average_short_long_request(
                    policies=['OFFLINE_DQN', 'ARS', 'OFFLINE_DQN_DUPL_10_TRAIN', 'OFFLINE_DQN_DUPL_20_TRAIN', 'OFFLINE_DQN_EXPLR_0_TRAIN', 'OFFLINE_DQN_EXPLR_10_TRAIN', 'OFFLINE_DQN_EXPLR_20_TRAIN', ])
                plotter.plot_latency_over_time_short_long_request(
                    policies=['OFFLINE_DQN', 'ARS', 'OFFLINE_DQN_DUPL_10_TRAIN', 'OFFLINE_DQN_DUPL_20_TRAIN', 'OFFLINE_DQN_EXPLR_0_TRAIN', 'OFFLINE_DQN_EXPLR_10_TRAIN', 'OFFLINE_DQN_EXPLR_20_TRAIN', ])
                plotter.generate_plots()
            except Exception as e:
                print(e)
                print(f'Warning, import failed for {base_path}, continuing')
                continue


run_plots()

# base_path = Path(
# f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{i}/variable_long_task_fraction_updated_long_tasks_20.00_util_20.00_long_tasks')
# for base_path in ['/dev/shm/outputs/story_experiments/58/base_45.00_util_30.00_long_tasks_532508/', '/dev/shm/outputs/story_experiments/58/base_45.00_util_0.00_long_tasks_532508/',
#                     '/dev/shm/outputs/story_experiments/10/base_45.00_util_30.00_long_tasks_532508/']:
#     base_path = Path(base_path)
#     data_folder = base_path / 'data'
#     plot_folder = base_path / 'plots'

#     plotter = ExperimentPlot(plot_folder=plot_folder, data_folder=data_folder, state_parser=state_parser)

#     plotter.from_csv()
#     plotter.generate_plots()
# plotter.plot_latency_over_time_short_long_request(
#     policies=['OFFLINE_DQN', 'ARS', 'OFFLINE_DQN_DUPL_10_TRAIN', 'OFFLINE_DQN_DUPL_20_TRAIN', 'OFFLINE_DQN_DUPL_30_TRAIN', 'OFFLINE_DQN_EXPLR_10_TRAIN', 'OFFLINE_DQN_EXPLR_20_TRAIN', 'OFFLINE_DQN_EXPLR_30_TRAIN'])


# for (long_tasks, i) in [('0.00', 9), ('10.00', 10), ('20.00', 11), ('30.00', 12), ('40.00', 13), ('50.00', 14), ('60.00', 15), ('70.00', 16), ('80.00', 17)]:  # [('10.00', 1),
#     additional_data_path = Path(
#         f'/home/jonas/projects/absim/outputs/new_benchmarks/{i}/base_45.00_util_{long_tasks}_long_tasks/data')
#     additional_data_df = pd.read_csv(additional_data_path / f'data.csv')
#     additional_data_df = additional_data_df[additional_data_df['Policy'] == 'DQN']
#     additional_data_df['Policy'] = 'DQN_OPTIMIZED'

#     for j in [3]:
#         # base_path = Path(f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{i}/{MODE}')
#         base_path = Path(f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{j}/base_45.00_util_{long_tasks}_long_tasks')

#         data_folder = base_path / 'data'
#         plot_folder = base_path / 'plots'

#         plotter = ExperimentPlot(plot_folder=plot_folder, data_folder=data_folder)

#         plotter.from_csv()
#         plotter.add_data_from_df(additional_data=additional_data_df)

#         plotter.generate_plots()
#         plotter.save_stats_to_file()


# [('0.00', 18), ('10.00', 19), ('20.00', 20), ('30.00', 21), ('40.00', 22), ('50.00', 23), ('60.00', 24), ('70.00', 25), ('80.00', 26)]
# for (long_tasks, i) in [('0.00', 18), ('20.00', 20), ('40.00', 22), ('60.00', 24), ('80.00', 26)]:  # [('10.00', 1),
#     additional_data_path = Path(
#         f'/home/jonas/projects/absim/outputs/new_benchmarks/{i}/base_45.00_util_{long_tasks}_long_tasks/data')
#     additional_data_df = pd.read_csv(additional_data_path / f'data.csv')
#     additional_data_df = additional_data_df[additional_data_df['Policy'] == 'DQN']
#     additional_data_df['Policy'] = 'DQN_OPTIMIZED'

#     for j in [4]:
#         # base_path = Path(f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{i}/{MODE}')
#         base_path = Path(f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{j}/base_45.00_util_{long_tasks}_long_tasks')

#         data_folder = base_path / 'data'
#         plot_folder = base_path / 'plots'

#         plotter = ExperimentPlot(plot_folder=plot_folder, data_folder=data_folder)

#         plotter.from_csv()
#         plotter.add_data_from_df(additional_data=additional_data_df)

#         plotter.generate_plots()
#         plotter.save_stats_to_file()


# for (long_tasks, i) in [('0.00', 18), ('10.00', 19), ('20.00', 20), ('30.00', 21), ('40.00', 22), ('50.00', 23), ('60.00', 24), ('70.00', 25), ('80.00', 26)]:  # [('10.00', 1),
#     # additional_data_path = Path(
#     #     f'/home/jonas/projects/absim/outputs/new_benchmarks/{i}/base_45.00_util_{long_tasks}_long_tasks/data')
#     # additional_data_df = pd.read_csv(additional_data_path / f'data.csv')
#     # additional_data_df = additional_data_df[additional_data_df['Policy'] == 'DQN']
#     # additional_data_df['Policy'] = 'DQN_OPTIMIZED'

#     for j in [1]:
#         # base_path = Path(f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{i}/{MODE}')
#         base_path = Path(f'/home/jonas/projects/absim/outputs/{EXPERIMENT}/{j}/base_45.00_util_{long_tasks}_long_tasks')

#         data_folder = base_path / 'data'
#         plot_folder = base_path / 'plots'

#         plotter = ExperimentPlot(plot_folder=plot_folder, data_folder=data_folder)

#         plotter.from_csv()
#         # plotter.add_data_from_df(additional_data=additional_data_df)

#         plotter.generate_plots()
#         plotter.save_stats_to_file()
