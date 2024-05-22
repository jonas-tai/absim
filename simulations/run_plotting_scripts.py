import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

POLICY_ORDER = ["DQN", "ARS", "round_robin", "random"]

POLICY_COLORS = {
    "ARS": "C0",
    "random": "C1",
    "DQN": "C2",
    "round_robin": "C3",
    'DQN_EXPLR': "C4",
}

mode = 'test'
base_path = Path('/home/jonas/projects/absim/plots/165')
df = pd.read_csv(base_path / f'{mode}_data.csv')

out_folder = base_path / mode
os.makedirs(out_folder, exist_ok=True)


def plot(df):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')

    print(f'Mean and median latency')
    # print(df[df['Epoch'] > 20].groupby(['Policy'])['Latency'].mean())
    # print(df[df['Epoch'] > 20].groupby(['Policy'])['Latency'].median())

    print(df.groupby(['Policy'])['Latency'].mean())
    print(df.groupby(['Policy'])['Latency'].median())

    sns.lineplot(df, x="Epoch", y="Latency", hue="Policy", ax=axes, palette=POLICY_COLORS)
    # plt.yscale('log')
    axes.get_legend().remove()

    fig.legend(loc='lower center', ncols=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # plt.ylim(top=150)

    plt.savefig(out_folder / 'mean_latency.pdf')
    plt.savefig(out_folder / 'mean_latency.jpg')

    return fig, axes


def boxplot_latency(df):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')

    # Plotting boxplot
    sns.boxplot(x="Policy", y="Latency", data=df, ax=axes, hue="Policy", palette=POLICY_COLORS, order=POLICY_ORDER)
    plt.yscale('log')
    axes.set_title('Latency Distribution by Policy')

    fig.legend(loc='lower center', ncols=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    plt.savefig(out_folder / 'boxplot.pdf')
    plt.savefig(out_folder / 'boxplot.jpg')

    return fig, axes


def plot_episode(df, epoch: int):
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
    sns.scatterplot(df[df['Epoch'] == epoch], x="Time", y="Latency",
                    hue="Policy", ax=axes, palette=POLICY_COLORS)
    axes.get_legend().remove()

    fig.legend(loc='lower center', ncols=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # plt.ylim(top=50)
    return fig, axes


def plot_quantile(df, quantile: float, title: str):
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
    print(f'Quantile: {quantile}')
    # print(df[df['Epoch'] > 20].groupby(['Policy'])['Latency'].quantile(quantile))
    print(df.groupby(['Policy'])['Latency'].quantile(quantile))

    quantiles = df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

    sns.lineplot(data=quantiles, x="Epoch", y=f'Latency', hue="Policy", ax=axes, palette=POLICY_COLORS)
    plt.title(title)
    # plt.yscale('log')
    axes.get_legend().remove()

    fig.legend(loc='lower center', ncols=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # plt.ylim(top=150)

    plt.savefig(out_folder / f'p_{quantile * 100}.pdf')
    plt.savefig(out_folder / f'p_{quantile * 100}.jpg')
    return fig, axes


def plot_average_quantile_bar(df, quantile: float):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(figsize=(10, 6), dpi=200)

    # Calculate the quantile latency for each policy and epoch
    quantile_latency = df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

    # Calculate the mean of the quantile latency over all epochs for each policy
    mean_quantile_latency = quantile_latency.groupby('Policy')['Latency'].mean().reset_index()

    # Create bar plot
    sns.barplot(data=mean_quantile_latency, x='Policy', hue='Policy', y='Latency',
                palette=POLICY_COLORS, ax=axes, order=POLICY_ORDER)
    axes.set_title(f'Average {quantile*100:.0f}th Quantile Latency by Policy')

    plt.tight_layout()

    plt.savefig(out_folder / f'bar_p_{quantile * 100}.pdf')
    plt.savefig(out_folder / f'bar_p_{quantile * 100}.jpg')

    return fig, axes


def plot_average_latency_bar(df):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(figsize=(10, 6), dpi=200)

    # Calculate the mean of the quantile latency over all epochs for each policy
    mean_quantile_latency = df.groupby('Policy')['Latency'].mean().reset_index()

    # Create bar plot
    sns.barplot(data=mean_quantile_latency, x='Policy', hue='Policy', y='Latency',
                palette=POLICY_COLORS, ax=axes, order=POLICY_ORDER)
    axes.set_title(f'Average Latency by Policy over all test runs')

    plt.tight_layout()

    plt.savefig(out_folder / f'bar_mean.pdf')
    plt.savefig(out_folder / f'bar_mean.jpg')

    return fig, axes


print(f'Analyzing output for {base_path}')

fig, ax = plot(df)

fig, ax = boxplot_latency(df)
fig, ax = plot_average_latency_bar(df)

fig, ax = plot_quantile(df, 0.90, title='p90 Latency')
fig, ax = plot_quantile(df, 0.95, title='p95 Latency')
fig, ax = plot_quantile(df, 0.99, title='p99 Latency')

fig, ax = plot_average_quantile_bar(df, 0.90)
fig, ax = plot_average_quantile_bar(df, 0.95)
fig, ax = plot_average_quantile_bar(df, 0.99)


# plt_episode = 400 - 1
# fig, ax = plot_episode(df=df, epoch=plt_episode)
# plt.savefig(out_folder / f'output_train_{plt_episode}_epoch.pdf')
# plt.savefig(out_folder / f'output_train_{plt_episode}_epoch.jpg')
