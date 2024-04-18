import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import pandas as pd

policy_colors = {
    "ARS": "C0",
    "random": "C1",
    "DQN": "C2"
}


def plot(df):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
    sns.lineplot(df, x="Epoch", y="Latency", hue="Policy", ax=axes, palette=policy_colors)
    plt.yscale('log')
    axes.get_legend().remove()

    fig.legend(loc='lower center', ncols=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.ylim(top=150)

    return fig, axes


def plot_episode(df, epoch: int):
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
    sns.scatterplot(df[df['Epoch'] == epoch], x="Time", y="Latency",
                    hue="Policy", ax=axes, palette=policy_colors)
    axes.get_legend().remove()

    fig.legend(loc='lower center', ncols=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.ylim(top=50)
    return fig, axes


def plot_quantile(df, quantile: float, title: str):
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
    quantiles = df.groupby(['Policy', 'Epoch'])['Latency'].quantile(quantile).reset_index()

    sns.lineplot(data=quantiles, x="Epoch", y=f'Latency', hue="Policy", ax=axes, palette=policy_colors)
    plt.title(title)
    plt.yscale('log')
    axes.get_legend().remove()

    fig.legend(loc='lower center', ncols=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.ylim(top=150)

    return fig, axes


base_path = Path('/home/jonas/projects/absim/plots/XXXXX')
df = pd.read_csv(base_path / 'train_data.csv')

out_folder = base_path / 'report'

os.makedirs(out_folder, exist_ok=True)


fig, ax = plot(df)
plt.savefig(out_folder / 'output_train.pdf')
plt.savefig(out_folder / 'output_train.jpg')

fig, ax = plot_quantile(df, 0.90, title='p90 Latency')
plt.savefig(out_folder / 'output_train_p_90.pdf')
plt.savefig(out_folder / 'output_train_p_90.jpg')

fig, ax = plot_quantile(df, 0.95, title='p95 Latency')
plt.savefig(out_folder / 'output_train_p_95.pdf')
plt.savefig(out_folder / 'output_train_p_95.jpg')

fig, ax = plot_quantile(df, 0.99, title='p99 Latency')
plt.savefig(out_folder / 'output_train_p_99.pdf')
plt.savefig(out_folder / 'output_train_p_99.jpg')

plt_episode = 400 - 1
fig, ax = plot_episode(df=df, epoch=plt_episode)
plt.savefig(out_folder / f'output_train_{plt_episode}_epoch.pdf')
plt.savefig(out_folder / f'output_train_{plt_episode}_epoch.jpg')


fig, axes = plot(df)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
