import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from simulations.plotting import ExperimentPlot

for i in range(106, 107):
    mode = 'test'
    base_path = Path(f'/home/jonas/projects/absim/outputs/{i}/{mode}')

    data_folder = base_path / 'data'
    plot_folder = base_path / 'plots'

    plotter = ExperimentPlot(plot_folder=plot_folder, data_folder=data_folder)

    plotter.from_csv()

    plotter.generate_plots()
