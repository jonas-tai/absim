import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


mode = 'test'
base_path = Path('/home/jonas/projects/absim/outputs/165')
df = pd.read_csv(base_path / f'{mode}_data.csv')

out_folder = base_path / mode
os.makedirs(out_folder, exist_ok=True)
