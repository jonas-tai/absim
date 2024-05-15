import os
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, mode: str, data_path: Path, target_col: str, seed: int, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.target_column = target_col
        self.df = pd.read_csv(data_path)

        # split the dataset into train - val - test with the ratio 60 - 20 - 20
        assert mode in ["train", "val", "test"], "wrong mode for dataset given"

        train, test = np.split(self.df.sample(frac=1, random_state=seed), [int(.8 * len(self.df))])

        # train, val, test = np.split(self.df.sample(frac=1,  random_state=seed), [
        #                             int(.6 * len(self.df)), int(.8 * len(self.df))])
        if mode == "train":
            self.df = train
        elif mode == "val":
            raise Exception('Validation not implemented')
            self.df = val
        elif mode == "test":
            self.df = test

        self.data = self.df.loc[:, self.df.columns != self.target_column]
        self.targets = self.df[self.target_column]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        features = self.data.iloc[idx].values.astype(np.float32)
        label = self.targets.iloc[idx]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label
