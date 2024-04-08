from global_sim import Simulation
import numpy as np


class Monitor():
    def __init__(self, name=""):
        self.name = name
        self.data = []

    def observe(self, y, t=None):
        if t is None:
            self.data.append((y, Simulation.now))
        else:
            self.data.append((y, t))

    def get_data(self):
        return [x[0] for x in self.data]

    def mean(self):
        return np.mean(self.get_data())

    def percentile(self, p):
        return np.percentile(self.get_data(), p)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
