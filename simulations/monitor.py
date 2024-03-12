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

    def mean(self):
        return np.mean([x[0] for x in self.data])

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)