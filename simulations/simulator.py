import random
import simpy
import numpy as np


class Simulation(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.random = random.Random()
        self.np_random = np.random.default_rng()

    def set_seed(self, seed):
        self.random = random.Random(seed)
        self.np_random = np.random.default_rng(seed)
