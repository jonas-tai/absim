import random
import simpy
import numpy as np


class Simulation(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.random = random.Random()
        self.random_strategy = random.Random()
        self.random_permutations = random.Random()
        self.random_exploration = random.Random()
        self.np_random = np.random.default_rng()

    def set_seed(self, seed):
        self.random = random.Random(seed + 1000)
        self.np_random = np.random.default_rng(seed + 1001)
        self.random_strategy = random.Random(seed + 1002)
        self.random_permutations = random.Random(seed + 1003)
        self.random_exploration = random.Random(seed + 1004)
