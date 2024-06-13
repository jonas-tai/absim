import random
from collections import namedtuple, deque

import torch

# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# faster implementation https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


random.seed(42)


class ReplayMemory(object):
    def __init__(self, max_size, summary) -> None:
        self.memory = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.summary = summary
        # self.memory = deque([], maxlen=capacity)

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, latency: torch.Tensor) -> None:
        transition = Transition(state, action, next_state, latency)
        self.memory[self.index] = transition
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size
        self.summary.add(state)

    def sample(self, batch_size):
        indices = random.sample(range(self.size), int(batch_size))
        return [self.memory[index] for index in indices]

    def __len__(self):
        return self.size
