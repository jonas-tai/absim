import random
from collections import namedtuple, deque

import torch

# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, latency: torch.Tensor) -> None:
        transition = Transition(state, action, next_state, latency)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
