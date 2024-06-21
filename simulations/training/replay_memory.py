from pathlib import Path
import pickle
import random
from collections import namedtuple, deque

import torch

from simulations.models.dqn import SummaryStats

# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# faster implementation https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


random.seed(42)


class ReplayMemory(object):
    def __init__(self, max_size: int, always_use_newest: bool = False) -> None:
        self.memory = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.newest = None
        self.always_use_newest = always_use_newest  # Use https://arxiv.org/pdf/1712.01275
        # self.memory = deque([], maxlen=capacity)

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, latency: torch.Tensor) -> None:
        transition = Transition(state, action, next_state, latency)
        self.push_transition(transition=transition)

    def push_transition(self, transition: Transition) -> None:
        self.memory[self.index] = transition
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size
        self.newest = transition

    def sample(self, batch_size):
        if self.always_use_newest:
            indices = random.sample(range(self.size), int(batch_size) - 1)
            return [self.newest] + [self.memory[index] for index in indices]
        else:
            indices = random.sample(range(self.size), int(batch_size))
            return [self.memory[index] for index in indices]

    def __len__(self):
        return self.size

    def save_to_file(self, model_folder: Path) -> None:
        file_path = model_folder / 'replay_memory_state.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(model_folder: Path):
        file_path = model_folder / 'replay_memory_state.pkl'
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class ReplayMemoryWithSummary(ReplayMemory):
    def __init__(self, max_size: int, summary: SummaryStats, always_use_newest: bool = False) -> None:
        super().__init__(max_size=max_size, always_use_newest=always_use_newest)
        self.summary = summary

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, latency: torch.Tensor) -> None:
        super().push(state=state, action=action, next_state=next_state, latency=latency)
        self.summary.add(state)

    def save_to_file(self, model_folder: Path) -> None:
        file_path = model_folder / 'replay_memory_with_summary_state.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(model_folder: Path):
        file_path = model_folder / 'replay_memory_with_summary_state.pkl'
        with open(file_path, 'rb') as f:
            return pickle.load(f)
