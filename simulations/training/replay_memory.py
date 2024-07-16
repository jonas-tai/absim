from pathlib import Path
import pickle
import random
from collections import namedtuple, deque
from typing import List

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
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        # self.memory = deque([], maxlen=capacity)

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, latency: torch.Tensor) -> None:
        transition = Transition(state, action, next_state, latency)
        self.push_transition(transition=transition)

    def push_transition(self, transition: Transition) -> None:
        self.memory[self.index] = transition
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size
        self.newest = transition

    def get_stored_transitions(self) -> List[Transition]:
        return self.memory[:self.size]

    def extend_buffer(self, transitions: List[Transition]) -> None:
        # Fills empty slots of the memory with new elements and extends the buffer to the size needed
        num_trans_fittig = min(self.max_size - self.size, len(transitions))
        self.memory[self.size:(self.size + num_trans_fittig)] = transitions[:num_trans_fittig]
        self.size += num_trans_fittig

        added_len = len(transitions) - num_trans_fittig
        self.max_size += added_len
        self.size += added_len
        self.memory += transitions[num_trans_fittig:]

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
    def load_from_file(model_folder: Path, size: int, replay_always_use_newest: bool):
        file_path = model_folder / 'replay_memory_state.pkl'
        with open(file_path, 'rb') as f:
            replay_memory = pickle.load(f)
            replay_memory.always_use_newest = replay_always_use_newest
            if size < replay_memory.max_size:
                replay_memory.memory = replay_memory.memory[:size]
            elif size > replay_memory.max_size:
                replay_memory.memory = replay_memory.memory + ([None] * (size - replay_memory.max_size))
            replay_memory.max_size = size
            replay_memory.size = min(size, replay_memory.size)
            replay_memory.index = min(size, replay_memory.index)
            replay_memory.to_device()
            return replay_memory

    def to_device(self) -> None:
        """Move all tensors in the replay memory to the specified device."""
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        for i in range(self.size):
            transition = self.memory[i]

            if transition is not None:
                self.memory[i] = Transition(
                    state=transition.state.to(device),
                    action=transition.action.to(device),
                    next_state=transition.next_state.to(device),
                    reward=transition.reward.to(device)
                )
        if self.newest is not None:
            self.newest = Transition(
                state=self.newest.state.to(device),
                action=self.newest.action.to(device),
                next_state=self.newest.next_state.to(device),
                reward=self.newest.reward.to(device)
            )


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
    def load_from_file(model_folder: Path, size: int):
        file_path = model_folder / 'replay_memory_with_summary_state.pkl'
        with open(file_path, 'rb') as f:
            replay_memory = pickle.load(f)
            if size < replay_memory.size:
                replay_memory.memory = replay_memory.memory[:size]
            elif size > replay_memory.size:
                replay_memory.memory = replay_memory.memory + [None] * (size - replay_memory.size)
            replay_memory.size = size
            replay_memory.index = 0

            replay_memory.to_device()
            return replay_memory

    def to_device(self) -> None:
        """Move all tensors in the replay memory to the specified device."""
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        for i in range(self.size):
            transition = self.memory[i]
            if transition is not None:
                self.memory[i] = Transition(
                    state=transition.state.to(device),
                    action=transition.action.to(device),
                    next_state=transition.next_state.to(device),
                    reward=transition.reward.to(device)
                )
        if self.newest is not None:
            self.newest = Transition(
                state=self.newest.state.to(device),
                action=self.newest.action.to(device),
                next_state=self.newest.next_state.to(device),
                reward=self.newest.reward.to(device)
            )
