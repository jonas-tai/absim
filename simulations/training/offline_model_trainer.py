import json
import os
import math
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple
import constants as const

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

from simulations.training.replay_memory import ReplayMemory, Transition
from simulations.models.dqn import DQN
from simulations.state import State, StateParser
from collections import defaultdict

from simulations.task import Task
from simulations.training.norm_stats import NormStats
# from simulations.training.training_data_collector import TrainingDataCollector

MODEL_TRAINER_JSON = 'model_trainer.json'


class OfflineTrainer:
    def __init__(self, state_parser: StateParser, model_structure: str, n_actions: int, add_retrain_to_expert_buffer: bool,
                 replay_always_use_newest: bool, offline_train_epoch_len: int, replay_mem_retrain_expert_fraction: float,
                 use_sliding_retrain_memory: bool, sliding_window_mem_size: int,
                 batch_size=128, gamma=0.8, eps_start=0.2, eps_end=0.2, eps_decay=1000, tau=0.005, lr=1e-4,
                 tau_decay=10, clipping_value=1):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.state_parser = state_parser
        self.offline_train_epoch_len = offline_train_epoch_len

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.TAU_DECAY = tau_decay
        self.LR = lr
        self.CLIPPING_VALUE = clipping_value
        self.replay_mem_retrain_expert_fraction = replay_mem_retrain_expert_fraction

        # Make sure we dont add data to expert data if using sliding retrain memory
        assert not (add_retrain_to_expert_buffer and use_sliding_retrain_memory)
        self.add_retrain_to_expert_buffer = add_retrain_to_expert_buffer
        self.use_sliding_retrain_memory = use_sliding_retrain_memory
        self.sliding_replay_mem_size = sliding_window_mem_size

        self.expert_batch_size = int(self.BATCH_SIZE * self.replay_mem_retrain_expert_fraction)
        self.retrain_batch_size = self.BATCH_SIZE - self.expert_batch_size
        # self.lr_scheduler_step_size = lr_scheduler_step_size
        # self.lr_scheduler_gamma = lr_scheduler_gamma

        self.model_folder: Path | None = None
        self.expert_data_folder: Path | None = None

        self.explore_actions_episode = 0
        self.exploit_actions_episode = 0

        self.losses = []
        self.grads = []
        self.mean_value = []
        self.reward_logs = []

        # num servers
        self.n_actions = n_actions
        self.n_observations = self.state_parser.get_state_size()
        self.model_structure = model_structure

        self.reward_mean = torch.zeros(1)
        self.reward_std = torch.ones(1)

        self.feature_mean = torch.zeros(self.n_observations)
        self.feature_std = torch.ones(self.n_observations)

        self.do_active_retraining = False

        self.policy_net = DQN(self.n_observations, n_actions, model_structure=model_structure).to(self.device)
        self.target_net = DQN(self.n_observations, n_actions, model_structure=model_structure).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=self.lr_scheduler_step_size, gamma=self.lr_scheduler_gamma)

        if self.use_sliding_retrain_memory:
            self.retrain_memory = ReplayMemory(max_size=self.sliding_replay_mem_size,
                                               always_use_newest=replay_always_use_newest)
        else:
            self.retrain_memory = ReplayMemory(max_size=self.offline_train_epoch_len,
                                               always_use_newest=replay_always_use_newest)
        # Cached copy to avoid loading the expert memory from file multiple times
        self.expert_memory_unmodified = None
        self.expert_memory = None
        self.replay_always_use_newest = replay_always_use_newest

        self.steps_done = 0
        self.actions_chosen = defaultdict(int)

    def save_models_and_stats(self, model_folder: Path):
        torch.save(self.policy_net.state_dict(), model_folder / 'policy_model_weights.pth')
        torch.save(self.policy_net.state_dict(), model_folder / 'target_model_weights.pth')

        self.save_model_trainer_stats(model_folder)
        # self.retrain_memory.save_to_file(model_folder=model_folder)

    def save_model_trainer_stats(self, data_folder: Path):
        model_trainer_json = {
            "steps_done": self.steps_done,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "reward_mean": self.reward_mean.tolist(),
            "reward_std": self.reward_std.tolist(),
            "expert_data_folder": str(self.expert_data_folder)
        }

        # To get the final JSON string
        with open(data_folder / MODEL_TRAINER_JSON, 'w') as f:
            json.dump(model_trainer_json, f)

    def set_model_folder(self, model_folder: Path) -> None:
        self.model_folder = model_folder

    def load_models(self):
        if self.model_folder is None:
            raise Exception('Error, model path is none')
        self.load_models_from_file(model_folder=self.model_folder)

    def load_models_from_file(self, model_folder: Path):
        self.load_stats_from_file(model_folder)

        if self.expert_memory_unmodified is None and self.do_active_retraining:
            self.init_expert_data_from_csv()
        elif self.expert_memory_unmodified is not None and self.add_retrain_to_expert_buffer:
            self.expert_memory = self.expert_memory_unmodified.copy()

        # Reinit retrain memory
        # TODO: Change
        # This does not account for duplication, is later initialized with correct value
        if self.use_sliding_retrain_memory:
            self.retrain_memory = ReplayMemory(max_size=self.sliding_replay_mem_size,
                                               always_use_newest=self.replay_always_use_newest)
        else:
            self.retrain_memory = ReplayMemory(max_size=self.offline_train_epoch_len,
                                               always_use_newest=self.replay_always_use_newest)

        if self.expert_memory_unmodified is not None and self.use_sliding_retrain_memory:
            # TODO: Change into using lastest element
            for i in range(self.sliding_replay_mem_size):
                self.retrain_memory.push_transition(self.expert_memory_unmodified.memory[i])

        policy_net = DQN(self.n_observations, self.n_actions,
                         model_structure=self.model_structure).to(self.device)
        policy_net.load_state_dict(torch.load(model_folder / 'policy_model_weights.pth'))
        self.policy_net = policy_net

        # Target net gts normalized values so should not normalize!
        target_net = DQN(self.n_observations, self.n_actions,
                         model_structure=self.model_structure).to(self.device)
        target_net.load_state_dict(torch.load(model_folder / 'target_model_weights.pth'))
        self.target_net = target_net

        # Set optimizer to new model
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        # Note: Learning rate scheduler is currently not called in test epochs!
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=self.lr_scheduler_step_size, gamma=self.lr_scheduler_gamma)

    def load_stats_from_file(self, data_folder: Path):
        with open(data_folder / MODEL_TRAINER_JSON, 'r') as f:
            data = json.load(f)

        self.steps_done = data['steps_done']
        self.expert_data_folder = Path(data['expert_data_folder'])
        self.feature_mean = torch.tensor(data['feature_mean'], dtype=torch.float32, device=self.device)
        self.feature_std = torch.tensor(data['feature_std'], dtype=torch.float32, device=self.device)
        self.reward_mean = torch.tensor(data['reward_mean'], dtype=torch.float32, device=self.device)
        self.reward_std = torch.tensor(data['reward_std'], dtype=torch.float32, device=self.device)

    def select_action(self, state: State, simulation, random_decision: int, task: Task) -> torch.Tensor:
        # random_decision int handed in from outside to ensure its the same decision that random strategy would take

        sample = simulation.random_exploration.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = self.state_parser.state_to_tensor(state=state)
                norm_state = self.normalize_state(state=state)
                q_values = self.policy_net(norm_state)
                task.set_q_values(q_values=q_values)
                action_chosen = q_values.max(1).indices.view(1, 1)
                self.exploit_actions_episode += 1
        else:
            self.explore_actions_episode += 1
            action_chosen = torch.tensor([[random_decision]], device=self.device,
                                         dtype=torch.long)

        self.steps_done += 1
        self.actions_chosen[action_chosen.item()] += 1
        return action_chosen

    def run_offline_retraining_epoch(self, transitions: List[Transition], norm_stats: NormStats = None) -> None:
        if norm_stats is not None:
            self.reward_mean = norm_stats.reward_mean
            self.feature_mean = norm_stats.feature_mean
            self.reward_std = norm_stats.reward_std
            self.feature_std = norm_stats.feature_std

        if not self.use_sliding_retrain_memory:
            self.retrain_memory = ReplayMemory(max_size=len(
                transitions), always_use_newest=self.replay_always_use_newest)
        for transition in transitions:
            norm_transition = self.normalize_transition(transition=transition)

            # Store the normalized transition in memory
            self.retrain_memory.push_transition(transition=norm_transition)

        print(f'Number of steps: {len(transitions)}')
        for _ in range(self.offline_train_epoch_len):
            self.training_step()
        # print('Offline epoch finished')

        if self.add_retrain_to_expert_buffer:
            # Add retrain data to expert buffer for next retraining
            self.expert_memory.extend_buffer(transitions=self.retrain_memory.get_stored_transitions())

    def init_expert_data_from_csv(self, expert_data_folder: Path = None) -> None:
        if expert_data_folder is not None:
            self.expert_data_folder = expert_data_folder

        action_reward_policy_df = pd.read_csv(self.expert_data_folder / const.ACTION_REWARD_POLICY_FILE)
        state_df = pd.read_csv(self.expert_data_folder / const.STATE_FILE)
        next_state_df = pd.read_csv(self.expert_data_folder / const.NEXT_STATE_FILE)

        # Make sure that data is aligned properly
        assert len(action_reward_policy_df) == len(state_df) and len(state_df) == len(next_state_df)

        self.reward_mean = torch.tensor([action_reward_policy_df['reward'].mean()],
                                        dtype=torch.float32, device=self.device)
        self.reward_std = torch.tensor([action_reward_policy_df['reward'].std()],
                                       dtype=torch.float32, device=self.device)

        self.feature_mean = torch.tensor(state_df.mean(), dtype=torch.float32, device=self.device)
        self.feature_std = torch.tensor(state_df.std(), dtype=torch.float32, device=self.device)

        self.expert_memory_unmodified = ReplayMemory(max_size=len(state_df),
                                                     always_use_newest=self.replay_always_use_newest)

        for action_reward_policy_row, state_row, next_state_row in zip(action_reward_policy_df.itertuples(index=False), state_df.itertuples(index=False), next_state_df.itertuples(index=False)):
            action = torch.tensor([[action_reward_policy_row.action]], device=self.device)
            reward = torch.tensor([[action_reward_policy_row.reward]], dtype=torch.float32, device=self.device)
            state = torch.tensor([state_row], dtype=torch.float32, device=self.device)
            next_state = torch.tensor([next_state_row], dtype=torch.float32, device=self.device)
            transition = Transition(state=state, action=action, next_state=next_state, reward=reward)
            norm_transition = self.normalize_transition(transition=transition)

            # Store the normalized transition in memory
            self.expert_memory_unmodified.push_transition(transition=norm_transition)
        self.expert_memory = self.expert_memory_unmodified.copy()

    def train_model_from_expert_data_epoch(self, train_steps: int) -> None:
        for _ in range(train_steps):
            self.training_step()
        # print('Offline epoch finished')

    def normalize_state(self, state):
        epsilon = 1e-8  # A small value to avoid division by zero
        norm_state = (state - self.feature_mean) / (self.feature_std + epsilon)
        return norm_state

    def normalize_transition(self, transition: Transition) -> Transition:
        norm_state = self.normalize_state(state=transition.state)
        norm_next_state = self.normalize_state(state=transition.next_state)
        norm_reward = (transition.reward - self.reward_mean) / self.reward_std
        return Transition(state=norm_state, action=transition.action, next_state=norm_next_state, reward=norm_reward)

    def training_step(self):
        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        tau = self.TAU + (1 - self.TAU) * math.exp(-1. * self.steps_done / self.TAU_DECAY)

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if self.do_active_retraining:
            if len(self.retrain_memory) < self.retrain_batch_size:
                return
            transitions = self.expert_memory.sample(self.expert_batch_size) + \
                self.retrain_memory.sample(self.retrain_batch_size)
        else:
            if len(self.expert_memory) < self.BATCH_SIZE:
                return
            transitions = self.expert_memory.sample(self.BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # reward_batch = (reward_batch - self.reward_mean) / self.reward_std

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        next_state_values = next_state_values.unsqueeze(1)

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        self.reward_logs.append(reward_batch.mean().item())
        self.mean_value.append(next_state_values.mean().item())

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()

        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        grads = [
            param.grad.detach().flatten()
            for param in self.policy_net.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()

        self.losses.append(loss.item())
        self.grads.append(norm.item())

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.CLIPPING_VALUE)
        self.optimizer.step()

    def reset_episode_counters(self) -> None:
        self.explore_actions_episode = 0
        self.exploit_actions_episode = 0

    def reset_training_stats(self) -> None:
        self.losses = []
        self.grads = []
        self.mean_value = []
        self.reward_logs = []

        self.steps_done = 0
        self.actions_chosen = defaultdict(int)

    def plot_grads_and_losses(self, plot_path: Path, file_prefix: str):
        PLOT_OUT_FOLDER = 'model_stats'

        os.makedirs(plot_path / PLOT_OUT_FOLDER, exist_ok=True)
        os.makedirs(plot_path / f'pdfs' / PLOT_OUT_FOLDER, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.clf()
        plt.plot(range(len(self.losses)), self.losses)
        plt.savefig(plot_path / f'pdfs/{PLOT_OUT_FOLDER}/{file_prefix}_losses.pdf')
        plt.savefig(plot_path / f'{PLOT_OUT_FOLDER}/{file_prefix}_losses.jpg')
        plt.close()

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.plot(range(len(self.grads)), self.grads)
        plt.savefig(plot_path / f'pdfs/{PLOT_OUT_FOLDER}/{file_prefix}_grads.pdf')
        plt.savefig(plot_path / f'{PLOT_OUT_FOLDER}/{file_prefix}_grads.jpg')
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.plot(range(len(self.reward_logs)), self.reward_logs)
        plt.savefig(plot_path / f'pdfs/{PLOT_OUT_FOLDER}/{file_prefix}_rewards.pdf')
        plt.savefig(plot_path / f'{PLOT_OUT_FOLDER}/{file_prefix}_rewards.jpg')
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.plot(range(len(self.mean_value)), self.mean_value)
        plt.savefig(plot_path / f'pdfs/{PLOT_OUT_FOLDER}/{file_prefix}_mean_value.pdf')
        plt.savefig(plot_path / f'{PLOT_OUT_FOLDER}/{file_prefix}_mean_value.jpg')
        plt.close()

    def print_weights(self):
        self.policy_net.print_weights()
