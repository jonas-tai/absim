import random
import math
from collections import namedtuple
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

from replay_memory import ReplayMemory, Transition
from simulations.models.dqn import DQN, SummaryStats
from simulations.state import State, StateParser
from collections import defaultdict

StateAction = namedtuple('StateAction', ('state', 'action'))


class Trainer:
    def __init__(self, state_parser: StateParser, model_structure: str, n_actions: int, batch_size=128, gamma=0.8, eps_start=0.2, eps_end=0.2,
                 eps_decay=1000, tau=0.005, lr=1e-4, tau_decay=10, lr_scheduler_step_size=50, lr_scheduler_gamma=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_parser = state_parser

        self.task_to_state_action: Dict[str, StateAction] = {}
        self.task_to_next_state: Dict[str, torch.Tensor] = {}
        self.task_to_reward: Dict[str, torch.Tensor] = {}
        self.last_task_id: str = None

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.TAU_DECAY = tau_decay
        self.LR = lr

        self.explore_actions_episode = 0
        self.exploit_actions_episode = 0

        self.losses = []
        self.grads = []
        self.mean_value = []
        self.reward_logs = []

        # num servers
        self.n_actions = n_actions

        n_observations = self.state_parser.get_state_size()

        self.eval_mode = False
        self.policy_net = DQN(n_observations, n_actions, model_structure=model_structure).to(self.device)
        self.target_net = DQN(n_observations, n_actions, model_structure=model_structure).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
        self.memory = ReplayMemory(10000, self.policy_net.summary)

        self.steps_done = 0
        self.actions_chosen = defaultdict(int)

        self.reward_stats = SummaryStats(1)

    def record_state_and_action(self, task_id: str, state: State, action: int) -> None:
        if self.eval_mode:
            return
        action = torch.tensor([[action]], device=self.device)
        state = self.state_parser.state_to_tensor(state=state)
        self.task_to_state_action[task_id] = StateAction(state, action)

        if self.last_task_id is not None:
            self.task_to_next_state[self.last_task_id] = state
            if self.last_task_id in self.task_to_reward:
                # Reward (latency) of last task already present and not pushed to memory yet
                self.training_step(task_id=self.last_task_id)
        self.last_task_id = task_id

    def execute_step_if_state_present(self, task_id: str, latency: int) -> None:
        if self.eval_mode:
            return
        self.task_to_reward[task_id] = torch.tensor([[- latency]], device=self.device, dtype=torch.float32)
        if task_id not in self.task_to_next_state:
            # Next state not present because request finished before next request arrived
            return
        self.training_step(task_id)

    def push_to_memory(self, task_id: str) -> None:
        state_action = self.task_to_state_action[task_id]
        next_state = self.task_to_next_state[task_id]
        reward = self.task_to_reward[task_id]
        self.reward_stats.add(reward)
        self.memory.push(state_action.state, state_action.action, next_state, reward)

    def clean_up_after_step(self, task_id: str) -> None:
        del self.task_to_state_action[task_id]
        del self.task_to_next_state[task_id]
        del self.task_to_reward[task_id]

    def training_step(self, task_id: str):

        # Store the transition in memory
        self.push_to_memory(task_id)

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
        self.clean_up_after_step(task_id=task_id)

    def select_action(self, state: State):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                q_values = self.policy_net(self.state_parser.state_to_tensor(state=state))
                # print(q_values)
                action_chosen = q_values.max(1).indices.view(1, 1)
                self.exploit_actions_episode += 1
        else:
            self.explore_actions_episode += 1
            action_chosen = torch.tensor([[random.randint(0, self.n_actions - 1)]], device=self.device,
                                         dtype=torch.long)

        if not self.eval_mode:
            self.steps_done += 1
            self.actions_chosen[action_chosen.item()] += 1
        return action_chosen

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
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

        reward_batch = (reward_batch - self.reward_stats.means) * self.reward_stats.inv_sqrt_sd()

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
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def reset_episode_counters(self) -> None:
        self.explore_actions_episode = 0
        self.exploit_actions_episode = 0

    def plot_grads_and_losses(self, plot_path: Path):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.clf()
        plt.plot(range(len(self.losses)), self.losses)
        plt.savefig(plot_path / 'pdfs/losses.pdf')
        plt.savefig(plot_path / 'losses.jpg')

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.plot(range(len(self.grads)), self.grads)
        plt.savefig(plot_path / 'pdfs/grads.pdf')
        plt.savefig(plot_path / 'grads.jpg')

        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.plot(range(len(self.reward_logs)), self.reward_logs)
        plt.savefig(plot_path / 'pdfs/rewards.pdf')
        plt.savefig(plot_path / 'rewards.jpg')

        fig, ax = plt.subplots(figsize=(8, 4), dpi=200, nrows=1, ncols=1, sharex='all')
        plt.plot(range(len(self.mean_value)), self.mean_value)
        plt.savefig(plot_path / 'pdfs/mean_value.pdf')
        plt.savefig(plot_path / 'mean_value.jpg')

    def print_weights(self):
        self.policy_net.print_weights()
