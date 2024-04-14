import random
import math
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from model import DQN, SummaryStats
from simulations.state import State
from collections import defaultdict
import numpy as np

StateAction = namedtuple('StateAction', ('state', 'action'))


class Trainer:
    def __init__(self, n_actions, batch_size=1024, gamma=0.99, eps_start=0.9, eps_end=0.1,
                 eps_decay=10000, tau=0.001, lr=1e-4, tau_decay=10):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.task_to_state_action: Dict[str, StateAction] = {}
        self.task_to_next_state: Dict[str, torch.Tensor] = {}
        self.task_to_reward: Dict[str, torch.Tensor] = {}
        self.last_task_id: str = None

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU_DECAY = tau_decay
        self.TAU = tau
        self.LR = lr
        self.losses1 = []
        self.losses2 = []
        self.grads1 = []
        self.grads2 = []
        self.mean_value_1 = []
        self.mean_value_2 = []
        self.reward_logs = []
        # num servers
        self.n_actions = n_actions

        n_observations = State.get_state_size(num_servers=n_actions)

        self.model1 = DQN(n_observations, n_actions).to(self.device)
        self.model2 = DQN(n_observations, n_actions).to(self.device)

        self.model2.summary = self.model1.summary

        self.policy_net = self.model1

        # self.target_net.load_state_dict(self.policy_net.state_dict())

        self.model1_optimizer = optim.Adam(self.model1.parameters(), lr=self.LR, weight_decay=1e-2)
        self.model2_optimizer = optim.Adam(self.model2.parameters(), lr=self.LR, weight_decay=1e-2)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=lr / 100)
        # self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer)

        self.memory = ReplayMemory(8192, self.policy_net.summary)

        self.steps_done = 0

        self.actions_chosen = defaultdict(int)

        self.reward_stats = SummaryStats(1)

    def record_state_and_action(self, task_id: str, state: State, action: int) -> None:
        action = torch.tensor([[action]], device=self.device)
        state = state.to_tensor()
        self.task_to_state_action[task_id] = StateAction(state, action)

        if self.last_task_id is not None:
            self.task_to_next_state[self.last_task_id] = state
            if self.last_task_id in self.task_to_reward:
                # Reward (latency) of last task already present and not pushed to memory yet
                self.training_step(task_id=self.last_task_id)
        self.last_task_id = task_id

    def execute_step_if_state_present(self, task_id: str, latency: int) -> None:
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
        # tau = self.TAU + (0.1 - self.TAU) * math.exp(-1. * self.steps_done / self.TAU_DECAY)
        #
        # target_net_state_dict = self.target_net.state_dict()
        # policy_net_state_dict = self.policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = \
        #         policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        # self.target_net.load_state_dict(target_net_state_dict)
        self.clean_up_after_step(task_id=task_id)

    def select_action(self, state: State):
        action_chosen = None
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_chosen = self.policy_net(state.to_tensor()).max(1).indices.view(1, 1)
        else:
            action_chosen = torch.tensor([[random.randint(0, self.n_actions - 1)]], device=self.device,
                                         dtype=torch.long)

        self.actions_chosen[action_chosen.item()] += 1
        return action_chosen

    def grad_norm(self, model):
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm

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
        curr_Q1 = self.model1.forward(state_batch).gather(1, action_batch)
        curr_Q2 = self.model2.forward(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values1 = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values2 = torch.zeros(self.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            next_state_values1[non_final_mask] = self.model1.forward(non_final_next_states).max(1).values
            next_state_values2[non_final_mask] = self.model2.forward(non_final_next_states).max(1).values

        # Compute the expected Q values
        next_state_values = torch.min(next_state_values1, next_state_values2).unsqueeze(1)

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        self.reward_logs.append(reward_batch.mean().item())
        self.mean_value_1.append(next_state_values1.mean().item())
        self.mean_value_2.append(next_state_values2.mean().item())

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()

        loss1 = criterion(curr_Q1, expected_state_action_values)
        loss2 = criterion(curr_Q2, expected_state_action_values)

        self.model1_optimizer.zero_grad()
        loss1.backward()
        self.losses1.append(loss1.item())
        self.grads1.append(self.grad_norm(self.model1).item())
        torch.nn.utils.clip_grad_value_(self.model1.parameters(), 1)
        self.model1_optimizer.step()

        self.model2_optimizer.zero_grad()
        loss2.backward()
        self.losses2.append(loss2.item())
        self.grads2.append(self.grad_norm(self.model2).item())
        torch.nn.utils.clip_grad_value_(self.model2.parameters(), 1)
        self.model2_optimizer.step()
