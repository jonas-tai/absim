import json
import os
import pandas as pd
import torch

from pathlib import Path
from typing import Dict, List
from simulations.models.dqn import SummaryStats
from simulations.state import StateParser
from simulations.task import Task
from simulations.training.replay_memory import Transition

MODEL_TRAINER_JSON = 'training_data_collector.json'
STATE_FILE = 'state_data.csv'
ACTION_REWARD_POLICY_FILE = 'action_reward_policy_data.csv'
NEXT_STATE_FILE = 'next_state_data.csv'


class TrainingDataCollector:
    def __init__(self, state_parser: StateParser, n_actions: int, summary_stats_max_size: int, data_folder: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_parser = state_parser
        self.data_folder = data_folder

        self.task_id_to_action: Dict[str, int] = {}
        self.task_id_to_next_state: Dict[str, torch.Tensor] = {}
        self.task_id_to_rewards: Dict[str, torch.Tensor] = {}
        self.task_id_to_policy: Dict[str, str] = {}
        self.last_task: Task = None
        self.summary_stats_max_size = summary_stats_max_size

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.policies = []

        # num servers
        self.n_actions = n_actions
        self.n_observations = self.state_parser.get_state_size()

        self.reward_stats = SummaryStats(max_size=summary_stats_max_size, size=1)
        self.feature_stats = SummaryStats(max_size=summary_stats_max_size, size=self.n_observations)

        self.logged_transitions = 0

    def save_training_data_collector_stats(self):
        os.makedirs(self.data_folder, exist_ok=True)

        feature_stats_json = self.feature_stats.to_dict()
        reward_stats_json = self.reward_stats.to_dict()

        model_trainer_json = {
            "logged_transitions": self.logged_transitions,
            "feature_summary_stats": feature_stats_json,
            "reward_summary_stats": reward_stats_json
        }

        # To get the final JSON string
        with open(self.data_folder / MODEL_TRAINER_JSON, 'w') as f:
            json.dump(model_trainer_json, f)

    def load_stats_from_file(self) -> None:
        with open(self.data_folder / MODEL_TRAINER_JSON, 'r') as f:
            data = json.load(f)

        self.logged_transitions = data['logged_transitions']
        self.feature_stats = SummaryStats.from_dict(data=data['feature_summary_stats'])
        self.reward_stats = SummaryStats.from_dict(data=data['reward_summary_stats'])

    def save_training_data(self) -> None:
        os.makedirs(self.data_folder, exist_ok=True)

        data = {
            'action': self.actions,
            'reward': [reward.squeeze().numpy() for reward in self.rewards],
            'policy': self.policies,
        }

        action_reward_policy_df = pd.DataFrame(data)
        file_path = self.data_folder / ACTION_REWARD_POLICY_FILE
        action_reward_policy_df.to_csv(file_path, index=False)

        data = [state.numpy() for state in self.states]
        state_df = pd.DataFrame(data)
        file_path = self.data_folder / STATE_FILE
        state_df.to_csv(file_path, index=False)

        data = [next_state.numpy() for next_state in self.next_states]
        next_state_df = pd.DataFrame(data)
        file_path = self.data_folder / NEXT_STATE_FILE
        next_state_df.to_csv(file_path, index=False)

    def read_training_data_from_csv(self, train_data_folder: Path) -> List[Transition]:
        action_reward_policy_df = pd.read_csv(train_data_folder / ACTION_REWARD_POLICY_FILE)
        state_df = pd.read_csv(train_data_folder / STATE_FILE)
        next_state_df = pd.read_csv(train_data_folder / NEXT_STATE_FILE)

        # Make sure that data is aligned properly
        assert len(action_reward_policy_df) == len(state_df) and len(state_df) == len(next_state_df)

        transitions = []

        for action_reward_policy_row, state_row, next_state_row in zip(action_reward_policy_df.itertuples(index=False), state_df.itertuples(index=False), next_state_df.itertuples(index=False)):
            action = torch.tensor([[action_reward_policy_row.action]], dtype=torch.float32, device=self.device)
            reward = torch.tensor([[action_reward_policy_row.reward]], dtype=torch.float32, device=self.device)
            state = torch.tensor(state_row, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state_row, dtype=torch.float32, device=self.device)

            transition = Transition(state=state, action=action, next_state=next_state, reward=reward)
            transitions.append(transition)
        return transitions

    def log_state_and_action(self, task: Task, action: int, policy: str) -> None:
        state = self.state_parser.state_to_tensor(state=task.get_state())
        self.task_id_to_action[task.id] = action
        self.task_id_to_policy[task.id] = policy

        # Only do this if this is not the first task of the epoch and not a duplicate task
        if (self.last_task is not None) and (not task.is_duplicate):
            self.task_id_to_next_state[self.last_task.id] = state

            if self.last_task.has_duplicate:
                self.task_id_to_next_state[self.last_task.duplicate_task.id] = state

            # Check if reward (latency) of last task already present and not pushed to memory yet
            if self.last_task.id in self.task_id_to_rewards:
                self.process_complete_transition(task=self.last_task)
            if self.last_task.has_duplicate and self.last_task.duplicate_task.id in self.task_id_to_rewards:
                self.process_complete_transition(task=self.last_task.duplicate_task)
        if not task.is_duplicate:
            self.last_task = task

    def log_completion(self, task: Task, latency: int) -> None:
        # Saving the negative reward
        self.task_id_to_rewards[task.id] = torch.tensor([[- latency]], device=self.device, dtype=torch.float32)
        if task.id not in self.task_id_to_next_state:
            # Next state not present because request finished before next request arrived
            return
        self.process_complete_transition(task=task)

    def log_transition(self, task: Task) -> None:
        # Log transitions and maintain summary stats
        state = self.state_parser.state_to_tensor(state=task.get_state())
        self.states.append(state.squeeze())
        self.feature_stats.add(state)

        self.actions.append(self.task_id_to_action[task.id])
        self.next_states.append(self.task_id_to_next_state[task.id].squeeze())
        self.policies.append(self.task_id_to_policy[task.id])

        reward = self.task_id_to_rewards[task.id]
        self.rewards.append(reward)
        self.reward_stats.add(reward)

        self.logged_transitions += 1

    def clean_up_transition(self, task: Task) -> None:
        del self.task_id_to_action[task.id]
        del self.task_id_to_rewards[task.id]
        del self.task_id_to_next_state[task.id]
        del self.task_id_to_policy[task.id]

    def process_complete_transition(self, task: Task):
        # Store the transition in memory
        self.log_transition(task=task)
        self.clean_up_transition(task=task)

    def reset_episode_counters(self) -> None:
        self.task_id_to_action = {}
        self.task_id_to_next_state = {}
        self.task_id_to_rewards = {}
        self.task_id_to_policy = {}
        self.last_task = None
