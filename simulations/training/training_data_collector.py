import json
import os
import pandas as pd
import torch

from pathlib import Path
from typing import Dict, List, Tuple
from simulations.models.dqn import SummaryStats
from simulations.state import StateParser
from simulations.task import Task
from simulations.training.norm_stats import NormStats
from simulations.training.offline_model_trainer import OfflineTrainer
from simulations.training.replay_memory import Transition

MODEL_TRAINER_JSON = 'training_data_collector.json'
STATE_FILE = 'state_data.csv'
ACTION_REWARD_POLICY_FILE = 'action_reward_policy_data.csv'
NEXT_STATE_FILE = 'next_state_data.csv'


class TrainingDataCollector:
    def __init__(self, offline_trainer: OfflineTrainer, state_parser: StateParser, n_actions: int, summary_stats_max_size: int, offline_train_batch_size: int, data_folder: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_parser = state_parser
        self.data_folder = data_folder
        self.offline_trainer = offline_trainer

        self.task_id_to_action: Dict[str, int] = {}
        self.task_id_to_next_state: Dict[str, torch.Tensor] = {}
        self.task_id_to_rewards: Dict[str, torch.Tensor] = {}
        self.task_id_to_policy: Dict[str, str] = {}
        self.last_task: Task = None
        self.summary_stats_max_size = summary_stats_max_size
        self.current_train_batch = 0
        self.offline_train_batch_size = offline_train_batch_size

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.policies = []
        self.train_batches = []

        self.all_states = []
        self.all_actions = []
        self.all_next_states = []
        self.all_rewards = []
        self.all_policies = []
        self.all_train_batches = []

        # num servers
        self.n_actions = n_actions
        self.n_observations = self.state_parser.get_state_size()

        self.logged_transitions = 0

    def save_training_data_collector_stats(self):
        os.makedirs(self.data_folder, exist_ok=True)

        model_trainer_json = {
            "logged_transitions": self.logged_transitions,
        }

        # To get the final JSON string
        with open(self.data_folder / MODEL_TRAINER_JSON, 'w') as f:
            json.dump(model_trainer_json, f)

    def load_stats_from_file(self) -> None:
        with open(self.data_folder / MODEL_TRAINER_JSON, 'r') as f:
            data = json.load(f)

        self.logged_transitions = data['logged_transitions']

    def next_train_batch_is_ready(self) -> bool:
        return len(self.rewards) >= self.offline_train_batch_size

    def end_train_batch(self) -> List[Transition]:
        transitions = self.convert_current_train_batch_to_transitions()

        self.all_states += self.states[:self.offline_train_batch_size]
        self.all_actions += self.actions[:self.offline_train_batch_size]
        self.all_next_states += self.next_states[:self.offline_train_batch_size]
        self.all_rewards += self.rewards[:self.offline_train_batch_size]
        self.all_policies += self.policies[:self.offline_train_batch_size]
        self.all_train_batches += self.train_batches[:self.offline_train_batch_size]

        self.states = self.states[self.offline_train_batch_size:]
        self.actions = self.actions[self.offline_train_batch_size:]
        self.next_states = self.next_states[self.offline_train_batch_size:]
        self.rewards = self.rewards[self.offline_train_batch_size:]
        self.policies = self.policies[self.offline_train_batch_size:]
        self.current_train_batch += 1

        self.train_batches = [self.current_train_batch for _ in self.policies]

        return transitions

    def convert_current_train_batch_to_transitions(self) -> List[Transition]:
        return [Transition(state=state, action=torch.tensor([[action]],  device=self.device),
                           next_state=next_state,
                           reward=reward)
                for state, action, next_state, reward in zip(self.states[:self.offline_train_batch_size],
                                                             self.actions[:self.offline_train_batch_size],
                                                             self.next_states[:self.offline_train_batch_size],
                                                             self.rewards[:self.offline_train_batch_size])]

    def end_train_episode(self) -> None:
        self.all_states += self.states
        self.all_actions += self.actions
        self.all_next_states += self.next_states
        self.all_rewards += self.rewards
        self.all_policies += self.policies
        self.all_train_batches += self.train_batches

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.policies = []
        self.train_batches = []

        self.current_train_batch += 1

    def save_training_data(self) -> None:
        os.makedirs(self.data_folder, exist_ok=True)

        self.all_states += self.states
        self.all_actions += self.actions
        self.all_next_states += self.next_states
        self.all_rewards += self.rewards
        self.all_policies += self.policies
        self.all_train_batches += self.train_batches

        data = {
            'action': self.all_actions,
            'reward': [reward.squeeze().numpy() for reward in self.all_rewards],
            'policy': self.all_policies,
            'train_batch': self.all_train_batches
        }

        action_reward_policy_df = pd.DataFrame(data)
        file_path = self.data_folder / ACTION_REWARD_POLICY_FILE
        action_reward_policy_df.to_csv(file_path, index=False)

        data = [state.squeeze().numpy() for state in self.all_states]
        state_df = pd.DataFrame(data)
        file_path = self.data_folder / STATE_FILE
        state_df.to_csv(file_path, index=False)

        data = [next_state.squeeze().numpy() for next_state in self.all_next_states]
        next_state_df = pd.DataFrame(data)
        file_path = self.data_folder / NEXT_STATE_FILE
        next_state_df.to_csv(file_path, index=False)

    def read_training_data_from_csv(self, train_data_folder: Path) -> Tuple[List[Transition], NormStats]:
        action_reward_policy_df = pd.read_csv(train_data_folder / ACTION_REWARD_POLICY_FILE)
        state_df = pd.read_csv(train_data_folder / STATE_FILE)
        next_state_df = pd.read_csv(train_data_folder / NEXT_STATE_FILE)

        # Make sure that data is aligned properly
        assert len(action_reward_policy_df) == len(state_df) and len(state_df) == len(next_state_df)

        transitions = []

        reward_mean = torch.tensor([action_reward_policy_df['reward'].mean()], dtype=torch.float32, device=self.device)
        reward_std = torch.tensor([action_reward_policy_df['reward'].std()], dtype=torch.float32, device=self.device)

        feature_mean = torch.tensor(state_df.mean(), dtype=torch.float32, device=self.device)
        feature_std = torch.tensor(state_df.std(), dtype=torch.float32, device=self.device)

        norm_stats = NormStats(reward_mean=reward_mean, reward_std=reward_std,
                               feature_mean=feature_mean, feature_std=feature_std)

        for action_reward_policy_row, state_row, next_state_row in zip(action_reward_policy_df.itertuples(index=False), state_df.itertuples(index=False), next_state_df.itertuples(index=False)):
            action = torch.tensor([[action_reward_policy_row.action]], device=self.device)
            reward = torch.tensor([[action_reward_policy_row.reward]], dtype=torch.float32, device=self.device)
            state = torch.tensor([state_row], dtype=torch.float32, device=self.device)
            next_state = torch.tensor([next_state_row], dtype=torch.float32, device=self.device)

            transition = Transition(state=state, action=action, next_state=next_state, reward=reward)
            transitions.append(transition)
        return transitions, norm_stats

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
        self.states.append(state)

        self.actions.append(self.task_id_to_action[task.id])
        self.next_states.append(self.task_id_to_next_state[task.id])
        self.policies.append(self.task_id_to_policy[task.id])
        self.train_batches.append(self.current_train_batch)

        reward = self.task_id_to_rewards[task.id]
        self.rewards.append(reward)

        self.logged_transitions += 1
        if self.offline_trainer.do_active_retraining and len(self.rewards) >= self.offline_train_batch_size:
            print('Retraining')
            train_batch_transitions = self.end_train_batch()
            print(len(train_batch_transitions))
            self.offline_trainer.run_offline_training_epoch(transitions=train_batch_transitions)

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
