import json
import torch.nn as nn
import torch.nn.functional as F
import torch


# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class SummaryStats:
    # https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time
    def __init__(self,  max_size: int, size=1):
        self.max_size = max_size
        self.means = torch.zeros(size)
        self.S = torch.ones(size)
        self.n = 0
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    def add(self, x):
        # stop adding new stats after 1000
        # TODO: Tune this parameter
        if self.n >= self.max_size:
            return

        x = x.flatten()
        prev_mean = self.means
        self.n += 1
        self.means += (x - self.means) / self.n
        self.S += (x - self.means) * (x - prev_mean)

    def inv_sqrt_sd(self):
        return torch.rsqrt(self.S / self.n)

    def to_dict(self):
        # Convert tensors to lists for JSON serialization
        return {
            'means': self.means.tolist(),
            'S': self.S.tolist(),
            'n': self.n,
            'max_size': self.max_size
        }

    @classmethod
    def from_dict(cls, data):
        # Initialize an instance with the specified size
        max_size = data['max_size']
        obj = cls(max_size=max_size)
        # Load data from the dictionary
        obj.means = torch.tensor(data['means'], device=obj.device)
        obj.S = torch.tensor(data['S'], device=obj.device)
        obj.n = data['n']
        return obj

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, summary_stats: SummaryStats = None, n_hidden: int = 128, model_structure: str = "linear"):
        super(DQN, self).__init__()
        self.model_structure = model_structure
        if self.model_structure == 'three_layers':
            self.layer1 = nn.Linear(n_observations, n_hidden)
            self.layer2 = nn.Linear(n_hidden, n_hidden)
            self.layer3 = nn.Linear(n_hidden, n_actions)
        elif self.model_structure == 'linear':
            self.layer3 = nn.Linear(n_observations, n_actions)
        else:
            raise Exception(f'Unknown model structure {self.model_structure}')
        self.summary = summary_stats

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if self.summary is not None:
            x = (x - self.summary.means) * self.summary.inv_sqrt_sd()
        if self.model_structure == 'three_layers':
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def print_weights(self):
        layer3_weights = self.layer3.weight.data
        layer3_bias = self.layer3.bias.data

        # Print or inspect the weights and bias
        print(f"Layer 3 Weights: {layer3_weights}")
        print(f"\nLayer 3 Bias: {layer3_bias}")

    def get_summary_stats(self):
        if self.summary is None:
            raise Exception('DQN does not have summary stats')
        return self.summary
