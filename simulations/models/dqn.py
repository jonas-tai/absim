import torch.nn as nn
import torch.nn.functional as F
import torch


# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int, n_hidden: int = 128, model_structure: str = "linear"):
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
        self.summary = SummaryStats(n_observations)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
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


class SummaryStats:
    # https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time
    def __init__(self, size):
        self.means = torch.zeros(size)
        self.S = torch.ones(size)
        self.n = 0

    def add(self, x):
        # stop adding new stats after 10000
        if self.n >= 1000:
            return

        x = x.flatten()
        prev_mean = self.means
        self.n += 1
        self.means += (x - self.means) / self.n
        self.S += (x - self.means) * (x - prev_mean)

    def inv_sqrt_sd(self):
        return torch.rsqrt(self.S / self.n)
