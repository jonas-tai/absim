import torch.nn as nn
import torch.nn.functional as F
import torch


# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_dims=64):
        super(DQN, self).__init__()
        activations = nn.LeakyReLU

        self.layers = nn.Sequential(
            nn.Linear(n_observations, hidden_dims),
            activations(),
            nn.Linear(hidden_dims, hidden_dims),
            activations(),
            nn.Linear(hidden_dims, hidden_dims),
            activations(),
            nn.Linear(hidden_dims, n_actions)
        )

        self.summary = SummaryStats(n_observations)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = (x - self.summary.means) * self.summary.inv_sqrt_sd()
        return self.layers(x)


class SummaryStats():
    # https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time
    def __init__(self, size):
        self.means = torch.zeros(size)
        self.S = torch.ones(size)
        self.n = 0

    def add(self, x):
        # stop adding new stats after 10000
        if self.n >= 10000:
            return

        x = x.flatten()
        prev_mean = self.means
        self.n += 1
        self.means += (x - self.means) / self.n
        self.S += (x - self.means) * (x - prev_mean)

    def inv_sqrt_sd(self):
        return torch.rsqrt(self.S / self.n)
