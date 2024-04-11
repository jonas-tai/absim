import torch.nn as nn
import torch.nn.functional as F
import torch


# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.summary = SummaryStats(n_observations)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = (x - self.summary.means) * self.summary.inv_sqrt_sd()

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


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
