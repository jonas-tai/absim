import torch.nn as nn
import torch.nn.functional as F
import torch


# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class Classifier(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=8):
        super(Classifier, self).__init__()

        self.layer1 = nn.Linear(n_observations, n_hidden)
        # self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_actions)
        # self.layer3 = nn.Linear(n_observations, n_actions)
        # self.summary = SummaryStats(n_observations)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = (x - self.summary.means) * self.summary.inv_sqrt_sd()

        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def print_weights(self):
        layer3_weights = self.layer3.weight.data
        layer3_bias = self.layer3.bias.data

        # Print or inspect the weights and bias
        print(f"Layer 3 Weights: {layer3_weights}")
        print(f"\nLayer 3 Bias: {layer3_bias}")
