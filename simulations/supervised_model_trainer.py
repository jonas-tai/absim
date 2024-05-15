from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from model import DQN
from simulations.data.csv_dataset import CSVDataset
from simulations.state import State, StateParser
import torch.utils.data.dataloader as dataloader


class SupervisedModelTrainer:
    def __init__(self, n_labels, out_folder: Path, data_path: Path, state_parser: StateParser, seed: int,
                 target_col: str = 'Replica', print_interval: int = 100, batch_size=128,  lr=1e-4) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = seed
        self.out_folder = out_folder
        self.state_parser = state_parser
        self.print_interval = print_interval

        self.trainset = CSVDataset(
            mode='train',
            data_path=data_path,
            seed=seed,
            target_col=target_col,
            transform=torch.from_numpy
        )

        self.trainloader = dataloader.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset = CSVDataset(
            mode='test',
            data_path=data_path,
            seed=seed,
            target_col=target_col,
            transform=torch.from_numpy
        )

        self.testloader = dataloader.DataLoader(self.testset,  batch_size=batch_size, shuffle=True)

        # val_data = CSVDataset(
        #     mode='val',
        #     data_path=data_path,
        #     target_col='Replica_id',
        #     transform=ToTensor()
        # )

        self.BATCH_SIZE = batch_size
        self.LR = lr

        self.losses = []
        self.grads = []
        self.mean_value = []
        self.reward_logs = []

        self.mode = "train"
        self.n_feats = self.state_parser.get_state_size()

        self.net = DQN(self.n_feats, n_labels).to(self.device)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.LR, amsgrad=True)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()

        self.steps_done = 0

    def train_model(self, epochs: int) -> None:
        print(f'Training model for {epochs} epochs')
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                features, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(features)
                loss = self.criterion(outputs, labels)
                loss.backward()

                grads = [
                    param.grad.detach().flatten()
                    for param in self.net.parameters()
                    if param.grad is not None
                ]
                norm = torch.cat(grads).norm()
                self.grads.append(norm.item())
                self.losses.append(loss.item())

                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % self.print_interval == (self.print_interval - 1):
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / self.print_interval:.3f}')

                    running_loss = 0.0
        print('Finished Training')

    def test_model(self) -> None:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                features, labels = data
                # calculate outputs by running features through the network
                outputs = self.net(features)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test data: {100 * correct // total} %')

    def export_model(self) -> None:
        PATH = self.out_folder / 'model.pth'
        torch.save(self.net.state_dict(), PATH)

    def select_action(self, state: State):
        features = self.state_parser.state_to_tensor(state=state)
        output = self.net(features)
        _, predicted = torch.max(output, 1)
        return predicted

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
