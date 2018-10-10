from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20*20*20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.selu(x)
        x = self.conv2(x)
        x = F.selu(x)
        x = x.view(-1, 20*20*20)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer, epoch):
    model.train()
    loss_f = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        loss_f = loss.item()     
    return loss_f


def main():
    epochs = 20
    batch_size = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    sensitive = []
    losses = []
    for i in range(len(batch_size)):
        model = CNN().cuda()
        optimizer = optim.Adam(model.parameters())
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),  batch_size=batch_size[i], shuffle=True)
        for epoch in range(1, epochs):
            loss = train(model, train_loader, optimizer, epoch)
        losses.append(loss)
        print("loss", loss)
        sensitive1 = get_sensitive(model)
        print('sensitive', sensitive1)
        sensitive.append(sensitive1)
    batch_size_log = np.log10(batch_size)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(batch_size_log, sensitive, '-', color='b', label='sensitive')
    ax1.set_ylabel('sensitive')
    ax1.set_xlabel('batch_size(log)')
    ax2 = ax1.twinx()  # this is the important function
    plt.plot(batch_size_log, losses, '-', color='r', label='loss')
    ax2.set_ylabel('loss')
    plt.legend()
    plt.show()


def get_sensitive(model):
    grad_all = 0
    for p in model.parameters():
        if p.grad is not None:
            grad = 0.0
            grad = (p.grad ** 2).sum()
        grad_all += grad
    grad_all = grad_all ** 0.5
    return grad_all


if __name__ == '__main__':
    main()

