from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20*20*20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 20*20*20)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(torch.ones_like(loss), create_graph=True)
        grad_all = Variable(torch.zeros(1), requires_grad=True).cuda()
        for p in model.parameters():
            if p.grad is not None:
                grad_all = grad_all + (p.grad ** 2).sum()
        grad_all.backward()
        optimizer.step()
        
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def main():
    epochs = 10
    batch_size = 1000
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    model = CNN().cuda()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, epochs):
        train(model, train_loader, optimizer, epoch)
        grad_norm = get_gradient(model)
        print('grad_norm', grad_norm)


def get_gradient(model):
    grad_all = 0
    for p in model.parameters():
        if p.grad is not None:
            grad = 0.0
            grad = (p.grad ** 2).sum()
        grad_all += grad
    return grad_all


if __name__ == '__main__':
    main()

