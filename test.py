from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import torch
from torch.utils.data import DataLoader

from torch.nn import Module, Linear, Flatten, Softmax, CrossEntropyLoss, Conv2d, ReLU, BatchNorm2d, Sequential, MaxPool2d
from torch.optim import Adam

from tqdm import tqdm

train_dataset = MNIST('./data/mnist', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST('./data/mnist', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

class Net(Module):
    def __init__(self, n=1) -> None:
        super(Net, self).__init__()
        
        self.core = Sequential(
            Conv2d(1, 32 * n, 5, 1),
            ReLU(),
            BatchNorm2d(32 * n),
            MaxPool2d(2, 2),
            Conv2d(32 * n, 64 * n, 5, 1),
            ReLU(),
            BatchNorm2d(64 * n),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(4 * 4 * 64 * n, 128 * n),
            ReLU(),
            Linear(128*n, 10),
            Softmax(1)
        )

    def forward(self, input):
        return self.core(input)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Net().to(device)

optim = Adam(net.parameters())

criterion = CrossEntropyLoss()

for epoch in range(2):

    counter = tqdm(train_loader)

    for x, y in counter:
        x = x.to(device)
        y = y.to(device)

        pred = net(x)

        loss = criterion(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            correct = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                a = (torch.argmax(net(x), dim=-1) == y)
                
                correct += torch.sum(a)

        counter.set_description(f"On epoch {epoch}, loss is : {loss.detach().item():4f}, accuracy is {correct / len(test_loader.dataset)}")