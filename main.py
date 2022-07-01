from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from diffusion.algorithm import BaseDiffusion
from diffusion.model import BaseNet, BaseUnet

def transpose(img):
    return torch.transpose(torch.transpose(img, 0, -1), 0, 1)

transform = Compose([
    ToTensor(),
    Normalize(.5, .5)
])

BATCH_SIZE = 128

train_dataset = CIFAR10('./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10('./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

def main():

    net = BaseUnet()

    diffusion = BaseDiffusion(net)

    for epoch in range(10):
        diffusion.train(train_loader)

    x, _ = diffusion.sample(torch.ones((1, 3, 32, 32), device=diffusion.device))

    x = torch.squeeze(x, dim=0).cpu() / 2 + .5

    plt.imshow(transpose(x.clip(0, 1)), cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()