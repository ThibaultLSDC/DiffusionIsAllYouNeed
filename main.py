from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from diffusion.algorithm import BaseDiffusion
from diffusion.model import BaseNet

def transpose(img):
    return torch.transpose(torch.transpose(img, 0, -1), 0, 1)

transform = Compose([
    ToTensor(),
    Normalize(.5, .5)
])

BATCH_SIZE = 256

train_dataset = MNIST('./data', train=True, transform=transform, download=True)
test_dataset = MNIST('./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

def main():
    x, y = next(iter(train_loader))

    x = x[0]
    y = y[0]

    # plt.title(str(y.item()))
    # plt.imshow(transpose(x), cmap='gray')
    # plt.show()

    net = BaseNet()

    diffusion = BaseDiffusion(net)

    print(diffusion.alphas)
    print(diffusion.sigmas.shape)
    print(diffusion.sigmas[0])

    for epoch in range(3):
        diffusion.train(train_loader)

    x, _ = diffusion.sample(torch.ones((1, 1, 28, 28), device=diffusion.device))

    x = torch.squeeze(x, dim=0).cpu()
    plt.imshow(transpose(x), cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()