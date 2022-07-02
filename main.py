from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.utils import save_image

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from diffusion.algorithm import BaseDiffusion
from diffusion.model import AttentionUnet, BaseNet, BaseUnet

from argparse import ArgumentParser

## Parsing
parser = ArgumentParser()
parser.add_argument('-l', '--load-last', action='store_true', help="Loading last trained model")
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs in this run', default=100)

args = parser.parse_args()

def transpose(img):
    return torch.transpose(torch.transpose(img, 0, -1), 0, 1)

transform = Compose([
    ToTensor(),
    Normalize(.5, .5)
])

BATCH_SIZE = 128

train_dataset = CIFAR10('./data', train=True, transform=transform, download=False)
test_dataset = CIFAR10('./data', train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

def main():

    net = AttentionUnet()

    diffusion = BaseDiffusion(net)

    if args.load_last:

        ckpt = torch.load('./models/latest')
        diffusion.model.load_state_dict(ckpt['model_state_dict'])
        diffusion.optim.load_state_dict(ckpt['optimizer_state_dict'])
        loss_hist = ckpt['loss_history']
        epochs_done = ckpt['epoch']
    else:
        loss_hist = []
        epochs_done = 0

    for epoch in range(epochs_done, epochs_done + args.epochs):
        loss = diffusion.train(train_loader, epoch)

        loss_hist.append(loss)

        x, _ = diffusion.sample(torch.ones((1, 3, 32, 32), device=diffusion.device))

        x = torch.squeeze(x, dim=0).cpu() / 2 + .5

        save_image(x.clip(0, 1), f"./data/output/{epoch+1}.png")

        plt.plot(loss_hist)

        plt.savefig('./data/output/curve.png')

    
        if epoch % 10 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion.model.state_dict(),
            'optimizer_state_dict': diffusion.optim.state_dict(),
            'loss_history': loss_hist,
            }, f"./models/{epoch+1}")
            torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion.model.state_dict(),
            'optimizer_state_dict': diffusion.optim.state_dict(),
            'loss_history': loss_hist,
            }, f"./models/latest")

if __name__ == '__main__':
    main()