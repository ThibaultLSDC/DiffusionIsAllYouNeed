from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.utils import save_image, make_grid

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from diffusion.ddpm import BaseDiffusion
from diffusion.model import AttentionUnet, BaseNet, BaseUnet

from argparse import ArgumentParser

## Parsing
parser = ArgumentParser()
parser.add_argument('-l', '--load-last', action='store_true', help="Loading last trained model")
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs in this run', default=100)
parser.add_argument('-s', '--sample-rate', type=int, help='Sample frequence', default=5)

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

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"UNet has {pytorch_total_params} trainable parameters")

    diffusion = BaseDiffusion(net)

    if args.load_last:

        ckpt = torch.load('./models/latest')
        diffusion.model.load_state_dict(ckpt['model_state_dict'])
        diffusion.ema.load_state_dict(ckpt['ema_state_dict'])
        diffusion.optim.load_state_dict(ckpt['optimizer_state_dict'])
        x_T = ckpt['x_T']
        loss_hist = ckpt['loss_history']
        epochs_done = ckpt['epoch']
    else:
        loss_hist = []
        epochs_done = 0
        x_T = torch.randn((16, 3, 32, 32))

    for epoch in range(epochs_done, epochs_done + args.epochs):
        loss = diffusion.train(train_loader, epoch)

        loss_hist.append(loss)

        if epoch % args.sample_rate == 0:
            x, _ = diffusion.sample(x_T)
            x = torch.squeeze(x, dim=0).cpu() / 2 + .5

            save_image(make_grid(x.clip(0, 1), nrow=4), f"./data/output/{epoch+1}.png")

        plt.semilogy(loss_hist)

        plt.savefig('./data/output/curve.png')


        torch.save({
        'epoch': epoch,
        'model_state_dict': diffusion.model.state_dict(),
        'ema_state_dict': diffusion.ema.state_dict(),
        'optimizer_state_dict': diffusion.optim.state_dict(),
        'x_T': x_T,
        'loss_history': loss_hist,
        }, f"./models/latest")
    
        if epoch % 50 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion.model.state_dict(),
            'ema_state_dict': diffusion.ema.state_dict(),
            'optimizer_state_dict': diffusion.optim.state_dict(),
            'x_T': x_T,
            'loss_history': loss_hist,
            }, f"./models/{epoch+1}")

if __name__ == '__main__':
    main()