from torchvision.utils import save_image

import os

import torch

for path in os.listdir('./data/output'):
    print(path)
    img = torch.load('./data/output/' + path)
    img = torch.transpose(torch.transpose(img, 0, -1), 1, -1)
    print(img.shape)
    save_image(img, './data/output/' + path + '.png')

import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()