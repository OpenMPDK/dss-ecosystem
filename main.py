'''Train CIFAR10 with PyTorch.'''
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision as Tv
import torchvision.transforms as transforms

from torchvision.utils import make_grid
from matplotlib.pyplot import imshow, figure
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse

from model import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--mx_ep', default = 30, type = int, help = 'number of epochs')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = Tv.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_dl = T.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = Tv.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_dl = T.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')

vae = VAE()
vae = vae.to(device)
if device == 'cuda':
    vae = nn.DataParallel(vae)
    #cudnn.benchmark = True

opt = optim.Adam(vae.parameters(), lr = args.lr)

#image generation
latent_dim = vae.module.latent_dim
figure(figsize=(8, 3), dpi=300)
num_preds = 16
p = T.distributions.Normal(T.zeros([latent_dim]), T.ones([latent_dim]))
z = p.rsample((num_preds,))

# Training
for epoch in range(args.mx_ep):
    print('\nEpoch: %d' % epoch)

    for batch_idx, (inputs, _) in enumerate(train_dl):
        
        opt.zero_grad()
        x = inputs.to(device)

        loss, summaries = vae.module.loss_function(x)
        loss.backward()
        
        with T.no_grad():
            opt.step()
            opt.zero_grad()

        if(batch_idx==0):

            print("Epoch number: ", epoch, "loss: ", loss.item())
            
            with T.no_grad():
                pred = vae.module.decoder(z.to(device)).cpu()

            img = make_grid(pred, normalize = True).permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.savefig("images/images_epoch{}_step{}.png".format(epoch, batch_idx))
            print("summaries:", summaries)
    















    







