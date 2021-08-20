'''Train CIFAR10 with PyTorch.'''
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import pandas as pd


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
parser.add_argument("--lr", default=1e-4, type=float, help='learning rate')
parser.add_argument("--mx_ep", default = 30, type = int, help = 'number of epochs')

parser.add_argument("--worker", type=int, default=2, help="the number of workers, default is 2")
parser.add_argument("--batch", type=int, default=128, help="batch size")
parser.add_argument("--client", type=int, default=0, help="the number of dss client instances")
parser.add_argument("--model", type=str, default="VAE", help="model name" )
parser.add_argument("--GPU", type=int, default=1, help="number of GPUs, default is 1" )

args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    trainset, batch_size=args.batch, shuffle=True, num_workers=args.worker)

testset = Tv.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_dl = T.utils.data.DataLoader(
    testset, batch_size=args.batch, shuffle=False, num_workers=args.worker)


# Model
print('==> Building model..')

vae = VAE()

vae = vae.to(device)
if device == 'cuda' and args.GPU > 1:
    vae = nn.DataParallel(vae, device_ids = range(0, args.GPU))
    #cudnn.benchmark = True
vae = vae.cuda()

opt = optim.Adam(vae.parameters(), lr = args.lr)

#image generation
if args.GPU>1:
    latent_dim = vae.module.latent_dim
else:
    latent_dim = vae.latent_dim 
#latent_dim = vae.latent_dim

figure(figsize=(8, 3), dpi=300)
num_preds = 16
p = T.distributions.Normal(T.zeros([latent_dim]), T.ones([latent_dim]))
z = p.rsample((num_preds,))

# Training
current_time = time.time()
total_time =0
batch_time = AverageMeter('Time', ':6.3f')
dataloading_time = AverageMeter('Data', ':6.3f')

metric_df = pd.DataFrame(columns=["epoch", "iteration", "dataloading_time", "training_time", "batch_time"])


for epoch in range(args.mx_ep):
    print('\nEpoch: %d' % epoch)
    train_loss = 0

    df_temp = pd.DataFrame(columns=["epoch", "iteration", "dataloading_time", "training_time", "batch_time"])

    progress = ProgressMeter(len(train_dl),
        [batch_time, dataloading_time],
        prefix="Epoch: [{}]".format(epoch))
    
    end = time.time()
    for batch_idx, (inputs, _) in enumerate(train_dl):
        opt.zero_grad()

        #print("inputs: ", inputs.shape)
        x = inputs.to(device)

        data_time = time.time() - end
        dataloading_time.update(data_time)
        
        x_hat, z_latent, mu, log_var = vae(x)
        if(args.GPU==1):
            loss, summaries = vae.loss_function(x_hat, x, mu, log_var)
        else:
            loss, summaries = vae.module.loss_function(x_hat, x, mu, log_var)


        loss.backward()
        
        with T.no_grad():
            opt.step()
            opt.zero_grad()

        if(batch_idx==0):

            print(epoch, batch_idx, loss.item())
            with T.no_grad():
                if(args.GPU>1):
                    pred = vae.module.decoder(z.to(device)).cpu()
                else:
                    pred = vae.decoder(z.to(device)).cpu()
            img = make_grid(pred, normalize = True).permute(1, 2, 0).numpy()
            #print("x shape", x.shape)
            plt.imshow(img)
            plt.savefig("images/images_epoch{}_step{}.png".format(epoch, batch_idx))
            print("summaries:", summaries)
            if epoch>0:
                
                total_time += time.time() - current_time
                print("epoch: ", epoch, "total time: ", total_time, "batch time: ", time.time() - current_time)
            
            current_time = time.time()
        
        itr_time = time.time() - end
        batch_time.update(itr_time)
        end = time.time()

        df_temp.loc[len(df_temp), :] = np.array([epoch, batch_idx, data_time, itr_time - data_time, itr_time])
        if batch_idx % 30 == 0:
            progress.display(batch_idx)
    # prof.export_chrome_trace("trace_epoch{}_iter{}.json".format(epoch, i))
    metric_df = metric_df.append(df_temp, ignore_index = True)    
metric_df.to_csv("{}_batch{}_worker{}_GPU{}_client{}_epoch{}_iteration{}.csv".format(args.model, args.batch, args.worker, args.GPU, args.client ,epoch, batch_idx), index=False)
















    













