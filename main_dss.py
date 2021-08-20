from dss_dataloader import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#from torch.profiler import profile, record_function, ProfilerActivity

from model import *

parser = argparse.ArgumentParser(description="a simple AI model on cifar100")
parser.add_argument("--worker", "-w", type=int, default=4, help="the number of workers, default is 0")
parser.add_argument("--batch", "-b", type=int, default=16, help="batch size")
parser.add_argument("--client", "-c", type=int, default=1, help="the number of dss client instances")
parser.add_argument("--epochs", "-e", type=int, default=1, help="epoches of training")
parser.add_argument("--model","-m", type=str, default="vae", help="model name" )
parser.add_argument('--lr', default=1e-4, type=float,
                    metavar='LR', help='learning rate', dest='lr')

#parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

args = parser.parse_args()

# specify S3 storage parameters
access_key = "minio"
secret_key = "minio123"
endpoint = "http://202.0.0.3:9000" #this needs to be changed
option = dss.clientOption()
option.maxConnections = 1

# create dataloader instance
data_loader = torch.utils.data.DataLoader


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# create dataset instance for training and test
# train_ds = DSSIterableDataset(prefix="train", access_key=access_key, secret_key=secret_key, endpoint=endpoint, dss_option=option, transforms=transform)
#test_ds = DSSDataset(prefix="map-train", access_key=access_key, secret_key=secret_key, endpoint=endpoint, dss_option=option, client_num=args.client, transforms=transform)
train_ds = DSSDataset(prefix="cifar100-train", access_key=access_key, secret_key=secret_key, endpoint=endpoint, dss_option=option, client_num=args.client, transforms=transform_train)

train_loader = data_loader(train_ds, batch_size=args.batch, num_workers=args.worker, shuffle=True, collate_fn=collate_fn)
#test_loader = data_loader(test_ds, batch_size=args.batch, num_workers=args.worker)


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


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    metric_df = pd.DataFrame(columns=["epoch", "iteration", "dataloading_time", "training_time", "batch_time"])

    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        dataloading_time = time.time() - end
        data_time.update(dataloading_time)

        iamges = images.to(device)
        
        # compute loss
        optimizer.zero_grad()
        loss, summaries = model.loss_function(images)

        # measure record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step

        loss.backward()

        with torch.no_grad():
            optimizer.zero_grad()
            optimizer.step()

        # measure elapsed time
        itr_time = time.time() - end
        batch_time.update(itr_time)
        end = time.time()

        metric_df.loc[len(metric_df), :] = np.array([epoch, i, dataloading_time, itr_time-dataloading_time, itr_time])
        if i % 10 == 0:
            progress.display(i)
    # prof.export_chrome_trace("trace_epoch{}_iter{}.json".format(epoch, i))
    metric_df.to_csv("{}_batch{}_worker{}_client{}_epoch{}_iteration{}.csv".format(args.model, args.batch, args.worker, args.client ,epoch, i), index=False)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = models.__dict__[args.arch]()




if args.model == "vae":
    model = VAE()
    model = model.to(device)
    
 # define  optimizer   

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


for epoch in range(args.epochs):
    # if args.distributed:
    #     train_sampler.set_epoch(epoch)

    # train for one epoch
    print("epoch: {}".format(epoch))
    train(train_loader, model, optimizer, epoch, args)


    
















# def upload_AI_dataset(data_folder):
#     data_folder = "/opt/ansible/examples/data/cifar100/cifar100-preprocessed"
#     train_path = os.path.join(data_folder, "train")
#     test_path = os.path.join(data_folder, "test")
#     for i in os.listdir(train_path):
#         prefix = "train"
#         key = "{}-image-{}".format(prefix, i)
#         file_name = os.path.join(train_path, i)
#         try:
#             ds.client.putObject( key, file_name)
#         except Exception as e:
#             print("error in putting {}: {}".format(key, e))

#     for i in os.listdir(test_path):
#         prefix = "test"
#         key = "{}-image-{}".format(prefix, i)
#         file_name = os.path.join(test_path, i)
#         try:
#             ds.client.putObject( key, file_name)
#         except Exception as e:
#             print("error in putting {}: {}".format(key, e))

# index = 0
# data_folder = "/opt/ansible/examples/data/mapillary/"
# for target, continent in enumerate(os.listdir(data_folder)):
#     for i in range(5000):
#         for fig in  os.listdir(os.path.join(data_folder, continent)):
#             file_name = os.path.join(data_folder, continent, fig)
#             key = "{}-image-{}_{}".format("map-train", index , target)
#             try:
#                 train_ds.client.putObject( key, file_name)
#             except Exception as e:
#                 print("error in putting {}: {}".format(key, e))
#             index+=1
