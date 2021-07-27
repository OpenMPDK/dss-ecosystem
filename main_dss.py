from dss_dataloader import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as multiprocessing
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

parser = argparse.ArgumentParser(description="a simple AI model on cifar100")
parser.add_argument("--worker", "-w", type=int, default=2, help="the number of workers, default is 0")
parser.add_argument("--batch", "-b", type=int, default=16, help="batch size")
parser.add_argument("--epochs", "-e", type=int, default=1, help="epoches of training")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
args = parser.parse_args()

# specify S3 storage parameters
access_key = "minio"
secret_key = "minio123"
endpoint = "http://202.0.0.3:9000"
option = dss.clientOption()
option.maxConnections = 1

# create dataloader instance
data_loader = torch.utils.data.DataLoader


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

# create dataset instance for training and test
train_ds = DSSIterableDataset(prefix="train", access_key=access_key, secret_key=secret_key, endpoint=endpoint, dss_option=option, transforms=transform)
test_ds = DSSIterableDataset(prefix="train", access_key=access_key, secret_key=secret_key, endpoint=endpoint, dss_option=option)

train_loader = data_loader(train_ds, batch_size=args.batch, num_workers=args.worker)
test_loader = data_loader(test_ds, batch_size=args.batch, num_workers=args.worker)

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    metric_df = pd.DataFrame(columns=["epoch", "iteration", "dataloading_time", "training_time", "batch_time"])

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        dataloading_time = time.time() - end
        data_time.update(dataloading_time)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        itr_time = time.time() - end
        batch_time.update(itr_time)
        end = time.time()

        metric_df.loc[len(metric_df), :] = np.array([epoch, i, dataloading_time, itr_time-dataloading_time, itr_time])
        if i % 10 == 0:
            progress.display(i)
    metric_df.to_csv("epoch{}_iteration{}.csv".format(epoch, i), index=False)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# model = models.__dict__[args.arch]()
model = models.resnet18(pretrained=False)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

for epoch in range(args.epochs):
    # if args.distributed:
    #     train_sampler.set_epoch(epoch)
    # adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    print("epoch: {}".format(epoch))
    train(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation set
    # acc1 = validate(val_loader, model, criterion, args)

    # remember best acc@1 and save checkpoint
    # is_best = acc1 > best_acc1
    # best_acc1 = max(acc1, best_acc1)

    
















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
