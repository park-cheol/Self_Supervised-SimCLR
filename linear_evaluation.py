import argparse
import os
import random
import shutil
import sys
import datetime
import time
import warnings
from enum import Enum
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

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

from models.SimCLR import SimCLR
from models.logistic_regression import LogisticRegression
from utils.accuracy import accuracy
from utils.utils import create_data_loaders_from_arrays
from utils.inference import inference
from data.transforms import Transforms


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out-dim', default=128, type=int)
parser.add_argument('--temperature', default=0.07, type=float)

# parser.add_argument('--fp16-precision', action='store_true',
#                     help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    summary = SummaryWriter()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # Load SimCLR
    simclr = SimCLR(encoder_name=args.arch, out_dim=args.out_dim).cuda(args.gpu)
    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.resume, map_location=loc)
    simclr.load_state_dict(checkpoint['G_A2B'])
    simclr.eval() # Freeze Weight

    # create Model for linear evaluation
    model = LogisticRegression(simclr.in_features, n_classes=10) # cifar-10 STL-10

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Dataset And Dataloader
    cudnn.benchmark = True

    if args.dataset_name == 'stl10':
        data_transforms = Transforms(size=96).test_transform
        train_dataset = datasets.STL10(args.data, split='train', transform=data_transforms,
                                       download=True)
        test_dataset = datasets.STL10(args.data, split='test', transform=data_transforms,
                                      download=True)

    elif args.dataset_name == 'cifar10':
        data_transforms = Transforms(size=32).test_transform
        train_dataset = datasets.CIFAR10(args.data, train=True, transform=data_transforms,
                                         download=True)
        test_dataset = datasets.CIFAR10(args.data, train=False, transform=data_transforms,
                                        download=True)
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, drop_last=True,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    print("Representation from Pretrained SimCLR")
    train_x, train_y = inference(train_loader, simclr, args.gpu)
    test_x, test_y = inference(test_loader, simclr, args.gpu)

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_x, train_y,
                                                                        test_x, test_y,
                                                                        args.batch_size)

    # Train
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(arr_train_loader, model, criterion, optimizer, epoch, args, summary)
        print("Testing")
        acc1, acc5, loss = test(arr_test_loader, model, criterion, epoch, args, summary)
        print(" Epoch [%d] | loss: %f | Acc@1: %f | Acc@5: %f |"
              % (epoch + 1, loss, acc1, acc5))
        if acc1 >= best_acc1:
            print("Record Best Acc1")
            best_acc1 = acc1
    print("BEST ACC1 :", best_acc1)


def train(dataloader, model, criterion, optimizer, epoch, args, summary):
    model.train()

    for i, (image, target) in enumerate(dataloader):
        start_time = time.time()
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(image)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        niter = epoch * len(dataloader) + i
        summary.add_scalar('Train/loss', loss.item(), niter)
        summary.add_scalar('Train/acc1', acc1[0].item(), niter)
        summary.add_scalar('Train/acc5', acc5[0].item(), niter)

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | loss: %f | Acc@1: %f | Acc@5: %f |"
                  % (epoch + 1, i, len(dataloader), loss, acc1[0], acc5[0]))

    elapse = datetime.timedelta(seconds=time.time() - start_time)
    print(f"걸린 시간: ", elapse)


def test(dataloader, model, criterion, epoch, args, summary):
    loss_avg = 0
    acc1_avg = 0
    acc5_avg = 0
    model.eval()

    for i, (image, target) in tqdm(enumerate(dataloader)):
        model.zero_grad()
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(image)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_avg += acc1[0]
        acc5_avg += acc5[0]
        loss_avg += loss

        niter = epoch * len(dataloader) + i
        summary.add_scalar('Test/loss', loss.item(), niter)
        summary.add_scalar('Test/acc1', acc1[0].item(), niter)
        summary.add_scalar('Test/acc5', acc5[0].item(), niter)

    summary.add_scalar('Test/avg_acc5', acc1_avg, epoch)
    summary.add_scalar('Test/avg_acc5', acc5_avg, epoch)

    return acc1_avg / len(dataloader), acc1_avg / len(dataloader), loss_avg / len(dataloader)


if __name__ == "__main__":
    main()