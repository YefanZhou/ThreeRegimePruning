import errno
import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from three_regime_taxonomy.models import cifar as models

__all__ = ['get_model', 'get_dataloader',  'mkdir_p', 'AverageMeter', 'get_conv_zero_param']




def get_dataloader(args, with_data_augmentation=True, trainset_shuffle=True, droplast=False):
    print('==> Preparing dataset %s' % args.dataset)
    if 'cifar' in args.dataset:
        CIFAR_MEAN= [0.4914, 0.4822, 0.4465]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
        if with_data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    elif 'svhn' in args.dataset:
        SVHN_MEAN = [0.4377, 0.4438, 0.4728]
        SVHN_STD = [0.1980, 0.2010, 0.1970]
        if with_data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(SVHN_MEAN, SVHN_STD),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(SVHN_MEAN, SVHN_STD),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
            ])

    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        trainset = dataloader(root=args.data_path, train=True, download=True, transform=transform_train)
        testset = dataloader(root=args.data_path, train=False, download=True, transform=transform_test)

    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        trainset = dataloader(root=args.data_path, train=True, download=True, transform=transform_train)
        testset = dataloader(root=args.data_path, train=False, download=True, transform=transform_test)
    
    elif args.dataset == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10
        trainset = dataloader(root=args.data_path, split='train', download=True, transform=transform_train)
        testset = dataloader(root=args.data_path, split='test', download=True, transform=transform_test)
    
    
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=trainset_shuffle, num_workers=args.workers, drop_last=droplast)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    return trainloader, testloader, num_classes


def get_model(args, num_classes):
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch == 'preresnet':
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch == 'preresnet_width':
        print(f"init a width model {args.width_scale}")
        model = models.__dict__[args.arch](depth=args.depth, k=args.width_scale, num_classes=num_classes)
        
    elif args.arch.startswith('vgg'):
        model = models.__dict__[args.arch](num_classes=num_classes)

    else:
        raise NotImplementedError

    return model


def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
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