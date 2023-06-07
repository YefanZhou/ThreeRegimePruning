from __future__ import print_function

import sys
import argparse
import os
import random
import shutil
import time

import torch

from os.path import join
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.utils.prune as torch_prune

from three_regime_taxonomy.utils import get_dataloader, get_model, Bar, Logger, AverageMeter, accuracy, mkdir_p
from three_regime_taxonomy.layer_adaptive_sparsity.pruners import prune_weights_reparam
from three_regime_taxonomy.layer_adaptive_sparsity.utils import get_model_total_sparsity, get_modules


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options

parser.add_argument('--iterations', default=62400, type=int, metavar='N',
                    help='number of total iterations to run')

parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')

parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--optimizer', default="SGD", type=str, help='adam, sgd')

# Architecture
parser.add_argument('--arch', default='preresnet', type=str)
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--width-scale', type=float, default=1)
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--save_dir', default='test_checkpoint/', type=str)
parser.add_argument('--data-path', default='/work/yefan0726/data/cifar', type=str)



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True

best_acc = 0  # best test accuracy
curr_iter = 0

def main():
    global best_acc
    global curr_iter
    if os.path.exists(join(args.save_dir, 'finetune_stats.npy')):
        print("The results already exist")
        sys.exit(0)

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)

    # Data
    trainloader, testloader, num_classes = get_dataloader(args, droplast=True)

    # Model
    model = get_model(args, num_classes)
    print(model)
    model = torch.nn.DataParallel(model).cuda()
    
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()

    # Resume
    title = f'{args.dataset}-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'

        prune_weights_reparam(model)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

        for m in get_modules(model):
            torch_prune.custom_from_mask(m, 'weight', m.weight_mask)
 
        config = {}
        config['pruned_layer_indices'] = []
        layer_idx = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                config['pruned_layer_indices'].append(layer_idx)
                layer_idx += 1

        # measure remaining parameter percent
        remaining_parameter_percent = get_model_total_sparsity(model, config['pruned_layer_indices'])
        print(f"Model Density: {(remaining_parameter_percent * 100):.2f},  Loading Checkpoints From", args.resume)
    else:
        raise NotImplementedError
    
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    elif args.optimizer == 'Adam':
        print(f"Use optimizer: {args.optimizer} to retrain")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(args.iterations * 0.5), int(args.iterations * 0.75)], gamma=args.gamma)


    before_test_loss, before_test_acc = test(testloader, model, criterion, -1, use_cuda)
    print(f'before finetuning: test loss: {before_test_loss},  test acc: {before_test_acc}')
    logger = Logger(os.path.join(args.save_dir, 'log_finetune.txt'), title=title)
    logger.set_names(['Learning Rate', 'Epoch', 'Num Iter', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, scheduler.get_last_lr()[0]))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, scheduler, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([scheduler.get_last_lr()[0], epoch+1, curr_iter, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
  
        if curr_iter >= args.iterations:
            break

    logger.close()
    save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'curr_iter': curr_iter,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=args.save_dir, end_of_training=True)
    
    finetune_stats_dict = {'before_finetune_test_acc':before_test_acc, 
                         'before_finetune_test_loss':before_test_loss, 
                            'best_acc': best_acc, 
                            'final_test_acc': test_acc, 'final_test_loss':test_loss, 
                            'final_train_acc':train_acc, 'final_train_loss':train_loss, 'scratch': args.scratch, 
                            'epoch': epoch+1}

    np.save(join(args.save_dir, 'finetune_stats.npy'), finetune_stats_dict)
    print(f"Saving to {args.save_dir}")



def train(trainloader, model, criterion, optimizer, scheduler,  epoch, use_cuda):
    global curr_iter
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        curr_iter += 1

        if curr_iter >= args.iterations:
            break


    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)



def save_checkpoint(state, is_best, checkpoint, filename='finetuned.pth.tar', end_of_training=False):
    if end_of_training:
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)



if __name__ == '__main__':
    main()
