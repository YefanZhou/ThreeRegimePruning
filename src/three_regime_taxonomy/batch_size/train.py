from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm

from pathlib import Path

from three_regime_taxonomy.utils import SAM, get_dataloader, get_model, Bar, Logger, AverageMeter, accuracy, mkdir_p


parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--iterations', default=62400, type=int, metavar='N', help='number of total iterations to run')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--optimizer', default="SGD", type=str, help='adam, sgd')

parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='preresent')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--width-scale', type=float, default=1)
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--save_dir', default='', type=str)
parser.add_argument('--data-path', default='', type=str)

parser.add_argument('--rho', default=0.05, type=float, help='rho for SAM')
parser.add_argument("--adaptive", default=False, type=bool, help="use adaptive SAM")


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

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
    if os.path.exists(os.path.join(args.save_dir, f"checkpoint_iter_{args.iterations}.pth.tar")):
        print("=====> results exists <=====")
        sys.exit(0)

    global best_acc
    global curr_iter
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Data
    trainloader, testloader, num_classes = get_dataloader(args, droplast=True)

    # Model
    model  =  get_model(args, num_classes)

    print(model)
    model = torch.nn.DataParallel(model).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SAM':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(args.iterations * 0.5), int(args.iterations * 0.75)], gamma=args.gamma)

    # Resume
    title = f'{args.dataset}-' + args.arch
    if args.resume:
        # Load checkpoint.
        print(f'==> Resuming from checkpoint..{args.resume}')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # use given save dir
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        print("start_epoch", checkpoint['epoch'])
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(args.save_dir)

    logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Epoch', 'Num Iter', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return



    train_loss, train_acc = test(trainloader, model, criterion, start_epoch, use_cuda)
    test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    if scheduler is None:
        logger.append([args.lr, 0, curr_iter, train_loss, test_loss, train_acc, test_acc])
    else:
        logger.append([scheduler.get_last_lr()[0], 0, curr_iter, train_loss, test_loss, train_acc, test_acc])

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        if scheduler is None:
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, args.lr ))
        else:
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, scheduler.get_last_lr()[0] ))
        
        if args.optimizer == 'SAM':
            train_loss, train_acc = train_sam(trainloader, model, criterion, optimizer, scheduler, epoch, use_cuda)
        else:
            train_loss, train_acc = train(trainloader, model, criterion, optimizer, scheduler, epoch, use_cuda)
            
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        if scheduler is None:
            logger.append([args.lr, epoch + 1, curr_iter, train_loss, test_loss, train_acc, test_acc])
        else:
            logger.append([scheduler.get_last_lr()[0], epoch+1, curr_iter, train_loss, test_loss, train_acc, test_acc])


        if curr_iter >= args.iterations:
            break
        
    logger.close()

    if curr_iter == args.iterations:
        filepath = os.path.join(args.save_dir, f"checkpoint_iter_{curr_iter}.pth.tar")
        print(f"\n====================================\n")
        print(f"Iteration Checkpoint: Saving to {filepath}")
        print(f"\n====================================\n")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': None,
            'curr_iter': curr_iter,
            'best_acc': -1,
            'optimizer' : optimizer.state_dict(),
        }, filepath)
    else:
        raise ValueError('Final number of iterations fails to match the target')


def train(trainloader, model, criterion, optimizer, scheduler, epoch, use_cuda):
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
    print(args)
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

        if scheduler is not None:
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


def train_sam(trainloader, model, criterion, optimizer, scheduler, epoch, use_cuda):
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
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # first forward-backward step
        enable_running_stats(model)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        loss.backward()
        optimizer.first_step(zero_grad=True)
        disable_running_stats(model)
        adv_outputs = model(inputs)
        adv_loss = criterion(adv_outputs, targets)

        adv_loss.backward()
        optimizer.second_step(zero_grad=True)

        if scheduler is not None:
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


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


if __name__ == '__main__':
    main()
