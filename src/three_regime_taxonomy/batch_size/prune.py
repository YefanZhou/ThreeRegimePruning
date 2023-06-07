from __future__ import print_function

import argparse
import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from three_regime_taxonomy.utils import get_dataloader, get_model, Bar, Logger, AverageMeter, accuracy, mkdir_p
from three_regime_taxonomy.layer_adaptive_sparsity.pruners import weight_pruner_loader, prune_weights_reparam
from three_regime_taxonomy.layer_adaptive_sparsity.utils import get_model_total_sparsity

from os.path import join

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch',   metavar='ARCH', default='preresent')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--save_dir', default='test_checkpoint/', type=str)
parser.add_argument('--data-path', default='/work/yefan0726/data/cifar', type=str)
parser.add_argument('--width-scale', type=float, default=1)

# Some parameters to cover pruning
parser.add_argument('--percent', default=0.6, type=float, help='percentage of weight to prune')
parser.add_argument("--prune_layer_sparsity", type=str, choices=['unif', 'glob'], help="uniform pruning, or global pruning", default='unif')


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


def main():
    global best_acc
    if os.path.exists(join(args.save_dir, "prune_score.npy")):
        print("results exists")
        sys.exit(0)

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)

    # Data
    _, testloader, num_classes = get_dataloader(args)
    model =  get_model(args, num_classes)
    
    print(model)

    model = torch.nn.DataParallel(model).cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())  /1000000.0))

    criterion = nn.CrossEntropyLoss()
    # Resume
    title = f'{args.dataset}-' + args.arch
    if args.resume:
        # Load checkpoint.
        print(f'==> Resuming from checkpoint from {args.resume}..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    print('Evaluation only')
    test_loss0, test_acc0 = test(testloader, model, criterion, start_epoch, use_cuda)
    print('Before pruning: Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss0, test_acc0))


    # initialize layer mask with all 1s
    config = {}
    config['pruned_layer_indices'] = []
    layer_idx = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            config['pruned_layer_indices'].append(layer_idx)
            layer_idx += 1


    config['prune_layer_sparsity'] = args.prune_layer_sparsity
    config['pruning_ratio'] = args.percent

    start_time = time.time()

    print(config)
    pruner = weight_pruner_loader(config)
    prune_weights_reparam(model)

    remained_params_lst, ori_params_lst = pruner(model, config['pruning_ratio'], config['pruned_layer_indices'])

    model.zero_grad()
    elapsed_time = (time.time() - start_time)/60
    total_unpruned_parameters = sum(remained_params_lst)
    total_parameters = sum(ori_params_lst)
    # measure remaining parameter percent
    remaining_parameter_percent = get_model_total_sparsity(model, config['pruned_layer_indices'])
    print(f"Model Density After Pruning: {(remaining_parameter_percent * 100):.2f} %")

    np.savez(join(args.save_dir, "prune_stats.npz"), 
                            total_indices_params=total_parameters,
                            remained_indices_params=total_unpruned_parameters,
                            numel_gt_total_params=sum(p.numel() for p in model.parameters()),
                            total_remaining=remaining_parameter_percent, 
                            layerwise_remaining=remained_params_lst, 
                            layerwise_origin=ori_params_lst, 
                            elapsed_time_minute = elapsed_time,
                            prune_arg_percent=args.percent)
    

    print('\nTesting')
    test_loss1, test_acc1 = test(testloader, model, criterion, start_epoch, use_cuda)
    print('After Pruning: Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss1, test_acc1))
    save_checkpoint({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'acc': test_acc1,
            'best_acc': 0.,
        }, False, checkpoint=args.save_dir)

    with open(os.path.join(args.save_dir, 'prune.txt'), 'w') as f:
        f.write('Before pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss0, test_acc0))
        f.write('After pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss1, test_acc1))

    perf_dict = {'test_loss_before':test_loss0 , 'test_loss_after':test_loss1 ,  'test_acc_before':test_acc0 , 'test_acc_after':test_acc1}
    
    np.save(join(args.save_dir, "prune_score.npy"), perf_dict)
    print(f"Saving to {args.save_dir}")

    return


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

def save_checkpoint(state, is_best, checkpoint, filename='pruned.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


if __name__ == '__main__':
    main()