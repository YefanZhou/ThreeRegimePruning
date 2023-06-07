"""Test Jensen-Shannon Divergence or CKA and JSD similarity between two models
"""
from __future__ import print_function

import argparse
import os
import random
import time
import numpy as np
from os.path import join

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as torch_prune


from three_regime_taxonomy.loss_landscape_measure.cka_utils import *
from three_regime_taxonomy.models import cifar as models
from three_regime_taxonomy.utils import Bar, AverageMeter, accuracy, mkdir_p, get_dataloader
from three_regime_taxonomy.utils import get_model as get_base_model
from three_regime_taxonomy.layer_adaptive_sparsity.pruners import prune_weights_reparam, prune_weights_remove_reparam
from three_regime_taxonomy.layer_adaptive_sparsity.utils import get_model_total_sparsity, get_modules


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--resume_ckpt_1', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_ckpt_2', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--manualSeed', type=int, help='manual seed')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='preresnet')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--width-scale', type=float, default=1)
parser.add_argument('--save_dir', default='test_checkpoint/', type=str)
parser.add_argument('--data-path', default='/work/yefan0726/data/cifar', type=str)


# Some parameters to cover cka
parser.add_argument("--with-data-augmentation", action='store_true', default=False)
parser.add_argument("--CKA-batches", type=int, default=100, help='number of batches for computing CKA')
parser.add_argument("--CKA-repeat-runs", type=int, default=1, help='number of repeat for CKA')
parser.add_argument('--flattenHW', default = False, action = 'store_true', help = 'flatten the height and width dimension while only comparing the channel dimension')
parser.add_argument('--not-input', dest='not_input', default = False, action='store_true', help='no CKA computation on input data')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
use_cuda = torch.cuda.is_available()
torch.backends.cudnn.deterministic = True
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)
    # Data
    trainloader, testloader, num_classes = get_dataloader(args, args.with_data_augmentation, trainset_shuffle=False)

    # Model 
    model1 = get_model(args, num_classes, args.resume_ckpt_1)
    model2 = get_model(args, num_classes, args.resume_ckpt_2)
    criterion = nn.CrossEntropyLoss()

    model1_test_loss, model1_test_acc = test(testloader, model1, criterion, -1, use_cuda)
    model2_test_loss, model2_test_acc = test(testloader, model2, criterion, -1, use_cuda)
    
    print(f'model 1: test loss: {model1_test_loss},  test acc: {model1_test_acc} \n model 2: test loss: {model2_test_loss},  test acc: {model2_test_acc}')

    model1.eval()
    model2.eval()

    print("Measuring CKA similarity")
    cka_from_features_average = []

    for _ in range(args.CKA_repeat_runs):
        cka_from_features = []
        latent_all_1, latent_all_2 = all_latent(model1, model2, trainloader, num_batches = args.CKA_batches, args=args)
        for name in latent_all_1.keys():
            if args.flattenHW:
                cka_from_features.append(feature_space_linear_cka(latent_all_1[name], latent_all_2[name]))
            else:
                cka_from_features.append(cka_compute(gram_linear(latent_all_1[name]), gram_linear(latent_all_2[name])))
            
        cka_from_features_average.append(cka_from_features)

    cka_from_features_average = np.mean(np.array(cka_from_features_average), axis=0)
    print('cka_from_features shape: ', cka_from_features_average.shape)

    save_folder = f"batches{args.CKA_batches}_repeat{args.CKA_repeat_runs}_flat{args.flattenHW}"

    np.save(join(args.save_dir, f'cka_similarity_{save_folder}.npy'), cka_from_features_average)
    print(f"Saving to {join(args.save_dir, f'cka_similarity_{save_folder}.npy')}")


def get_model(args, num_classes, resume_path):
    # Model
    model = get_base_model(args, num_classes)  
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Resume
    if resume_path:
        # Load checkpoint and sparse masks
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(resume_path), 'Error: no checkpoint directory found!'
            
        print("Loading pruned model from: ", resume_path)
        prune_weights_reparam(model)
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        for m in get_modules(model):
            torch_prune.custom_from_mask(m, 'weight', m.weight_mask)

        # Check sparsity
        pruned_layer_indices = []
        layer_idx = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                pruned_layer_indices.append(layer_idx)
                layer_idx += 1

        # Measure remaining parameter percent
        density = get_model_total_sparsity(model, pruned_layer_indices)
        print(f"Loading Checkpoints from {resume_path}, Model Density: {density * 100:.2f} %, Saved Test Acc: {checkpoint['acc']:.2f}")
    
    return model



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

def save_checkpoint(state, is_best, checkpoint, filename='finetuned.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
