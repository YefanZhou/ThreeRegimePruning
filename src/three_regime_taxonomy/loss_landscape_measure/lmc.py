"""Test mode connectivity between two models
"""
from __future__ import print_function
import argparse
import os
import random
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as torch_prune

import three_regime_taxonomy.models.cifar.curves as curves 
from three_regime_taxonomy.models import cifar as models
import three_regime_taxonomy.loss_landscape_measure.mc_utils as mc_utils
from three_regime_taxonomy.layer_adaptive_sparsity.pruners import prune_weights_reparam, prune_weights_remove_reparam
from three_regime_taxonomy.layer_adaptive_sparsity.utils import get_model_total_sparsity, get_modules
from three_regime_taxonomy.utils import get_dataloader, mkdir_p
from three_regime_taxonomy.utils import get_model as get_base_model


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--train-batch', default=64, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N', help='test batchsize')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--resume_ckpt_1', default='',  type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_ckpt_2', default='',  type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='preresnet')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--width-scale', type=float, default=1)
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--save_dir', default='test_checkpoint/', type=str)
parser.add_argument('--data-path', default='/work/yefan0726/data/cifar', type=str)

# Some parameters to cover lmc
parser.add_argument("--with-data-augmentation", action='store_true', default=False)
parser.add_argument("--num-points",  type=int, default=11, help='number of interpolated points in the curve')
parser.add_argument('--curve',       type=str, default=None, metavar='CURVE', help='curve type to use (default: None)')
parser.add_argument('--num_bends',   type=int, default=3, metavar='N',help='number of curve bends (default: 3)')
parser.add_argument('--fix_start',   action='store_true',              help='fix start point (default: off)')
parser.add_argument('--fix_end',     action='store_true',                help='fix end point (default: off)')
parser.add_argument('--to_eval',     type=str,             default=None, help='curve network to be evaluated')

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
cudnn.benchmark = True

best_acc = 0  # best test accuracy

def main():
    curve = getattr(curves, 'Bezier')
    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)

    # Data
    trainloader, testloader, num_classes = get_dataloader(args, args.with_data_augmentation, trainset_shuffle=False)

    # Model 
    print('Import two endpoint models...')
    curve_arch = args.arch + '_curve'
    model = curves.CurveNet(
        num_classes,
        curve,
        models.__dict__[curve_arch].curve,
        args.num_bends,
        args.fix_start,
        args.fix_end,
        architecture_kwargs=get_model_kwargs(args),
    )
    model1 = get_model(args, num_classes, args.resume_ckpt_1)
    model2 = get_model(args, num_classes, args.resume_ckpt_2)
    for m, k in [(model1, 0), (model2, args.num_bends - 1)]:
        model.import_base_parameters(m, k)
    print('---------> Linear initialize the middle points.')
    model.init_linear()

    args.to_eval = mc_utils.save_checkpoint(
        args.save_dir,
        -1,
        model_state=model.state_dict()
    )
    print(f"---------> Evaluating network: {args.to_eval}")

    model = curves.CurveNet(
            num_classes,
            curve,
            models.__dict__[curve_arch].curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=get_model_kwargs(args),
        )
    model.cuda()
    checkpoint = torch.load(args.to_eval)
    model.load_state_dict(checkpoint['model_state'])

    criterion = F.cross_entropy
    regularizer = curves.l2_regularizer(args.weight_decay)

    T       = args.num_points
    ts      = np.linspace(0.0, 1.0, T)
    tr_loss = np.zeros(T)
    tr_nll  = np.zeros(T)
    tr_acc  = np.zeros(T)
    te_loss = np.zeros(T)
    te_nll  = np.zeros(T)
    te_acc  = np.zeros(T)
    tr_err  = np.zeros(T)
    te_err  = np.zeros(T)
    dl      = np.zeros(T)

    previous_weights = None
    columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test loss', 'Test nll', 'Test error (%)']

    t = torch.FloatTensor([0.0]).cuda()

    for i, t_value in enumerate(ts):
        print(f"Sampleing Points at t = {t_value:.2f}")
        t.data.fill_(t_value)
        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt( np.sum(np.square(weights - previous_weights)) )
        previous_weights = weights.copy()

        mc_utils.update_bn(trainloader, model, t=t)
        tr_res = mc_utils.test(trainloader, model, criterion, regularizer, t=t)
        te_res = mc_utils.test(testloader, model, criterion, regularizer, t=t)
        tr_loss[i] = tr_res['loss']
        tr_nll[i] = tr_res['nll']
        tr_acc[i] = tr_res['accuracy']
        tr_err[i] = 100.0 - tr_acc[i]
        te_loss[i] = te_res['loss']
        te_nll[i] = te_res['nll']
        te_acc[i] = te_res['accuracy']
        te_err[i] = 100.0 - te_acc[i]

        values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_loss[i], te_nll[i], te_err[i]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
    tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

    te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
    te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
    te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)

    print('Length: %.2f' % np.sum(dl))
    print(tabulate.tabulate([
                ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
                ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
                ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
                ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
            ], [
                '', 'start', 'end', 'min', 'max', 'avg', 'int'
            ], tablefmt='simple', floatfmt='10.4f'))

    np.savez(
        os.path.join(args.save_dir, f'curve_points_{args.num_points}.npz'),
        ts=ts,
        dl=dl,
        tr_loss=tr_loss,
        tr_loss_min=tr_loss_min,
        tr_loss_max=tr_loss_max,
        tr_loss_avg=tr_loss_avg,
        tr_loss_int=tr_loss_int,
        tr_nll=tr_nll,
        tr_nll_min=tr_nll_min,
        tr_nll_max=tr_nll_max,
        tr_nll_avg=tr_nll_avg,
        tr_nll_int=tr_nll_int,
        tr_acc=tr_acc,
        tr_err=tr_err,
        tr_err_min=tr_err_min,
        tr_err_max=tr_err_max,
        tr_err_avg=tr_err_avg,
        tr_err_int=tr_err_int,
        te_loss=te_loss,
        te_loss_min=te_loss_min,
        te_loss_max=te_loss_max,
        te_loss_avg=te_loss_avg,
        te_loss_int=te_loss_int,
        te_nll=te_nll,
        te_nll_min=te_nll_min,
        te_nll_max=te_nll_max,
        te_nll_avg=te_nll_avg,
        te_nll_int=te_nll_int,
        te_acc=te_acc,
        te_err=te_err,
        te_err_min=te_err_min,
        te_err_max=te_err_max,
        te_err_avg=te_err_avg,
        te_err_int=te_err_int,
    )
    print(f"Saving Final Stats to {os.path.join(args.save_dir, f'curve_points_{args.num_points}.npz')}")


def get_model_kwargs(args): 
    if args.arch.startswith('densenet'):
        kwargs = {
                    'depth':args.depth,
                    'growthRate':args.growthRate,
                    'compressionRate':args.compressionRate,
                    'dropRate':args.drop,
            }
    elif args.arch == 'preresnet_width':
        print(f"init a width model {args.width_scale}")
        kwargs = {
                    'depth':args.depth,
                    'k':args.width_scale
            }
    elif args.arch.startswith('vgg'):
        configuration = {11: 'A', 13: 'B', 16: 'D', 19: 'E'}
        batch_norm = 'bn' in args.arch
        kwargs = {
                  "config": configuration[args.depth],
                  "batch_norm":  batch_norm
                }
    elif args.arch == 'preresnet':
        kwargs = {
                    "depth":args.depth,
                }
    else:
        kwargs = {}
    return kwargs


def get_model(args, num_classes, resume_path):
    # Model
    print("==> creating model '{}'".format(args.arch))
    model = get_base_model(args, num_classes)  
    model = torch.nn.DataParallel(model)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # Resume
    if resume_path:
        # Load checkpoint and sparse masks
        assert os.path.isfile(resume_path), 'Error: no checkpoint directory found!'
        prune_weights_reparam(model)
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        for m in get_modules(model):
            torch_prune.custom_from_mask(m, 'weight', m.weight_mask)

        print("remove weights_orig, apply mask to weights")
        prune_weights_remove_reparam(model)

        # Check sparsity
        pruned_layer_indices = []
        layer_idx = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                pruned_layer_indices.append(layer_idx)
                layer_idx += 1
        
        density = get_model_total_sparsity(model, pruned_layer_indices)
        print(f"Loading Checkpoints from {resume_path}, Model Density: {density * 100:.2f} %, Saved Test Acc: {checkpoint['acc']:.2f}")
    
    print("============================")
    return model


def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int




if __name__ == '__main__':
    main()
