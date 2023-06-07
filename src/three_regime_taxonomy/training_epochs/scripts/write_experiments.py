import os
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--ckpt-path', default='/data/yefan0726/checkpoints_public', type=str, help='checkpoint path')
parser.add_argument('--data-path', default='/data/yefan0726/data_public',        type=str, help='data path')
parser.add_argument('--all',                   action='store_true',               default=False )
parser.add_argument('--train',                 action='store_true',               default=False )
parser.add_argument('--prune',                 action='store_true',               default=False )
parser.add_argument('--retrain',               action='store_true',               default=False )
parser.add_argument('--lmc',                   action='store_true',               default=False )
parser.add_argument('--cka',                   action='store_true',               default=False )
parser.add_argument('--prune-ratio-lst',      type=float,    nargs='+',           default=[0.95, 0.94, 0.93, 0.92, 0.9, 0.86, 0.8, 0.6, 0.2])
parser.add_argument('--earlystop-epoch-lst',  type=int,     nargs='+',             default=range(10, 170, 10))
parser.add_argument('--optimizer',            type=str,                            default='SGD')
parser.add_argument('--rho-lst',              type=float,  nargs='+',           default=[0.6])


args = parser.parse_args()
##################################### Hyperparameters Settings ##################################################
############################################################################################################
############################################################################################################
# train
task='cv'
dataset='cifar10'
arch='preresnet'
depth=20
batch_size=64
train_seed_lst=[1]
optimizer=args.optimizer # / Adam / SAM
rho_lst=args.rho_lst
initial_lr=0.1
stage='train'
epochs=160
earlystop_interval=10
width_scale=1
gpu_id=0

# prune
earlystop_epoch_lst=args.earlystop_epoch_lst  #, 0
prune_distri='unif'
prune_ratio_lst=args.prune_ratio_lst #, 

# retrain
retrain_optimizer='SGD' # SGD 
retrain_initial_lr=0.1
retrain_epochs=160
retrain_seed_lst=[1, 2, 3]
retrain_seed_pair_lst=[[1, 2], [1, 3], [2, 3]]

# lmc
num_points=11
#####################################################################################################################
#####################################################################################################################



script_path = os.path.dirname(os.path.abspath(__file__))

with open(f'{script_path}/three_stage.sh', 'w') as f:
    f.write(f'export ckpt_path={args.ckpt_path}\n')
    f.write(f'export data_path={args.data_path}\n')
    f.write(f'\n')


############################################################################
########################      Train   ######################################
############################################################################

if args.all or args.train:
    with open(f'{script_path}/config/train.txt', 'w') as f:
        train_count = 1
        for train_seed in train_seed_lst:
            for rho in rho_lst:
                arguments = f'{task} {dataset} {arch} {depth} {batch_size} {train_seed} {optimizer} {initial_lr} {stage} {epochs} {earlystop_interval} {width_scale} {gpu_id} {rho}\n'
                f.write(arguments)
                train_count += 1

    with open(f'{script_path}/three_stage.sh', 'a+') as f:
        f.write(f'# First Stage: training...\n')
        f.write(f'source {script_path}/train.sh {train_count-1}\n')


############################################################################
########################      Prune   ######################################
############################################################################

if args.all or args.prune:
    last_stage='train'
    stage='prune'
    with open(f'{script_path}/config/prune.txt', 'w') as f:
        prune_count = 1
        for earlystop_epoch in earlystop_epoch_lst:
            for train_seed in train_seed_lst:
                for prune_ratio in prune_ratio_lst:
                    for rho in rho_lst:
                        arguments = f'{task} {dataset} {arch} {depth} {batch_size} {train_seed} {optimizer} {initial_lr} {last_stage} {stage} {width_scale} {gpu_id} {earlystop_epoch} {prune_distri} {prune_ratio} {rho}\n'
                        f.write(arguments)
                        prune_count += 1

    with open(f'{script_path}/three_stage.sh', 'a+') as f:
        f.write(f'# Second Stage: pruning...\n')
        f.write(f'source {script_path}/prune.sh {prune_count-1}\n')



############################################################################
########################      Retrain   ####################################
############################################################################

if args.all or args.retrain:
    last_stage='prune'
    stage='retrain'

    with open(f'{script_path}/config/retrain.txt', 'w') as f:
        retrain_count = 1
        for earlystop_epoch in earlystop_epoch_lst:
            for train_seed in train_seed_lst:
                for retrain_seed in retrain_seed_lst:
                    for prune_ratio in prune_ratio_lst:
                        for rho in rho_lst:
                            arguments = f'{task} {dataset} {arch} {depth} {batch_size} {train_seed} {retrain_seed} {optimizer} {retrain_initial_lr} {last_stage} {stage} {width_scale} {gpu_id} {earlystop_epoch} {prune_distri} {prune_ratio} {retrain_epochs} {rho} {retrain_optimizer}\n'
                            f.write(arguments)
                            retrain_count += 1

    with open(f'{script_path}/three_stage.sh', 'a+') as f:
        f.write(f'# Third Stage: retraining...\n')
        f.write(f'source {script_path}/retrain.sh {retrain_count-1}\n')

############################################################################
########################      LMC   ####################################
############################################################################

if args.all or args.lmc:
    last_stage = 'retrain'

    with open(f'{script_path}/config/lmc.txt', 'w') as f:
        lmc_count=1
        for earlystop_epoch in earlystop_epoch_lst:
            for train_seed in train_seed_lst:
                for retrain_seed_pair in retrain_seed_pair_lst:
                    for prune_ratio in prune_ratio_lst:
                        for rho in rho_lst:
                            arguments = f'{task} {dataset} {arch} {depth} {batch_size} {train_seed} {optimizer} {retrain_initial_lr} {last_stage} {width_scale} {gpu_id} {earlystop_epoch} {prune_distri} {prune_ratio} {retrain_epochs} {retrain_seed_pair[0]} {retrain_seed_pair[1]} {num_points} {rho}\n'

                            f.write(arguments)
                            lmc_count += 1

    with open(f'{script_path}/three_stage.sh', 'a+') as f:
        f.write(f'# LMC measurement...\n')
        f.write(f'source {script_path}/lmc.sh {lmc_count-1}\n')


    
############################################################################
########################      CKA   ####################################
############################################################################

if args.all or args.cka:
    last_stage = 'retrain'

    with open(f'{script_path}/config/cka.txt', 'w') as f:
        cka_count=1
        for earlystop_epoch in earlystop_epoch_lst:
            for train_seed in train_seed_lst:
                for retrain_seed_pair in retrain_seed_pair_lst:
                    for prune_ratio in prune_ratio_lst:
                        for rho in rho_lst:
                            arguments = f'{task} {dataset} {arch} {depth} {batch_size} {train_seed} {optimizer} {retrain_initial_lr} {last_stage} {width_scale} {gpu_id} {earlystop_epoch} {prune_distri} {prune_ratio} {retrain_epochs} {retrain_seed_pair[0]} {retrain_seed_pair[1]} {rho}\n'
                            f.write(arguments)
                            cka_count += 1

    with open(f'{script_path}/three_stage.sh', 'a+') as f:
        f.write(f'# CKA measurement...\n')
        f.write(f'source {script_path}/cka.sh {cka_count-1}\n')
