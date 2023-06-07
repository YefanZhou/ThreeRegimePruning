#!/bin/bash

# export ckpt_path=/data/yefan0726/checkpoints_public
# export data_path=/data/yefan0726/data_public

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1


end=$1

root=$(pwd)

for TASK_ID in $(seq 1 "$end")
    do  
        cfg=$(sed -n "$TASK_ID"p ${root}/scripts/config/train.txt)

        task=$(echo $cfg | cut -f 1 -d ' ')
        dataset=$(echo $cfg | cut -f 2 -d ' ')
        arch=$(echo $cfg | cut -f 3 -d ' ')
        depth=$(echo $cfg | cut -f 4 -d ' ')
        batch_size=$(echo $cfg | cut -f 5 -d ' ')
        seed=$(echo $cfg | cut -f 6 -d ' ')
        optimizer=$(echo $cfg | cut -f 7 -d ' ')
        initial_lr=$(echo $cfg | cut -f 8 -d ' ')
        stage=$(echo $cfg | cut -f 9 -d ' ')
        epochs=$(echo $cfg | cut -f 10 -d ' ')
        width_scale=$(echo $cfg | cut -f 11 -d ' ')
        gpu_id=$(echo $cfg | cut -f 12 -d ' ')
        num_iterations=$(echo $cfg | cut -f 13 -d ' ')
        rho=$(echo $cfg | cut -f 14 -d ' ')

        if [ "$optimizer" = "SAM" ]; then
            echo "Optimizer is SAM"
            save_dir=${ckpt_path}/${task}/${dataset}/${stage}/${optimizer}_rho${rho}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}_batchsize${batch_size}
        else
            echo "Optimizer is not SAM"
            save_dir=${ckpt_path}/${task}/${dataset}/${stage}/${optimizer}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}_batchsize${batch_size}
        fi
        
        data_dir=${data_path}/${task}/${dataset}

        CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
                    --dataset ${dataset} \
                    --arch ${arch} \
                    --depth ${depth} \
                    --manualSeed ${seed} \
                    --save_dir ${save_dir} \
                    --data-path ${data_dir} \
                    --epochs ${epochs} \
                    --width-scale ${width_scale} \
                    --lr ${initial_lr} \
                    --optimizer ${optimizer} \
                    --iterations ${num_iterations} \
                    --train-batch ${batch_size} \
                    --rho ${rho}
    done