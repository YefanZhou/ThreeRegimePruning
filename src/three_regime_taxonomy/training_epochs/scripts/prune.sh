#!/bin/bash


# export ckpt_path=/data/yefan0726/checkpoints_public
# export data_path=/data/yefan0726/data_public

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

end=$1 

root=$(pwd)


for TASK_ID in $(seq 1 "$end")
    do
        cfg=$(sed -n "$TASK_ID"p ${root}/scripts/config/prune.txt)

        task=$(echo $cfg | cut -f 1 -d ' ')
        dataset=$(echo $cfg | cut -f 2 -d ' ')
        arch=$(echo $cfg | cut -f 3 -d ' ')
        depth=$(echo $cfg | cut -f 4 -d ' ')
        batch_size=$(echo $cfg | cut -f 5 -d ' ')
        seed=$(echo $cfg | cut -f 6 -d ' ')
        optimizer=$(echo $cfg | cut -f 7 -d ' ')
        initial_lr=$(echo $cfg | cut -f 8 -d ' ')
        last_stage=$(echo $cfg | cut -f 9 -d ' ')
        stage=$(echo $cfg | cut -f 10 -d ' ')
        width_scale=$(echo $cfg | cut -f 11 -d ' ')
        gpu_id=$(echo $cfg | cut -f 12 -d ' ')
        earlystop_epoch=$(echo $cfg | cut -f 13 -d ' ')
        prune_distri=$(echo $cfg | cut -f 14 -d ' ')
        prune_ratio=$(echo $cfg | cut -f 15 -d ' ')
        rho=$(echo $cfg | cut -f 16 -d ' ')


        data_dir=${data_path}/${task}/${dataset}
        if [ "$optimizer" = "SAM" ]; then
            echo "Optimizer is SAM"
            load_dir=${ckpt_path}/${task}/${dataset}/${last_stage}/${optimizer}_rho${rho}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}/checkpoint_epoch_${earlystop_epoch}.pth.tar
            save_dir=${ckpt_path}/${task}/${dataset}/${stage}/${optimizer}_rho${rho}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}
        else
            echo "Optimizer is not SAM"
            load_dir=${ckpt_path}/${task}/${dataset}/${last_stage}/${optimizer}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}/checkpoint_epoch_${earlystop_epoch}.pth.tar
            save_dir=${ckpt_path}/${task}/${dataset}/${stage}/${optimizer}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}
        fi
        
        save_dir=${save_dir}/${prune_distri}/p_ratio_${prune_ratio}/epoch_${earlystop_epoch}


        CUDA_VISIBLE_DEVICES=${gpu_id} python prune.py \
                    --dataset ${dataset} \
                    --arch ${arch} \
                    --depth ${depth} \
                    --manualSeed ${seed} \
                    --resume ${load_dir} \
                    --data-path ${data_dir} \
                    --percent ${prune_ratio} \
                    --prune_layer_sparsity ${prune_distri} \
                    --save_dir ${save_dir}

    done