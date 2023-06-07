#!/bin/bash


# export ckpt_path=/data/yefan0726/checkpoints_public
# export data_path=/data/yefan0726/data_public

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1


end=$1 


root=$(pwd)

for TASK_ID in $(seq 1 "$end")
    do
        cfg=$(sed -n "$TASK_ID"p ${root}/scripts/config/retrain.txt)

        task=$(echo $cfg | cut -f 1 -d ' ')
        dataset=$(echo $cfg | cut -f 2 -d ' ')
        arch=$(echo $cfg | cut -f 3 -d ' ')
        depth=$(echo $cfg | cut -f 4 -d ' ')
        batch_size=$(echo $cfg | cut -f 5 -d ' ')
        seed=$(echo $cfg | cut -f 6 -d ' ')
        retrain_seed=$(echo $cfg | cut -f 7 -d ' ')
        optimizer=$(echo $cfg | cut -f 8 -d ' ')
        initial_lr=$(echo $cfg | cut -f 9 -d ' ')
        last_stage=$(echo $cfg | cut -f 10 -d ' ')
        stage=$(echo $cfg | cut -f 11 -d ' ')
        width_scale=$(echo $cfg | cut -f 12 -d ' ')
        gpu_id=$(echo $cfg | cut -f 13 -d ' ')
        earlystop_epoch=$(echo $cfg | cut -f 14 -d ' ')
        prune_distri=$(echo $cfg | cut -f 15 -d ' ')
        prune_ratio=$(echo $cfg | cut -f 16 -d ' ')
        epochs=$(echo $cfg | cut -f 17 -d ' ')
        rho=$(echo $cfg | cut -f 18 -d ' ')
        retrain_optimizer=$(echo $cfg | cut -f 19 -d ' ')


        data_dir=${data_path}/${task}/${dataset}
        
        if [ "$optimizer" = "SAM" ]; then
            echo "Train (first stage) Optimizer is SAM"
            load_dir=${ckpt_path}/${task}/${dataset}/${last_stage}/${optimizer}_rho${rho}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}
            save_dir=${ckpt_path}/${task}/${dataset}/${stage}/${optimizer}_rho${rho}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}
        else
            echo "Train (first stage) Optimizer is not SAM"
            load_dir=${ckpt_path}/${task}/${dataset}/${last_stage}/${optimizer}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}
            save_dir=${ckpt_path}/${task}/${dataset}/${stage}/${optimizer}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}
        fi
        
        load_dir=${load_dir}/${prune_distri}/p_ratio_${prune_ratio}/epoch_${earlystop_epoch}/pruned.pth.tar
        save_dir=${save_dir}/${prune_distri}/p_ratio_${prune_ratio}/epoch_${earlystop_epoch}/retrain_epoch${epochs}/retrain_seed${retrain_seed}



        CUDA_VISIBLE_DEVICES=${gpu_id} python retrain.py \
                    --dataset ${dataset} \
                    --arch ${arch} \
                    --depth ${depth} \
                    --manualSeed ${retrain_seed} \
                    --resume ${load_dir} \
                    --data-path ${data_dir} \
                    --epochs ${epochs} \
                    --schedule 80 120 \
                    --train-batch ${batch_size} \
                    --lr ${initial_lr} \
                    --optimizer ${retrain_optimizer} \
                    --save_dir ${save_dir}

    done