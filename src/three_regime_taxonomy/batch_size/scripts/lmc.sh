#!/bin/bash


export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

end=$1 
root=$(pwd)

for TASK_ID in $(seq 1 "$end")
    do
        cfg=$(sed -n "$TASK_ID"p ${root}/scripts/config/lmc.txt)

        task=$(echo $cfg | cut -f 1 -d ' ')
        dataset=$(echo $cfg | cut -f 2 -d ' ')
        arch=$(echo $cfg | cut -f 3 -d ' ')
        depth=$(echo $cfg | cut -f 4 -d ' ')
        batch_size=$(echo $cfg | cut -f 5 -d ' ')
        seed=$(echo $cfg | cut -f 6 -d ' ')
        optimizer=$(echo $cfg | cut -f 7 -d ' ')
        initial_lr=$(echo $cfg | cut -f 8 -d ' ')
        last_stage=$(echo $cfg | cut -f 9 -d ' ')
        width_scale=$(echo $cfg | cut -f 10 -d ' ')
        gpu_id=$(echo $cfg | cut -f 11 -d ' ')
        prune_distri=$(echo $cfg | cut -f 12 -d ' ')
        prune_ratio=$(echo $cfg | cut -f 13 -d ' ')
        epochs=$(echo $cfg | cut -f 14 -d ' ')
        retrain_seed_1=$(echo $cfg | cut -f 15 -d ' ')
        retrain_seed_2=$(echo $cfg | cut -f 16 -d ' ')
        num_points=$(echo $cfg | cut -f 17 -d ' ')
        rho=$(echo $cfg | cut -f 18 -d ' ')

        if [ "$optimizer" = "SAM" ]; then
            echo "Train Optimizer (first stage) is SAM"
            load_dir=${ckpt_path}/${task}/${dataset}/${last_stage}/${optimizer}_rho${rho}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}_batchsize${batch_size}
        else
            echo "Train Optimizer (first stage) is not SAM"
            load_dir=${ckpt_path}/${task}/${dataset}/${last_stage}/${optimizer}/lr_${initial_lr}/${arch}${depth}_width${width_scale}_seed${seed}_batchsize${batch_size}
        fi
        
        load_dir=${load_dir}/${prune_distri}/p_ratio_${prune_ratio}/retrain_epoch${epochs}

        load_dir_1=${load_dir}/retrain_seed${retrain_seed_1}/finetuned.pth.tar
        load_dir_2=${load_dir}/retrain_seed${retrain_seed_2}/finetuned.pth.tar
        data_dir=${data_path}/${task}/${dataset}

        echo $load_dir_1
        echo $load_dir_2
        CUDA_VISIBLE_DEVICES=${gpu_id} python ../loss_landscape_measure/lmc.py \
                    --dataset ${dataset} \
                    --arch ${arch} \
                    --depth ${depth} \
                    --manualSeed ${seed} \
                    --resume_ckpt_1 ${load_dir_1} \
                    --resume_ckpt_2 ${load_dir_2} \
                    --data-path ${data_dir} \
                    --save_dir ${load_dir}/lmc/seed${retrain_seed_1}_seed${retrain_seed_2} \
                    --fix_start \
                    --fix_end \
                    --num-points 11 

    done