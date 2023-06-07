# [A Three-regime Model of Network Pruning [ICML 2023]](https://arxiv.org/abs/2305.18383)

**[Yefan Zhou](https://yefanzhou.github.io/), [Yaoqing Yang](https://sites.google.com/site/yangyaoqingcmu/), [Arin Chang](https://arinchang.github.io/), [Michael Mahoney](https://www.stat.berkeley.edu/~mmahoney/)**

## Introduction

We build a three-regime model by taxonomizing the global structure of the pruned NN loss landscape. Our model reveals that the dichotomous effect of high temperature is associated with transitions between distinct types of global structures in the post-pruned model. Our new insights lead to new practical approaches of hyperparameter tuning and model selection to improve pruning.  
Please see the full paper on [ArXiv](https://arxiv.org/abs/2305.18383).

<img src="https://github.com/YefanZhou/ThreeRegimePruning/blob/main/visualization/figure1.png" alt="Image description" width="1000" height="290">

**(Figure 1 Overview of three-regime model.)**  The three regimes of pruning obtained by varying temperature-like parameters (in the dense pre-pruned model) and load-like parameters (in the sparse post-pruned model).



<img src="https://github.com/YefanZhou/ThreeRegimePruning/blob/main/visualization/figure2.png" alt="Image description" width="1000" height="190">

**(Figure 2 Empirical Results.)**  Partitioning the 2D model density (load) – training epoch (temperature) diagram into three regimes. Models are trained with PreResNet-20 on CIFAR-10.



## Installation

```bash
conda create -n three_regime python=3.8
conda activate three_regime
pip install -r requirements.txt
cd src
pip install -e .
```


## Experiments

Generate the three-regime phase plots by varying **temperature** and **load**

```bash
cd src/three_regime_taxonomy/{training_epochs, batch_size}
python scripts/write_experiments.py \
            --ckpt-path {your_checkpoints_path} \
            --data-path {your_dataset_path} \
            --all \
            --earlystop-epoch-lst 10 20 ... 160 \         # Varying temperature via training epochs 
            # OR
            --batch-size-lst 16 32 ... 512 \              # Varying temperature via batch size
            # OR
            --optimizer 'SAM' --rho-lst 0.1 0.2 ... 0.8 \ # Varying temperature via SAM rho
            --prune-ratio-lst 0.95 0.94 ... 0.2 \         # Varying load via model density (pruning ratio)  
bash scripts/three_regime.sh
```

**Notes**: We use [Slurm](https://slurm.schedmd.com/documentation.html) to parallelize the experiments, but didn't include the scripts for simplicity of code. Please adapt the code based on the parallelization mechanism that works for you.



## Visualization

Download the provided checkpoints and metrics
```bash
bash download_ckpt.sh
```

Run the notebook  to reproduce the Figure 2.

```bash
cd visualization
load_temperature_plots.ipynb
```



## Acknowledgment

Thanks to the great open-source codebases [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning),  [LAMP](https://github.com/jaeho-lee/layer-adaptive-sparsity),  [loss_landscape_taxonomy](https://github.com/nsfzyzz/loss_landscape_taxonomy).



## Citation 

We would appreciate it if you could cite the following paper if you found the repository useful for your work:

```bash
@inproceedings{zhou2023three,
  title={A Three-regime model of network pruning},
  author={Zhou, Yefan and Yang, Yaoqing and Chang, Arin and Mahoney W Michael},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

