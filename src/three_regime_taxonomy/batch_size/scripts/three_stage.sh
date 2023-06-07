export ckpt_path=/data/yefan0726/checkpoints_public
export data_path=/data/yefan0726/data_public

# First Stage: training...
source /home/eecs/yefan0726/ww_prune/ThreeRegimePruning/src/three_regime_taxonomy/batch_size/scripts/train.sh 1
# Second Stage: pruning...
source /home/eecs/yefan0726/ww_prune/ThreeRegimePruning/src/three_regime_taxonomy/batch_size/scripts/prune.sh 1
# Third Stage: retraining...
source /home/eecs/yefan0726/ww_prune/ThreeRegimePruning/src/three_regime_taxonomy/batch_size/scripts/retrain.sh 2
# LMC measurement...
source /home/eecs/yefan0726/ww_prune/ThreeRegimePruning/src/three_regime_taxonomy/batch_size/scripts/lmc.sh 3
# CKA measurement...
source /home/eecs/yefan0726/ww_prune/ThreeRegimePruning/src/three_regime_taxonomy/batch_size/scripts/cka.sh 3