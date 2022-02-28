#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres gpu:2
#SBATCH --time 1-00:00
#SBATCH --mem=64GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

. /mnt/storage/software/languages/anaconda/Anaconda3-2019.07/etc/profile.d/conda.sh
conda deactivate
which python
conda activate epic-100
which python

python src/scripts/train_mtrn.py datasets/epic-100/video_id_features/p01_features.pkl datasets/epic-100/models/ --epoch 200 --type "verb"
python src/scripts/train_mtrn.py datasets/epic-100/video_id_features/p01_features.pkl datasets/epic-100/models/ --epoch 200 --type "noun"
