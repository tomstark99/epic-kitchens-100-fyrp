#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --nodes 2
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

python src/scripts/train_mtrn.py datasets/epic-100/features/67217_train_features.pkl datasets/epic-100/models/ --val-features-pkl datasets/epic-100/features/9668_val_features.pkl --epoch 1000 --type "verb" --learning-rate 3e-5 --batch-size 64
python src/scripts/train_mtrn.py datasets/epic-100/features/67217_train_features.pkl datasets/epic-100/models/ --val-features-pkl datasets/epic-100/features/9668_val_features.pkl --epoch 1000 --type "noun" --learning-rate 3e-5 --batch-size 64
