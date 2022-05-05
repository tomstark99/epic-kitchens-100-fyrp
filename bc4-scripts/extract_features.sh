#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --nodes 3
#SBATCH --gres gpu:2
#SBATCH --time 7-00:00
#SBATCH --mem=100GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

. /mnt/storage/software/languages/anaconda/Anaconda3-2019.07/etc/profile.d/conda.sh
conda deactivate
which python
conda activate epic-100
which python

python src/scripts/extract_features.py datasets/epic-100/gulp/train/ checkpoints/trn_rgb.ckpt datasets/epic-100/features/o_features.pkl || date
