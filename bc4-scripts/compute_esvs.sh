#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time 2-00:00
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

python src/scripts/compute_esvs.py datasets/epic-100/features/9668_val_features.pkl datasets/epic-100/models/ datasets/epic-100/labels/verb_class_priors_val.csv datasets/epic-100/labels/noun_class_priors_val.csv datasets/epic-100/esvs/f_val_mtrn-esv-n_frames=8.pkl --sample-n-frames 8
