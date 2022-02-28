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

for n in $(seq 1 8); do
    python src/scripts/compute_esvs.py datasets/epic-100/video_id_features/p01_features.pkl datasets/epic-100/models/ datasets/epic-100/labels/verb_class_priors.csv datasets/epic-100/labels/noun_class_priors.csv datasets/epic-100/esvs/mtrn-esv-n_frames=$n.pkl --sample-n-frames $n
done
