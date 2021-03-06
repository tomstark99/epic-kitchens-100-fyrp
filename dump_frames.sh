#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time 7-00:00
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

python src/scripts/dump_frames_from_gulp_dir.py /mnt/storage/scratch/wp13824/vil/datasets/epic-100/gulp/rgb_train/ /mnt/storage/scratch/jb18789/datasets/epic-100/video_frames/
