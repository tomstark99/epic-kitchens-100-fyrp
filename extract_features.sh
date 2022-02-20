#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres gpu:2
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

# lost job 10032270
# last run job 10034983 (4 workers)
# last run job 10034986 (no workers)
# last run job 10034987 (1 worker)
# last run job 10034988 (2 workers)
# last run job 10034989 (3 workers)
python src/scripts/extract_features.py datasets/epic-100/gulp/train/ checkpoints/trn_rgb.ckpt datasets/epic-100/features/features.pkl || date
