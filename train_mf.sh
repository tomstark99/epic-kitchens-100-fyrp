#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time 5-00:00
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

python src/scripts/train_mf.py datasets/epic-100/mf_features/audio_slowfast_visual_mformer_features_train.hdf5 datasets/epic-100/features/67217_train_features.pkl datasets/epic-100/mf_features/audio_slowfast_visual_mformer_features_val.hdf5 datasets/epic-100/features/9668_val_features.pkl datasets/epic-100/models/ --epoch 200 --type "verb" --learning-rate 1e-6 --batch-size 512 --hidden-layer-size 256 --dropout-probability 0.5
python src/scripts/train_mf.py datasets/epic-100/mf_features/audio_slowfast_visual_mformer_features_train.hdf5 datasets/epic-100/features/67217_train_features.pkl datasets/epic-100/mf_features/audio_slowfast_visual_mformer_features_val.hdf5 datasets/epic-100/features/9668_val_features.pkl datasets/epic-100/models/ --epoch 200 --type "noun" --learning-rate 1e-6 --batch-size 512 --hidden-layer-size 256 --dropout-probability 0.5

