#!/usr/bin/env bash
#SBATCH --partition cpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --time 0-08:00
#SBATCH --mem=32GB

for n in $(seq 7 8); 
do 
	python src/scripts/compute_dummy_esvs.py datasets/epic-100/gulp/train/ datasets/epic-100/esvs/mtrn-esv-n_frames=$n.pkl --sample-n-frames $n
done
