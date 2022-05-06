#!/usr/bin/env bash
#SBATCH --partition cpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --time 1-00:00
#SBATCH --mem=32GB

python src/scripts/extract_verb_noun_links.py datasets/epic-100/gulp/train datasets/epic-100/labels/verb_noun.pkl datasets/epic-100/labels/EPIC_100_verb_classes.csv datasets/epic-100/labels/EPIC_100_noun_classes.csv

python src/scripts/extract_verb_noun_links.py datasets/epic-100/gulp/train datasets/epic-100/labels/verb_noun_classes.pkl datasets/epic-100/labels/EPIC_100_verb_classes.csv datasets/epic-100/labels/EPIC_100_noun_classes.csv --classes True

# python src/scripts/extract_verb_noun_links.py datasets/epic-100/gulp/train datasets/epic-100/labels/verb_noun_classes_narration.pkl datasets/epic-100/labels/EPIC_100_verb_classes.csv datasets/epic-100/labels/EPIC_100_noun_classes.csv --classes True --narration-id True
