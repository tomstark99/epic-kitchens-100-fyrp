echo "computing verb class priors..."
python src/scripts/compute_verb_class_priors.py datasets/epic-100/labels/p01.pkl datasets/epic-100/labels/verb_class_priors.csv
echo "computing noun class priors..."
python src/scripts/compute_noun_class_priors.py datasets/epic-100/labels/p01.pkl datasets/epic-100/labels/noun_class_priors.csv
echo "done..."
