file_name=$1
if [ -z $file_name ]
then
	file_name=p01_features.pkl
fi
echo "extracting features..."

if [[ $(which python) =~ ^.*(epic-100).*$ ]];
then
        echo "envorinment active..."
else
        conda activate epic-100
fi
python src/scripts/extract_features.py datasets/epic-100/gulp/train checkpoints/trn_rgb.ckpt datasets/epic-100/features/$file_name
