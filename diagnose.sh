
dataset='utilitarian'

output_dir="../score_curation_results/"


echo "*** processing dataset: ${dataset} ***"


python3 diagnose.py \
    --config template.py \
    --dataset_name $dataset \
    --output_dir $output_dir
    # --dataset_path \
    # --feature_keywords \
    # --score_keywords \

