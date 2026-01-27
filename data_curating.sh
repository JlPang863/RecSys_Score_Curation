
dataset='utilitarian'

dataset_path="${dataset}.json" 

output_dir="score_curation_results/"
feature_keywords='embed_text'
score_keywords="bin_score"

echo "*** Processing dataset: ${dataset} ***"

#####################################
##### Raw Score Error Detection #####
#####################################
python3 diagnose.py \
    --config template.py \
    --dataset_name $dataset \
    --output_dir $output_dir \
    --feature_keywords $feature_keywords \
    --score_keywords $score_keywords \
    --dataset_path $dataset_path


#####################################
##########  Score Curation ##########
#####################################
python score_generation.py \
    --dataset_name $dataset \
    --score_keywords $score_keywords \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --dataset_path $dataset_path
