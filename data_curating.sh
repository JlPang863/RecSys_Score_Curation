#!/usr/bin/env bash
export PYTHONPATH=$(pwd)

############################
# Config
############################
ROOT_DATA_PATH="raw_data"

DATASET="utilitarian"
DATASET_PATH="${ROOT_DATA_PATH}/${DATASET}.json"
OUTPUT_DIR="result/"
FEATURE_KEY="embed_text"
SCORE_KEY="bin_score"
CONFIG="template.py"

echo "======================================"
echo "*** Processing dataset: ${DATASET} ***"
echo "======================================"

############################
# Step 1: Raw Score Diagnosis
############################
echo "*** [1/2] Running score diagnosis... *** "
python score_curation/data_diagnose.py \
    --config "${CONFIG}" \
    --dataset_name "${DATASET}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --feature_key "${FEATURE_KEY}" \
    --score_key "${SCORE_KEY}"

############################
# Step 2: Score Curation
############################
echo "*** [2/2] Running score curation... ***"
python score_curation/data_curation.py \
    --dataset_name "${DATASET}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --score_key "${SCORE_KEY}"

echo "✅ Pipeline finished successfully."
