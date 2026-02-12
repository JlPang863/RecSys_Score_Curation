#!/usr/bin/env bash
set -e
export PYTHONPATH=$(pwd)

############################
# Config
############################
ROOT_DATA_PATH="raw_data"
ROOT_RESULT_PATH="runs"

DATASET="tulu_300k_with_embeddings"
DATASET_PATH="${ROOT_DATA_PATH}/${DATASET}.parquet"

EMBEDDING_KEY="embeddings"
LABEL_KEY="gpt_scores"          # gpt_scores | llama_scores | mistral_scores
PREDICTION_KEY="proxy_knn_label"
SEED=3
KNN_K=50
TAU=0.1
NUM_CLASSES=6
KNN_BATCH_SIZE=1024
DATA_POOL_SIZE=3000000

# Train ratios to evaluate
TRAIN_RATIOS=(0.01 0.02 0.05 0.10 0.15 0.20 0.30 0.50 0.60 0.70 0.80 0.90)

# Base output directory for this experiment
EXPERIMENT_DIR="${ROOT_RESULT_PATH}/proxy_knn_${LABEL_KEY}"
mkdir -p "${EXPERIMENT_DIR}"

echo "======================================"
echo "*** Proxy kNN Label Generation ***"
echo "*** Multiple Train Ratios ***"
echo "======================================"
echo "dataset_path=${DATASET_PATH}"
echo "data_pool_size=${DATA_POOL_SIZE}"
echo "label_key=${LABEL_KEY}"
echo "train_ratios=${TRAIN_RATIOS[*]}"
echo "======================================"

# Loop over train ratios
for TRAIN_RATIO in "${TRAIN_RATIOS[@]}"; do
    # Format train ratio for directory name (e.g., 0.01 -> tr001, 0.10 -> tr010)
    TR_NAME=$(printf "tr%03d" $(echo "${TRAIN_RATIO} * 100" | bc | cut -d. -f1))
    OUTPUT_DIR="${EXPERIMENT_DIR}/${TR_NAME}"

    echo ""
    echo "--------------------------------------"
    echo "Running with TRAIN_RATIO=${TRAIN_RATIO} (${TR_NAME})"
    echo "Output: ${OUTPUT_DIR}"
    echo "--------------------------------------"

    python partial_labeling/proxy_label_generation.py \
        --dataset_path "${DATASET_PATH}" \
        --embedding_key "${EMBEDDING_KEY}" \
        --label_key "${LABEL_KEY}" \
        --prediction_key "${PREDICTION_KEY}" \
        --train_ratio "${TRAIN_RATIO}" \
        --seed "${SEED}" \
        --knn_k "${KNN_K}" \
        --tau "${TAU}" \
        --num_classes "${NUM_CLASSES}" \
        --knn_batch_size "${KNN_BATCH_SIZE}" \
        --data_pool_size "${DATA_POOL_SIZE}" \
        --output_dir "${OUTPUT_DIR}" || {
        echo "FAILED at TRAIN_RATIO=${TRAIN_RATIO}"
        exit 1
    }

    echo "Finished TRAIN_RATIO=${TRAIN_RATIO}"
done

echo ""
echo "======================================"
echo "All runs completed. Generating plots..."
echo "======================================"

# Generate comparison plots
python partial_labeling/plot_metrics.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --output_dir "${EXPERIMENT_DIR}/plots"

echo ""
echo "Done! Results saved to:"
echo "  - Metrics: ${EXPERIMENT_DIR}/*/metrics.json"
echo "  - Plots: ${EXPERIMENT_DIR}/plots/"
