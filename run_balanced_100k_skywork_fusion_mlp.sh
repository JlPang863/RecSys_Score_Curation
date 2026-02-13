#!/usr/bin/env bash
export PYTHONPATH=$(pwd)

############################
# Config
############################
ROOT_RESULT_PATH="runs"

DATASET_PATH="raw_data/tulu_balanced_100k.parquet"
EMBEDDING_NPY="raw_data/embedding_cache/Skywork_balanced_100k_embeddings.npy"
LABEL_KEY="gpt_scores"
PREDICTION_KEY="proxy_supervised_label"
SEED=3
NUM_CLASSES=6
DATA_POOL_SIZE=100000

# Model config
MODEL="mlp"
HIDDEN_DIM=256
EPOCHS=100
BATCH_SIZE=512
LR=1e-3

# Train ratios to evaluate
TRAIN_RATIOS=(0.10 0.20 0.30 0.50 0.70 0.90)

echo "============================================================"
echo "*** Balanced 100k — Skywork Fusion MLP ***"
echo "============================================================"
echo "dataset=${DATASET_PATH}"
echo "embedding_npy=${EMBEDDING_NPY}"
echo "model=${MODEL}, hidden_dim=${HIDDEN_DIM}, epochs=${EPOCHS}"
echo "data_pool_size=${DATA_POOL_SIZE} (balanced: 16667 per class)"
echo "class 5 oversampled: 3829 -> 16667 (4.35x)"
echo "train_ratios=${TRAIN_RATIOS[*]}"
echo "============================================================"

for TRAIN_RATIO in "${TRAIN_RATIOS[@]}"; do
    TR_NAME=$(python3 -c "print(f'tr{int(round(${TRAIN_RATIO}*100)):03d}')")
    OUTPUT_DIR="${ROOT_RESULT_PATH}/balanced_100k_skywork_fusion_mlp_${TR_NAME}"

    echo ""
    echo "--------------------------------------"
    echo "Running with TRAIN_RATIO=${TRAIN_RATIO} (${TR_NAME})"
    echo "Output: ${OUTPUT_DIR}"
    echo "--------------------------------------"

    CUDA_VISIBLE_DEVICES=4 python partial_labeling/supervised_proxy.py \
        --dataset_path "${DATASET_PATH}" \
        --embedding_npy "${EMBEDDING_NPY}" \
        --use_text_features \
        --label_key "${LABEL_KEY}" \
        --prediction_key "${PREDICTION_KEY}" \
        --train_ratio "${TRAIN_RATIO}" \
        --seed "${SEED}" \
        --num_classes "${NUM_CLASSES}" \
        --data_pool_size "${DATA_POOL_SIZE}" \
        --model "${MODEL}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --output_dir "${OUTPUT_DIR}" || {
        echo "FAILED at TRAIN_RATIO=${TRAIN_RATIO}"
        exit 1
    }

    echo "Finished TRAIN_RATIO=${TRAIN_RATIO}"
done

echo ""
echo "============================================================"
echo "All runs completed!"
echo "============================================================"
