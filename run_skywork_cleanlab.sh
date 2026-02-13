#!/usr/bin/env bash
export PYTHONPATH=$(pwd)

############################
# Config
############################
ROOT_RESULT_PATH="runs"

DATASET_PATH="raw_data/tulu_300k_with_embeddings.parquet"
EMBEDDING_NPY="raw_data/embedding_cache/Skywork_Skywork-Reward-Llama-3.1-8B_300932_embeddings.npy"
LABEL_KEY="gpt_scores"
SEED=3
NUM_CLASSES=6
DATA_POOL_SIZE=30000

# MLP config
HIDDEN_DIM=256
EPOCHS=100
BATCH_SIZE=512
LR=1e-3

# Cleanlab config
CV_FOLDS=5

# Train ratios to evaluate
TRAIN_RATIOS=(0.10 0.20 0.30)

# Clean methods to try
CLEAN_METHODS=(remove relabel)

echo "============================================================"
echo "*** Skywork Cleanlab Label Noise Detection + Cleaning ***"
echo "============================================================"
echo "embedding_npy=${EMBEDDING_NPY}"
echo "hidden_dim=${HIDDEN_DIM}, epochs=${EPOCHS}"
echo "cv_folds=${CV_FOLDS}"
echo "data_pool_size=${DATA_POOL_SIZE}"
echo "train_ratios=${TRAIN_RATIOS[*]}"
echo "clean_methods=${CLEAN_METHODS[*]}"
echo "============================================================"

for TRAIN_RATIO in "${TRAIN_RATIOS[@]}"; do
    TR_NAME=$(python3 -c "print(f'tr{int(round(${TRAIN_RATIO}*100)):03d}')")

    for CLEAN_METHOD in "${CLEAN_METHODS[@]}"; do
        OUTPUT_DIR="${ROOT_RESULT_PATH}/skywork_cleanlab_${CLEAN_METHOD}_${TR_NAME}"

        echo ""
        echo "--------------------------------------"
        echo "Running: TRAIN_RATIO=${TRAIN_RATIO}, method=${CLEAN_METHOD}"
        echo "Output: ${OUTPUT_DIR}"
        echo "--------------------------------------"

        CUDA_VISIBLE_DEVICES=4 python partial_labeling/cleanlab_proxy.py \
            --dataset_path "${DATASET_PATH}" \
            --embedding_npy "${EMBEDDING_NPY}" \
            --use_text_features \
            --label_key "${LABEL_KEY}" \
            --train_ratio "${TRAIN_RATIO}" \
            --seed "${SEED}" \
            --num_classes "${NUM_CLASSES}" \
            --data_pool_size "${DATA_POOL_SIZE}" \
            --hidden_dim "${HIDDEN_DIM}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --cv_folds "${CV_FOLDS}" \
            --clean_method "${CLEAN_METHOD}" \
            --output_dir "${OUTPUT_DIR}" || {
            echo "FAILED at TRAIN_RATIO=${TRAIN_RATIO}, method=${CLEAN_METHOD}"
            exit 1
        }

        echo "Finished TRAIN_RATIO=${TRAIN_RATIO}, method=${CLEAN_METHOD}"
    done
done

echo ""
echo "============================================================"
echo "All cleanlab runs completed!"
echo "============================================================"
