#!/usr/bin/env bash
# Continue from where run_all_150k_experiments.sh failed (Focal onwards)
set -e
export PYTHONPATH=$(pwd)

DATASET_PATH="raw_data/tulu_300k_with_embeddings.parquet"
EMBEDDING_NPY="raw_data/embedding_cache/Skywork_Skywork-Reward-Llama-3.1-8B_300932_embeddings.npy"
LABEL_KEY="gpt_scores"
SEED=3
NUM_CLASSES=6
DATA_POOL_SIZE=150000
ROOT="runs"
HIDDEN_DIM=256
EPOCHS=100
BATCH_SIZE=512
LR=1e-3
TRAIN_RATIOS=(0.10 0.20 0.30)

########################################################################
# Focal loss + class-weight
########################################################################
echo "*** Focal Loss + Class Weight (150k) ***"
for TR in "${TRAIN_RATIOS[@]}"; do
    TR_NAME=$(python3 -c "print(f'tr{int(round(${TR}*100)):03d}')")
    OUT="${ROOT}/skywork_focal_150k_${TR_NAME}"
    echo "--- Focal ${TR_NAME} ---"
    CUDA_VISIBLE_DEVICES=4 python partial_labeling/supervised_proxy.py \
        --dataset_path "${DATASET_PATH}" \
        --embedding_npy "${EMBEDDING_NPY}" \
        --use_text_features \
        --label_key "${LABEL_KEY}" \
        --train_ratio "${TR}" \
        --seed "${SEED}" \
        --num_classes "${NUM_CLASSES}" \
        --data_pool_size "${DATA_POOL_SIZE}" \
        --model mlp \
        --hidden_dim "${HIDDEN_DIM}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --class_weight \
        --focal_loss --focal_gamma 2.0 \
        --output_dir "${OUT}"
    echo "Done: ${OUT}"
done

########################################################################
# Oversample + class-weight
########################################################################
echo "*** Oversample + Class Weight (150k) ***"
for TR in "${TRAIN_RATIOS[@]}"; do
    TR_NAME=$(python3 -c "print(f'tr{int(round(${TR}*100)):03d}')")
    OUT="${ROOT}/skywork_oversample_150k_${TR_NAME}"
    echo "--- Oversample ${TR_NAME} ---"
    CUDA_VISIBLE_DEVICES=4 python partial_labeling/supervised_proxy.py \
        --dataset_path "${DATASET_PATH}" \
        --embedding_npy "${EMBEDDING_NPY}" \
        --use_text_features \
        --label_key "${LABEL_KEY}" \
        --train_ratio "${TR}" \
        --seed "${SEED}" \
        --num_classes "${NUM_CLASSES}" \
        --data_pool_size "${DATA_POOL_SIZE}" \
        --model mlp \
        --hidden_dim "${HIDDEN_DIM}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --class_weight \
        --oversample_train \
        --output_dir "${OUT}"
    echo "Done: ${OUT}"
done

########################################################################
# Ordinal Regression
########################################################################
echo "*** Ordinal Regression (150k) ***"
for TR in "${TRAIN_RATIOS[@]}"; do
    TR_NAME=$(python3 -c "print(f'tr{int(round(${TR}*100)):03d}')")
    OUT="${ROOT}/skywork_ordinal_150k_${TR_NAME}"
    echo "--- Ordinal ${TR_NAME} ---"
    CUDA_VISIBLE_DEVICES=4 python partial_labeling/supervised_proxy.py \
        --dataset_path "${DATASET_PATH}" \
        --embedding_npy "${EMBEDDING_NPY}" \
        --use_text_features \
        --label_key "${LABEL_KEY}" \
        --train_ratio "${TR}" \
        --seed "${SEED}" \
        --num_classes "${NUM_CLASSES}" \
        --data_pool_size "${DATA_POOL_SIZE}" \
        --model ordinal \
        --hidden_dim "${HIDDEN_DIM}" \
        --num_layers 4 \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr 5e-4 \
        --output_dir "${OUT}"
    echo "Done: ${OUT}"
done

########################################################################
# Cleanlab (150k, remove + relabel)
########################################################################
echo "*** Cleanlab (150k) ***"
for TR in "${TRAIN_RATIOS[@]}"; do
    TR_NAME=$(python3 -c "print(f'tr{int(round(${TR}*100)):03d}')")
    for METHOD in remove relabel; do
        OUT="${ROOT}/skywork_cleanlab_${METHOD}_150k_${TR_NAME}"
        echo "--- Cleanlab ${METHOD} ${TR_NAME} ---"
        CUDA_VISIBLE_DEVICES=4 python partial_labeling/cleanlab_proxy.py \
            --dataset_path "${DATASET_PATH}" \
            --embedding_npy "${EMBEDDING_NPY}" \
            --use_text_features \
            --label_key "${LABEL_KEY}" \
            --train_ratio "${TR}" \
            --seed "${SEED}" \
            --num_classes "${NUM_CLASSES}" \
            --data_pool_size "${DATA_POOL_SIZE}" \
            --hidden_dim "${HIDDEN_DIM}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --clean_method "${METHOD}" \
            --output_dir "${OUT}"
        echo "Done: ${OUT}"
    done
done

echo "============================================================"
echo "All remaining 150k experiments completed!"
echo "============================================================"
