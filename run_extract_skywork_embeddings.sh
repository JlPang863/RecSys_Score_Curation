#!/usr/bin/env bash
export PYTHONPATH=$(pwd)

############################
# Config
############################
DATASET_PATH="raw_data/tulu_300k_with_embeddings.parquet"
MESSAGES_KEY="messages"

# Skywork Reward Model
MODEL_NAME="Skywork/Skywork-Reward-Llama-3.1-8B"

# Extraction settings
BATCH_SIZE=16
MAX_LENGTH=2048
DTYPE="bf16"
DATA_POOL_SIZE=300000

# Parallel config
NUM_GPUS=8
GPU_IDS=(0 1 2 3 4 5 6 7)

echo "======================================"
echo "*** Skywork Reward Embedding Extraction ***"
echo "*** ${NUM_GPUS}-GPU Parallel Mode ***"
echo "======================================"
echo "model=${MODEL_NAME}"
echo "dataset=${DATASET_PATH}"
echo "batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}, dtype=${DTYPE}"
echo "data_pool_size=${DATA_POOL_SIZE}, shards=${NUM_GPUS}"
echo "======================================"

# Launch one process per GPU
PIDS=()
FAILED=0

for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
    GPU=${GPU_IDS[$SHARD_ID]}
    echo "[shard ${SHARD_ID}] launching on GPU ${GPU} ..."

    CUDA_VISIBLE_DEVICES=${GPU} python partial_labeling/extract_reward_embeddings.py \
        --model_name "${MODEL_NAME}" \
        --dataset_path "${DATASET_PATH}" \
        --messages_key "${MESSAGES_KEY}" \
        --batch_size "${BATCH_SIZE}" \
        --max_length "${MAX_LENGTH}" \
        --dtype "${DTYPE}" \
        --pooling "last_token" \
        --data_pool_size "${DATA_POOL_SIZE}" \
        --shard_id "${SHARD_ID}" \
        --num_shards "${NUM_GPUS}" \
        > "logs/skywork_shard_${SHARD_ID}.log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${NUM_GPUS} shards launched. PIDs: ${PIDS[*]}"
echo "Waiting for all shards to finish..."

# Wait and check each process
for i in $(seq 0 $((NUM_GPUS - 1))); do
    wait ${PIDS[$i]}
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "[shard ${i}] FAILED (exit code ${STATUS}). See logs/skywork_shard_${i}.log"
        FAILED=1
    else
        echo "[shard ${i}] done."
    fi
done

if [ ${FAILED} -ne 0 ]; then
    echo "ERROR: one or more shards failed. Check logs/ for details."
    exit 1
fi

echo ""
echo "All shards completed. Merging embeddings..."

# Merge all shard .npy files into one
python3 -c "
import numpy as np
from pathlib import Path

model_short = '${MODEL_NAME}'.replace('/', '_')
pool_tag = '_${DATA_POOL_SIZE}'
cache_dir = Path('raw_data/embedding_cache')

shards = []
for i in range(${NUM_GPUS}):
    path = cache_dir / f'{model_short}{pool_tag}_shard{i}_embeddings.npy'
    print(f'[merge] loading {path}')
    shards.append(np.load(path))

merged = np.concatenate(shards, axis=0)
output_path = cache_dir / f'{model_short}{pool_tag}_embeddings.npy'
np.save(output_path, merged)
print(f'[merge] saved {merged.shape} ({merged.nbytes / 1024**2:.1f} MB) to {output_path}')

# Clean up shard files
for i in range(${NUM_GPUS}):
    path = cache_dir / f'{model_short}{pool_tag}_shard{i}_embeddings.npy'
    path.unlink()
    print(f'[merge] removed {path}')
"

echo ""
echo "Done! Final embeddings saved to raw_data/embedding_cache/"
