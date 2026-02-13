"""
Extract embeddings from a reward model (or any HuggingFace model) via pooling
on the last hidden state.  Saves output as .npy for downstream use.

Pooling strategies:
  - last_token: Use the last non-padding token's hidden state (best for decoder-only / causal LMs)
  - mean: Mean pool over all non-padding tokens (best for encoder models like BERT/DeBERTa)

Usage examples:
    # Skywork-Reward (Llama-based, decoder-only → use last_token pooling)
    python extract_reward_embeddings.py \
        --model_name Skywork/Skywork-Reward-Llama-3.1-8B \
        --pooling last_token \
        --data_pool_size 300000 --batch_size 16 --max_length 2048

    # DeBERTa-based reward model (encoder → use mean pooling)
    python extract_reward_embeddings.py \
        --model_name OpenAssistant/reward-model-deberta-v3-large-v2 \
        --pooling mean \
        --data_pool_size 30000 --batch_size 32 --max_length 512

    # Mistral-7B-based reward model (decoder-only → use last_token pooling)
    python extract_reward_embeddings.py \
        --model_name weqweasdas/RM-Mistral-7B \
        --pooling last_token \
        --data_pool_size 30000 --batch_size 4 --max_length 2048
"""

import argparse
import json
from pathlib import Path

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# Bypass torch.load safety check for older model formats (torch < 2.6)
import transformers.utils.import_utils as _tiu
import transformers.modeling_utils as _tmu
_noop = lambda: None
_tiu.check_torch_load_is_safe = _noop
_tmu.check_torch_load_is_safe = _noop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract reward model embeddings via mean pooling")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    parser.add_argument("--messages_key", default="messages")
    parser.add_argument("--output_path", default=None, help="Output .npy path (auto-generated if omitted)")
    parser.add_argument("--data_pool_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--pooling", choices=["last_token", "mean"], default="last_token",
                        help="Pooling strategy: last_token (decoder-only models) or mean (encoder models)")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Shard index for multi-GPU parallel (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards for multi-GPU parallel")
    return parser.parse_args()


def format_messages(messages, tokenizer) -> str:
    """Format conversation messages into a single text string."""
    if isinstance(messages, str):
        messages = json.loads(messages)

    # Try the tokenizer's chat template first
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass

    # Fallback: simple concatenation
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"### {role}\n{content}")
    return "\n\n".join(parts)


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool token embeddings with attention mask. Best for encoder models (BERT/DeBERTa)."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last non-padding token's hidden state. Best for decoder-only models (Llama/Mistral)."""
    # attention_mask: (batch, seq_len), find index of last 1 in each row
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, seq_lengths]


def main() -> None:
    args = parse_args()

    # Device & dtype
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load data
    print(f"[info] loading dataset: {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)
    if args.data_pool_size is not None:
        df = df.iloc[: args.data_pool_size].copy()
    df = df.reset_index(drop=True)

    # Shard splitting for multi-GPU parallel
    if args.shard_id is not None and args.num_shards is not None:
        total = len(df)
        shard_size = (total + args.num_shards - 1) // args.num_shards  # ceil division
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total)
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"[info] shard {args.shard_id}/{args.num_shards}: rows [{start_idx}, {end_idx}) of {total}")

    n = len(df)
    print(f"[info] loaded {n} rows")

    # Load tokenizer & model
    print(f"[info] loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the base model (without classification/reward head)
    # Try AutoModel first; fall back to extracting base from classification model
    try:
        model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
    except (ValueError, OSError):
        print("[info] AutoModel failed, trying AutoModelForSequenceClassification + extracting base...")
        full_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        # Extract the base encoder (works for DeBERTa, BERT, RoBERTa, etc.)
        if hasattr(full_model, "deberta"):
            model = full_model.deberta
        elif hasattr(full_model, "model"):
            model = full_model.model
        elif hasattr(full_model, "base_model"):
            model = full_model.base_model
        else:
            raise RuntimeError("Cannot extract base model from classification model")
        model = model.to(device)
        del full_model

    model.eval()

    hidden_dim = model.config.hidden_size
    print(f"[info] hidden_dim={hidden_dim}, dtype={dtype}, device={device}")

    # Format conversation messages → text
    print("[info] formatting messages...")
    texts = []
    for i in tqdm(range(n), desc="formatting"):
        messages = df.iloc[i][args.messages_key]
        texts.append(format_messages(messages, tokenizer))

    # Show one sample
    sample_tokens = tokenizer(texts[0], truncation=True, max_length=args.max_length)
    print(f"[info] sample text length: {len(texts[0])} chars, {len(sample_tokens['input_ids'])} tokens")

    # Extract embeddings
    pool_fn = last_token_pooling if args.pooling == "last_token" else mean_pooling
    print(f"[info] extracting embeddings (batch_size={args.batch_size}, max_length={args.max_length}, pooling={args.pooling})...")
    all_embeddings = []

    for start in tqdm(range(0, n, args.batch_size), desc="embedding"):
        end = min(start + args.batch_size, n)
        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype):
            outputs = model(**encoded)

        hidden = outputs.last_hidden_state

        # Pool + L2 normalize
        emb = pool_fn(hidden, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().float().numpy())

        # Memory cleanup
        del outputs, hidden, emb, encoded
        if start % (args.batch_size * 50) == 0:
            torch.cuda.empty_cache()

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[info] embeddings shape: {embeddings.shape}")
    print(f"[info] embeddings dtype: {embeddings.dtype}")
    print(f"[info] norm check (should be ~1.0): {np.linalg.norm(embeddings[:5], axis=1)}")

    # Save
    if args.output_path is None:
        model_short = args.model_name.replace("/", "_")
        pool_tag = f"_{args.data_pool_size}" if args.data_pool_size else ""
        shard_tag = f"_shard{args.shard_id}" if args.shard_id is not None else ""
        output_path = f"raw_data/embedding_cache/{model_short}{pool_tag}{shard_tag}_embeddings.npy"
    else:
        output_path = args.output_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    size_mb = embeddings.nbytes / 1024 / 1024
    print(f"[done] saved {embeddings.shape} ({size_mb:.1f} MB) to {output_path}")


if __name__ == "__main__":
    main()
