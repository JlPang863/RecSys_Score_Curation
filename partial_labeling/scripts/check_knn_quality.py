"""
Check kNN neighbor quality for the proxy kNN algorithm.

Goal: Determine whether L2-normalized embeddings capture label-related
information by examining how often nearest neighbors share labels
(or have similar labels) with each test query point.

Dataset : raw_data/tulu_300k_with_embeddings.parquet
Embedding: "embeddings" column (1024-dim, L2-normalised)
Label    : "gpt_scores" column  (integer 0-5)
"""

import time
import numpy as np
import pandas as pd
import torch

# ── 0. Config ────────────────────────────────────────────────────────
PARQUET_PATH = "raw_data/tulu_300k_with_embeddings.parquet"
SUBSET_N = 50_000          # rows to load (speed)
TRAIN_FRAC = 0.10          # 10 % train, 90 % test
SEED = 3
N_QUERY = 1_000            # test points to sample
K = 50                     # neighbours to retrieve
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ── 1. Load data ─────────────────────────────────────────────────────
t0 = time.time()
df = pd.read_parquet(PARQUET_PATH).head(SUBSET_N)
print(f"Loaded {len(df)} rows in {time.time()-t0:.1f}s")

embeddings = np.stack(df["embeddings"].values).astype(np.float32)   # (N, 1024)
labels = df["gpt_scores"].values.astype(np.int64)                   # (N,)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Label distribution in subset:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  label {u}: {c:>6d}  ({100*c/len(labels):5.1f}%)")

# ── 2. Train / Test split ───────────────────────────────────────────
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(df))
n_train = int(len(df) * TRAIN_FRAC)
train_idx, test_idx = idx[:n_train], idx[n_train:]

train_emb = torch.tensor(embeddings[train_idx], device=DEVICE)   # (n_train, 1024)
train_lab = labels[train_idx]                                      # numpy

test_emb_all = embeddings[test_idx]
test_lab_all = labels[test_idx]

# Sample N_QUERY test points
query_sel = rng.choice(len(test_idx), size=min(N_QUERY, len(test_idx)), replace=False)
query_emb = torch.tensor(test_emb_all[query_sel], device=DEVICE)  # (N_QUERY, 1024)
query_lab = test_lab_all[query_sel]                                 # numpy

print(f"\nTrain size : {len(train_idx)}")
print(f"Test  size : {len(test_idx)}")
print(f"Query size : {len(query_sel)}")
print(f"K          : {K}")

# ── 3. Find K nearest neighbours (cosine sim = dot product) ─────────
t0 = time.time()

# Process in batches to avoid OOM
BATCH = 200
all_topk_indices = []

for start in range(0, len(query_emb), BATCH):
    q_batch = query_emb[start:start+BATCH]              # (B, 1024)
    sims = q_batch @ train_emb.T                         # (B, n_train)
    _, topk_idx = sims.topk(K, dim=1)                    # (B, K)
    all_topk_indices.append(topk_idx.cpu().numpy())

topk_indices = np.concatenate(all_topk_indices, axis=0)  # (N_QUERY, K)
print(f"kNN search done in {time.time()-t0:.1f}s")

# Retrieve neighbour labels
neighbour_labels = train_lab[topk_indices]                # (N_QUERY, K)

# ── 4. Compute statistics ───────────────────────────────────────────

# 4a. Exact-match fraction: what fraction of K neighbours share the query label?
exact_match = (neighbour_labels == query_lab[:, None]).astype(np.float32)
per_query_exact = exact_match.mean(axis=1)                # (N_QUERY,)

# 4b. Absolute label distance
abs_dist = np.abs(neighbour_labels.astype(np.float32) - query_lab[:, None].astype(np.float32))
per_query_mean_dist = abs_dist.mean(axis=1)

# 4c. Neighbour label mode vs query label
from scipy import stats as sp_stats
neighbour_mode = sp_stats.mode(neighbour_labels, axis=1, keepdims=False).mode
mode_matches_query = (neighbour_mode == query_lab).mean()

# 4d. Mean neighbour label vs query label (regression-style)
neighbour_mean_label = neighbour_labels.astype(np.float32).mean(axis=1)
correlation = np.corrcoef(query_lab.astype(np.float32), neighbour_mean_label)[0, 1]

# ── 5. Print results ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  kNN NEIGHBOUR QUALITY REPORT")
print("=" * 65)

print(f"\n--- Exact Label Match (fraction of {K} neighbours with same label) ---")
print(f"  Mean   : {per_query_exact.mean():.4f}")
print(f"  Median : {np.median(per_query_exact):.4f}")
print(f"  Std    : {per_query_exact.std():.4f}")
print(f"  Min    : {per_query_exact.min():.4f}")
print(f"  Max    : {per_query_exact.max():.4f}")

print(f"\n--- Absolute Label Distance (|query_label - neighbour_label|) ---")
print(f"  Mean   : {per_query_mean_dist.mean():.4f}")
print(f"  Median : {np.median(per_query_mean_dist):.4f}")
print(f"  Std    : {per_query_mean_dist.std():.4f}")

print(f"\n--- Mode of Neighbour Labels Matches Query Label ---")
print(f"  Accuracy: {mode_matches_query:.4f}  ({100*mode_matches_query:.1f}%)")

print(f"\n--- Correlation: mean(neighbour labels) vs query label ---")
print(f"  Pearson r: {correlation:.4f}")

# ── 6. Breakdown by query label ─────────────────────────────────────
print(f"\n--- Per-label breakdown ---")
print(f"{'Label':>5s} | {'Count':>5s} | {'ExactMatch':>10s} | {'MeanAbsDist':>11s} | {'MeanNbrLabel':>12s}")
print("-" * 60)
for lab in sorted(np.unique(query_lab)):
    mask = query_lab == lab
    n = mask.sum()
    em = per_query_exact[mask].mean()
    md = per_query_mean_dist[mask].mean()
    mnl = neighbour_mean_label[mask].mean()
    print(f"{lab:>5d} | {n:>5d} | {em:>10.4f} | {md:>11.4f} | {mnl:>12.4f}")

# ── 7. Random baseline comparison ───────────────────────────────────
print(f"\n--- Random Baseline (expected if embeddings carried no info) ---")
# If we picked K neighbours uniformly at random from train
train_label_dist = np.bincount(train_lab, minlength=6) / len(train_lab)
random_exact = 0.0
random_abs_dist = 0.0
for ql in range(6):
    q_frac = (query_lab == ql).sum() / len(query_lab)
    random_exact += q_frac * train_label_dist[ql]
    for nl in range(6):
        random_abs_dist += q_frac * train_label_dist[nl] * abs(ql - nl)

print(f"  Expected exact-match fraction : {random_exact:.4f}")
print(f"  Expected mean abs distance    : {random_abs_dist:.4f}")
print(f"  Actual  exact-match fraction  : {per_query_exact.mean():.4f}")
print(f"  Actual  mean abs distance     : {per_query_mean_dist.mean():.4f}")
print(f"  Lift in exact-match (actual / random): {per_query_exact.mean()/random_exact:.2f}x")
print(f"  Reduction in distance (random / actual): {random_abs_dist/per_query_mean_dist.mean():.2f}x")

print("\n" + "=" * 65)
print("Done.")
