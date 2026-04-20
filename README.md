# Recommendation System Data Score Curation for LLMs

This project applies the DS2 (ICLR 2025) framework for data score curation to recommendation systems, and explores proxy label generation for scaling quality scoring to large unlabeled datasets.

**Google Search Team — LLM Data Augmentation Project**

**Manager**: Shuang Yang

**Mentor**: Jingjing Liu

---

## Project Progress

- [x] **~2026.03** — Pipeline refactor: pre-computed embeddings support (`embedding_path`), removed multimodal code, single-file embedding storage, lazy imports for CPU-only curation
- [x] **~2026.03** — Scaled Skywork Fusion MLP to 300k full dataset — Acc **0.6009**, F1 **0.5271**, AUC **0.8952**
- [x] **~2026.02** — Comprehensive method comparison: Ensemble, Cleanlab, Self-Training, Label Propagation, C&S/APPNP, Ordinal Regression, etc.
- [x] **~2026.02** — Supervised classifiers (MLP/ResNet) with feature fusion (Skywork embedding + 63 text features)
- [x] **~2026.02** — kNN proxy label generation with BGE and Skywork Reward embeddings
- [x] **2026.02.04** — Applied Score Curation on **Activity Timeline** dataset
- [x] **2026.01.31** — Applied Score Curation on **Utilitarian** dataset
- [x] **2026.01.26** — Refactored [DS2](https://github.com/UCSC-REAL/DS2) codebase, modularized pipeline, added Python API

---

## Project Structure

```
RecSys_Score_Curation/
├── score_curation/           # Core DS2-lite pipeline (diagnosis + curation)
├── docta/                    # DS2 core library (embedding, kNN, HOC, detection)
├── partial_labeling/         # Proxy label generation system
│   ├── scripts/              #   All experiment shell scripts
│   └── *.py                  #   kNN, MLP, ensemble, cleanlab, graph methods, etc.
├── raw_data/                 # Input datasets & embedding caches
├── results/                  # Experiment outputs
├── run_pipeline.py           # Main ScoreCuration pipeline runner
├── data_curating.sh          # Main data curating shell script
└── template.py               # Config template
```

---

## Part 1: Score Curation (DS2-lite)

The DS2-lite pipeline consists of two stages:

1. **Score Diagnosis** — Detect mislabeled/inconsistent scores using SimiFeat + HOC noise estimation
2. **Score Curation** — Revise noisy scores via confidence-based correction + diversity-aware signals

Output fields: `curated_score`, `diversity_score`, `final_curated_score`

### Option A: Shell Script

```bash
bash data_curating.sh
```

### Option B: Python API (Recommended)

```python
from score_curation import ScoreCurationPipeline
import os

pipeline = ScoreCurationPipeline(
    config_path="template.py",
    dataset_name="utilitarian",
    dataset_path=os.path.join("raw_data", "utilitarian.json"),
    feature_key="embed_text",
    score_key="bin_score",
    output_dir="results/",
)

outputs = pipeline.run()
curated_dataset = outputs["dataset"]
report = outputs["report"]
```

### Using Pre-computed Embeddings (No GPU Required)

The pipeline supports loading pre-computed embeddings via `embedding_path`, skipping GPU-based encoding entirely. The `.pt` file should contain a `CustomizedDataset` with `feature` (embeddings) and `label` (scores) attributes.

The embedding file can come from any source — for example, running the normal pipeline (which saves `embedded_{dataset_name}.pt` automatically), or your own embedding extraction script.

```python
pipeline = ScoreCurationPipeline(
    config_path="template.py",
    dataset_name="utilitarian",
    dataset_path="raw_data/utilitarian.json",
    output_dir="results/",
    embedding_path="path/to/embedded_utilitarian.pt",  # skip GPU encoding
)
outputs = pipeline.run()
```

---

## Part 2: Partial Labeling — Proxy Label Generation

**Problem**: LLM-based quality scoring at RecSys scale (millions of samples) is too expensive. Given ~10% labeled data, can we generate reliable proxy labels for the remaining 90%?

**Dataset**: 300K instruction-tuning conversations (tulu_300k), 6-class quality scores (0-5), highly imbalanced (class 2/3 ~30% each, class 5 ~1.3%).

### Methods Explored

| Method | Accuracy | Macro F1 | AUC | Verdict |
|--------|:---:|:---:|:---:|---------|
| **Skywork Fusion MLP** (emb + text feat) | **0.6009** | **0.5271** | **0.8952** | Best overall |
| Balanced dataset (18k) | 0.5720 | 0.5677 | 0.8850 | Best F1 (class imbalance is main issue) |
| Skywork kNN | 0.5144 | 0.4119 | — | Decent baseline |
| BGE kNN | 0.4826 | 0.3876 | — | Weak — embedding lacks quality signal |
| Self-Training / Ensemble / Class Weight | — | — | — | Limited improvement |
| Cleanlab / C&S / APPNP / Label Propagation | — | — | — | Negative — all graph/noise methods fail |

### Key Findings

1. **Data pool size > Embedding quality > Training data amount > Model capacity**: 300k 10% train outperforms 30k 90% train
2. **Skywork Reward embedding > BGE semantic embedding**: +3% accuracy for kNN. Reward model hidden states carry quality signals
3. **Early Fusion (embedding + text features) is the only effective enhancement**: all "advanced" methods fail to beat the simple Fusion MLP baseline
4. **Class imbalance is the main bottleneck for Macro F1**: balanced 18k achieves F1=0.57 vs imbalanced 30k F1=0.35

### Quick Start

```bash
# Fusion MLP on 300k (best method)
bash partial_labeling/scripts/run_skywork_fusion_mlp_300k.sh

# kNN proxy labels
python partial_labeling/run_proxy_knn_experiment.py \
  --config template.py \
  --dataset_path raw_data/tulu_300k_with_embeddings.parquet \
  --output_dir runs/proxy_knn_skywork \
  --teacher_score_key gpt_scores \
  --num_classes 6 --budget 0.10 --knn_k 50 --tau 0.1 \
  --mode dev_teacher
```

See [partial_labeling/experimental_results.md](partial_labeling/experimental_results.md) for full experiment details.


[Google Extension Proposal](https://docs.google.com/document/d/1O85_mGAal_rNCqa7xvykEGI50E0bWnnJ3HXPPmkky5o/edit?tab=t.0)
