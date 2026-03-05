# Recommendation System Data Score Curation for LLMs

This project applies the DS2 (ICLR 2025) framework for data score curation to recommendation systems, and explores proxy label generation for scaling quality scoring to large unlabeled datasets.

**Google Search Team — LLM Data Augmentation Project**

**Manager**: Shuang Yang

**Mentor**: Jingjing Liu

---

## Project Progress

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

---

## Part 2: Partial Labeling — Proxy Label Generation

**Problem**: LLM-based quality scoring at RecSys scale (millions of samples) is too expensive. Given ~10% labeled data, can we generate reliable proxy labels for the remaining 90%?

**Dataset**: 300K instruction-tuning conversations (tulu_300k), 6-class quality scores (0-5), highly imbalanced (class 2/3 ~30% each, class 5 ~1.3%).

### Methods Explored

| Method | Best Accuracy | Best Macro F1 | Verdict |
|--------|:---:|:---:|---------|
| **Skywork Fusion MLP** (embedding + text features) | **0.6009** | **0.5271** | Best overall |
| Skywork kNN (reward embedding) | 0.5144 | 0.4119 | Decent baseline |
| BGE kNN (semantic embedding) | 0.4826 | 0.3876 | Weak — embedding lacks quality signal |
| Balanced dataset (18k) | 0.5720 | 0.5677 | Best F1 (proves class imbalance is main issue) |
| Self-Training (5 rounds) | — | +2.5% F1 | Limited (pseudo-label accuracy ~50%) |
| Ensemble (kNN + MLP) | +0.1% Acc | -1% F1 | Ineffective |
| Class Weight / Oversample | -5% Acc | +1% F1 | Limited |
| Cleanlab noise cleaning | -0.5% Acc | -2% F1 | Negative |
| Label Propagation / C&S / APPNP | -5~6% Acc | -7~12% F1 | Negative |
| Ordinal Regression | -5% Acc | -12% F1 | Negative |
| Multi-annotator label fusion | -3~8% Acc | -7~19% F1 | Negative |

### Key Findings

1. **Data pool size is the biggest lever**: 300k 10% train (30k samples) outperforms 30k 90% train (27k samples) — larger pool provides more representative test sets and more minority-class samples
2. **Embedding quality matters**: Skywork Reward embedding > BGE semantic embedding (+3% accuracy on average for kNN). Reward model hidden states carry quality signals
3. **Shallow text features are surprisingly strong**: 35 hand-crafted features (length, formatting, structure) alone beat 1024-dim BGE embedding. Quality correlates with surface features
4. **Early Fusion is the only effective enhancement**: embedding + text features complement each other. All "advanced" methods (ensemble, cleanlab, self-training, graph methods, etc.) fail to beat the simple Fusion MLP baseline
5. **Factor ranking**: Data pool size > Embedding quality > Training data amount > Model capacity = Feature count

### Quick Start — Fusion MLP (Best Method)

```bash
# Run Skywork Fusion MLP on 300k dataset (10% train)
bash partial_labeling/scripts/run_skywork_fusion_mlp_300k.sh
```

### Quick Start — kNN Proxy Labels

```bash
# Run kNN proxy label experiment with pre-computed embeddings
python partial_labeling/run_proxy_knn_experiment.py \
  --config template.py \
  --dataset_path raw_data/tulu_300k_with_embeddings.parquet \
  --output_dir runs/proxy_knn_skywork \
  --feature_key embed_text \
  --teacher_score_key gpt_scores \
  --num_classes 6 \
  --budget 0.10 \
  --seed 3 \
  --knn_k 50 \
  --tau 0.1 \
  --mode dev_teacher \
  --output_score_key bin_score_proxy
```

### Detailed Documentation

- [partial_labeling/experimental_results.md](partial_labeling/experimental_results.md) — Full experiment log with all tables, per-class accuracy, and analysis
- [partial_labeling/quickstart.md](partial_labeling/quickstart.md) — Task specification and developer guide
