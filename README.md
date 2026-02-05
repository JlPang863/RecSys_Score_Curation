# Recommendation System Data Score Curation for LLMs


This project applies the DS2 (ICLR 2025) framework for data score curation to recommendation systems, evaluating whether the approach generalizes effectively beyond its original setting.

**Google Search Team — LLM Data Augmentation Project**

**Manager**: Shuang Yang

**Mentor**: Jingjing Liu

---

## 🚀 Project Progress

- [x] **2026.02.04** — Applied Score Curation on **Activity Timeline** dataset (with info: activity timeline, title)
- [x] **2026.01.31** — Applied Score Curation on **Utilitarian** dataset (with info: title, salient labels)
- [x] **2026.01.26** — Refactored and adapted the original [DS2](https://github.com/UCSC-REAL/DS2) codebase for recommendation system data  
- [x] Modularized the workflow into diagnosis and curation stages  
- [x] Added an end-to-end pipeline interface for programmatic usage  

---

## 🔧 Overview

The project consists of two main stages:

1. **Score Diagnosis**
   - Detect mislabeled or inconsistent scores
   - Identify rare or long-tail samples using embedding similarity
   - Produce a diagnostic report

2. **Score Curation**
   - Revise noisy scores using confidence-based correction
   - Incorporate diversity-aware signals
   - Write curated scores back into the dataset

The final dataset includes three additional fields:
- `curated_score`
- `diversity_score`
- `final_curated_score`

## 🧩 Option 1: Run the Full Pipeline via Script 

For most users, the easiest way to run the entire pipeline is via the provided shell script:

```bash
bash data_curating.sh
```

All corresponding curation report files can be found in the path `result`.


---

## 🧩 Option 2: Use the Python Pipeline Interface (Recommended)

For programmatic usage (e.g., research workflows, notebooks, or system integration), the pipeline can be invoked directly in Python:

```python

from pipeline_utils import ScoreCurationPipeline
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

This interface runs both diagnosis and curation sequentially and returns:

-  curated dataset

- full diagnosis report