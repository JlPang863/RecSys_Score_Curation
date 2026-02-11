# Partial Labeling + Propagation for `RecSys_Score_Curation`

> Goal: plug a "partial LLM scoring + graph propagation" stage into the current DS2-lite workflow **without changing core code**, so labeling cost is much lower.

---

## 1. New idea in one sentence

Only score a small subset with LLM (for example 10%), propagate labels to the rest using embedding neighborhoods, then run the existing pipeline as-is:

- `score_curation/data_diagnose.py`
- `score_curation/data_curation.py`
- `score_curation/pipeline_utils.py`

---

## 2. Why this is compatible with current code

In the current implementation, `run_diagnose` and `run_curation` require a **fully populated discrete score field** (like `bin_score`) for every sample.  
So the key is not changing DS2 logic, but preparing a full proxy score field first, e.g. `bin_score_proxy`:

1. LLM scores only a budgeted subset.
2. Propagation fills scores for all remaining samples.
3. Pipeline runs with `score_key="bin_score_proxy"`.

This reuses existing diagnosis, transition matrix estimation, curation, and diversity scoring end-to-end.

---

## 3. Existing capabilities in this repo you can reuse

- Preprocessing: `score_curation/data_preprocess.py`
  - Converts `label_score -> bin_score`
  - Merges text fields into `embed_text`
- Diagnosis: `score_curation/data_diagnose.py`
  - Embedding generation
  - HOC transition matrix estimation
  - SimiFeat noisy-label detection
- Curation: `score_curation/data_curation.py`
  - Confidence-threshold score correction
  - Diversity score
  - Outputs `*_curated.json`
- One-shot interface: `score_curation/pipeline_utils.py` (`ScoreCurationPipeline`)

---

## 4. Integration plan (no core code changes)

## Stage A: preprocess raw data (existing code)

Input example: `raw_data/timeline_label_raw.json`  
Output example: `raw_data/timeline_label_partial_base.json`

```bash
python - <<'PY'
from score_curation.data_preprocess import preprocess_dataset

preprocess_dataset(
    input_path="raw_data/timeline_label_raw.json",
    output_path="raw_data/timeline_label_partial_base.json",
    num_classes=6,
    score_key="label_score",
    feature_keys=["history", "label_title"],
    output_score_key="bin_score",
    output_feature_key="embed_text",
)
PY
```

---

## Stage B: create full proxy score field `bin_score_proxy` (entry point for new idea)

### Minimal runnable smoke test (today)

First copy `bin_score` to `bin_score_proxy`, only to validate the pipeline connection:

```bash
python - <<'PY'
import json

in_path = "raw_data/timeline_label_partial_base.json"
out_path = "raw_data/timeline_label_partial_proxy.json"

with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for x in data:
    x["bin_score_proxy"] = int(x["bin_score"])
    x["proxy_confidence"] = 1.0
    x["proxy_source"] = "smoke_copy"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"saved: {out_path}, n={len(data)}")
PY
```

### Real partial-labeling version (your new idea)

Replace `smoke_copy` with:

1. Budgeted sampling for subset `L` (for example 10%)
2. LLM scoring only on `L`
3. Embedding-neighborhood propagation to produce full `bin_score_proxy`

Recommended fields:

- `bin_score_proxy`: full proxy label used by DS2
- `proxy_confidence`: propagation confidence (e.g., max class probability)
- `proxy_source`: `llm_seed` or `propagated`

---

## Stage C: run existing DS2-lite pipeline directly

```bash
python - <<'PY'
from score_curation import ScoreCurationPipeline

pipeline = ScoreCurationPipeline(
    config_path="template.py",
    dataset_name="timeline_label_partial",
    dataset_path="raw_data/timeline_label_partial_proxy.json",
    feature_key="embed_text",
    score_key="bin_score_proxy",   # key point: use proxy score field
    output_dir="results",
    num_classes=6,
    confidence_prob=0.5,
)

outputs = pipeline.run()
print("done, dataset_size:", len(outputs["dataset"]))
PY
```

---

## Stage D: output artifacts

After running, under `results/timeline_label_partial/` you should get:

- `dataset_timeline_label_partial.pt`
- `embedded_timeline_label_partial_*.pt`
- `timeline_label_partial_report.pt`
- `timeline_label_partial_curated.json`

Final curated JSON includes:

- `diversity_score`
- `curated_score`
- `final_curated_score`

---

## 5. Recommended propagation baseline (easiest fit with current repo)

For each unlabeled sample `u`, find top-k neighbors in labeled set `L` and do weighted voting:

\[
p_u(c) = \frac{\sum_{i \in \text{NN}_k(u)\cap L} w_i \cdot \mathbf{1}[y_i=c]}
{\sum_{i \in \text{NN}_k(u)\cap L} w_i},\quad
w_i=\exp(\text{sim}(u,i)/\tau)
\]

- `bin_score_proxy = argmax_c p_u(c)`
- `proxy_confidence = max_c p_u(c)`
- Fallback to global class prior if no labeled neighbors are found

The output format remains fully compatible with `ScoreCurationPipeline`.

---

## 6. Evaluation suggestions for this idea

- Cost: number of LLM calls (`budget * N`)
- Label quality: agreement between `bin_score_proxy` and full labels `bin_score` (offline check)
- Downstream utility: compare selection quality driven by `final_curated_score` against full-scoring baseline

---

## 7. Current boundary (important)

This repo currently does **not** include a ready-made script for "budgeted sampling + LLM scoring + propagation".  
But once you generate a full `bin_score_proxy` field in data, the DS2-lite pipeline can be reused directly without changing `score_curation` core modules.

---

## 8. Optional next engineering step

Later you can add a standalone script (for example `scripts/build_proxy_scores.py`) to generate `bin_score_proxy` automatically, then feed it into the existing pipeline.  
For now, manual generation in notebook/script is enough.
