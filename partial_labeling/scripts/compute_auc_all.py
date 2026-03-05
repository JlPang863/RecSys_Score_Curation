"""
Compute macro AUC (one-vs-rest) for all saved experiment runs.

Loads model.pt + reconstructs data pipeline from metrics.json,
computes probabilities on the test set, then computes AUC.

Usage:
    python compute_auc_all.py
    python compute_auc_all.py --run_dirs runs/skywork_fusion_mlp_150k_tr030
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "partial_labeling"))

from partial_labeling.supervised_proxy import (
    LinearProbe,
    MLPClassifier,
    OrdinalModel,
    ResNetMLP,
)
from partial_labeling.proxy_label_generation import (
    load_dataframe,
    split_indices,
)

# ── Global caches ──────────────────────────────────────────
_df_cache = {}          # (dataset_path, pool_size) -> df
_emb_cache = {}         # (npy_path, pool_size) -> np.ndarray
_text_feat_cache = {}   # (dataset_path, pool_size) -> (feats, names)


def _get_df(dataset_path: str, pool_size: int | None):
    key = (dataset_path, pool_size)
    if key not in _df_cache:
        print(f"  [cache] loading dataframe {dataset_path} pool={pool_size} ...")
        df = load_dataframe(dataset_path)
        if pool_size is not None:
            df = df.iloc[:pool_size].copy()
        df = df.reset_index(drop=True)
        _df_cache[key] = df
    return _df_cache[key]


def _get_emb(npy_path: str, pool_size: int | None):
    key = (npy_path, pool_size)
    if key not in _emb_cache:
        print(f"  [cache] loading embeddings {npy_path} ...")
        emb = np.load(npy_path).astype(np.float32)
        if pool_size is not None:
            emb = emb[:pool_size]
        _emb_cache[key] = emb
    return _emb_cache[key]


def _get_text_feats(dataset_path: str, pool_size: int | None):
    key = (dataset_path, pool_size)
    if key not in _text_feat_cache:
        print(f"  [cache] extracting text features (pool={pool_size}) ...")
        from partial_labeling.feature_extraction import extract_features_batch
        df = _get_df(dataset_path, pool_size)
        feats, names = extract_features_batch(df, "messages", show_progress=True)
        _text_feat_cache[key] = (feats, names)
    return _text_feat_cache[key]


def get_probs(model, embeddings, device, is_ordinal=False, batch_size=2048):
    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for start in range(0, embeddings.size(0), batch_size):
            x = embeddings[start:start + batch_size].to(device)
            if is_ordinal:
                probs = model.predict_probs(x)
            else:
                probs = F.softmax(model(x), dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def compute_auc_for_run(run_dir: Path, device: torch.device) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    model_path = run_dir / "model.pt"

    if not metrics_path.exists():
        print(f"  SKIP {run_dir.name}: no metrics.json")
        return None
    if not model_path.exists():
        print(f"  SKIP {run_dir.name}: no model.pt")
        return None

    with open(metrics_path) as f:
        cfg = json.load(f)

    if "dataset_path" not in cfg:
        print(f"  SKIP {run_dir.name}: non-standard metrics format")
        return None

    # Already computed?
    if "auc_macro" in cfg and cfg["auc_macro"] is not None:
        print(f"  CACHED {run_dir.name}: auc_macro={cfg['auc_macro']:.4f}")
        return {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "accuracy": cfg.get("accuracy"),
            "macro_f1": cfg.get("macro_f1"),
            "auc_macro": cfg["auc_macro"],
            "auc_weighted": cfg.get("auc_weighted"),
            "per_class_auc": {str(c): cfg.get(f"class_{c}_auc") for c in range(cfg["num_classes"])},
            "test_rows": cfg.get("test_rows"),
            "train_ratio": cfg["train_ratio"],
            "data_pool_size": cfg.get("data_pool_size"),
        }

    dataset_path = cfg["dataset_path"]
    label_key = cfg["label_key"]
    train_ratio = cfg["train_ratio"]
    seed = cfg.get("seed", 3)
    num_classes = cfg["num_classes"]
    data_pool_size = cfg.get("data_pool_size")
    model_type = cfg["model"]
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg.get("num_layers")
    dropout = cfg.get("dropout", 0.1)

    # Embedding source
    if "balanced_18k" in dataset_path:
        embedding_npy = "raw_data/embedding_cache/Skywork_balanced_18k_embeddings.npy"
    elif "balanced_100k" in dataset_path:
        embedding_npy = "raw_data/embedding_cache/Skywork_balanced_100k_embeddings.npy"
    else:
        embedding_npy = "raw_data/embedding_cache/Skywork_Skywork-Reward-Llama-3.1-8B_300932_embeddings.npy"

    # Load cached data
    df = _get_df(dataset_path, data_pool_size)
    labels = np.array([int(v) for v in df[label_key].tolist()], dtype=np.int64)
    train_idx, test_idx = split_indices(len(df), train_ratio, seed)
    test_labels = labels[test_idx]

    emb = _get_emb(embedding_npy, data_pool_size)

    # Text features
    run_name = run_dir.name
    use_text_features = any(k in run_name for k in
                            ["fusion", "classweight", "focal", "oversample", "ordinal", "balanced"])

    if use_text_features:
        from partial_labeling.feature_extraction import normalize_features
        text_feats, _ = _get_text_feats(dataset_path, data_pool_size)
        train_tf, test_tf = normalize_features(text_feats[train_idx], text_feats[test_idx])
        test_input = np.concatenate([emb[test_idx], test_tf], axis=1)
    else:
        test_input = emb[test_idx]

    input_dim = test_input.shape[1]
    is_ordinal = model_type == "ordinal"

    # Build model
    if model_type == "linear":
        model = LinearProbe(input_dim, num_classes, dropout=dropout)
    elif model_type == "mlp":
        model = MLPClassifier(input_dim, hidden_dim, num_classes, dropout=dropout)
    elif model_type == "resnet":
        model = ResNetMLP(input_dim, hidden_dim, num_classes,
                          num_layers=num_layers or 4, dropout=dropout)
    elif model_type == "ordinal":
        model = OrdinalModel(input_dim, hidden_dim, num_classes,
                             num_layers=num_layers or 4, dropout=dropout)
    else:
        print(f"  SKIP {run_dir.name}: unknown model '{model_type}'")
        return None

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    test_emb_t = torch.from_numpy(test_input.astype(np.float32))
    probs = get_probs(model, test_emb_t, device, is_ordinal=is_ordinal)

    # AUC
    try:
        auc_macro = float(roc_auc_score(test_labels, probs, multi_class="ovr", average="macro"))
    except ValueError as e:
        print(f"  WARN {run_dir.name}: AUC failed: {e}")
        auc_macro = None

    try:
        auc_weighted = float(roc_auc_score(test_labels, probs, multi_class="ovr", average="weighted"))
    except ValueError:
        auc_weighted = None

    per_class_auc = {}
    for c in range(num_classes):
        binary = (test_labels == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            per_class_auc[c] = None
        else:
            try:
                per_class_auc[c] = float(roc_auc_score(binary, probs[:, c]))
            except ValueError:
                per_class_auc[c] = None

    acc_str = f"{cfg.get('accuracy', 0):.4f}"
    f1_str = f"{cfg.get('macro_f1', 0):.4f}"
    auc_m_str = f"{auc_macro:.4f}" if auc_macro is not None else "N/A"
    auc_w_str = f"{auc_weighted:.4f}" if auc_weighted is not None else "N/A"
    print(f"  {run_dir.name}: acc={acc_str}, f1={f1_str}, AUC_macro={auc_m_str}, AUC_weighted={auc_w_str}")

    result = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "accuracy": cfg.get("accuracy"),
        "macro_f1": cfg.get("macro_f1"),
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "per_class_auc": {str(k): v for k, v in per_class_auc.items()},
        "test_rows": len(test_labels),
        "train_ratio": train_ratio,
        "data_pool_size": data_pool_size,
    }

    # Update metrics.json
    cfg["auc_macro"] = auc_macro
    cfg["auc_weighted"] = auc_weighted
    for c_str, auc_val in per_class_auc.items():
        cfg[f"class_{c_str}_auc"] = auc_val
    with open(metrics_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="*", default=None)
    parser.add_argument("--output", default="results/auc_all_results.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        run_dirs = sorted(Path("runs").glob("skywork_*"))

    print(f"Found {len(run_dirs)} directories\n")

    all_results = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        result = compute_auc_for_run(run_dir, device)
        if result:
            all_results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"{'Run':<50} {'Acc':>8} {'F1':>8} {'AUC_m':>8} {'AUC_w':>8}")
    print(f"{'-'*100}")
    for r in sorted(all_results, key=lambda x: x.get("auc_macro") or 0, reverse=True):
        auc_m = f"{r['auc_macro']:.4f}" if r['auc_macro'] else "N/A"
        auc_w = f"{r['auc_weighted']:.4f}" if r['auc_weighted'] else "N/A"
        print(f"{r['run_name']:<50} {r.get('accuracy', 0):>8.4f} {r.get('macro_f1', 0):>8.4f} {auc_m:>8} {auc_w:>8}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
