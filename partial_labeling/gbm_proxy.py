"""
Gradient Boosting proxy label generation using LightGBM.
Supports embedding-only, text-features-only, and fusion modes.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score

from proxy_label_generation import (
    confusion_matrix,
    load_dataframe,
    macro_f1,
    resolve_num_classes,
    split_indices,
    stack_embedding_column,
    to_serializable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GBM proxy label generation")
    parser.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    parser.add_argument("--embedding_key", default="embeddings")
    parser.add_argument("--label_key", default="gpt_scores")
    parser.add_argument("--prediction_key", default="proxy_gbm_label")
    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--data_pool_size", type=int, default=None)

    # Feature mode
    parser.add_argument("--use_text_features", action="store_true",
                        help="Use embedding + text features")
    parser.add_argument("--text_features_only", action="store_true",
                        help="Use ONLY text features")
    parser.add_argument("--messages_key", default="messages")

    # LightGBM params
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--class_weight", choices=["balanced", "none"], default="none")

    parser.add_argument("--output_dir", default="runs/gbm_proxy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.dataset_path)
    if args.data_pool_size is not None:
        df = df.iloc[:args.data_pool_size].copy()
    df = df.reset_index(drop=True)

    labels = np.array([int(v) for v in df[args.label_key].tolist()], dtype=np.int64)
    train_idx, test_idx = split_indices(len(df), args.train_ratio, args.seed)
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # Build features
    if args.text_features_only:
        from feature_extraction import extract_features_batch, normalize_features
        text_feats, feat_names = extract_features_batch(df, args.messages_key)
        train_input = text_feats[train_idx]
        test_input = text_feats[test_idx]
        feature_names = feat_names
        print(f"[info] text features only: {len(feat_names)} features")
    elif args.use_text_features:
        from feature_extraction import extract_features_batch
        embeddings = stack_embedding_column(df, args.embedding_key, show_progress=True)
        text_feats, feat_names = extract_features_batch(df, args.messages_key)
        train_input = np.concatenate([embeddings[train_idx], text_feats[train_idx]], axis=1)
        test_input = np.concatenate([embeddings[test_idx], text_feats[test_idx]], axis=1)
        emb_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
        feature_names = emb_names + feat_names
        print(f"[info] embedding dim={embeddings.shape[1]} + text features={len(feat_names)} "
              f"= {train_input.shape[1]}")
    else:
        embeddings = stack_embedding_column(df, args.embedding_key, show_progress=True)
        train_input = embeddings[train_idx]
        test_input = embeddings[test_idx]
        feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]

    num_classes = resolve_num_classes(args.num_classes, train_labels, test_labels)
    print(f"[info] input_dim={train_input.shape[1]}, num_classes={num_classes}")
    print(f"[info] train={len(train_idx)}, test={len(test_idx)}")

    # Class weights
    cw = None
    if args.class_weight == "balanced":
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        total = counts.sum()
        cw = {c: total / (num_classes * counts[c]) for c in range(num_classes)}
        print(f"[info] class_weights={cw}")

    # Train LightGBM
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": args.seed,
        "verbose": -1,
        "n_jobs": 8,
    }
    if cw is not None:
        params["class_weight"] = cw

    print("[info] training LightGBM...")
    model = lgb.LGBMClassifier(**params)

    # Use early stopping with validation set
    eval_set = [(test_input, test_labels)]
    model.fit(
        train_input, train_labels,
        eval_set=eval_set,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    best_iter = model.best_iteration_
    print(f"[info] best iteration: {best_iter}")

    # Predict
    proxy_pred = model.predict(test_input).astype(np.int64)
    proxy_probs = model.predict_proba(test_input)
    proxy_conf = proxy_probs.max(axis=1).astype(np.float32)

    # Metrics
    conf_mat = confusion_matrix(test_labels, proxy_pred, num_classes)
    accuracy = float((proxy_pred == test_labels).mean())
    mf1 = macro_f1(conf_mat)
    mae = float(np.abs(proxy_pred - test_labels).mean())

    metrics = {
        "dataset_path": args.dataset_path,
        "label_key": args.label_key,
        "num_rows": len(df),
        "data_pool_size": args.data_pool_size,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "num_classes": num_classes,
        "model": "lightgbm",
        "input_dim": int(train_input.shape[1]),
        "use_text_features": args.use_text_features,
        "text_features_only": args.text_features_only,
        "n_estimators": args.n_estimators,
        "best_iteration": best_iter,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "class_weight": args.class_weight,
        "accuracy": accuracy,
        "macro_f1": mf1,
        "mean_abs_error": mae,
        "confusion_matrix": conf_mat.tolist(),
    }

    for c in range(num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            metrics[f"class_{c}_acc"] = float((proxy_pred[mask] == c).mean())
            metrics[f"class_{c}_count"] = int(mask.sum())

    # Feature importance (top 30)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[::-1][:30]
        metrics["top_features"] = [
            {"name": feature_names[i], "importance": int(importance[i])}
            for i in top_idx
        ]
        print("\n[feature importance - top 20]")
        for rank, i in enumerate(top_idx[:20]):
            print(f"  {rank+1:2d}. {feature_names[i]:30s}  imp={importance[i]}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[done] metrics: {metrics_path}")
    print(f"[summary] accuracy={accuracy:.4f}, macro_f1={mf1:.4f}, mae={mae:.4f}")

    print("\n[per-class accuracy]")
    for c in range(num_classes):
        acc_key = f"class_{c}_acc"
        cnt_key = f"class_{c}_count"
        if acc_key in metrics:
            print(f"  class {c}: acc={metrics[acc_key]:.4f}  (n={metrics[cnt_key]})")


if __name__ == "__main__":
    main()
