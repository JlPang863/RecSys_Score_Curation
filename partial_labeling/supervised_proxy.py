"""
Supervised proxy label generation.
Train a Linear Probe, MLP, ResNet-MLP, or Ordinal model on labeled embeddings,
predict on unlabeled data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from proxy_label_generation import (
    confusion_matrix,
    load_dataframe,
    macro_f1,
    resolve_device,
    resolve_num_classes,
    split_indices,
    stack_embedding_column,
    to_serializable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised proxy label generation")
    parser.add_argument("--dataset_path", default="raw_data/tulu_300k_with_embeddings.parquet")
    parser.add_argument("--embedding_key", default="embeddings")
    parser.add_argument("--embedding_npy", default=None,
                        help="Load embeddings from .npy file instead of parquet column")
    parser.add_argument("--extra_embedding_npy", default=None,
                        help="Extra .npy embeddings to concatenate (e.g. combine Skywork + BGE)")
    parser.add_argument("--label_key", default="gpt_scores")
    parser.add_argument("--prediction_key", default="proxy_supervised_label")
    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--data_pool_size", type=int, default=None)

    # Model
    parser.add_argument("--model", choices=["linear", "mlp", "resnet", "ordinal"], default="mlp",
                        help="linear: logistic regression; mlp: 2-layer MLP; "
                             "resnet: deep residual MLP; ordinal: ordinal regression")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of residual blocks for resnet model")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--class_weight", action="store_true",
                        help="Use inverse class frequency as loss weight")
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma for focal loss")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0 = no smoothing)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Mixup alpha (0 = disabled)")

    # Feature fusion
    parser.add_argument("--use_text_features", action="store_true",
                        help="Extract and concatenate shallow text features with embeddings")
    parser.add_argument("--text_features_only", action="store_true",
                        help="Use ONLY text features (no embeddings)")
    parser.add_argument("--messages_key", default="messages",
                        help="Column name for messages in dataset")

    parser.add_argument("--oversample_train", action="store_true",
                        help="Oversample minority classes in train set to balance class counts")
    parser.add_argument("--label_fusion", default=None,
                        choices=["majority", "median", "consensus"],
                        help="Multi-annotator label fusion: "
                             "majority=mode of gpt/llama/mistral; "
                             "median=median of 3; "
                             "consensus=only keep samples where >=2 agree")
    # Self-training arguments
    parser.add_argument("--self_train_rounds", type=int, default=0,
                        help="Number of iterative self-training rounds (0=disabled)")
    parser.add_argument("--conf_threshold", type=float, default=0.9,
                        help="Confidence threshold for majority classes")
    parser.add_argument("--minority_conf", type=float, default=0.7,
                        help="Confidence threshold for minority classes")
    parser.add_argument("--minority_max_freq", type=float, default=0.10,
                        help="Classes with frequency < this are treated as minority")

    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="runs/supervised_proxy")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


# ============================================================
# Models
# ============================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetMLP(nn.Module):
    """Deep residual MLP with LayerNorm and GELU."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h)


class OrdinalModel(nn.Module):
    """Ordinal regression: predict K-1 cumulative thresholds.
    P(y > k) = sigmoid(f(x) - threshold_k), k=0..K-2.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # Single scalar output (latent score)
        self.score_head = nn.Linear(hidden_dim, 1)
        # K-1 learnable thresholds, initialized evenly spaced
        init_thresholds = torch.linspace(-2, 2, num_classes - 1)
        self.thresholds = nn.Parameter(init_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.blocks(h)
        h = self.norm(h)
        score = self.score_head(h)  # (B, 1)
        # Cumulative logits: P(y > k) = sigmoid(score - threshold_k)
        cum_logits = score - self.thresholds.unsqueeze(0)  # (B, K-1)
        return cum_logits

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        cum_logits = self.forward(x)  # (B, K-1)
        cum_probs = torch.sigmoid(cum_logits)  # P(y > k)
        # P(y = k) = P(y > k-1) - P(y > k), with P(y > -1) = 1
        probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_classes - 1):
            probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
        probs[:, -1] = cum_probs[:, -1]
        # Clamp for numerical stability
        probs = probs.clamp(min=1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


# ============================================================
# Loss functions
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        return focal.mean()


class OrdinalLoss(nn.Module):
    """Binary cross-entropy on cumulative logits."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, cum_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # cum_logits: (B, K-1), targets: (B,) integers
        # Build binary targets: target_k = 1 if y > k, else 0
        binary_targets = torch.zeros_like(cum_logits)
        for k in range(self.num_classes - 1):
            binary_targets[:, k] = (targets > k).float()
        return F.binary_cross_entropy_with_logits(cum_logits, binary_targets)


# ============================================================
# Mixup
# ============================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float,
               num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation. Returns mixed x and soft label targets."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    # Soft labels
    y_onehot = F.one_hot(y, num_classes).float()
    y_index_onehot = F.one_hot(y[index], num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_index_onehot
    return mixed_x, mixed_y


# ============================================================
# Training
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None,
    device: torch.device,
    num_classes: int,
    show_progress: bool = True,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    is_ordinal = args.model == "ordinal"

    # Build criterion
    if is_ordinal:
        criterion = OrdinalLoss(num_classes)
    elif args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights,
                              label_smoothing=args.label_smoothing)
    else:
        cw = class_weights.to(device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in tqdm(range(args.epochs), desc="training", disable=not show_progress):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            if args.mixup_alpha > 0 and not is_ordinal:
                mixed_x, mixed_y = mixup_data(x_batch, y_batch, args.mixup_alpha, num_classes)
                logits = model(mixed_x)
                loss = -torch.sum(mixed_y * F.log_softmax(logits, dim=1)) / logits.size(0)
            else:
                output = model(x_batch)
                loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

            # Training accuracy (always with original data)
            with torch.no_grad():
                if is_ordinal:
                    train_probs = model.predict_probs(x_batch)
                    pred_batch = train_probs.argmax(dim=1)
                else:
                    pred_batch = model(x_batch).argmax(dim=1)
                correct += (pred_batch == y_batch).sum().item()
                total += x_batch.size(0)
        scheduler.step()

        train_acc = correct / total
        train_loss = total_loss / total

        # Validate (batch-wise to avoid OOM on large test sets)
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = val_embeddings.size(0)
            for v_start in range(0, val_total, args.batch_size):
                v_end = min(v_start + args.batch_size, val_total)
                val_x = val_embeddings[v_start:v_end].to(device)
                val_y = val_labels[v_start:v_end].to(device)
                if is_ordinal:
                    val_probs = model.predict_probs(val_x)
                    val_pred = val_probs.argmax(dim=1)
                else:
                    val_pred = model(val_x).argmax(dim=1)
                val_correct += (val_pred == val_y).sum().item()
            val_acc = val_correct / val_total

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_acc": best_val_acc, "history": history}


def predict(
    model: nn.Module,
    embeddings: torch.Tensor,
    device: torch.device,
    is_ordinal: bool = False,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_pred = []
    all_conf = []
    with torch.no_grad():
        for start in range(0, embeddings.size(0), batch_size):
            x = embeddings[start:start + batch_size].to(device)
            if is_ordinal:
                probs = model.predict_probs(x)
            else:
                probs = F.softmax(model(x), dim=1)
            pred = probs.argmax(dim=1)
            conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
            all_pred.append(pred.cpu().numpy())
            all_conf.append(conf.cpu().numpy())
    return np.concatenate(all_pred).astype(np.int64), np.concatenate(all_conf).astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.disable_tqdm

    # Load data
    df = load_dataframe(args.dataset_path)
    if args.data_pool_size is not None:
        df = df.iloc[:args.data_pool_size].copy()
    df = df.reset_index(drop=True)

    # Label fusion: combine multiple annotator scores
    _consensus_keep_idx = None  # track filtered indices for .npy embeddings
    if args.label_fusion is not None:
        score_cols = ["gpt_scores", "llama_scores", "mistral_scores"]
        scores = np.stack([df[c].values.astype(int) for c in score_cols], axis=1)  # (N, 3)

        if args.label_fusion == "median":
            labels = np.median(scores, axis=1).astype(np.int64)
            print(f"[info] label_fusion=median: using median of {score_cols}")
        elif args.label_fusion == "majority":
            from collections import Counter
            labels = np.array([
                Counter(row.tolist()).most_common(1)[0][0] for row in scores
            ], dtype=np.int64)
            print(f"[info] label_fusion=majority: using majority vote of {score_cols}")
        elif args.label_fusion == "consensus":
            # Keep only samples where >=2 annotators agree
            agree_mask = (
                (scores[:, 0] == scores[:, 1]) |
                (scores[:, 0] == scores[:, 2]) |
                (scores[:, 1] == scores[:, 2])
            )
            from collections import Counter
            fused = np.array([
                Counter(row.tolist()).most_common(1)[0][0] for row in scores
            ], dtype=np.int64)
            _consensus_keep_idx = np.where(agree_mask)[0]
            df = df.iloc[_consensus_keep_idx].reset_index(drop=True)
            labels = fused[_consensus_keep_idx]
            print(f"[info] label_fusion=consensus: kept {len(_consensus_keep_idx)}/{len(agree_mask)} "
                  f"({len(_consensus_keep_idx)/len(agree_mask)*100:.1f}%) where >=2 agree")

        print(f"[info] fused label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    else:
        labels = np.array([int(v) for v in df[args.label_key].tolist()], dtype=np.int64)

    train_idx, test_idx = split_indices(len(df), args.train_ratio, args.seed)
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # Load embeddings (from .npy file or parquet column)
    def _load_embeddings():
        if args.embedding_npy:
            emb = np.load(args.embedding_npy).astype(np.float32)
            if args.data_pool_size is not None:
                emb = emb[: args.data_pool_size]
            if _consensus_keep_idx is not None:
                emb = emb[_consensus_keep_idx]
            print(f"[info] loaded embeddings from {args.embedding_npy}: {emb.shape}")
            return emb
        return stack_embedding_column(df, args.embedding_key, show_progress=show_progress)

    def _load_extra_embeddings():
        if not args.extra_embedding_npy:
            return None
        emb = np.load(args.extra_embedding_npy).astype(np.float32)
        if args.data_pool_size is not None:
            emb = emb[: args.data_pool_size]
        if _consensus_keep_idx is not None:
            emb = emb[_consensus_keep_idx]
        print(f"[info] loaded extra embeddings from {args.extra_embedding_npy}: {emb.shape}")
        return emb

    # Build input features
    if args.text_features_only:
        from feature_extraction import extract_features_batch, normalize_features
        text_feats, feat_names = extract_features_batch(df, args.messages_key, show_progress)
        train_input = text_feats[train_idx]
        test_input = text_feats[test_idx]
        train_input, test_input = normalize_features(train_input, test_input)
        print(f"[info] text features only: {len(feat_names)} features")
        print(f"[info] feature names: {feat_names}")
    elif args.use_text_features:
        from feature_extraction import extract_features_batch, normalize_features
        embeddings = _load_embeddings()
        extra_emb = _load_extra_embeddings()
        text_feats, feat_names = extract_features_batch(df, args.messages_key, show_progress)
        # Normalize text features before concatenation (embeddings already L2-normed)
        train_tf, test_tf = normalize_features(text_feats[train_idx], text_feats[test_idx])
        parts_train = [embeddings[train_idx]]
        parts_test = [embeddings[test_idx]]
        dim_desc = f"emb={embeddings.shape[1]}"
        if extra_emb is not None:
            parts_train.append(extra_emb[train_idx])
            parts_test.append(extra_emb[test_idx])
            dim_desc += f" + extra_emb={extra_emb.shape[1]}"
        parts_train.append(train_tf)
        parts_test.append(test_tf)
        dim_desc += f" + text={len(feat_names)}"
        train_input = np.concatenate(parts_train, axis=1)
        test_input = np.concatenate(parts_test, axis=1)
        print(f"[info] {dim_desc} = {train_input.shape[1]}")
        print(f"[info] feature names: {feat_names}")
    else:
        embeddings = _load_embeddings()
        extra_emb = _load_extra_embeddings()
        if extra_emb is not None:
            embeddings = np.concatenate([embeddings, extra_emb], axis=1)
            print(f"[info] concatenated embeddings: {embeddings.shape}")
        train_input = embeddings[train_idx]
        test_input = embeddings[test_idx]

    num_classes = resolve_num_classes(args.num_classes, train_labels, test_labels)

    device = resolve_device(args.device)
    input_dim = train_input.shape[1]
    is_ordinal = args.model == "ordinal"

    print(f"[info] input_dim={input_dim}, num_classes={num_classes}")
    print(f"[info] train={len(train_labels)}, test={len(test_idx)}")
    print(f"[info] model={args.model}, hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}, epochs={args.epochs}")

    # Fixed evaluation data (never changes across self-training rounds)
    eval_input = test_input
    eval_labels = test_labels
    eval_emb_t = torch.from_numpy(eval_input.astype(np.float32))
    eval_labels_t = torch.from_numpy(eval_labels).long()

    # ── Self-training state ───────────────────────────────────────
    total_rounds = args.self_train_rounds + 1  # round 0 = baseline
    cur_train_input = train_input.copy()
    cur_train_labels = train_labels.copy()

    # Track which eval samples have been pseudo-labeled
    unlabeled_mask = np.ones(len(eval_labels), dtype=bool)

    # Detect minority classes from the original label distribution
    orig_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    orig_freq = orig_counts / orig_counts.sum()
    minority_classes = set(
        c for c in range(num_classes)
        if orig_freq[c] < args.minority_max_freq
    )
    if args.self_train_rounds > 0:
        print(f"\n[self-train] rounds={args.self_train_rounds}, "
              f"conf_threshold={args.conf_threshold}, "
              f"minority_conf={args.minority_conf}")
        print(f"[self-train] minority classes (freq<{args.minority_max_freq}): "
              f"{sorted(minority_classes)} "
              f"(freqs: {[f'{orig_freq[c]:.3f}' for c in sorted(minority_classes)]})")

    round_metrics_list = []

    for st_round in range(total_rounds):
        round_label = f"round {st_round}/{args.self_train_rounds}"
        if args.self_train_rounds > 0:
            print(f"\n{'='*60}")
            print(f"[self-train] {round_label}, "
                  f"train_size={len(cur_train_labels)}, "
                  f"unlabeled={unlabeled_mask.sum()}")
            print(f"{'='*60}")

        # ── Apply oversample_train each round ─────────────────────
        round_train_input = cur_train_input
        round_train_labels = cur_train_labels
        if args.oversample_train:
            rng_os = np.random.RandomState(args.seed + st_round)
            counts_os = np.bincount(round_train_labels, minlength=num_classes)
            target_os = int(counts_os.max())
            new_idx = []
            for c in range(num_classes):
                c_idx = np.where(round_train_labels == c)[0]
                if len(c_idx) == 0:
                    continue
                if len(c_idx) < target_os:
                    extra = rng_os.choice(c_idx, size=target_os - len(c_idx), replace=True)
                    c_idx = np.concatenate([c_idx, extra])
                new_idx.append(c_idx)
            new_idx = np.concatenate(new_idx)
            rng_os.shuffle(new_idx)
            round_train_input = round_train_input[new_idx]
            round_train_labels = round_train_labels[new_idx]
            if st_round == 0:
                print(f"[info] oversample_train: {len(cur_train_labels)} -> "
                      f"{len(round_train_labels)} (target={target_os}/class)")

        # ── Prepare tensors ───────────────────────────────────────
        train_emb_t = torch.from_numpy(round_train_input.astype(np.float32))
        train_labels_t = torch.from_numpy(round_train_labels).long()

        train_dataset = TensorDataset(train_emb_t, train_labels_t)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)

        # ── Class weights ─────────────────────────────────────────
        class_weights = None
        if args.class_weight:
            cw_counts = np.bincount(round_train_labels, minlength=num_classes).astype(np.float64)
            cw_counts = np.maximum(cw_counts, 1.0)
            inv_freq = 1.0 / cw_counts
            inv_freq = inv_freq / inv_freq.sum() * num_classes
            class_weights = torch.from_numpy(inv_freq).float()
            if st_round == 0:
                print(f"[info] class_weights="
                      f"{[round(w, 4) for w in class_weights.tolist()]}")

        # ── Build fresh model each round ──────────────────────────
        if args.model == "linear":
            model = LinearProbe(input_dim, num_classes, dropout=args.dropout)
        elif args.model == "mlp":
            model = MLPClassifier(input_dim, args.hidden_dim, num_classes,
                                  dropout=args.dropout)
        elif args.model == "resnet":
            model = ResNetMLP(input_dim, args.hidden_dim, num_classes,
                              num_layers=args.num_layers, dropout=args.dropout)
        elif args.model == "ordinal":
            model = OrdinalModel(input_dim, args.hidden_dim, num_classes,
                                 num_layers=args.num_layers, dropout=args.dropout)

        param_count = sum(p.numel() for p in model.parameters())
        if st_round == 0:
            print(f"[info] model parameters: {param_count:,}")

        # ── Train ─────────────────────────────────────────────────
        train_result = train_model(
            model=model,
            train_loader=train_loader,
            val_embeddings=eval_emb_t,
            val_labels=eval_labels_t,
            args=args,
            class_weights=class_weights,
            device=device,
            num_classes=num_classes,
            show_progress=show_progress,
        )

        # ── Predict on eval set ───────────────────────────────────
        proxy_pred, proxy_conf = predict(model, eval_emb_t, device,
                                         is_ordinal=is_ordinal)

        # ── Metrics ───────────────────────────────────────────────
        conf_mat = confusion_matrix(eval_labels, proxy_pred, num_classes)
        accuracy = float((proxy_pred == eval_labels).mean())
        mf1 = macro_f1(conf_mat)
        mae = float(np.abs(proxy_pred - eval_labels).mean())

        round_info = {
            "round": st_round,
            "train_size": len(round_train_labels),
            "train_size_before_oversample": len(cur_train_labels),
            "unlabeled_remaining": int(unlabeled_mask.sum()),
            "accuracy": accuracy,
            "macro_f1": mf1,
            "mean_abs_error": mae,
            "best_val_acc": train_result["best_val_acc"],
        }
        for c in range(num_classes):
            mask_c = eval_labels == c
            if mask_c.sum() > 0:
                round_info[f"class_{c}_acc"] = float(
                    (proxy_pred[mask_c] == c).mean())

        round_metrics_list.append(round_info)

        print(f"[{round_label}] acc={accuracy:.4f}, F1={mf1:.4f}, "
              f"mae={mae:.4f}, train={len(round_train_labels)}")

        # ── Pseudo-label selection (skip last round) ──────────────
        if st_round < args.self_train_rounds:
            # Select high-confidence predictions from unlabeled pool
            selected = []
            for i in range(len(eval_labels)):
                if not unlabeled_mask[i]:
                    continue
                pred_class = int(proxy_pred[i])
                thr = (args.minority_conf if pred_class in minority_classes
                       else args.conf_threshold)
                if proxy_conf[i] >= thr:
                    selected.append(i)

            selected = np.array(selected, dtype=np.int64)
            if len(selected) == 0:
                print(f"[self-train] no new pseudo-labels selected, stopping early")
                break

            pseudo_input = eval_input[selected]
            pseudo_labels = proxy_pred[selected]

            # Per-class breakdown of pseudo-labels
            pl_counts = np.bincount(pseudo_labels, minlength=num_classes)
            # Accuracy of pseudo-labels (vs ground truth)
            pl_acc = float((pseudo_labels == eval_labels[selected]).mean())
            round_info["pseudo_added"] = int(len(selected))
            round_info["pseudo_accuracy"] = pl_acc
            round_info["pseudo_per_class"] = pl_counts.tolist()

            print(f"[self-train] +{len(selected)} pseudo-labels "
                  f"(pseudo_acc={pl_acc:.4f}), "
                  f"per_class={pl_counts.tolist()}")

            # Add to training set
            cur_train_input = np.concatenate(
                [cur_train_input, pseudo_input], axis=0)
            cur_train_labels = np.concatenate(
                [cur_train_labels, pseudo_labels], axis=0)
            unlabeled_mask[selected] = False

    # ── Final metrics & save ──────────────────────────────────────
    metrics = {
        "dataset_path": args.dataset_path,
        "embedding_key": args.embedding_key,
        "label_key": args.label_key,
        "num_rows": len(df),
        "data_pool_size": args.data_pool_size,
        "train_rows_initial": len(train_labels),
        "train_rows_final": len(cur_train_labels),
        "test_rows": len(test_idx),
        "oversample_train": args.oversample_train,
        "label_fusion": args.label_fusion,
        "self_train_rounds": args.self_train_rounds,
        "conf_threshold": args.conf_threshold,
        "minority_conf": args.minority_conf,
        "minority_classes": sorted(minority_classes) if args.self_train_rounds > 0 else None,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "num_classes": num_classes,
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers if args.model in ("resnet", "ordinal") else None,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "class_weight": args.class_weight,
        "focal_loss": args.focal_loss,
        "focal_gamma": args.focal_gamma if args.focal_loss else None,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "param_count": param_count,
        "best_val_acc": train_result["best_val_acc"],
        "accuracy": accuracy,
        "macro_f1": mf1,
        "mean_abs_error": mae,
        "confusion_matrix": conf_mat.tolist(),
    }

    # Per-class accuracy
    for c in range(num_classes):
        mask = eval_labels == c
        if mask.sum() > 0:
            metrics[f"class_{c}_acc"] = float((proxy_pred[mask] == c).mean())
            metrics[f"class_{c}_count"] = int(mask.sum())

    # Save
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    history_path = output_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(train_result["history"], f, indent=2)

    # Save round metrics (self-training progression)
    if args.self_train_rounds > 0:
        round_path = output_dir / "round_metrics.json"
        with round_path.open("w", encoding="utf-8") as f:
            json.dump(round_metrics_list, f, indent=2, ensure_ascii=False)
        print(f"[done] round_metrics: {round_path}")

    # Save predictions
    predictions_path = output_dir / "test_with_proxy.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for local_i, global_i in enumerate(
            tqdm(test_idx.tolist(), desc="write predictions",
                 disable=not show_progress)
        ):
            row = df.iloc[global_i].to_dict()
            row.pop(args.embedding_key, None)
            row = {k: to_serializable(v) for k, v in row.items()}
            row[args.prediction_key] = int(proxy_pred[local_i])
            row["proxy_confidence"] = float(proxy_conf[local_i])
            row["ground_truth_label"] = int(eval_labels[local_i])
            row["split"] = "test"
            row["row_index"] = int(global_i)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save model
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    print(f"\n[done] metrics: {metrics_path}")
    print(f"[done] predictions: {predictions_path}")
    print(f"[done] model: {model_path}")
    print(
        f"[summary] accuracy={accuracy:.4f}, macro_f1={mf1:.4f}, "
        f"mae={mae:.4f}, best_val_acc={train_result['best_val_acc']:.4f}"
    )

    # Per-class summary
    print("\n[per-class accuracy]")
    for c in range(num_classes):
        acc_key = f"class_{c}_acc"
        cnt_key = f"class_{c}_count"
        if acc_key in metrics:
            print(f"  class {c}: acc={metrics[acc_key]:.4f}  (n={metrics[cnt_key]})")


if __name__ == "__main__":
    main()
