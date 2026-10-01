#!/usr/bin/env python
"""Run the official DCdetector architecture on QAR flight-level windows.

The upstream evaluation mixes test scores into its threshold and applies
label-aware point adjustment.  This adapter intentionally keeps only the
architecture and contrastive objective: training/checkpoint selection use
normal train/validation flights, the threshold is fitted on normal validation
flights, and the held-out flight labels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--win-size", type=int, default=200)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[4, 5, 10])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--temperature", type=float, default=50.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_bounds(n: int) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = max(1, min(int(n * 0.7), n - 2))
    val_end = max(train_end + 1, min(int(n * 0.8), n - 1))
    return train_end, val_end


def ordered_split(indices: np.ndarray, time_keys: np.ndarray, sources: np.ndarray):
    ordered = sorted(
        (int(i) for i in indices),
        key=lambda i: (int(time_keys[i]), str(sources[i]), i),
    )
    train_end, val_end = split_bounds(len(ordered))
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


class FlightWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, labels: np.ndarray, indices: list[int], win_size: int):
        self.x = x
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.win_size = int(win_size)
        source_len = int(x.shape[1])
        self.sample_points = np.rint(np.linspace(0, source_len - 1, self.win_size)).astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        seq = np.asarray(self.x[index, self.sample_points], dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(seq), int(self.labels[index] != 0)


def kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    value = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(value, dim=-1), dim=1)


def normalized_prior(prior: torch.Tensor, win_size: int) -> torch.Tensor:
    denom = torch.sum(prior, dim=-1, keepdim=True).clamp_min(1e-12)
    return prior / denom.repeat(1, 1, 1, win_size)


def contrastive_loss(model, inputs: torch.Tensor) -> torch.Tensor:
    series_list, prior_list = model(inputs)
    series_loss = torch.zeros((), device=inputs.device)
    prior_loss = torch.zeros((), device=inputs.device)
    for series, prior in zip(series_list, prior_list):
        prior_norm = normalized_prior(prior, inputs.shape[1])
        series_loss = series_loss + (
            torch.mean(kl_loss(series, prior_norm.detach()))
            + torch.mean(kl_loss(prior_norm.detach(), series))
        )
        prior_loss = prior_loss + (
            torch.mean(kl_loss(prior_norm, series.detach()))
            + torch.mean(kl_loss(series.detach(), prior_norm))
        )
    count = max(1, len(prior_list))
    return prior_loss / count - series_loss / count


def symmetric_divergence(model, inputs: torch.Tensor) -> torch.Tensor:
    """Positive validation objective for checkpoint selection.

    The upstream min-max expression has an exactly zero scalar value because
    the two terms differ only in gradient detachment.  It can train, but its
    numeric value cannot rank checkpoints.  Validation therefore uses the
    corresponding symmetric KL divergence without the cancelling subtraction.
    """
    series_list, prior_list = model(inputs)
    value = torch.zeros((), device=inputs.device)
    for series, prior in zip(series_list, prior_list):
        prior_norm = normalized_prior(prior, inputs.shape[1])
        value = value + (
            torch.mean(kl_loss(series, prior_norm))
            + torch.mean(kl_loss(prior_norm, series))
        )
    return value / max(1, len(prior_list))


@torch.no_grad()
def validation_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    values = []
    for inputs, _ in loader:
        values.append(float(symmetric_divergence(model, inputs.float().to(device)).item()))
    return float(np.mean(values)) if values else float("inf")


@torch.no_grad()
def flight_scores(model, loader: DataLoader, device: torch.device, temperature: float):
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for inputs, batch_labels in loader:
        inputs = inputs.float().to(device)
        series_list, prior_list = model(inputs)
        point_energy = None
        for series, prior in zip(series_list, prior_list):
            prior_norm = normalized_prior(prior, inputs.shape[1])
            divergence = (
                kl_loss(series, prior_norm.detach())
                + kl_loss(prior_norm, series.detach())
            ) * temperature
            point_energy = divergence if point_energy is None else point_energy + divergence
        # Preserve the upstream score definition while aggregating one score
        # per independently labelled flight window.
        point_score = torch.softmax(-point_energy, dim=-1)
        flight_score = torch.quantile(point_score, 0.99, dim=-1)
        scores.append(flight_score.cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(np.int64)


def main() -> None:
    args = parse_args()
    for patch in args.patch_sizes:
        if args.win_size % patch:
            raise ValueError(f"win_size={args.win_size} must be divisible by patch_size={patch}")
    if not torch.cuda.is_available():
        raise RuntimeError("The official RevIN implementation constructs CUDA tensors in forward; a CUDA device is required")
    seed_everything(args.seed)

    source_root = args.source_root.resolve()
    if not (source_root / "model" / "DCdetector.py").is_file():
        raise FileNotFoundError(source_root)
    sys.path.insert(0, str(source_root))
    from model.DCdetector import DCdetector  # pylint: disable=import-error,import-outside-toplevel

    cache = np.load(args.cache, allow_pickle=False, mmap_mode="r")
    x = cache["x"]
    labels = cache["labels"].astype(np.int64, copy=False)
    sources = cache["sources"].astype(str) if "sources" in cache.files else np.arange(len(labels)).astype(str)
    time_keys = cache["time_keys"].astype(np.int64, copy=False) if "time_keys" in cache.files else np.arange(len(labels), dtype=np.int64)
    all_indices = np.arange(len(labels), dtype=np.int64)
    normal_train, normal_val, normal_test = ordered_split(all_indices[labels == 0], time_keys, sources)
    _, _, fault_test = ordered_split(all_indices[labels != 0], time_keys, sources)
    test_indices = sorted(normal_test + fault_test, key=lambda i: (int(time_keys[i]), str(sources[i]), i))

    train_ds = FlightWindowDataset(x, labels, normal_train, args.win_size)
    val_ds = FlightWindowDataset(x, labels, normal_val, args.win_size)
    test_ds = FlightWindowDataset(x, labels, test_indices, args.win_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda:0")
    model = DCdetector(
        win_size=args.win_size,
        enc_in=int(x.shape[2]),
        c_out=int(x.shape[2]),
        n_heads=args.n_heads,
        d_model=args.d_model,
        e_layers=args.e_layers,
        patch_size=args.patch_sizes,
        channel=int(x.shape[2]),
        d_ff=args.d_ff,
        dropout=args.dropout,
        output_attention=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best.pt"
    best_val = float("inf")
    bad_epochs = 0
    started = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_values = []
        for inputs, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(model, inputs.float().to(device))
            loss.backward()
            optimizer.step()
            train_values.append(float(loss.item()))
        val_loss = validation_loss(model, val_loader, device)
        train_loss = float(np.mean(train_values))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch + 1} train_loss={train_loss:.8f} val_loss={val_loss:.8f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.5 ** epoch)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    val_scores, val_labels = flight_scores(model, val_loader, device, args.temperature)
    if np.any(val_labels != 0):
        raise RuntimeError("validation threshold set must contain normal flights only")
    threshold = float(np.percentile(val_scores, args.threshold_percentile))
    test_scores, truth = flight_scores(model, test_loader, device, args.temperature)
    pred = (test_scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    precision, recall, binary_f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    metrics = {
        "dataset": args.dataset,
        "model": "DCdetector",
        "seed": args.seed,
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(binary_f1),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, test_scores)) if np.unique(truth).size == 2 else float("nan"),
        "pr_auc": float(average_precision_score(truth, test_scores)) if np.any(truth == 1) else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "threshold": threshold,
        "threshold_source": "normal_validation",
        "threshold_percentile": args.threshold_percentile,
        "best_val_loss": best_val,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - started,
        "train_normal": len(train_ds),
        "val_normal": len(val_ds),
        "test_samples": len(test_ds),
        "source_commit": "9d107cae518ef24ee2eb2ad1b92107cdcd09ea10",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["score", "truth", "prediction"])
        writer.writerows(zip(test_scores.tolist(), truth.tolist(), pred.tolist()))
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
