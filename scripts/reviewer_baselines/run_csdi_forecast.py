#!/usr/bin/env python3
"""Run the official CSDI forecasting model on one QAR dataset/anchor cell.

This adapter deliberately preserves the submitted Table 3 protocol:

* one compact transition window per sample, shape ``(80, C)``;
* the first 60 points are observed and the last 20 are forecast;
* per-class chronological 70/10/20 split with the legacy cumulative boundary;
* all channels are forecast jointly;
* MAE, RMSE, and ``Loss=(MAE+RMSE)/2`` are computed over every test target.

Only the data adapter, training loop, and result serialization are local.  The
network and diffusion sampler are imported unchanged from the official CSDI
repository (https://github.com/ermongroup/CSDI).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import DATASET_LABELS, per_class_chronological_split


class QARCSDIDataset(Dataset):
    def __init__(self, x: np.ndarray, mask: np.ndarray, indices: np.ndarray):
        self.x = x
        self.mask = mask
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        data = np.asarray(self.x[index, :80], dtype=np.float32)
        time_mask = np.asarray(self.mask[index, :80], dtype=np.float32)
        observed_mask = np.broadcast_to(time_mask[:, None], data.shape).copy()
        condition_mask = observed_mask.copy()
        condition_mask[60:] = 0.0
        return {
            "observed_data": torch.from_numpy(data),
            "observed_mask": torch.from_numpy(observed_mask),
            "gt_mask": torch.from_numpy(condition_mask),
            "timepoints": torch.arange(80, dtype=torch.float32),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cache = np.load(path, allow_pickle=False)
    x = np.asarray(cache["x"], dtype=np.float32)
    if x.ndim != 3 or x.shape[1] < 80:
        raise ValueError(f"Expected x=(N,T>=80,C), got {x.shape} in {path}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    labels = (
        np.asarray(cache["labels"], dtype=np.int64)
        if "labels" in cache.files
        else np.zeros(x.shape[0], dtype=np.int64)
    )
    mask = (
        np.asarray(cache["mask"], dtype=np.float32)
        if "mask" in cache.files
        else np.ones(x.shape[:2], dtype=np.float32)
    )
    if mask.ndim == 3:
        mask = (mask.min(axis=-1) > 0).astype(np.float32)
    if mask.shape[:2] != x.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match x {x.shape}")
    sources = (
        np.asarray(cache["sources"]).astype(str)
        if "sources" in cache.files
        else np.arange(x.shape[0]).astype(str)
    )
    time_keys = (
        np.asarray(cache["time_keys"], dtype=np.int64)
        if "time_keys" in cache.files
        else np.arange(x.shape[0], dtype=np.int64)
    )
    valid_indices = np.flatnonzero(mask[:, :80].sum(axis=1) >= 80)
    relative = per_class_chronological_split(
        labels[valid_indices], time_keys[valid_indices], sources[valid_indices]
    )
    splits = {name: valid_indices[indices] for name, indices in relative.items()}
    return x, mask, splits


def average_train_loss(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            # CSDI_Forecasting uses the fixed gt_mask for its "test" target
            # strategy.  is_train=1 samples one diffusion step per batch.
            loss = model(batch, is_train=1)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = int(batch["observed_data"].shape[0])
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, n_samples: int) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    n_values = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        samples, observed, target_mask, _, _ = model.evaluate(batch, n_samples)
        # Official output: samples=(B,S,C,T), observed=(B,C,T).
        prediction = samples.mean(dim=1)
        active = target_mask.bool()
        residual = prediction - observed
        sum_abs += float(residual.abs()[active].sum().cpu())
        sum_sq += float(residual.square()[active].sum().cpu())
        n_values += int(active.sum().cpu())
        predictions.append(prediction[:, :, 60:80].permute(0, 2, 1).cpu().numpy())
        targets.append(observed[:, :, 60:80].permute(0, 2, 1).cpu().numpy())
    mae = sum_abs / n_values
    mse = sum_sq / n_values
    rmse = float(np.sqrt(mse))
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "loss": float((mae + rmse) / 2.0),
        "n_target_values": int(n_values),
    }
    return metrics, np.concatenate(predictions), np.concatenate(targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--csdi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.csdi_root.resolve()))
    from main_model import CSDI_Forecasting

    set_seed(args.seed)
    device = torch.device(args.device)
    x, mask, splits = load_cache(args.cache)
    loaders = {
        name: DataLoader(
            QARCSDIDataset(x, mask, indices),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, indices in splits.items()
    }

    config = {
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.learning_rate,
        },
        "diffusion": {
            "layers": args.layers,
            "channels": args.channels,
            "nheads": args.heads,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": args.diffusion_steps,
            "schedule": "quad",
            "is_linear": True,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 128,
            "featureemb": 16,
            "target_strategy": "test",
            # No feature sub-sampling: every QAR cache has <=20 channels.
            "num_sample_features": int(x.shape[2]),
        },
    }
    model = CSDI_Forecasting(config, device, target_dim=int(x.shape[2])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_model.pth"

    best_val = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = average_train_loss(model, loaders["train"], optimizer, device, True)
        val_loss = average_train_loss(model, loaders["validation"], optimizer, device, False)
        row = {"epoch": epoch, "train_diffusion_loss": train_loss, "val_diffusion_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    metrics, predictions, targets = evaluate(model, loaders["test"], args.n_samples)
    metrics.update(
        {
            "model": "CSDI",
            "official_source_commit": "7f24a436f08d98853a6b43d4f7f04e5a65ecdf27",
            "dataset": args.dataset,
            "dataset_label": DATASET_LABELS.get(args.dataset, args.dataset),
            "anchor": args.anchor,
            "seed": args.seed,
            "protocol": "Table3_fullshot_60to20_legacy_per_class_chrono",
            "cache": str(args.cache),
            "cache_shape": list(map(int, x.shape)),
            "train_samples": int(splits["train"].size),
            "validation_samples": int(splits["validation"].size),
            "test_samples": int(splits["test"].size),
            "epochs_completed": len(history),
            "best_val_diffusion_loss": float(best_val),
            "n_diffusion_samples": args.n_samples,
            "elapsed_seconds": float(time.time() - start),
            "config": config,
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        predictions=predictions.astype(np.float32),
        targets=targets.astype(np.float32),
        test_indices=splits["test"],
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
