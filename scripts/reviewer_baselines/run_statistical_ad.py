#!/usr/bin/env python3
"""Full-flight statistical anomaly baseline for the submitted Table 4 split.

Each sensor has an independent robust normal range estimated from normal
training flights only. A flight score is the configured percentile of all
absolute robust z-scores across its valid time points and sensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reviewer_baselines.common import (  # noqa: E402
    DATASETS,
    cache_path,
    parse_list,
    per_class_chronological_split,
    precision_at_recall_floor,
)


def fit_reference(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.where(mask[:, :, None] > 0, x, np.nan)
    center = np.nanmedian(values, axis=(0, 1))
    mad = np.nanmedian(np.abs(values - center[None, None, :]), axis=(0, 1))
    scale = np.maximum(1.4826 * mad, 1e-3)
    return center.astype(np.float32), scale.astype(np.float32)


def score_in_chunks(
    x: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    percentile: float,
    chunk_size: int,
) -> np.ndarray:
    output = []
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        values = np.asarray(x[selected], dtype=np.float32)
        valid = np.asarray(mask[selected]) > 0
        residual = np.abs((values - center[None, None, :]) / scale[None, None, :])
        residual = np.where(valid[:, :, None], residual, np.nan)
        output.append(np.nanpercentile(residual, percentile, axis=(1, 2)))
    return np.concatenate(output).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--score-percentile", type=float, default=99.0)
    parser.add_argument("--validation-threshold-percentile", type=float, default=99.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in parse_list(args.datasets):
        path = cache_path(args.cache_root, dataset)
        with np.load(path, allow_pickle=False) as archive:
            x = np.asarray(archive["x"], dtype=np.float32)
            labels = np.asarray(archive["labels"], dtype=np.int64)
            mask = (
                np.asarray(archive["mask"], dtype=np.float32)
                if "mask" in archive.files
                else np.ones(x.shape[:2], dtype=np.float32)
            )
            sources = (
                np.asarray(archive["sources"]).astype(str)
                if "sources" in archive.files
                else np.arange(labels.size).astype(str)
            )
            time_keys = (
                np.asarray(archive["time_keys"], dtype=np.int64)
                if "time_keys" in archive.files
                else np.arange(labels.size, dtype=np.int64)
            )
        split = per_class_chronological_split(labels, time_keys, sources)
        train_normal = split["train"][labels[split["train"]] == 0]
        val_normal = split["validation"][labels[split["validation"]] == 0]
        test = split["test"]
        center, scale = fit_reference(x[train_normal], mask[train_normal])
        validation_scores = score_in_chunks(
            x, mask, val_normal, center, scale, args.score_percentile, args.chunk_size
        )
        test_scores = score_in_chunks(
            x, mask, test, center, scale, args.score_percentile, args.chunk_size
        )
        y_test = (labels[test] != 0).astype(np.int8)
        operating = precision_at_recall_floor(y_test, test_scores)
        val_threshold = float(np.percentile(validation_scores, args.validation_threshold_percentile))
        row = {
            "dataset": dataset,
            "model": "Robust univariate threshold",
            "seed": 2021,
            "sequence_length": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "train_normal": int(train_normal.size),
            "validation_normal": int(val_normal.size),
            "test_samples": int(test.size),
            "score_percentile": args.score_percentile,
            "validation_threshold_percentile": args.validation_threshold_percentile,
            "validation_threshold": val_threshold,
            "auroc": float(roc_auc_score(y_test, test_scores)),
            "auprc": float(average_precision_score(y_test, test_scores)),
            **operating,
        }
        result_dir = args.output_dir / dataset
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(
            result_dir / "scores.npz",
            labels=y_test,
            scores=test_scores,
            test_indices=test,
            validation_normal_scores=validation_scores,
            center=center,
            scale=scale,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with (args.output_dir / "all_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
