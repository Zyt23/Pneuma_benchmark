#!/usr/bin/env python3
"""Zero-shot MOMENT reconstruction detector on complete QAR flight sequences.

MOMENT-1-large has a 512-step native context. Each complete flight is therefore
covered by consecutive 512-step chunks (the final chunk is mask-padded), and a
single flight score is the configured percentile of all reconstruction errors
from all valid chunks, channels, and time points. No fault sample is used for
training or adaptation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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


@torch.inference_mode()
def score_flights(
    model,
    x: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    context_length: int,
    batch_size: int,
    score_percentile: float,
) -> np.ndarray:
    result: list[np.ndarray] = []
    for batch_start in range(0, len(indices), batch_size):
        selected = indices[batch_start : batch_start + batch_size]
        batch_x = np.asarray(x[selected], dtype=np.float32)
        batch_mask = np.asarray(mask[selected], dtype=np.float32)
        batch_errors: list[np.ndarray] = []
        for start in range(0, batch_x.shape[1], context_length):
            end = min(start + context_length, batch_x.shape[1])
            valid_length = end - start
            segment = np.zeros(
                (batch_x.shape[0], context_length, batch_x.shape[2]), dtype=np.float32
            )
            segment_mask = np.zeros((batch_x.shape[0], context_length), dtype=np.float32)
            segment[:, :valid_length] = batch_x[:, start:end]
            segment_mask[:, :valid_length] = batch_mask[:, start:end]
            tensor = torch.from_numpy(segment).permute(0, 2, 1).to(device)
            input_mask = torch.from_numpy(segment_mask).to(device)
            outputs = model.detect_anomalies(
                x_enc=tensor,
                input_mask=input_mask,
                anomaly_criterion="mse",
            )
            error = outputs.anomaly_scores.detach().float().cpu().numpy()
            if error.shape != tensor.shape:
                raise ValueError(
                    f"unexpected MOMENT anomaly score shape {error.shape}; expected {tuple(tensor.shape)}"
                )
            error = error[:, :, :valid_length]
            valid = segment_mask[:, None, :valid_length] > 0
            batch_errors.append(np.where(valid, error, np.nan))
        flattened = np.concatenate(batch_errors, axis=2)
        result.append(np.nanpercentile(flattened, score_percentile, axis=(1, 2)))
        print(
            f"scored {min(batch_start + batch_size, len(indices))}/{len(indices)} flights",
            flush=True,
        )
    return np.concatenate(result).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--model", default="AutonLab/MOMENT-1-large")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-percentile", type=float, default=99.0)
    parser.add_argument(
        "--summary-name",
        default="all_metrics.csv",
        help="Root-level summary filename; use a dataset-specific name for parallel runs.",
    )
    args = parser.parse_args()

    from momentfm import MOMENTPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()
    model = MOMENTPipeline.from_pretrained(
        args.model,
        model_kwargs={"task_name": "reconstruction"},
        cache_dir=str(args.cache_dir),
    )
    model.init()
    model = model.to(device).eval()
    load_seconds = time.time() - started

    rows = []
    for dataset in parse_list(args.datasets):
        dataset_started = time.time()
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
        test = split["test"]
        y_test = (labels[test] != 0).astype(np.int8)
        scores = score_flights(
            model,
            x,
            mask,
            test,
            device,
            args.context_length,
            args.batch_size,
            args.score_percentile,
        )
        operating = precision_at_recall_floor(y_test, scores)
        row = {
            "dataset": dataset,
            "model": "MOMENT-1-large (zero-shot AD)",
            "seed": 2021,
            "sequence_length": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "test_samples": int(test.size),
            "native_context_length": args.context_length,
            "coverage": "non-overlapping chunks plus masked final chunk",
            "score": f"p{args.score_percentile:g} point-channel reconstruction MSE",
            "model_load_seconds": load_seconds,
            "elapsed_seconds": time.time() - dataset_started,
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
            scores=scores,
            test_indices=test,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with (args.output_dir / args.summary_name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
