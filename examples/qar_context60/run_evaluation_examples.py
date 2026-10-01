#!/usr/bin/env python3
"""Run small evaluation examples for the QAR context60 protocol.

This script is deliberately lightweight.  It evaluates simple deterministic
forecasting and classification baselines on the representative compact caches
created by `make_representative_data.py`, then writes table-shaped CSV outputs
that mirror the paper-table aggregation logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Callable

import numpy as np


ANCHORS = ["predict_2_3", "predict_4_5", "predict_5_6", "predict_8_9"]
DATASET = "dataset_demo"
DATASET_LABEL = "DEMO-QAR"
MAIN_MODELS = ["NaiveLast", "LinearDrift"]
UNI_MULTI_MODELS = ["NaiveLast", "CoupledNaive"]


def read_int(cache: np.lib.npyio.NpzFile, name: str, default: int) -> int:
    if name not in cache.files:
        return default
    value = np.asarray(cache[name]).reshape(-1)
    return int(value[0])


def load_forecast_cache(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    cache = np.load(path, allow_pickle=False)
    x = np.asarray(cache["x"], dtype=np.float64)
    mask = np.asarray(cache["mask"], dtype=np.float64) if "mask" in cache.files else np.ones(x.shape[:2])
    seq_len = read_int(cache, "seq_len", x.shape[1] - 20)
    pred_len = read_int(cache, "pred_len", 20)
    required = seq_len + pred_len
    valid = mask[:, :required].sum(axis=1) >= required
    x = x[valid, :required, :]
    if x.shape[0] == 0:
        raise ValueError(f"No valid rows in {path}")
    return x[:, :seq_len, :], x[:, seq_len:required, :], seq_len, pred_len


def metrics_from_arrays(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}")
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    rmse = float(math.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse, "loss": (mae + rmse) / 2.0}


def naive_last(context: np.ndarray, pred_len: int) -> np.ndarray:
    return np.repeat(context[:, -1:, :], pred_len, axis=1)


def linear_drift(context: np.ndarray, pred_len: int) -> np.ndarray:
    span = min(12, max(2, context.shape[1] // 4))
    slope = (context[:, -1:, :] - context[:, -span:-span + 1, :]) / float(span - 1)
    steps = np.arange(1, pred_len + 1, dtype=np.float64).reshape(1, pred_len, 1)
    return context[:, -1:, :] + steps * slope


def coupled_naive(context: np.ndarray, pred_len: int) -> np.ndarray:
    last = context[:, -1:, :]
    channel_mean = last.mean(axis=2, keepdims=True)
    return np.repeat(last + 0.15 * (channel_mean - last), pred_len, axis=1)


FORECASTERS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "NaiveLast": naive_last,
    "LinearDrift": linear_drift,
    "CoupledNaive": coupled_naive,
}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    values = [float(v) for v in values if not math.isnan(float(v))]
    return float(sum(values) / len(values)) if values else float("nan")


def write_result_arrays(result_root: Path, model: str, table: str, anchor: str, pred: np.ndarray, true: np.ndarray) -> str:
    result_dir = result_root / table / model / anchor / DATASET
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "pred.npy", pred.astype(np.float32))
    np.save(result_dir / "true.npy", true.astype(np.float32))
    metric = metrics_from_arrays(pred, true)
    np.save(
        result_dir / "metrics.npy",
        np.asarray([metric["mae"], metric["mse"], metric["rmse"]], dtype=np.float32),
    )
    return result_dir.as_posix()


def run_main_table(data_root: Path, output: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_rows: list[dict[str, object]] = []
    result_root = output / "result_arrays"
    for anchor in ANCHORS:
        cache_path = data_root / "context60" / "hist5" / anchor / DATASET / "qar_compact_shiftN80.npz"
        context, true, seq_len, pred_len = load_forecast_cache(cache_path)
        for model in MAIN_MODELS:
            pred = FORECASTERS[model](context, pred_len)
            metric = metrics_from_arrays(pred, true)
            result_dir = write_result_arrays(result_root, model, "main", anchor, pred, true)
            detail_rows.append(
                {
                    "history_count": 5,
                    "anchor": anchor,
                    "dataset": DATASET,
                    "dataset_label": DATASET_LABEL,
                    "model": model,
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "mae": metric["mae"],
                    "mse": metric["mse"],
                    "rmse": metric["rmse"],
                    "loss": metric["loss"],
                    "result_dir": result_dir,
                }
            )

    table_rows: list[dict[str, object]] = []
    for model in MAIN_MODELS:
        sub = [row for row in detail_rows if row["model"] == model]
        loss = mean([float(row["loss"]) for row in sub])
        mae = mean([float(row["mae"]) for row in sub])
        mse = mean([float(row["mse"]) for row in sub])
        rmse = mean([float(row["rmse"]) for row in sub])
        table_rows.append(
            {
                "model": model,
                "dataset": DATASET,
                "dataset_label": DATASET_LABEL,
                "n_anchors": len(sub),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "loss": loss,
                "score": 100.0 / (1.0 + loss),
            }
        )

    write_csv(
        output / "main_anchor_detail_example.csv",
        detail_rows,
        ["history_count", "anchor", "dataset", "dataset_label", "model", "seq_len", "pred_len", "mae", "mse", "rmse", "loss", "result_dir"],
    )
    write_csv(
        output / "main_table_example.csv",
        table_rows,
        ["model", "dataset", "dataset_label", "n_anchors", "mae", "mse", "rmse", "loss", "score"],
    )
    return table_rows, detail_rows


def run_hist_analysis(data_root: Path, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MAIN_MODELS:
        row: dict[str, object] = {"Foundation Model": model}
        for hist in [2, 5]:
            losses = []
            n_anchor_rows = 0
            for anchor in ANCHORS:
                cache_path = data_root / "context60" / f"hist{hist}" / anchor / DATASET / "qar_compact_shiftN80.npz"
                context, true, _, pred_len = load_forecast_cache(cache_path)
                pred = FORECASTERS[model](context, pred_len)
                losses.append(metrics_from_arrays(pred, true)["loss"])
                n_anchor_rows += 1
            row[f"hist{hist}"] = mean(losses)
            row[f"hist{hist}_anchor_rows"] = n_anchor_rows
        rows.append(row)
    write_csv(
        output / "hist_analysis_example.csv",
        rows,
        ["Foundation Model", "hist2", "hist2_anchor_rows", "hist5", "hist5_anchor_rows"],
    )
    return rows


def run_uni_multi(data_root: Path, output: Path) -> list[dict[str, object]]:
    detail_rows: list[dict[str, object]] = []
    for anchor in ANCHORS:
        cache_path = data_root / "context60" / "hist5" / anchor / DATASET / "qar_compact_shiftN80.npz"
        context, true, _, pred_len = load_forecast_cache(cache_path)
        n_features = true.shape[2]
        for model in UNI_MULTI_MODELS:
            pred_multi = FORECASTERS[model](context, pred_len)
            multi_metric = metrics_from_arrays(pred_multi, true)

            stitched = []
            for feature_idx in range(n_features):
                c_context = context[:, :, feature_idx : feature_idx + 1]
                c_pred = FORECASTERS[model](c_context, pred_len)
                stitched.append(c_pred)
            pred_uni = np.concatenate(stitched, axis=2)
            uni_metric = metrics_from_arrays(pred_uni, true)
            detail_rows.append(
                {
                    "dataset": DATASET,
                    "dataset_label": DATASET_LABEL,
                    "anchor": anchor,
                    "model": model,
                    "n_features": n_features,
                    "univariate_stitched_loss": uni_metric["loss"],
                    "multivariate_direct_loss": multi_metric["loss"],
                    "univariate_stitched_mae": uni_metric["mae"],
                    "multivariate_direct_mae": multi_metric["mae"],
                    "univariate_stitched_rmse": uni_metric["rmse"],
                    "multivariate_direct_rmse": multi_metric["rmse"],
                }
            )

    rows: list[dict[str, object]] = []
    for model in UNI_MULTI_MODELS:
        sub = [row for row in detail_rows if row["model"] == model]
        rows.append(
            {
                "model": model,
                "dataset": DATASET,
                "n_anchors": len(sub),
                "n_features": int(sub[0]["n_features"]) if sub else 0,
                "univariate_stitched_loss": mean([float(row["univariate_stitched_loss"]) for row in sub]),
                "multivariate_direct_loss": mean([float(row["multivariate_direct_loss"]) for row in sub]),
                "univariate_stitched_mae": mean([float(row["univariate_stitched_mae"]) for row in sub]),
                "multivariate_direct_mae": mean([float(row["multivariate_direct_mae"]) for row in sub]),
                "univariate_stitched_rmse": mean([float(row["univariate_stitched_rmse"]) for row in sub]),
                "multivariate_direct_rmse": mean([float(row["multivariate_direct_rmse"]) for row in sub]),
            }
        )

    write_csv(
        output / "uni_multi_detail_example.csv",
        detail_rows,
        [
            "dataset",
            "dataset_label",
            "anchor",
            "model",
            "n_features",
            "univariate_stitched_loss",
            "multivariate_direct_loss",
            "univariate_stitched_mae",
            "multivariate_direct_mae",
            "univariate_stitched_rmse",
            "multivariate_direct_rmse",
        ],
    )
    write_csv(
        output / "uni_multi_example.csv",
        rows,
        [
            "model",
            "dataset",
            "n_anchors",
            "n_features",
            "univariate_stitched_loss",
            "multivariate_direct_loss",
            "univariate_stitched_mae",
            "multivariate_direct_mae",
            "univariate_stitched_rmse",
            "multivariate_direct_rmse",
        ],
    )
    return rows


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision_1 = div(tp, tp + fp)
    recall_1 = div(tp, tp + fn)
    f1_1 = div(2.0 * precision_1 * recall_1, precision_1 + recall_1)
    precision_0 = div(tn, tn + fn)
    recall_0 = div(tn, tn + fp)
    f1_0 = div(2.0 * precision_0 * recall_0, precision_0 + recall_0)
    return {
        "accuracy": div(tp + tn, len(y_true)),
        "macro_f1": 0.5 * (f1_0 + f1_1),
        "precision_fault": precision_1,
        "recall_fault": recall_1,
        "f1_fault": f1_1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_classification(data_root: Path, output: Path) -> list[dict[str, object]]:
    cache_path = data_root / "classification" / DATASET / "qar_compact_shiftN80.npz"
    cache = np.load(cache_path, allow_pickle=False)
    x = np.asarray(cache["x"], dtype=np.float64)
    labels = np.asarray(cache["labels"], dtype=int)
    score = x[:, 55:, 0].mean(axis=1) - x[:, :25, 0].mean(axis=1)

    train_end = max(2, int(round(0.7 * len(labels))))
    test_start = max(train_end, int(round(0.8 * len(labels))))
    train_score = score[:train_end]
    train_labels = labels[:train_end]
    normal_mean = float(train_score[train_labels == 0].mean())
    fault_mean = float(train_score[train_labels == 1].mean())
    threshold = 0.5 * (normal_mean + fault_mean)
    test_labels = labels[test_start:]
    pred = (score[test_start:] > threshold).astype(int)
    metrics = binary_metrics(test_labels, pred)
    row = {
        "dataset": DATASET,
        "dataset_label": DATASET_LABEL,
        "model": "BleedPressureThreshold",
        "feature": "bleed_air_pressure",
        "train_samples": train_end,
        "test_samples": int(len(test_labels)),
        "threshold": threshold,
        **metrics,
    }
    write_csv(
        output / "classification_example.csv",
        [row],
        [
            "dataset",
            "dataset_label",
            "model",
            "feature",
            "train_samples",
            "test_samples",
            "threshold",
            "accuracy",
            "macro_f1",
            "precision_fault",
            "recall_fault",
            "f1_fault",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("examples/qar_context60/data"))
    parser.add_argument("--output", type=Path, default=Path("examples/qar_context60/outputs/smoke"))
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before running.")
    args = parser.parse_args()

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    main_rows, detail_rows = run_main_table(args.data_root, args.output)
    hist_rows = run_hist_analysis(args.data_root, args.output)
    uni_multi_rows = run_uni_multi(args.data_root, args.output)
    cls_rows = run_classification(args.data_root, args.output)

    summary = {
        "data_root": args.data_root.as_posix(),
        "output": args.output.as_posix(),
        "main_rows": len(main_rows),
        "main_anchor_rows": len(detail_rows),
        "hist_rows": len(hist_rows),
        "uni_multi_rows": len(uni_multi_rows),
        "classification_rows": len(cls_rows),
        "protocol": "80*hist + 60 context -> 20 target",
    }
    (args.output / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
