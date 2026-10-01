#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect QAR anomaly-detection metrics into dataset/model tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import pandas as pd


DEFAULT_MODELS = ["KANAD", "AnomalyTransformer", "TranAD", "USAD", "OmniAnomaly"]

DATASET_NAMES = {
    "dataset5": "320 takeoff pressure duct fault",
    "dataset6": "320 HPV valve fault",
    "dataset7": "320 PRV valve fault",
    "dataset8": "320 duct leakage",
    "dataset8-1": "320 PRV valve leakage",
    "dataset10": "321 pressure duct fault",
    "dataset11": "321 duct leakage",
    "dataset13": "787 air-compressor fault",
    "dataset14": "777 PRSOV fault",
}


def parse_setting(setting: str, run_tag: str, models: list[str]) -> tuple[str, str] | None:
    prefixes = [
        f"anomaly_detection_{run_tag}_",
        f"forecast_anomaly_detection_{run_tag}_",
    ]
    rest = None
    for prefix in prefixes:
        if setting.startswith(prefix):
            rest = setting[len(prefix):]
            break
    if rest is None:
        return None

    for model in sorted(models, key=len, reverse=True):
        markers = [
            f"_{model}_QAR_anomaly_",
            f"_{model}_QAR_forecast_anomaly_",
            f"_{model}_QAR_forecast_head_anomaly_",
            f"_{model}_QAR_forecast_",
        ]
        for marker in markers:
            if marker in rest:
                return rest.split(marker, 1)[0], model
    return None


def read_metric(path: Path) -> dict | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def float_value(value: object) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def int_value(value: object) -> int | None:
    number = float_value(value)
    if math.isnan(number):
        return None
    return int(number)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if math.isnan(value) else f"{value:.6f}"
            elif value is None:
                value = ""
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--run_tag")
    run_group.add_argument(
        "--run_tags",
        nargs="+",
        help="Collect several model-specific run tags into one table.",
    )
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_NAMES))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument(
        "--metric_file_name",
        default="anomaly_metrics.csv",
        help="Metric filename, e.g. anomaly_metrics.csv or forecast_anomaly_metrics.csv.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    run_tags = args.run_tags or [args.run_tag]
    records: list[dict] = []
    for metric_path in results_root.glob(f"*/{args.metric_file_name}"):
        row = read_metric(metric_path)
        if not row:
            continue
        parsed = None
        matched_run_tag = None
        for run_tag in run_tags:
            parsed = parse_setting(row.get("setting", metric_path.parent.name), run_tag, args.models)
            if parsed is None:
                parsed = parse_setting(metric_path.parent.name, run_tag, args.models)
            if parsed is not None:
                matched_run_tag = run_tag
                break
        if parsed is None:
            continue
        dataset, model = parsed
        if dataset not in args.datasets or model not in args.models:
            continue
        records.append(
            {
                "dataset": dataset,
                "fault": DATASET_NAMES.get(dataset, ""),
                "model": model,
                "run_tag": matched_run_tag,
                "accuracy": float_value(row.get("accuracy")),
                "balanced_accuracy": float_value(row.get("balanced_accuracy")),
                "precision": float_value(row.get("precision")),
                "recall": float_value(row.get("recall")),
                "f1": float_value(row.get("f1")),
                "macro_f1": float_value(row.get("macro_f1")),
                "roc_auc": float_value(row.get("roc_auc")),
                "pr_auc": float_value(row.get("pr_auc")),
                "raw_roc_auc": float_value(row.get("raw_roc_auc")),
                "raw_pr_auc": float_value(row.get("raw_pr_auc")),
                "true_counts": row.get("true_counts", ""),
                "pred_counts": row.get("pred_counts", ""),
                "TN": int_value(row.get("TN")),
                "FP": int_value(row.get("FP")),
                "FN": int_value(row.get("FN")),
                "TP": int_value(row.get("TP")),
                "threshold": float_value(row.get("threshold")),
                "threshold_source": row.get("threshold_source", ""),
                "threshold_percentile": float_value(row.get("threshold_percentile")),
                "level": row.get("level", row.get("score", "")),
                "score_direction": row.get("score_direction", ""),
                "normal_score_mean": float_value(row.get("normal_score_mean")),
                "fault_score_mean": float_value(row.get("fault_score_mean")),
                "normal_score_median": float_value(row.get("normal_score_median")),
                "fault_score_median": float_value(row.get("fault_score_median")),
                "normal_score_p95": float_value(row.get("normal_score_p95")),
                "fault_score_p95": float_value(row.get("fault_score_p95")),
                "threshold_val_accuracy": float_value(row.get("threshold_val_accuracy")),
                "threshold_val_precision": float_value(row.get("threshold_val_precision")),
                "threshold_val_recall": float_value(row.get("threshold_val_recall")),
                "threshold_val_f1": float_value(row.get("threshold_val_f1")),
                "setting": row.get("setting", metric_path.parent.name),
                "metric_file": str(metric_path),
            }
        )

    if not records:
        raise SystemExit(f"No anomaly metrics found for run_tags={run_tags!r} under {results_root}")

    df = pd.DataFrame(records)
    df["dataset_order"] = df["dataset"].map({d: i for i, d in enumerate(args.datasets)})
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(args.models)})
    df = df.sort_values(["dataset_order", "model_order"]).drop(columns=["dataset_order", "model_order"])

    all_csv = output_dir / "all_anomaly_metrics.csv"
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    for dataset, sub in df.groupby("dataset", sort=False):
        sub.to_csv(tables_dir / f"{dataset}_anomaly_metrics.csv", index=False, encoding="utf-8-sig")

    readme = output_dir / "README.md"
    with readme.open("w", encoding="utf-8") as f:
        f.write("# QAR anomaly detection\n\n")
        f.write("- Training uses normal-class windows only by default.\n")
        f.write("- Thresholds are selected on validation data, not on the test set.\n")
        f.write("- Test metrics are reported at the window level.\n\n")
        f.write(f"- run_tags: `{', '.join(run_tags)}`\n")
        f.write(f"- all metrics: `{all_csv.name}`\n\n")
        for dataset in args.datasets:
            sub = df[df["dataset"] == dataset]
            if sub.empty:
                continue
            f.write(f"## {dataset}: {DATASET_NAMES.get(dataset, '')}\n\n")
            cols = [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "true_counts",
                "pred_counts",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
            f.write(markdown_table(sub[cols].to_dict("records"), cols))
            f.write("\n\n")

    xlsx = output_dir / "QAR_anomaly_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_anomaly_metrics", index=False)
        for dataset in args.datasets:
            sub = df[df["dataset"] == dataset]
            if not sub.empty:
                safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", dataset)[:31]
                sub.to_excel(writer, sheet_name=safe_name, index=False)

    print(f"Wrote {all_csv}")
    print(f"Wrote {readme}")
    print(f"Wrote {xlsx}")


if __name__ == "__main__":
    main()
