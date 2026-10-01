#!/usr/bin/env python3
"""Run tiny end-to-end TSLib smoke experiments on representative QAR caches.

Unlike `run_evaluation_examples.py`, this script calls the repository entry
point `run.py`.  It is intended to verify that a fresh checkout can execute the
main task families with a small compact cache:

* long-term forecasting with the context60 history protocol;
* classification with a QAR compact cache;
* one-class anomaly detection with a QAR compact cache.

The smoke uses DLinear, one training epoch, CPU by default, and the synthetic
representative data from `make_representative_data.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


TASK_CHOICES = ("forecast", "classification", "anomaly")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def feature_count(cache_path: Path) -> int:
    cache = np.load(cache_path, allow_pickle=False)
    return int(cache["x"].shape[2])


def run_logged(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        proc = subprocess.run(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def common_model_args(args: argparse.Namespace, enc_in: int) -> list[str]:
    base = [
        "--model", "DLinear",
        "--enc_in", str(enc_in),
        "--dec_in", str(enc_in),
        "--c_out", str(enc_in),
        "--d_model", "16",
        "--d_ff", "32",
        "--e_layers", "1",
        "--d_layers", "1",
        "--n_heads", "2",
        "--top_k", "2",
        "--batch_size", str(args.batch_size),
        "--train_epochs", str(args.train_epochs),
        "--patience", "1",
        "--learning_rate", "0.001",
        "--num_workers", "0",
        "--itr", "1",
        "--des", "qar_context60_tslib_smoke",
        "--checkpoints", str(args.output / "checkpoints"),
    ]
    if args.cpu:
        base.append("--no_use_gpu")
    return base


def forecast_command(args: argparse.Namespace) -> tuple[str, list[str]]:
    cache_root = args.data_root / "context60" / "hist5" / "predict_2_3" / "dataset_demo"
    enc_in = feature_count(cache_root / "qar_compact_shiftN80.npz")
    command = [
        args.python,
        "-u",
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", "qar_context60_demo_forecast",
        "--data", "QAR_forecast",
        "--root_path", str(cache_root),
        "--data_path", "qar_compact_shiftN80.npz",
        "--features", "M",
        "--target", "bleed_air_pressure",
        "--freq", "h",
        "--seq_len", "460",
        "--label_len", "20",
        "--pred_len", "20",
    ]
    command.extend(common_model_args(args, enc_in))
    return "forecast", command


def classification_command(args: argparse.Namespace) -> tuple[str, list[str]]:
    cache_root = args.data_root / "classification" / "dataset_demo"
    enc_in = feature_count(cache_root / "qar_compact_shiftN80.npz")
    command = [
        args.python,
        "-u",
        "run.py",
        "--task_name", "classification",
        "--is_training", "1",
        "--model_id", "qar_context60_demo_classification",
        "--data", "QAR_shift",
        "--root_path", str(cache_root),
        "--phase_a_shift", "-80",
        "--seq_len", "80",
        "--class_weight", "balanced",
        "--early_stop_metric", "macro_f1",
        "--save_epoch_checkpoints", "0",
    ]
    command.extend(common_model_args(args, enc_in))
    return "classification", command


def anomaly_command(args: argparse.Namespace) -> tuple[str, list[str]]:
    cache_root = args.data_root / "classification" / "dataset_demo"
    enc_in = feature_count(cache_root / "qar_compact_shiftN80.npz")
    command = [
        args.python,
        "-u",
        "run.py",
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--model_id", "qar_context60_demo_anomaly",
        "--data", "QAR_anomaly",
        "--root_path", str(cache_root),
        "--data_path", "qar_compact_shiftN80.npz",
        "--features", "M",
        "--seq_len", "80",
        "--pred_len", "0",
        "--anomaly_threshold_source", "val",
        "--anomaly_threshold_percentile", "95",
        "--anomaly_score_direction", "high",
        "--anomaly_level", "window",
    ]
    command.extend(common_model_args(args, enc_in))
    return "anomaly", command


def ensure_data(args: argparse.Namespace, root: Path) -> None:
    if (args.data_root / "classification" / "dataset_demo" / "qar_compact_shiftN80.npz").exists():
        return
    command = [
        args.python,
        "examples/qar_context60/make_representative_data.py",
        "--clean",
        "--output",
        str(args.data_root),
    ]
    status = run_logged(command, root, args.output / "logs" / "make_representative_data.log", os.environ.copy())
    if status != 0:
        raise RuntimeError("representative data generation failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("examples/qar_context60/data"))
    parser.add_argument("--output", type=Path, default=Path("examples/qar_context60/outputs/tslib_smoke"))
    parser.add_argument("--tasks", nargs="+", choices=TASK_CHOICES, default=list(TASK_CHOICES))
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    if not args.data_root.is_absolute():
        args.data_root = root / args.data_root
    if not args.output.is_absolute():
        args.output = root / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_data(args, root)

    builders = {
        "forecast": forecast_command,
        "classification": classification_command,
        "anomaly": anomaly_command,
    }
    rows = []
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for task in args.tasks:
        label, command = builders[task](args)
        log_path = args.output / "logs" / f"{label}.log"
        status = run_logged(command, root, log_path, env)
        rows.append({"task": label, "status": status, "log": log_path.as_posix()})
        print(f"{label}: status={status} log={log_path.as_posix()}", flush=True)
        if status != 0 and not args.keep_going:
            break

    summary = {
        "python": args.python,
        "data_root": args.data_root.as_posix(),
        "output": args.output.as_posix(),
        "tasks": rows,
    }
    summary_path = args.output / "tslib_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if any(row["status"] != 0 for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
