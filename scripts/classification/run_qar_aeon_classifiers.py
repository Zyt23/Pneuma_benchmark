#!/usr/bin/env python3
"""Run aeon HIVE-COTE2 or InceptionTime with an explicit QAR split convention."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(os.environ.get("QAR_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classification.run_qar_rocket_classifiers import (  # noqa: E402
    load_cache,
    split_indices_by_flags,
    split_indices_per_class,
)
from utils.benchmark_metrics import precision_at_recall


DATASETS = [
    "dataset5", "dataset6", "dataset7", "dataset8", "dataset8-1",
    "dataset10", "dataset11", "dataset13", "dataset14",
]


def parse_list(text: str) -> list[str]:
    return [item for item in text.replace(",", " ").split() if item]


def make_classifier(args: argparse.Namespace):
    if args.model == "HIVE-COTE2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(
            time_limit_in_minutes=args.contract_minutes,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
    if args.model == "InceptionTime":
        from aeon.classification.deep_learning import InceptionTimeClassifier

        return InceptionTimeClassifier(
            n_classifiers=args.n_classifiers,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.seed,
            verbose=bool(args.verbose),
        )
    raise ValueError(args.model)


def score_rows(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    try:
        auroc = roc_auc_score(y_true, probability)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, probability)
    except ValueError:
        auprc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "fault_f1": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_one(args: argparse.Namespace, dataset: str) -> dict:
    cache_path = args.compact_root / dataset / "qar_compact_shiftN80.npz"
    x, y, keys, split_flags, meta = load_cache(cache_path)
    if split_flags is None:
        train_idx = split_indices_per_class(y, keys, "TRAIN")
        val_idx = split_indices_per_class(y, keys, "VAL")
        test_idx = split_indices_per_class(y, keys, "TEST")
    else:
        train_idx = split_indices_by_flags(split_flags, "TRAIN", keys)
        val_idx = split_indices_by_flags(split_flags, "VAL", keys)
        test_idx = split_indices_by_flags(split_flags, "TEST", keys)

    # aeon collections use (cases, channels, timepoints).  No metadata or
    # resampling is introduced. Split rounding remains an explicit option.
    # HIVE-COTE2's shapelet component compiles a float64 numba kernel; feeding
    # the float32 cache directly can create mixed-dtype shapelets in aeon.
    dtype = np.float64 if args.model == "HIVE-COTE2" else np.float32
    x_train = np.ascontiguousarray(x[train_idx].transpose(0, 2, 1), dtype=dtype)
    x_test = np.ascontiguousarray(x[test_idx].transpose(0, 2, 1), dtype=dtype)
    y_train = y[train_idx]
    y_test = y[test_idx]
    classifier = make_classifier(args)
    started = time.time()
    classifier.fit(x_train, y_train)
    fit_seconds = time.time() - started
    prediction = np.asarray(classifier.predict(x_test), dtype=np.int64)
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(classifier.predict_proba(x_test), dtype=np.float64)
        classes = np.asarray(getattr(classifier, "classes_", [0, 1]))
        positive = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    else:
        positive = prediction.astype(np.float64)
    elapsed = time.time() - started

    row = {
        "dataset": dataset,
        "model": args.model,
        "seed": args.seed,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "cache_shape": json.dumps(meta["shape"]),
        "fit_seconds": fit_seconds,
        "elapsed_seconds": elapsed,
        "contract_minutes": args.contract_minutes if args.model == "HIVE-COTE2" else "",
        "n_jobs": args.n_jobs,
        "epochs": args.epochs if args.model == "InceptionTime" else "",
        "n_classifiers": args.n_classifiers if args.model == "InceptionTime" else "",
        **score_rows(y_test, prediction, positive),
    }
    result_dir = args.output_dir / "results" / dataset / args.model
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        result_dir / "scores.npz",
        labels=y_test.astype(np.int8),
        predictions=prediction.astype(np.int8),
        positive_scores=positive.astype(np.float32),
    )
    (result_dir / 'ranking_metrics.json').write_text(
        json.dumps(precision_at_recall(y_test, positive), indent=2), encoding='utf-8')
    (result_dir / 'protocol.json').write_text(json.dumps({
        'seed': args.seed, 'split_boundary': args.split_boundary,
        'uses_split_flags': split_flags is not None,
        'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist(),
    }, indent=2), encoding='utf-8')
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--model", choices=["HIVE-COTE2", "InceptionTime"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--split-boundary', choices=['rocket_float', 'legacy_float', 'integer', 'floor_sum'],
                        default=os.environ.get('QAR_SPLIT_BOUNDARY', 'rocket_float'))
    parser.add_argument("--contract-minutes", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-classifiers", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    os.environ['QAR_SPLIT_BOUNDARY'] = args.split_boundary
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_path = args.output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as status:
        status.write("dataset\tmodel\tstatus\tmessage\n")
        for dataset in parse_list(args.datasets):
            try:
                print(f"START {dataset} {args.model} seed={args.seed}", flush=True)
                row = run_one(args, dataset)
                rows.append(row)
                status.write(f"{dataset}\t{args.model}\t0\tOK\n")
                status.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
            except Exception as exc:
                status.write(
                    f"{dataset}\t{args.model}\t1\t{type(exc).__name__}: "
                    f"{str(exc).replace(chr(9), ' ').replace(chr(10), ' ')}\n"
                )
                status.flush()
                raise
    if rows:
        with (args.output_dir / "all_metrics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
