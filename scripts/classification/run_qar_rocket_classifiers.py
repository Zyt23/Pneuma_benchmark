#!/usr/bin/env python
"""Run ROCKET-family classifiers on QAR compact classification caches.

This script is intentionally separate from the neural-network classification
entry point because MiniROCKET/MultiROCKET are sklearn-style feature transforms
rather than torch models.  It reuses the same compact cache format and either
the cached fixed split flags, when present, or the same per-class chronological
70/10/20 split used by ``QARFlightDatasetShift``.

Input cache:
    <compact_root>/<dataset>/qar_compact_shiftN80.npz

Expected arrays:
    x:      (N, T, C)
    labels: (N,)
    sources/time_keys optional, used only for chronological sorting.
    split_flags optional, 0/1/2 or TRAIN/VAL/TEST, used to force a fixed split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.engineering_metrics import binary_metrics_at_recall, binary_metrics_at_threshold
from utils.benchmark_metrics import precision_at_recall


DATASETS = [
    "dataset5",
    "dataset6",
    "dataset7",
    "dataset8",
    "dataset8-1",
    "dataset10",
    "dataset11",
    "dataset13",
    "dataset14",
]


def split_bounds_keep_test(n: int) -> tuple[int, int]:
    n = int(n)
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = int(n * 0.7)
    # Historical runners used different rounding conventions. Record the
    # selected convention instead of claiming all three give identical splits.
    boundary = os.environ.get('QAR_SPLIT_BOUNDARY', 'rocket_float')
    if boundary == 'integer':
        train_end, val_end = 7 * n // 10, 8 * n // 10
    elif boundary == 'legacy_float':
        val_end = int(n * (0.7 + 0.1))
    elif boundary == 'floor_sum':
        val_end = train_end + int(n * 0.1)
    elif boundary == 'rocket_float':
        val_end = int(n * 0.8)
    else:
        raise ValueError(f'Unknown split boundary: {boundary}')
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return train_end, val_end


def split_indices_per_class(labels: np.ndarray, keys: np.ndarray, flag: str) -> np.ndarray:
    flag = flag.upper()
    selected: list[int] = []
    labels = labels.astype(int, copy=False)
    for label in sorted(np.unique(labels).astype(int).tolist()):
        group = np.flatnonzero(labels == label)
        group = group[np.argsort(keys[group], kind="mergesort")]
        train_end, val_end = split_bounds_keep_test(len(group))
        if flag == "TRAIN":
            selected.extend(group[:train_end].tolist())
        elif flag in {"VAL", "VALI", "VALID", "VALIDATION"}:
            selected.extend(group[train_end:val_end].tolist())
        elif flag == "TEST":
            selected.extend(group[val_end:].tolist())
        else:
            raise ValueError(f"Unsupported split flag: {flag}")
    selected_arr = np.asarray(selected, dtype=np.int64)
    selected_arr = selected_arr[np.argsort(keys[selected_arr], kind="mergesort")]
    return selected_arr


def split_indices_by_flags(split_flags: np.ndarray, flag: str, keys: np.ndarray) -> np.ndarray:
    flag_upper = flag.upper()
    if flag_upper == "TRAIN":
        wanted = {"0", "TRAIN"}
    elif flag_upper in {"VAL", "VALI", "VALID", "VALIDATION"}:
        wanted = {"1", "VAL", "VALI", "VALID", "VALIDATION"}
    elif flag_upper == "TEST":
        wanted = {"2", "TEST"}
    else:
        raise ValueError(f"Unsupported split flag: {flag}")

    selected = []
    for i, raw in enumerate(split_flags):
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="ignore")
        else:
            text = str(raw)
        if text.strip().upper() in wanted:
            selected.append(i)
    selected_arr = np.asarray(selected, dtype=np.int64)
    if selected_arr.size:
        selected_arr = selected_arr[np.argsort(keys[selected_arr], kind="mergesort")]
    return selected_arr


def load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict]:
    cache = np.load(cache_path, allow_pickle=False)
    x = cache["x"].astype(np.float32, copy=False)
    y = cache["labels"].astype(np.int64, copy=False)
    if "time_keys" in cache.files:
        keys = cache["time_keys"].astype(np.int64, copy=False)
    else:
        keys = np.arange(y.shape[0], dtype=np.int64)
    split_flags = cache["split_flags"] if "split_flags" in cache.files else None
    meta = {
        "shape": list(x.shape),
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "cache_path": str(cache_path),
        "uses_split_flags": split_flags is not None,
    }
    return x, y, keys, split_flags, meta


def make_transform(model: str, n_kernels: int, n_jobs: int, random_state: int):
    from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

    if model == "MiniROCKET":
        return MiniRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)
    if model == "MultiROCKET":
        # Aeon MultiRocket output dimensionality is roughly n_kernels *
        # n_features_per_kernel, so keep this separately configurable by using
        # the same public n_kernels knob.
        return MultiRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)
    raise ValueError(f"Unsupported model: {model}")


def run_one(
    dataset: str,
    model: str,
    compact_root: Path,
    output_dir: Path,
    n_kernels: int,
    n_jobs: int,
    random_state: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict:
    cache_path = compact_root / dataset / "qar_compact_shiftN80.npz"
    x, y, keys, split_flags, meta = load_cache(cache_path)
    if split_flags is not None:
        train_idx = split_indices_by_flags(split_flags, "TRAIN", keys)
        val_idx = split_indices_by_flags(split_flags, "VAL", keys)
        test_idx = split_indices_by_flags(split_flags, "TEST", keys)
    else:
        train_idx = split_indices_per_class(y, keys, "TRAIN")
        val_idx = split_indices_per_class(y, keys, "VAL")
        test_idx = split_indices_per_class(y, keys, "TEST")

    rng = np.random.default_rng(random_state)
    if max_train_samples and len(train_idx) > max_train_samples:
        train_idx = np.sort(rng.choice(train_idx, size=max_train_samples, replace=False))
    if max_test_samples and len(test_idx) > max_test_samples:
        test_idx = np.sort(rng.choice(test_idx, size=max_test_samples, replace=False))

    # Aeon collection transformers expect (n_cases, n_channels, n_timepoints).
    x_train = np.transpose(x[train_idx], (0, 2, 1))
    x_val = np.transpose(x[val_idx], (0, 2, 1))
    x_test = np.transpose(x[test_idx], (0, 2, 1))
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    transform = make_transform(model, n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)
    clf = make_pipeline(
        transform,
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"),
    )
    clf.fit(x_train, y_train)
    val_pred = clf.predict(x_val).astype(int)
    pred = clf.predict(x_test).astype(int)
    if hasattr(clf, "decision_function"):
        val_positive_scores = np.asarray(clf.decision_function(x_val), dtype=np.float32).reshape(-1)
        positive_scores = np.asarray(clf.decision_function(x_test), dtype=np.float32).reshape(-1)
    elif hasattr(clf, "predict_proba"):
        val_positive_scores = np.asarray(clf.predict_proba(x_val), dtype=np.float32)[:, 1]
        positive_scores = np.asarray(clf.predict_proba(x_test), dtype=np.float32)[:, 1]
    else:
        val_positive_scores = val_pred.astype(np.float32)
        positive_scores = pred.astype(np.float32)
    val_engineering = binary_metrics_at_recall(y_val, val_positive_scores, target_recall=0.10)
    if np.isfinite(val_engineering["threshold"]):
        engineering = binary_metrics_at_threshold(
            y_test,
            positive_scores,
            threshold=val_engineering["threshold"],
            target_recall=0.10,
        )
        threshold_source = "val"
    else:
        engineering = binary_metrics_at_recall(y_test, positive_scores, target_recall=0.10)
        threshold_source = "test"

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=labels).ravel()
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    true_counts = np.bincount(y_test.astype(int), minlength=2).astype(int).tolist()
    pred_counts = np.bincount(pred.astype(int), minlength=2).astype(int).tolist()

    row = {
        "dataset": dataset,
        "model": model,
        "acc": float(acc),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "true_counts": json.dumps(true_counts, ensure_ascii=False),
        "pred_counts": json.dumps(pred_counts, ensure_ascii=False),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "n_kernels": int(n_kernels),
        "n_jobs": int(n_jobs),
        "cache_shape": json.dumps(meta["shape"]),
        "uses_split_flags": bool(meta["uses_split_flags"]),
        "engineering_target_recall": engineering["target_recall"],
        "engineering_threshold_source": threshold_source,
        "engineering_threshold": engineering["threshold"],
        "engineering_precision_at_recall": engineering["precision_at_recall"],
        "engineering_recall_at_threshold": engineering["recall_at_threshold"],
        "engineering_f1_at_recall": engineering["f1_at_recall"],
        "engineering_alert_rate": engineering["alert_rate"],
        "engineering_TN": engineering["TN_at_recall"],
        "engineering_FP": engineering["FP_at_recall"],
        "engineering_FN": engineering["FN_at_recall"],
        "engineering_TP": engineering["TP_at_recall"],
        "validation_engineering_threshold": val_engineering["threshold"],
        "validation_engineering_precision_at_recall": val_engineering["precision_at_recall"],
        "validation_engineering_recall_at_threshold": val_engineering["recall_at_threshold"],
        "validation_engineering_f1_at_recall": val_engineering["f1_at_recall"],
    }

    result_dir = output_dir / "results" / dataset / model
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result_classification.txt").write_text(
        "\n".join(
            [
                f"dataset:{dataset}",
                f"model:{model}",
                f"acc:{acc}",
                f"accuracy:{acc}",
                f"macro_f1:{macro_f1}",
                f"weighted_f1:{weighted_f1}",
                f"true counts:{true_counts}",
                f"pred counts:{pred_counts}",
                f"confusion matrix:{[[int(tn), int(fp)], [int(fn), int(tp)]]}",
                f"train_samples:{len(train_idx)}",
                f"val_samples:{len(val_idx)}",
                f"test_samples:{len(test_idx)}",
                f"n_kernels:{n_kernels}",
                f"cache_shape:{meta['shape']}",
                f"uses_split_flags:{meta['uses_split_flags']}",
                f"engineering threshold source:{threshold_source}",
                f"engineering target recall:{engineering['target_recall']}",
                f"engineering threshold:{engineering['threshold']}",
                f"engineering precision at recall:{engineering['precision_at_recall']}",
                f"engineering recall at threshold:{engineering['recall_at_threshold']}",
                f"engineering f1 at recall:{engineering['f1_at_recall']}",
                f"engineering alert rate:{engineering['alert_rate']}",
                f"engineering TN:{engineering['TN_at_recall']}",
                f"engineering FP:{engineering['FP_at_recall']}",
                f"engineering FN:{engineering['FN_at_recall']}",
                f"engineering TP:{engineering['TP_at_recall']}",
                f"validation engineering threshold:{val_engineering['threshold']}",
                f"validation engineering precision at recall:{val_engineering['precision_at_recall']}",
                f"validation engineering recall at threshold:{val_engineering['recall_at_threshold']}",
                f"validation engineering f1 at recall:{val_engineering['f1_at_recall']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        result_dir / "classification_scores.npz",
        positive_scores=positive_scores.astype(np.float32),
        labels=y_test.astype(np.int8),
        predictions=pred.astype(np.int8),
    )
    np.savez_compressed(
        result_dir / "classification_val_scores.npz",
        positive_scores=val_positive_scores.astype(np.float32),
        labels=y_val.astype(np.int8),
        predictions=val_pred.astype(np.int8),
    )
    ranking = precision_at_recall(y_test, positive_scores)
    (result_dir / 'ranking_metrics.json').write_text(
        json.dumps(ranking, indent=2), encoding='utf-8')
    (result_dir / 'protocol.json').write_text(json.dumps({
        'seed': random_state, 'split_boundary': os.environ.get('QAR_SPLIT_BOUNDARY', 'rocket_float'),
        'uses_split_flags': split_flags is not None,
        'train_indices': train_idx.tolist(), 'validation_indices': val_idx.tolist(),
        'test_indices': test_idx.tolist(),
    }, indent=2), encoding='utf-8')
    return row


def parse_list(text: str, default: Iterable[str]) -> list[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact_root", type=Path, default=Path("datasetall_tsfile_compact_custom_cls_chrono_20260711"))
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--models", default="MiniROCKET MultiROCKET")
    parser.add_argument("--output_dir", type=Path, default=Path("experiment_artifacts/QAR_benchmark_matrix_20260714/rocket_classification"))
    parser.add_argument("--n_kernels", type=int, default=10000)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--random_state", type=int, default=20260714)
    parser.add_argument('--split-boundary', choices=['rocket_float', 'legacy_float', 'integer', 'floor_sum'],
                        default=os.environ.get('QAR_SPLIT_BOUNDARY', 'rocket_float'))
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    args = parser.parse_args()
    os.environ['QAR_SPLIT_BOUNDARY'] = args.split_boundary

    datasets = parse_list(args.datasets, DATASETS)
    models = parse_list(args.models, ["MiniROCKET", "MultiROCKET"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_path = args.output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("dataset\tmodel\tstatus\tmessage\n")
        for dataset in datasets:
            for model in models:
                try:
                    print(f"START dataset={dataset} model={model}", flush=True)
                    row = run_one(
                        dataset=dataset,
                        model=model,
                        compact_root=args.compact_root,
                        output_dir=args.output_dir,
                        n_kernels=args.n_kernels,
                        n_jobs=args.n_jobs,
                        random_state=args.random_state,
                        max_train_samples=args.max_train_samples or None,
                        max_test_samples=args.max_test_samples or None,
                    )
                    rows.append(row)
                    summary.write(f"{dataset}\t{model}\t0\tOK\n")
                    summary.flush()
                    print(f"DONE dataset={dataset} model={model} acc={row['acc']:.6f} macro_f1={row['macro_f1']:.6f}", flush=True)
                except Exception as exc:  # keep sweep running
                    summary.write(f"{dataset}\t{model}\t1\t{type(exc).__name__}: {exc}\n")
                    summary.flush()
                    print(f"FAIL dataset={dataset} model={model}: {type(exc).__name__}: {exc}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# QAR ROCKET classification\n\n")
        f.write(f"- Compact root: `{args.compact_root}`\n")
        f.write(f"- Split: cached `split_flags` if present; otherwise per-class chronological 70/10/20 with `{args.split_boundary}` rounding.\n")
        f.write(f"- Models: `{', '.join(models)}`\n")
        f.write(f"- n_kernels: `{args.n_kernels}`; n_jobs: `{args.n_jobs}`\n")
        f.write("- Classifier: aeon ROCKET transform + StandardScaler(with_mean=False) + RidgeClassifierCV(class_weight='balanced').\n")
    if len(rows) != len(datasets) * len(models):
        raise SystemExit('One or more ROCKET runs failed; inspect summary.tsv')


if __name__ == "__main__":
    main()
