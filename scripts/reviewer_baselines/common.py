from __future__ import annotations

import math
from pathlib import Path

import numpy as np


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

DATASET_LABELS = {
    "dataset5": "320-DIFF",
    "dataset6": "320-HPV",
    "dataset7": "320-PRV",
    "dataset8": "320-WEAV",
    "dataset8-1": "320-LOW",
    "dataset10": "321-DIFF",
    "dataset11": "321-WEAV",
    "dataset13": "787-CAC",
    "dataset14": "777-PRSOV",
}


def parse_list(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def legacy_split_bounds(n: int) -> tuple[int, int]:
    """Submitted-main-table 70/10/20 boundary convention.

    The historical code used cumulative truncation for the validation end.
    This differs by one sample from ``int(.7*n) + int(.1*n)`` for some class
    sizes, so it is kept explicit for exact main-table comparability.
    """

    n = int(n)
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = max(1, min(int(n * 0.7), n - 2))
    val_end = max(train_end + 1, min(int(n * 0.8), n - 1))
    return train_end, val_end


def per_class_chronological_split(
    labels: np.ndarray,
    time_keys: np.ndarray,
    sources: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels).astype(np.int64, copy=False)
    time_keys = np.asarray(time_keys).astype(np.int64, copy=False)
    if sources is None:
        sources = np.arange(labels.size).astype(str)
    else:
        sources = np.asarray(sources).astype(str)

    parts: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for label in sorted(np.unique(labels).tolist()):
        group = np.flatnonzero(labels == label)
        ordered = sorted(
            (int(index) for index in group),
            key=lambda index: (int(time_keys[index]), str(sources[index]), index),
        )
        train_end, val_end = legacy_split_bounds(len(ordered))
        parts["train"].extend(ordered[:train_end])
        parts["validation"].extend(ordered[train_end:val_end])
        parts["test"].extend(ordered[val_end:])

    return {
        name: np.asarray(
            sorted(values, key=lambda index: (int(time_keys[index]), str(sources[index]), index)),
            dtype=np.int64,
        )
        for name, values in parts.items()
    }


def precision_at_recall_floor(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.10,
) -> dict[str, float | int]:
    """Precision at the first tied-score-safe cutoff reaching target recall.

    This is the submitted-paper definition of P@R10: sort fault-risk scores in
    descending order and stop at the first realizable threshold whose recall is
    at least the target.  Equal scores are always kept together.
    """

    y = np.asarray(labels).astype(np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError(f"labels/scores length mismatch: {y.size} vs {s.size}")
    if y.size == 0 or not np.any(y == 1):
        raise ValueError("P@R requires at least one positive test sample")
    s = np.nan_to_num(s, nan=-np.inf)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    group_ends = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    min_tp = max(1, int(math.ceil(float(target_recall) * int((y == 1).sum()))))

    n_pos = int((y == 1).sum())
    selected_end: int | None = None
    for end in group_ends:
        tp = int(tp_cum[end])
        if tp >= min_tp:
            selected_end = int(end)
            break
    if selected_end is None:
        raise RuntimeError("no valid P@R operating point")

    threshold = float(s_sorted[selected_end])
    pred = s >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    return {
        "target_recall": float(target_recall),
        "precision_at_recall": float(tp / (tp + fp)),
        "recall_at_threshold": float(tp / (tp + fn)),
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "n_test": int(y.size),
        "n_positive": n_pos,
    }


def cache_path(cache_root: Path, dataset: str) -> Path:
    path = cache_root / dataset / "qar_compact_shiftN80.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
