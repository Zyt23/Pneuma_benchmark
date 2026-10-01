"""Engineering operating-point metrics for fault warning.

The paper uses an engineering threshold: a model is considered useful once it
can recall at least 10% of future fault flights.  For reporting a single
Precision@Recall10 value, scores are scanned in descending order and the first
tied-score-safe operating point whose positive-class recall reaches the target
is selected.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def binary_metrics_at_recall(
    labels: Any,
    scores: Any,
    target_recall: float = 0.10,
    higher_is_positive: bool = True,
) -> dict[str, float | int]:
    """Compute binary metrics at a positive-class recall floor.

    ``scores`` must be a fault-risk score for class 1.  Higher scores mean more
    likely fault by default.  The threshold is the first descending-score
    cutoff that reaches ``target_recall``.  Equal scores are kept together so
    that the value does not depend on arbitrary sample order within a tie.
    """

    y = np.asarray(labels).astype(int).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"labels and scores have different lengths: {y.shape[0]} vs {s.shape[0]}")
    if y.size == 0:
        return _empty_result(target_recall)

    s = np.nan_to_num(s, nan=-np.inf if higher_is_positive else np.inf)
    decision = s if higher_is_positive else -s
    pos_scores = decision[y == 1]
    n_pos = int(pos_scores.size)
    n_total = int(y.size)
    if n_pos == 0:
        result = _empty_result(target_recall)
        result.update({
            "n_total": n_total,
            "n_pos": 0,
            "n_neg": int((y == 0).sum()),
        })
        return result

    min_tp = max(1, int(math.ceil(float(target_recall) * n_pos)))
    order = np.argsort(-decision, kind="mergesort")
    y_sorted = y[order]
    score_sorted = decision[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    group_ends = np.flatnonzero(np.r_[score_sorted[1:] != score_sorted[:-1], True])

    selected_end: int | None = None
    for end_idx in group_ends:
        tp_here = int(tp_cum[end_idx])
        if tp_here >= min_tp:
            selected_end = int(end_idx)
            break

    if selected_end is None:
        # Should only happen with pathological input, but keep a safe fallback.
        threshold_decision = float(np.sort(pos_scores)[::-1][min_tp - 1])
    else:
        threshold_decision = float(score_sorted[selected_end])
    pred = (decision >= threshold_decision).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    threshold = threshold_decision if higher_is_positive else -threshold_decision
    return {
        "target_recall": float(target_recall),
        "threshold": float(threshold),
        "precision_at_recall": precision,
        "recall_at_threshold": recall,
        "f1_at_recall": f1,
        "alert_rate": float((tp + fp) / n_total) if n_total else 0.0,
        "TN_at_recall": tn,
        "FP_at_recall": fp,
        "FN_at_recall": fn,
        "TP_at_recall": tp,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": int((y == 0).sum()),
    }


def binary_metrics_at_threshold(
    labels: Any,
    scores: Any,
    threshold: float,
    target_recall: float = 0.10,
    higher_is_positive: bool = True,
) -> dict[str, float | int]:
    """Compute binary metrics at a fixed deployment threshold.

    This is the leakage-safe companion to :func:`binary_metrics_at_recall`.
    A threshold can be selected on validation data, then applied unchanged to
    the held-out test split.
    """

    y = np.asarray(labels).astype(int).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"labels and scores have different lengths: {y.shape[0]} vs {s.shape[0]}")
    if y.size == 0:
        return _empty_result(target_recall)

    s = np.nan_to_num(s, nan=-np.inf if higher_is_positive else np.inf)
    if higher_is_positive:
        pred = (s >= float(threshold)).astype(int)
    else:
        pred = (s <= float(threshold)).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "target_recall": float(target_recall),
        "threshold": float(threshold),
        "precision_at_recall": precision,
        "recall_at_threshold": recall,
        "f1_at_recall": f1,
        "alert_rate": float((tp + fp) / int(y.size)) if y.size else 0.0,
        "TN_at_recall": tn,
        "FP_at_recall": fp,
        "FN_at_recall": fn,
        "TP_at_recall": tp,
        "n_total": int(y.size),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


def _empty_result(target_recall: float) -> dict[str, float | int]:
    return {
        "target_recall": float(target_recall),
        "threshold": float("nan"),
        "precision_at_recall": float("nan"),
        "recall_at_threshold": float("nan"),
        "f1_at_recall": float("nan"),
        "alert_rate": float("nan"),
        "TN_at_recall": 0,
        "FP_at_recall": 0,
        "FN_at_recall": 0,
        "TP_at_recall": 0,
        "n_total": 0,
        "n_pos": 0,
        "n_neg": 0,
    }
