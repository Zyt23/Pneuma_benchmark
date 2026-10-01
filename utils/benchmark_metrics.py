"""Explicit flight-level ranking metrics; not deployment-threshold fitting."""
import math
import numpy as np


def precision_at_recall(labels, scores, target_recall=0.10):
    """Highest score threshold reaching recall >= target, including all ties.

    Also report the precision envelope for auditing historical collectors. The
    envelope is a different metric and must not silently replace the first one.
    Test labels are used only to evaluate a ranking curve, never for training.
    """
    y = np.asarray(labels).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.shape != s.shape:
        raise ValueError('Expected nonempty, equally sized labels and scores')
    if not np.all(np.isin(y, [0, 1])) or not np.isfinite(s).all():
        raise ValueError('Labels must be binary and scores must be finite')
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0, 1]')
    positives = int(y.sum())
    base = dict(target_recall=float(target_recall), samples=int(y.size), positives=positives,
                definition='highest_threshold_reaching_target_recall_including_ties',
                threshold_role='ranking_evaluation_not_deployment_calibration')
    if positives == 0:
        return dict(base, defined=False, precision=None, achieved_recall=None,
                    threshold=None, tp=0, fp=0, max_precision_at_recall_ge_target=None)
    required = max(1, int(math.ceil(target_recall * positives)))
    threshold = float(np.sort(s[y == 1])[::-1][required - 1])
    predicted = s >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    order = np.argsort(-s, kind='stable')
    sorted_scores = s[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), y.size - 1]
    cumulative_tp = np.cumsum(y[order])[ends]
    recalls = cumulative_tp / positives
    precisions = cumulative_tp / (ends + 1)
    envelope = float(np.max(precisions[recalls >= target_recall]))
    return dict(base, defined=True, precision=float(tp / (tp + fp)),
                achieved_recall=float(tp / positives), threshold=threshold, tp=tp,
                fp=fp, max_precision_at_recall_ge_target=envelope)
