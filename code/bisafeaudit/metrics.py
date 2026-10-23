from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreMetrics:
    auprc: float
    achieved_error_recall: float
    threshold: float
    worst_group_error_recall: float
    worst_group_fpr: float
    fpr_gap: float
    flagged_count: int
    error_recall_by_group: dict[str, float]
    fpr_by_group: dict[str, float]


def _vector(values: np.ndarray, dtype: type) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    return array


def average_precision(error_mask: np.ndarray, scores: np.ndarray) -> float:
    errors = _vector(error_mask, bool)
    values = _vector(scores, float)
    if errors.shape != values.shape:
        raise ValueError("error_mask and scores must have the same shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    if not np.any(errors):
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    ranked_errors = errors[order]
    precision = np.cumsum(ranked_errors) / (np.arange(len(values)) + 1)
    return float(precision[ranked_errors].mean())


def threshold_at_error_recall(
    error_mask: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
) -> float:
    errors = _vector(error_mask, bool)
    values = _vector(scores, float)
    if errors.shape != values.shape:
        raise ValueError("error_mask and scores must have the same shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    error_scores = np.sort(values[errors])[::-1]
    if error_scores.size == 0:
        return float("nan")
    position = max(int(np.ceil(target_recall * error_scores.size)) - 1, 0)
    return float(error_scores[position])


def evaluate_scores(
    error_mask: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    target_recall: float = 0.70,
) -> ScoreMetrics:
    errors = _vector(error_mask, bool)
    values = _vector(scores, float)
    group_values = _vector(groups, object).astype(str)
    if errors.shape != values.shape or values.shape != group_values.shape:
        raise ValueError("error_mask, scores, and groups must have the same shape")
    threshold = threshold_at_error_recall(errors, values, target_recall)
    flagged = np.zeros(errors.shape, dtype=bool) if np.isnan(threshold) else values >= threshold
    recalls: dict[str, float] = {}
    fprs: dict[str, float] = {}
    for group in np.unique(group_values):
        selected = group_values == group
        group_errors = selected & errors
        group_correct = selected & ~errors
        recalls[group] = (
            float(flagged[group_errors].mean()) if np.any(group_errors) else float("nan")
        )
        fprs[group] = (
            float(flagged[group_correct].mean()) if np.any(group_correct) else float("nan")
        )
    finite_recalls = np.asarray([value for value in recalls.values() if np.isfinite(value)])
    finite_fprs = np.asarray([value for value in fprs.values() if np.isfinite(value)])
    worst_recall = float(finite_recalls.min()) if finite_recalls.size else float("nan")
    worst_fpr = float(finite_fprs.max()) if finite_fprs.size else float("nan")
    fpr_gap = (
        float(finite_fprs.max() - finite_fprs.min())
        if finite_fprs.size
        else float("nan")
    )
    achieved_recall = float(flagged[errors].mean()) if np.any(errors) else float("nan")
    return ScoreMetrics(
        auprc=average_precision(errors, values),
        achieved_error_recall=achieved_recall,
        threshold=threshold,
        worst_group_error_recall=worst_recall,
        worst_group_fpr=worst_fpr,
        fpr_gap=fpr_gap,
        flagged_count=int(flagged.sum()),
        error_recall_by_group=recalls,
        fpr_by_group=fprs,
    )
