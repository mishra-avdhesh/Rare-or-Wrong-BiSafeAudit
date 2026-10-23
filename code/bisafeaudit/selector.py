from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from bisafeaudit.metrics import ScoreMetrics, evaluate_scores


@dataclass(frozen=True)
class SelectionConfig:
    target_recall: float = 0.70
    auprc_tolerance: float = 0.005
    recall_tolerance: float = 0.02
    folds: int = 5
    seed: int = 0


@dataclass(frozen=True)
class CandidateEvaluation:
    name: str
    metrics: ScoreMetrics
    feasible: bool
    auprc_margin: float
    recall_margin: float
    fold_recall_margins: tuple[float, ...]


@dataclass(frozen=True)
class SelectionResult:
    anchor_name: str
    selected_name: str
    selected_scores: np.ndarray
    anchor_metrics: ScoreMetrics
    selected_metrics: ScoreMetrics
    evaluations: tuple[CandidateEvaluation, ...]

    @property
    def fallback_to_anchor(self) -> bool:
        return self.selected_name == self.anchor_name


def _scores(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("candidate scores must be finite one-dimensional arrays")
    return array


def _indices(values: np.ndarray, size: int) -> np.ndarray:
    indices = np.asarray(values, dtype=int)
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("at least two calibration indices are required")
    if np.unique(indices).size != indices.size:
        raise ValueError("calibration indices must be unique")
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("calibration indices are out of bounds")
    return indices


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    values = _scores(scores)
    if values.size <= 1:
        return np.zeros(values.size, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float) / (values.size - 1)
    return ranks


def build_rank_candidates(auditors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not auditors:
        raise ValueError("at least one auditor is required")
    ranked = {name: rank_normalize(values) for name, values in auditors.items()}
    lengths = {len(values) for values in ranked.values()}
    if len(lengths) != 1:
        raise ValueError("all auditor score vectors must have the same length")
    candidates = dict(ranked)
    for left, right in combinations(ranked, 2):
        candidates[f"rank_mean({left},{right})"] = 0.5 * (ranked[left] + ranked[right])
    return candidates


def _matching_max(values: Mapping[str, float], patterns: tuple[str, ...]) -> float:
    matches = [
        float(value)
        for name, value in values.items()
        if np.isfinite(value) and any(pattern in name.lower() for pattern in patterns)
    ]
    return max(matches) if matches else float("nan")


def _anchor_objective(metrics: ScoreMetrics) -> float:
    if not np.isfinite(metrics.auprc):
        return float("inf")
    objective = 1.0 - metrics.auprc
    tail_fpr = _matching_max(metrics.fpr_by_group, ("tail", "rare"))
    shift_fpr = _matching_max(
        metrics.fpr_by_group,
        ("shift", "target", "heldout", "held-out"),
    )
    if np.isfinite(tail_fpr):
        objective += tail_fpr
    if np.isfinite(shift_fpr):
        objective += shift_fpr
    if np.isfinite(metrics.fpr_gap):
        objective += 0.25 * metrics.fpr_gap
    return float(objective)


def select_anchor(
    baseline_candidates: Mapping[str, np.ndarray],
    calibration_indices: np.ndarray,
    error_mask: np.ndarray,
    groups: np.ndarray,
    config: SelectionConfig = SelectionConfig(),
) -> str:
    if not baseline_candidates:
        raise ValueError("baseline candidate library is empty")
    errors = np.asarray(error_mask, dtype=bool)
    group_values = np.asarray(groups)
    indices = _indices(calibration_indices, len(errors))
    if errors.ndim != 1 or group_values.shape != errors.shape:
        raise ValueError("error_mask and groups must be aligned one-dimensional arrays")
    best_name: str | None = None
    best_objective = float("inf")
    for name, full_scores in baseline_candidates.items():
        values = _scores(full_scores)
        if values.shape != errors.shape:
            raise ValueError("candidate scores must align with error_mask")
        metrics = evaluate_scores(
            errors[indices],
            values[indices],
            group_values[indices],
            config.target_recall,
        )
        objective = _anchor_objective(metrics)
        if objective < best_objective:
            best_name = name
            best_objective = objective
    if best_name is None:
        raise RuntimeError("no finite anchor candidate was found")
    return best_name


def _leave_one_fold_out_positions(
    errors: np.ndarray,
    folds: int,
    seed: int,
) -> list[np.ndarray]:
    fold_count = min(max(2, folds), len(errors))
    order = np.random.default_rng(seed).permutation(len(errors))
    partitions = np.array_split(order, fold_count)
    positions = np.arange(len(errors), dtype=int)
    subsets: list[np.ndarray] = []
    for held_out in partitions:
        keep = np.ones(len(errors), dtype=bool)
        keep[held_out] = False
        subset = positions[keep]
        if subset.size and np.unique(errors[subset]).size == 2:
            subsets.append(subset)
    return subsets


def select_qualified_auditor(
    candidates: Mapping[str, np.ndarray],
    anchor_name: str,
    calibration_indices: np.ndarray,
    error_mask: np.ndarray,
    groups: np.ndarray,
    config: SelectionConfig = SelectionConfig(),
) -> SelectionResult:
    if anchor_name not in candidates:
        raise ValueError("anchor candidate is not present in the expanded library")
    errors = np.asarray(error_mask, dtype=bool)
    group_values = np.asarray(groups)
    indices = _indices(calibration_indices, len(errors))
    if errors.ndim != 1 or group_values.shape != errors.shape:
        raise ValueError("error_mask and groups must be aligned one-dimensional arrays")
    prepared = {name: _scores(values) for name, values in candidates.items()}
    if any(values.shape != errors.shape for values in prepared.values()):
        raise ValueError("candidate scores must align with error_mask")
    calibration_errors = errors[indices]
    calibration_groups = group_values[indices]
    anchor_scores = prepared[anchor_name][indices]
    anchor_metrics = evaluate_scores(
        calibration_errors,
        anchor_scores,
        calibration_groups,
        config.target_recall,
    )
    fold_positions = _leave_one_fold_out_positions(
        calibration_errors,
        config.folds,
        config.seed,
    )
    anchor_fold_recalls = [
        evaluate_scores(
            calibration_errors[positions],
            anchor_scores[positions],
            calibration_groups[positions],
            config.target_recall,
        ).worst_group_error_recall
        for positions in fold_positions
    ]
    evaluations: list[CandidateEvaluation] = []
    feasible_names: list[str] = []
    for name, full_scores in prepared.items():
        candidate_scores = full_scores[indices]
        metrics = evaluate_scores(
            calibration_errors,
            candidate_scores,
            calibration_groups,
            config.target_recall,
        )
        auprc_margin = metrics.auprc - (
            anchor_metrics.auprc - config.auprc_tolerance
        )
        recall_margin = metrics.worst_group_error_recall - (
            anchor_metrics.worst_group_error_recall - config.recall_tolerance
        )
        fold_margins: list[float] = []
        for positions, anchor_recall in zip(
            fold_positions,
            anchor_fold_recalls,
            strict=True,
        ):
            fold_metrics = evaluate_scores(
                calibration_errors[positions],
                candidate_scores[positions],
                calibration_groups[positions],
                config.target_recall,
            )
            fold_margins.append(
                fold_metrics.worst_group_error_recall
                - (anchor_recall - config.recall_tolerance)
            )
        worst_fold_margin = min(fold_margins) if fold_margins else recall_margin
        feasible = bool(
            np.isfinite(metrics.auprc)
            and np.isfinite(metrics.worst_group_error_recall)
            and np.isfinite(metrics.worst_group_fpr)
            and auprc_margin >= -1e-12
            and recall_margin >= -1e-12
            and worst_fold_margin >= -1e-12
        )
        evaluations.append(
            CandidateEvaluation(
                name=name,
                metrics=metrics,
                feasible=feasible,
                auprc_margin=float(auprc_margin),
                recall_margin=float(recall_margin),
                fold_recall_margins=tuple(float(value) for value in fold_margins),
            )
        )
        if feasible:
            feasible_names.append(name)
    evaluation_by_name = {item.name: item for item in evaluations}
    selected_name = min(
        feasible_names,
        key=lambda name: (
            evaluation_by_name[name].metrics.worst_group_fpr,
            -evaluation_by_name[name].metrics.auprc,
            0 if name == anchor_name else 1,
            name,
        ),
    ) if feasible_names else anchor_name
    return SelectionResult(
        anchor_name=anchor_name,
        selected_name=selected_name,
        selected_scores=prepared[selected_name].copy(),
        anchor_metrics=anchor_metrics,
        selected_metrics=evaluation_by_name[selected_name].metrics,
        evaluations=tuple(evaluations),
    )


def fit_selector(
    baseline_auditors: Mapping[str, np.ndarray],
    expansion_auditors: Mapping[str, np.ndarray],
    calibration_indices: np.ndarray,
    error_mask: np.ndarray,
    groups: np.ndarray,
    config: SelectionConfig = SelectionConfig(),
) -> SelectionResult:
    overlapping = set(baseline_auditors) & set(expansion_auditors)
    if overlapping:
        names = ", ".join(sorted(overlapping))
        raise ValueError(f"auditor names must be unique: {names}")
    baseline_candidates = build_rank_candidates(baseline_auditors)
    anchor_name = select_anchor(
        baseline_candidates,
        calibration_indices,
        error_mask,
        groups,
        config,
    )
    expanded_candidates = build_rank_candidates(
        {**baseline_auditors, **expansion_auditors}
    )
    return select_qualified_auditor(
        expanded_candidates,
        anchor_name,
        calibration_indices,
        error_mask,
        groups,
        config,
    )
