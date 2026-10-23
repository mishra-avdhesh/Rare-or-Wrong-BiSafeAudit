from __future__ import annotations

import numpy as np

from bisafeaudit import SelectionConfig, build_rank_candidates, fit_selector


def test_candidate_library_sizes() -> None:
    values = np.arange(30, dtype=float)
    baseline = {f"base_{index}": values + index for index in range(17)}
    expanded = {f"score_{index}": values + index for index in range(20)}
    assert len(build_rank_candidates(baseline)) == 153
    assert len(build_rank_candidates(expanded)) == 210


def test_anchor_is_feasible_and_selection_is_deterministic() -> None:
    generator = np.random.default_rng(13)
    sample_count = 160
    errors = np.zeros(sample_count, dtype=bool)
    errors[::5] = True
    groups = np.asarray(["head", "medium", "tail", "shift"] * 40)
    signal = errors.astype(float)
    baseline = {
        "confidence": signal + generator.normal(0, 0.45, sample_count),
        "semantic": signal + generator.normal(0, 0.55, sample_count),
    }
    expansion = {
        "non_ot": signal + generator.normal(0, 0.50, sample_count),
        "balanced_ot": signal + generator.normal(0, 0.48, sample_count),
        "fixed_uot": signal + generator.normal(0, 0.52, sample_count),
    }
    calibration = np.arange(120)
    config = SelectionConfig(seed=5)
    first = fit_selector(
        baseline,
        expansion,
        calibration,
        errors,
        groups,
        config,
    )
    second = fit_selector(
        baseline,
        expansion,
        calibration,
        errors,
        groups,
        config,
    )
    anchor_evaluation = next(
        item for item in first.evaluations if item.name == first.anchor_name
    )
    assert anchor_evaluation.feasible
    assert first.selected_name == second.selected_name
    assert np.array_equal(first.selected_scores, second.selected_scores)
