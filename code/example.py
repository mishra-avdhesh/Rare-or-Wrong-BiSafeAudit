from __future__ import annotations

import numpy as np

from bisafeaudit import SelectionConfig, fit_selector


def main() -> None:
    generator = np.random.default_rng(7)
    sample_count = 200
    errors = generator.random(sample_count) < 0.20
    groups = np.asarray(["head", "medium", "tail", "shift"] * 50)
    signal = errors.astype(float)
    baseline = {
        "confidence": signal + generator.normal(0, 0.55, sample_count),
        "neighborhood": signal + generator.normal(0, 0.65, sample_count),
        "semantic": signal + generator.normal(0, 0.60, sample_count),
    }
    expansion = {
        "non_ot": signal + generator.normal(0, 0.50, sample_count),
        "balanced_ot": signal + generator.normal(0, 0.48, sample_count),
        "fixed_uot": signal + generator.normal(0, 0.52, sample_count),
    }
    calibration = np.sort(generator.choice(sample_count, 80, replace=False))
    result = fit_selector(
        baseline,
        expansion,
        calibration,
        errors,
        groups,
        SelectionConfig(seed=11),
    )
    print(f"anchor={result.anchor_name}")
    print(f"selected={result.selected_name}")
    print(f"fallback={result.fallback_to_anchor}")


if __name__ == "__main__":
    main()
