# Rare or Wrong? BiSafeAudit

This repository accompanies the ICVGIP 2026 paper **“Rare or Wrong? Constraint-Qualified Auditor Selection for Group-Preserving Visual Dataset Auditing”** by Avdhesh Mishra, Venkata Siddharth Dhara, and Pawan Kumar.

The code is a compact reference implementation of the finite BiSafeAudit selector. It operates on precomputed auditor score vectors and implements:

- empirical-rank normalization;
- individual and equal-weight pairwise candidates;
- calibration-only anchor selection;
- pooled AUPRC and worst-group error-recall constraints;
- leave-one-fold-out recall-stability checks;
- worst-group false-positive-rate minimization; and
- deterministic lexicographic tie-breaking.

The repository intentionally excludes datasets, extracted features, experimental results, logs, checkpoints, caches, trained weights, and cluster-specific job files.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -m pytest
python code/example.py
```

## Input contract

Each auditor is supplied as a one-dimensional NumPy score vector, with larger values indicating greater suspicion of label error. The selector additionally receives calibration indices, verified calibration error indicators, and operational group labels. Held-out error indicators are not used during selection.

The default protocol uses target recall 0.70, AUPRC tolerance 0.005, worst-group recall tolerance 0.02, and five stability folds.

## Supplementary material

The final supplementary PDF is stored under `supplementary/`.

## Citation

```text
Avdhesh Mishra, Venkata Siddharth Dhara, and Pawan Kumar.
Rare or Wrong? Constraint-Qualified Auditor Selection for Group-Preserving
Visual Dataset Auditing. ICVGIP 2026.
```
