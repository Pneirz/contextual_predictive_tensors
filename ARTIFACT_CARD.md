# Artifact Card

## Purpose

This artifact supports the empirical claims in "Contextual Predictive Tensors:
Beyond Scalar Feature Importance." It contains scripts, generated CSV outputs,
large compressed order-3 records, and supplemental figures for the null-swap
experiments.

## Contents

- `experiments/`: experiment scripts and shared null-swap implementation.
- `data/raw/`: redistributed public WDBC source data.
- `data/processed/`: generated CSV outputs small enough to keep uncompressed.
- `data/large_archives/`: compressed CSV outputs whose uncompressed versions
  exceed GitHub's single-file limit.
- `figures/`: supplemental figures generated from processed data.
- `MANIFEST.md`: mapping between paper experiments and artifact files.
- `THIRD_PARTY_ASSETS.md`: dataset and dependency license notes.
- `COMPUTE.md`: compute environment and timing manifest.
- `UNCERTAINTY_PLAN.md`: uncertainty-quantification plan and adversarial review.

## Reproducibility

Create an environment and install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Restore large uncompressed CSVs:

```bash
python experiments/unpack_large_data.py
```

Run the experiments in the order listed in `README.md` or use
`experiments/run_with_compute_log.py` to record wall-clock time and environment
metadata for each command.

## Uncertainty Status

The artifact includes raw `repeat,fold,delta` records and bootstrap summaries
for the main stochastic claims. Fixed-dataset summaries use repeat-level
bootstrap intervals. SCM and targeted DGP-level summaries use seed-level
bootstrap intervals. Post-selection maxima, such as WDBC lift candidates, are
reported as conditional on the selected context and are not multiplicity
adjusted.

The targeted outer-seed checks are in `outer_seed_uq_raw.csv` and
`outer_seed_uq_summary.csv`. The stronger 100-seed rerun used for DGP-level
manuscript claims is stored in `outer_seed_uq_raw_long.csv` and
`outer_seed_uq_summary_long.csv`; it was generated with the resumable runner
`experiments/outer_seed_uq_resumable.py`. Broader full-tensor outer-seed reruns
remain out of scope because their combinatorial cost is substantially larger.

## Ethics and Privacy

The artifact uses public or synthetic data and releases no pretrained model.
WDBC is an observational public biomedical dataset; the paper does not attempt
individual-level inference or causal discovery from WDBC.

## Licensing

See `LICENSE` and `THIRD_PARTY_ASSETS.md`.
