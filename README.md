# Contextual Predictive Tensors

Experiment code and generated artifacts for the contextual predictive tensor
experiments.

This repository contains only reproducibility materials: Python scripts,
processed outputs, raw public data used by the loaders, and generated
supplemental figures. Manuscript sources are intentionally not included.

## Layout

- `experiments/`: Python scripts used to generate the experiment tables and records.
- `data/raw/`: raw source data needed by the WDBC loader.
- `data/processed/`: generated CSVs that are small enough for GitHub.
- `data/large_archives/`: compressed CSV outputs that were larger than GitHub's 100 MB single-file limit when uncompressed.
- `figures/`: generated supplemental figures.
- `ARTIFACT_CARD.md`: scope, contents, ethics, and reproducibility notes.
- `THIRD_PARTY_ASSETS.md`: dataset and dependency licensing notes.
- `COMPUTE.md`: compute environment, timing procedure, and current measurement status.
- `UNCERTAINTY_PLAN.md`: interval-reporting plan and adversarial review.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Large data archives

Two raw order-3 CSVs exceed GitHub's single-file limit uncompressed:

- `null_swap_order3_tree.csv`
- `null_swap_order3_logreg.csv`

They are stored as ZIP archives under `data/large_archives/`. Most scripts that
only need to read `null_swap_order3_logreg.csv` can read the ZIP directly via
Pandas. To restore the uncompressed CSVs into `data/processed/`, run:

```bash
python experiments/unpack_large_data.py
```

## Reproducing artifacts

```bash
python experiments/null_swap_orders_monks1.py
python experiments/null_swap_order3_tree.py
python experiments/null_swap_order3_logreg.py
python experiments/null_swap_orders_wdbc.py
python experiments/exp5_sota_comparison.py
python experiments/exp6_mixed_dgp.py
python experiments/exp7_xor3_benchmark.py
python experiments/exp8_logloss_consistency.py
python experiments/exp9_scm_causal_control.py
python experiments/outer_seed_uq.py --n-seeds 20
python experiments/outer_seed_uq_resumable.py --n-seeds 100
```

The order-3 and WDBC scripts are the expensive runs. The current generated CSVs
are included so the experiment outputs can be inspected without rerunning
everything.

To record wall-clock time and environment metadata during reruns, wrap commands:

```bash
python experiments/run_with_compute_log.py -- python experiments/exp8_logloss_consistency.py
```

To build repeat-level bootstrap intervals from raw `repeat,fold,delta` outputs:

```bash
python experiments/summarize_uncertainty.py
```

For targeted manuscript-table intervals, pass only the relevant raw files:

```bash
python experiments/summarize_uncertainty.py --files exp8_logloss_consistency_raw.csv exp9_scm_causal_control_ns_raw.csv
```

For the stronger DGP-level interval checks used in the current manuscript,
summarize the 100-seed resumable rerun:

```bash
python experiments/summarize_uncertainty.py --files outer_seed_uq_raw_long.csv --n-bootstrap 5000 --output data/processed/outer_seed_uq_summary_long.csv
```

Current manuscript interval summaries are stored as:

- `data/processed/uncertainty_summary.csv` for MONK-1, XOR3, log-loss, and SCM null-swap raw records.
- `data/processed/uncertainty_exp6_mixed_dgp.csv` for the mixed-DGP null-swap raw records.
- `data/processed/uncertainty_order3_tree.csv` for the decision-tree order-3 stability table.
- `data/processed/uncertainty_order3_logreg.csv` for the logistic-regression order-3 comparison.
- `data/processed/uncertainty_wdbc.csv` for WDBC contextual activation candidates.
- `data/processed/uncertainty_seed_metrics.csv` for SCM ATE/MI/null-swap seed-bootstrap summaries.
- `data/processed/outer_seed_uq_raw.csv` and `data/processed/outer_seed_uq_summary.csv` for targeted 20-seed DGP-level checks.
- `data/processed/outer_seed_uq_raw_long.csv` and `data/processed/outer_seed_uq_summary_long.csv` for targeted 100-seed DGP-level checks.

## Licensing

Source code is released under the MIT License. Generated tables, figures, and
documentation are released under CC BY 4.0 unless otherwise noted. Third-party
datasets and dependencies retain their original licenses; see
`THIRD_PARTY_ASSETS.md`.
