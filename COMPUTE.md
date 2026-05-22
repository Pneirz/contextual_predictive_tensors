# Compute Manifest

This manifest records the compute assumptions and current measurement status
for reproducing the experiments.

## Observed Local Environment

- OS: Windows 11
- Python: 3.12.10
- Logical CPUs visible to Python: 20
- Physical memory visible to Python: approximately 31.7 GiB
- GPU requirement: none for reported experiments
- Default parallel backend: joblib/loky through scikit-learn-style CPU workers

The full `python experiments/null_swap_order3_tree.py` run with `n_jobs=-1`
completed in 8557.548 seconds on this environment. The generated CSV is included
as a compressed archive so reviewers can inspect the output without rerunning
that expensive stage.

The stronger targeted outer-seed UQ rerun uses
`experiments/outer_seed_uq_resumable.py`, which writes one CSV chunk per dataset
seed and then combines the chunks. The local chunk directory is intentionally
ignored by git; the combined 100-seed raw CSV and its summary are versioned.

## Experiment Commands

| Experiment family | Command | Expected workers | Measurement status |
| --- | --- | ---: | --- |
| MONK-1 | `python experiments/null_swap_orders_monks1.py` | CPU, serial in script | 4.953 s locally |
| Synthetic order-3 tree | `python experiments/null_swap_order3_tree.py` | CPU, up to `n_jobs=-1` | 8557.548 s locally |
| Synthetic order-3 logistic regression | `python experiments/null_swap_order3_logreg.py` | CPU, up to `n_jobs=-1` | 1963.167 s locally |
| WDBC tensor | `python experiments/null_swap_orders_wdbc.py` | CPU, up to `n_jobs=-1` | 358.416 s locally |
| Scalar baselines | `python experiments/exp5_sota_comparison.py` | CPU | 1037.205 s locally |
| Mixed DGP | `python experiments/exp6_mixed_dgp.py` | CPU, up to `n_jobs=-1` | 1965.972 s locally |
| Mixed DGP UQ summary | `python experiments/summarize_uncertainty.py --files exp6_mixed_dgp_ns_raw.csv --n-bootstrap 2000 --output data/processed/uncertainty_exp6_mixed_dgp.csv` | CPU | 4.827 s locally |
| XOR3 | `NULL_SWAP_N_JOBS=1 python experiments/exp7_xor3_benchmark.py` | CPU, serial null-swap for sandbox compatibility | 335.730 s locally |
| Log-loss convergence | `python experiments/exp8_logloss_consistency.py` | CPU | 8.488 s locally |
| SCM causal control | `python -c "import sys; sys.path.insert(0, 'experiments'); import exp9_scm_causal_control as e; e.N_JOBS=1; e.run_exp9()"` | CPU, serial null-swap for sandbox compatibility | 18.868 s locally |
| Outer-seed UQ | `python experiments/outer_seed_uq.py --n-seeds 20` | CPU | 1420.712 s locally |
| Outer-seed UQ, resumable long run | `python experiments/outer_seed_uq_resumable.py --n-seeds 100` | CPU | 7134.379 s locally |
| Targeted UQ summary | `python experiments/summarize_uncertainty.py --files null_swap_monks1_raw.csv exp7_xor3_records.csv exp8_logloss_consistency_raw.csv exp9_scm_causal_control_ns_raw.csv --n-bootstrap 2000 --output data/processed/uncertainty_summary.csv` | CPU | 4.099 s locally |
| Outer-seed UQ summary | `python experiments/summarize_uncertainty.py --files outer_seed_uq_raw.csv --n-bootstrap 2000 --output data/processed/outer_seed_uq_summary.csv` | CPU | 0.546 s locally |
| Outer-seed UQ long summary | `python experiments/summarize_uncertainty.py --files outer_seed_uq_raw_long.csv --n-bootstrap 5000 --output data/processed/outer_seed_uq_summary_long.csv` | CPU | 3.953 s locally |
| Order-3 tree UQ summary | `python experiments/summarize_uncertainty.py --files null_swap_order3_tree.csv --n-bootstrap 2000 --output data/processed/uncertainty_order3_tree.csv` | CPU | 42.553 s locally |
| WDBC UQ summary | `python experiments/summarize_uncertainty.py --files null_swap_wdbc.csv --n-bootstrap 2000 --output data/processed/uncertainty_wdbc.csv` | CPU | 16.353 s locally |
| Order-3 LR UQ summary | `python experiments/summarize_uncertainty.py --files null_swap_order3_logreg.csv --n-bootstrap 2000 --output data/processed/uncertainty_order3_logreg.csv` | CPU | 45.332 s locally |

## Timing Procedure

Use:

```bash
python experiments/run_with_compute_log.py -- python experiments/exp8_logloss_consistency.py
```

The wrapper appends timing and environment metadata to
`data/processed/compute_log.csv`. The log may also contain failed sandbox
attempts; the table above reports the successful reruns used for the artifact.
