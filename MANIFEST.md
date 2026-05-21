# Experiment Manifest

This manifest maps each experiment family to the files kept in this repository.

## Shared code

- Core null-swap implementation: `experiments/null_swap_core.py`
- Dataset loaders: `experiments/load_datasets.py`
- Compute logger: `experiments/run_with_compute_log.py`
- Uncertainty summarizer: `experiments/summarize_uncertainty.py`
- Targeted outer-seed UQ: `experiments/outer_seed_uq.py`
- Artifact metadata: `ARTIFACT_CARD.md`, `COMPUTE.md`, `UNCERTAINTY_PLAN.md`, `THIRD_PARTY_ASSETS.md`, `LICENSE`

## Experiment artifacts

1. MONK-1 marginal/contextual signal
   - Scripts: `experiments/null_swap_orders_monks1.py`, `experiments/load_datasets.py`
   - Data: `data/processed/monks1_full.csv`, `null_swap_order1_monks1.csv`, `null_swap_order2_monks1.csv`, `null_swap_order2_matrix_monks1.csv`, `null_swap_summary_monks1.csv`
   - Raw UQ output after rerun: `data/processed/null_swap_monks1_raw.csv`
   - Supplemental figure: `figures/null_swap_orders_monks1.png`

2. Synthetic sample-size stability and estimator dependence
   - Scripts: `experiments/null_swap_order3_tree.py`, `experiments/null_swap_order3_logreg.py`
   - Large outputs: `data/large_archives/null_swap_order3_tree.csv.zip`, `data/large_archives/null_swap_order3_logreg.csv.zip`
   - UQ summaries: `data/processed/uncertainty_order3_tree.csv`, `data/processed/uncertainty_order3_logreg.csv`

3. WDBC redundancy clusters and activation candidates
   - Scripts: `experiments/null_swap_orders_wdbc.py`, `experiments/load_datasets.py`
   - Raw data: `data/raw/breast_cancer_wisconsin_diagnostic.zip`, `data/raw/breast_cancer_wisconsin_diagnostic/`
   - Processed data: `data/processed/wdbc_full.csv`, `data/processed/null_swap_wdbc.csv`
   - UQ summary: `data/processed/uncertainty_wdbc.csv`

4. Scalar reductions
   - Script: `experiments/exp5_sota_comparison.py`
   - Data: `data/processed/exp5_monks1_comparison.csv`, `data/processed/exp5_makeclf_comparison.csv`
   - Depends on MONK-1 outputs and `null_swap_order3_logreg.csv.zip`

5. Mixed DGP
   - Script: `experiments/exp6_mixed_dgp.py`
   - Data: `data/processed/exp6_mixed_dgp_scores.csv`, `data/processed/exp6_threshold_sweep.csv`
   - Raw UQ output: `data/processed/exp6_mixed_dgp_ns_raw.csv`
   - UQ summary: `data/processed/uncertainty_exp6_mixed_dgp.csv`
   - Supplemental figure: `figures/exp6_threshold_sweep.png`

6. Pure XOR3
   - Script: `experiments/exp7_xor3_benchmark.py`
   - Data: `data/processed/exp7_xor3_records.csv`, `exp7_xor3_order1.csv`, `exp7_xor3_order2_inf.csv`, `exp7_xor3_order3_key.csv`
   - UQ note: `exp7_xor3_records.csv` now stores repeat/fold raw rows after rerun

7. Log-loss convergence
   - Script: `experiments/exp8_logloss_consistency.py`
   - Data: `data/processed/exp8_logloss_consistency.csv`
   - Raw UQ output: `data/processed/exp8_logloss_consistency_raw.csv`

8. SCM causal-control non-collapse
   - Script: `experiments/exp9_scm_causal_control.py`
   - Data: `data/processed/exp9_scm_causal_control.csv`
   - Raw UQ outputs: `data/processed/exp9_scm_causal_control_seed_summary.csv`, `data/processed/exp9_scm_causal_control_ns_raw.csv`

9. Targeted outer-seed UQ
   - Script: `experiments/outer_seed_uq.py`
   - Raw output: `data/processed/outer_seed_uq_raw.csv`
   - Summary: `data/processed/outer_seed_uq_summary.csv`
   - Scope: prespecified XOR3, XOR2 log-loss, and mixed-DGP tensor entries over 20 dataset seeds

10. Uncertainty summaries
   - Script: `experiments/summarize_uncertainty.py`
   - Data: `data/processed/uncertainty_summary.csv` for targeted MONK-1, XOR3, log-loss, and SCM null-swap intervals
   - Large-table summaries: `data/processed/outer_seed_uq_summary.csv`, `data/processed/uncertainty_exp6_mixed_dgp.csv`, `data/processed/uncertainty_order3_tree.csv`, `data/processed/uncertainty_order3_logreg.csv`, `data/processed/uncertainty_wdbc.csv`
   - Seed metric data: `data/processed/uncertainty_seed_metrics.csv` when seed-summary inputs exist
   - Current input coverage can be selected with `--files`; use targeted files for manuscript tables and `--files` omitted for a full raw-output sweep

11. Compute logs
   - Script: `experiments/run_with_compute_log.py`
   - Data: `data/processed/compute_log.csv` after wrapped reruns

## Intentionally excluded

- Manuscript sources and LaTeX bundles
- `__pycache__/`
- LaTeX build logs and auxiliary files
- Old bundle archives
- Generated PDFs and intermediate build outputs
- Uncompressed order-3 CSVs larger than 100 MB
