# Experiment Manifest

This manifest maps each experiment family to the files kept in this repository.

## Shared code

- Core null-swap implementation: `experiments/null_swap_core.py`
- Dataset loaders: `experiments/load_datasets.py`

## Experiment artifacts

1. MONK-1 marginal/contextual signal
   - Scripts: `experiments/null_swap_orders_monks1.py`, `experiments/load_datasets.py`
   - Data: `data/processed/monks1_full.csv`, `null_swap_order1_monks1.csv`, `null_swap_order2_monks1.csv`, `null_swap_order2_matrix_monks1.csv`, `null_swap_summary_monks1.csv`
   - Supplemental figure: `figures/null_swap_orders_monks1.png`

2. Synthetic sample-size stability and estimator dependence
   - Scripts: `experiments/null_swap_order3_v2.py`, `experiments/null_swap_order3_logreg.py`
   - Large outputs: `data/large_archives/null_swap_order3_v2.csv.zip`, `data/large_archives/null_swap_order3_logreg.csv.zip`

3. WDBC redundancy clusters and activation candidates
   - Scripts: `experiments/null_swap_orders_wdbc.py`, `experiments/load_datasets.py`
   - Raw data: `data/raw/breast_cancer_wisconsin_diagnostic.zip`, `data/raw/breast_cancer_wisconsin_diagnostic/`
   - Processed data: `data/processed/wdbc_full.csv`, `data/processed/null_swap_wdbc.csv`

4. Scalar reductions
   - Script: `experiments/exp5_sota_comparison.py`
   - Data: `data/processed/exp5_monks1_comparison.csv`, `data/processed/exp5_makeclf_comparison.csv`
   - Depends on MONK-1 outputs and `null_swap_order3_logreg.csv.zip`

5. Mixed DGP
   - Script: `experiments/exp6_mixed_dgp.py`
   - Data: `data/processed/exp6_mixed_dgp_scores.csv`, `data/processed/exp6_threshold_sweep.csv`
   - Supplemental figure: `figures/exp6_threshold_sweep.png`

6. Pure XOR3
   - Script: `experiments/exp7_xor3_benchmark.py`
   - Data: `data/processed/exp7_xor3_records.csv`, `exp7_xor3_order1.csv`, `exp7_xor3_order2_inf.csv`, `exp7_xor3_order3_key.csv`

7. Log-loss convergence
   - Script: `experiments/exp8_logloss_consistency.py`
   - Data: `data/processed/exp8_logloss_consistency.csv`

8. SCM causal-control non-collapse
   - Script: `experiments/exp9_scm_causal_control.py`
   - Data: `data/processed/exp9_scm_causal_control.csv`

## Intentionally excluded

- Manuscript sources and LaTeX bundles
- `__pycache__/`
- LaTeX build logs and auxiliary files
- Old bundle archives
- Generated PDFs and draft build outputs
- Uncompressed order-3 CSVs larger than 100 MB
