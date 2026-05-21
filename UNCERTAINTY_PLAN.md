# Uncertainty Quantification Plan

## What "complete UQ" Means Here

In this paper, complete uncertainty quantification means reporting intervals
for the empirical quantities that support each claim, while naming the source
of variability captured by each interval.

For null-swap tensor entries, the target estimand is the mean out-of-sample
loss contrast produced by a specified protocol. The main random components are:

1. train/test split variation across repeated folds;
2. null-permutation variation inside each training fold;
3. estimator randomness, when the estimator has a random state;
4. dataset-generation seed variation for synthetic DGPs;
5. post-selection variation for quantities chosen by maxima over contexts.

The current fold-level `mean +/- std` summaries capture only descriptive
dispersion over repeated folds. They should not be described as population
confidence intervals.

## Recommended Reporting Rule

For each prespecified tensor entry:

1. compute the mean delta within each repeat across its folds;
2. bootstrap the repeat-level means to obtain a percentile interval for the
   protocol mean;
3. report `mean [2.5%, 97.5%]`;
4. state that the interval captures split/null/estimator variation conditional
   on the sampled dataset, unless outer dataset seeds were rerun.

For synthetic claims about a DGP rather than a single realised dataset, add an
outer layer of independent dataset seeds and bootstrap over seeds first. This
is the stronger rerun plan and should be used for XOR3, log-loss convergence,
mixed DGP, and SCM summaries when time permits.

For WDBC, do not claim population DGP uncertainty from synthetic seeds. Report
repeat-level or patient-row bootstrap intervals conditional on this public
observational dataset, and label WDBC activation results as exploratory.

## Implementation Status

- `null_swap_core.explode_raw_deltas` standardizes conversion from stored
  per-task raw deltas to `repeat,fold,delta` rows.
- MONK-1 emits `null_swap_monks1_raw.csv` and has a logged rerun.
- Mixed DGP emitted `exp6_mixed_dgp_ns_raw.csv`; the full script later failed
  in mRMR's internal joblib parallelism on this Windows sandbox, after the raw
  null-swap UQ file had already been saved.
- XOR3 `exp7_xor3_records.csv` now emits repeat/fold raw rows and has a logged
  serial rerun.
- Log-loss convergence emits `exp8_logloss_consistency_raw.csv` and has a
  logged rerun.
- SCM emits both `exp9_scm_causal_control_seed_summary.csv` and
  `exp9_scm_causal_control_ns_raw.csv`.
- `summarize_uncertainty.py` computes bootstrap intervals over repeat means for
  fixed-dataset runs and over seed means when a `seed` column is present.
- Manuscript interval summaries have been generated for MONK-1/XOR3/log-loss/SCM
  (`uncertainty_summary.csv`), mixed DGP (`uncertainty_exp6_mixed_dgp.csv`),
  synthetic order-3 tree/logreg
  (`uncertainty_order3_tree.csv`, `uncertainty_order3_logreg.csv`), WDBC
  (`uncertainty_wdbc.csv`), and SCM seed metrics
  (`uncertainty_seed_metrics.csv`).
- Targeted outer-seed UQ has been generated in `outer_seed_uq_raw.csv` and
  `outer_seed_uq_summary.csv` for prespecified XOR3, XOR2 log-loss, and
  mixed-DGP entries over 20 dataset seeds.
- `run_with_compute_log.py` records wall-clock time and environment metadata for
  reruns.

## Adversarial Review of the Plan

- Fold deltas are correlated because folds share the same dataset and repeat
  structure. Treating all fold deltas as iid would make intervals too narrow.
- Max-over-context tables are post-selection summaries. Intervals around the
  selected maximum do not control the family-wise error of searching over many
  contexts.
- Synthetic fixed-DGP subsampling answers a stability question for one realised
  dataset. Fresh-seed reruns answer generalisation over the data-generating
  process. These are different claims and should not be conflated.
- RMSE intervals support operational stability in predictive space P. They do
  not support a conditional-mutual-information claim; that claim belongs to
  log-loss plus Bayes-consistency assumptions.
- SCM intervals quantify Monte Carlo stability of a constructed example, not
  validity of causal discovery on observational data.

## Rerun Priorities

1. Decide which outer-seed results should enter the main manuscript versus the
   supplement. The artifact now contains the 20-seed summaries.
2. Recompute paper tables after any additional outer-seed reruns.
3. Keep the previous descriptive standard deviations in supplemental CSVs, but
   remove ambiguity in the main paper tables.
