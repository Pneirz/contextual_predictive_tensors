"""Build repeat-level bootstrap intervals for raw null-swap CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"


RAW_FILES = [
    "null_swap_monks1_raw.csv",
    "null_swap_order3_tree.csv",
    "null_swap_order3_logreg.csv",
    "null_swap_wdbc.csv",
    "exp6_mixed_dgp_ns_raw.csv",
    "exp7_xor3_records.csv",
    "exp8_logloss_consistency_raw.csv",
    "exp9_scm_causal_control_ns_raw.csv",
    "outer_seed_uq_raw.csv",
    "outer_seed_uq_raw_long.csv",
]

SEED_METRIC_FILES = {
    "exp9_scm_causal_control_seed_summary.csv": [
        "ATE_do",
        "MI_marginal",
        "NS_delta_order1",
    ],
}


def bootstrap_interval(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_grouped(
    df: pd.DataFrame,
    source: str,
    group_cols: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if "seed" in group.columns:
            unit = "seed_mean"
            unit_values = group.groupby("seed")["delta"].mean().to_numpy(dtype=float)
        else:
            unit = "repeat_mean"
            unit_values = group.groupby("repeat")["delta"].mean().to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_interval(unit_values, n_bootstrap, rng)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update({
            "source_file": source,
            "mean_delta": float(group["delta"].mean()),
            "std_fold_delta": float(group["delta"].std(ddof=1)),
            "se_unit_mean": float(unit_values.std(ddof=1) / np.sqrt(len(unit_values)))
            if len(unit_values) > 1 else 0.0,
            "ci95_boot_low": ci_low,
            "ci95_boot_high": ci_high,
            "n_units": int(len(unit_values)),
            "n_repeats": int(group["repeat"].nunique()) if "repeat" in group.columns else 0,
            "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else 0,
            "n_folds_total": int(len(group)),
            "interval_unit": unit,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_file(path: Path, n_bootstrap: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"repeat", "fold", "delta", "target_feature"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    group_cols = []
    for col in [
        "experiment",
        "claim",
        "estimator",
        "n_samples",
        "n",
        "protocol",
        "loss",
        "calibrated",
        "order",
        "target_feature",
        "ground_truth",
        "role",
        "context_feature",
        "context_features",
    ]:
        if col in df.columns:
            group_cols.append(col)
    return summarize_grouped(df, path.name, group_cols, n_bootstrap, seed)


def summarize_seed_metric_file(
    path: Path,
    metric_cols: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seed" not in df.columns:
        raise ValueError(f"{path.name} must contain a seed column")

    rng = np.random.default_rng(seed)
    id_cols = [col for col in ["feature", "role", "in_C", "in_I_expected"] if col in df.columns]
    rows = []
    for metric in metric_cols:
        for keys, group in df.groupby(id_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            seed_values = group.groupby("seed")[metric].mean().to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_interval(seed_values, n_bootstrap, rng)
            row = {col: value for col, value in zip(id_cols, keys)}
            row.update({
                "source_file": path.name,
                "metric": metric,
                "mean_value": float(seed_values.mean()),
                "std_unit_value": float(seed_values.std(ddof=1)) if len(seed_values) > 1 else 0.0,
                "se_unit_mean": float(seed_values.std(ddof=1) / np.sqrt(len(seed_values)))
                if len(seed_values) > 1 else 0.0,
                "ci95_boot_low": ci_low,
                "ci95_boot_high": ci_high,
                "n_units": int(len(seed_values)),
                "interval_unit": "seed_mean",
            })
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "uncertainty_summary.csv")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional raw CSV filenames to summarize instead of the default list.",
    )
    parser.add_argument(
        "--include-seed-metrics",
        action="store_true",
        help=(
            "Also summarize configured seed-metric files. By default this is "
            "only done for a full raw-file sweep, not for targeted --files runs."
        ),
    )
    args = parser.parse_args()

    summaries = []
    raw_files = args.files if args.files is not None else RAW_FILES
    for name in raw_files:
        path = DATA_DIR / name
        if path.exists():
            print(f"Summarizing {name}...")
            summaries.append(summarize_file(path, args.n_bootstrap, args.seed))
        else:
            print(f"Skipping missing file: {name}")

    if not summaries:
        raise SystemExit("No raw files found to summarize.")

    out = pd.concat(summaries, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

    if args.include_seed_metrics or args.files is None:
        metric_summaries = []
        for name, metric_cols in SEED_METRIC_FILES.items():
            path = DATA_DIR / name
            if path.exists():
                print(f"Summarizing seed metrics in {name}...")
                metric_summaries.append(
                    summarize_seed_metric_file(path, metric_cols, args.n_bootstrap, args.seed)
                )
        if metric_summaries:
            metric_out = args.output.with_name("uncertainty_seed_metrics.csv")
            pd.concat(metric_summaries, ignore_index=True).to_csv(metric_out, index=False)
            print(f"Saved: {metric_out}")


if __name__ == "__main__":
    main()
