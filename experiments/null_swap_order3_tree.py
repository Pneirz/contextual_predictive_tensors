"""Null-swap order-3, fixed DGP subsampled, all per-fold deltas stored.

Generates make_classification once with n=16384, then subsamples to
512, 1024, 2048, 4096, 8192 so all sample sizes share the same DGP.

Each row in the output CSV corresponds to one (S,j) combination evaluated
at one CV fold — not an aggregate. Columns:
  n_samples, target_feature, ground_truth, context_features,
  repeat, fold, delta
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from null_swap_core import (
    compute_order_scores,
    make_decision_tree_estimator,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_FIG_DIR = _REPO_ROOT / "figures"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_FIG_DIR.mkdir(parents=True, exist_ok=True)

N_FULL = 16_384
SAMPLE_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
N_INFORMATIVE = 3
N_REDUNDANT = 7
N_NOISE = 20
N_FEATURES = N_INFORMATIVE + N_REDUNDANT + N_NOISE  # 30
N_REPEATS = 5
N_FOLDS = 5
RANDOM_SEED = 42


def generate_full_dataset() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=N_FULL,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        shuffle=False,
        random_state=RANDOM_SEED,
    )

    feature_names = [f"f{i:02d}" for i in range(N_FEATURES)]
    ground_truth: dict[str, str] = {}
    for i, name in enumerate(feature_names):
        if i < N_INFORMATIVE:
            ground_truth[name] = "informative"
        elif i < N_INFORMATIVE + N_REDUNDANT:
            ground_truth[name] = "redundant"
        else:
            ground_truth[name] = "noise"

    return X, y, feature_names, ground_truth


def subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    idx.sort()
    return X[idx], y[idx]


def explode_raw_deltas(rec: pd.DataFrame, n_repeats: int, n_folds: int) -> pd.DataFrame:
    """Expand raw_deltas list column into one row per (S,j,repeat,fold)."""
    rows = []
    for _, row in rec.iterrows():
        raw = row["raw_deltas"]
        for i, d in enumerate(raw):
            repeat = i // n_folds
            fold = i % n_folds
            rows.append({
                "target_feature": row["target_feature"],
                "context_features": row["context_features"],
                "repeat": repeat,
                "fold": fold,
                "delta": d,
            })
    return pd.DataFrame(rows)


def context_feature_set(value: object) -> frozenset[str]:
    """Return a context-feature set from an in-memory tuple or CSV string."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return frozenset(value)


def run_experiment() -> None:
    print(f"Generating full dataset: n={N_FULL}, seed={RANDOM_SEED}")
    X_full, y_full, feature_names, ground_truth = generate_full_dataset()

    all_rows = []
    for n in SAMPLE_SIZES:
        if n == N_FULL:
            X, y = X_full, y_full
        else:
            X, y = subsample(X_full, y_full, n)

        print(f"\n  n={n}")
        _, rec = compute_order_scores(
            X=X, y=y, feature_names=feature_names,
            order=3,
            estimator_factory=make_decision_tree_estimator,
            n_repeats=N_REPEATS, n_folds=N_FOLDS, random_seed=RANDOM_SEED,
            n_jobs=-1, aggregation="mean",
            max_evaluations=None,
            store_raw=True,
        )

        expanded = explode_raw_deltas(rec, N_REPEATS, N_FOLDS)
        expanded["n_samples"] = n
        expanded["ground_truth"] = expanded["target_feature"].map(ground_truth)
        all_rows.append(expanded)

    df = pd.concat(all_rows, ignore_index=True)
    df = df[["n_samples", "target_feature", "ground_truth", "context_features", "repeat", "fold", "delta"]]

    out_csv = _DATA_DIR / "null_swap_order3_tree.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.relative_to(_REPO_ROOT)}")
    print(f"Total rows: {len(df):,}  ({len(df) / (N_REPEATS * N_FOLDS):.0f} combinations x {N_REPEATS*N_FOLDS} folds)")

    # Show key informative combinations at each sample size
    print("\n=== Key informative combinations (other 2 inf as context) ===")
    inf = {"f00", "f01", "f02"}
    df["ctx_set"] = df["context_features"].apply(context_feature_set)
    for n in SAMPLE_SIZES:
        print(f"\n  n={n}:")
        sub = df[df["n_samples"] == n]
        for target in ["f00", "f01", "f02"]:
            ctx = inf - {target}
            rows = sub[(sub["target_feature"] == target) & (sub["ctx_set"] == frozenset(ctx))]
            mean_d = rows["delta"].mean()
            std_d = rows["delta"].std()
            print(f"    Delta[{sorted(ctx)}, {target}]: mean={mean_d:.4f}  std={std_d:.4f}  n_obs={len(rows)}")


if __name__ == "__main__":
    run_experiment()
