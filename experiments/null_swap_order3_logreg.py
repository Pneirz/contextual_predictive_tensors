"""Null-swap order 1-3 with LogisticRegression estimator, fixed DGP subsampled.

Same setup as null_swap_order3_tree.py but using LogisticRegression instead of
DecisionTreeClassifier. Tests whether model capacity (not structural order
limitations) explains why redundants are not suppressed at order 3.

DGP is linear (make_classification), so LogisticRegression should in principle
perfectly capture the linear redundancy given the parents in context.

Results saved to data/processed/null_swap_order3_logreg.csv.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from null_swap_core import (
    compute_order_scores,
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


def make_logreg_estimator(random_state: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


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
        for order in [1, 2, 3]:
            _, rec = compute_order_scores(
                X=X, y=y, feature_names=feature_names,
                order=order,
                estimator_factory=make_logreg_estimator,
                n_repeats=N_REPEATS, n_folds=N_FOLDS, random_seed=RANDOM_SEED,
                n_jobs=-1, aggregation="mean",
                max_evaluations=None,
                store_raw=True,
            )
            expanded = explode_raw_deltas(rec, N_REPEATS, N_FOLDS)
            expanded["n_samples"] = n
            expanded["order"] = order
            expanded["ground_truth"] = expanded["target_feature"].map(ground_truth)
            all_rows.append(expanded)

    df = pd.concat(all_rows, ignore_index=True)
    df = df[["n_samples", "order", "target_feature", "ground_truth",
             "context_features", "repeat", "fold", "delta"]]

    out_csv = _DATA_DIR / "null_swap_order3_logreg.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.relative_to(_REPO_ROOT)}")
    print(f"Total rows: {len(df):,}")

    # Key combinations
    print("\n=== Key informative combinations (other 2 inf as context), order 3 ===")
    inf = {"f00", "f01", "f02"}
    o3 = df[df["order"] == 3].copy()
    o3["ctx_set"] = o3["context_features"].apply(context_feature_set)
    for n in SAMPLE_SIZES:
        print(f"\n  n={n}:")
        sub = o3[o3["n_samples"] == n]
        for target in ["f00", "f01", "f02"]:
            ctx = inf - {target}
            rows = sub[(sub["target_feature"] == target) & (sub["ctx_set"] == frozenset(ctx))]
            print(f"    Delta[{sorted(ctx)}, {target}]: mean={rows['delta'].mean():.4f}  std={rows['delta'].std():.4f}")
        print("  Redundants (context={f00,f01}):")
        for target in ["f03", "f04", "f05", "f06", "f07", "f08", "f09"]:
            rows = sub[(sub["target_feature"] == target) & (sub["ctx_set"] == frozenset({"f00", "f01"}))]
            print(f"    Delta[{{f00,f01}}, {target}]: mean={rows['delta'].mean():.4f}  std={rows['delta'].std():.4f}")


if __name__ == "__main__":
    run_experiment()
