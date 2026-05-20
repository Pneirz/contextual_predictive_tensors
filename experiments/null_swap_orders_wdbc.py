"""Null-swap orders 1-3 on the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

Each row in the output CSV corresponds to one (S,j) combination evaluated at
one CV fold.  Columns: estimator, order, target_feature, context_features,
repeat, fold, delta.

Two estimators are run: DecisionTreeClassifier (dt) and
Pipeline(StandardScaler, LogisticRegression) (logreg).
Results saved to data/null_swap_wdbc.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from load_datasets import load_breast_cancer_wisconsin_diagnostic
from null_swap_core import (
    compute_order_scores,
    make_decision_tree_estimator,
    make_logreg_estimator,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = 5
N_FOLDS = 5
RANDOM_SEED = 42
ORDERS = [1, 2, 3]


def explode_raw_deltas(rec: pd.DataFrame, n_repeats: int, n_folds: int) -> pd.DataFrame:
    rows = []
    for _, row in rec.iterrows():
        for i, d in enumerate(row["raw_deltas"]):
            rows.append({
                "target_feature": row["target_feature"],
                "context_features": row["context_features"],
                "repeat": i // n_folds,
                "fold": i % n_folds,
                "delta": d,
            })
    return pd.DataFrame(rows)


def run_experiment() -> None:
    print("Loading UCI Breast Cancer Wisconsin (Diagnostic)...")
    X, y, feature_names, _ = load_breast_cancer_wisconsin_diagnostic(
        save_csv=True,
        download_if_missing=True,
    )
    print(f"  Shape: {X.shape}")
    print(f"  Positive rate (malignant): {y.mean():.3f}")

    estimators = [
        ("dt", make_decision_tree_estimator),
        ("logreg", make_logreg_estimator),
    ]

    all_rows = []
    for label, factory in estimators:
        print(f"\n--- Estimator: {label} ---")
        for order in ORDERS:
            _, rec = compute_order_scores(
                X=X,
                y=y,
                feature_names=feature_names,
                order=order,
                estimator_factory=factory,
                n_repeats=N_REPEATS,
                n_folds=N_FOLDS,
                random_seed=RANDOM_SEED,
                n_jobs=-1,
                aggregation="mean",
                max_evaluations=None,
                store_raw=True,
            )
            expanded = explode_raw_deltas(rec, N_REPEATS, N_FOLDS)
            expanded["estimator"] = label
            expanded["order"] = order
            all_rows.append(expanded)

    df = pd.concat(all_rows, ignore_index=True)
    df = df[["estimator", "order", "target_feature", "context_features",
             "repeat", "fold", "delta"]]

    out_csv = _DATA_DIR / "null_swap_wdbc.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.relative_to(_REPO_ROOT)}")
    print(f"Total rows: {len(df):,}")

    # Quick summary: top order-1 features per estimator
    print("\n--- Order-1 summary ---")
    o1 = df[df["order"] == 1].groupby(["estimator", "target_feature"])["delta"].mean()
    for label, _ in estimators:
        top = o1[label].sort_values(ascending=False).head(10)
        print(f"\n{label} top-10:")
        for feat, val in top.items():
            print(f"  {feat}: {val:.4f}")


if __name__ == "__main__":
    run_experiment()
