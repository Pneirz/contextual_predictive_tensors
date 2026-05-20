"""Experiment 5: Scalar baselines vs NS_order1 / NS_max on existing benchmarks.

Runs PI, SHAP, MI, mRMR on MONK-1 and make_classification.
Derives NS_order1 and NS_max from existing null-swap CSV outputs.
Saves comparison tables to data/exp5_monks1_comparison.csv and
data/exp5_makeclf_comparison.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_LARGE_DATA_DIR = _REPO_ROOT / "data" / "large_archives"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
K_INFORMATIVE = 3  # number of truly informative variables in each benchmark


def data_file(name: str) -> Path:
    """Return an uncompressed CSV path, or a single-file ZIP archive fallback."""
    path = _DATA_DIR / name
    if path.exists():
        return path
    archive = _LARGE_DATA_DIR / f"{name}.zip"
    if archive.exists():
        return archive
    raise FileNotFoundError(
        f"Missing {name}. Expected {_DATA_DIR / name} or {archive}."
    )


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def topk_metrics(
    scores: np.ndarray,
    feature_names: list[str],
    ground_truth: dict[str, str],
    k: int,
) -> dict:
    """Return Precision@k, Recall@k, F1@k, selected_redundant, selected_noise."""
    ranked_idx = np.argsort(scores)[::-1]
    top_k = {feature_names[i] for i in ranked_idx[:k]}
    informative = {f for f, r in ground_truth.items() if r == "informative"}
    redundant = {f for f, r in ground_truth.items() if r == "redundant"}
    noise = {f for f, r in ground_truth.items() if r == "noise"}

    tp = len(top_k & informative)
    precision = tp / k
    recall = tp / len(informative) if informative else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "Precision@k": round(precision, 3),
        "Recall@k": round(recall, 3),
        "F1@k": round(f1, 3),
        "selected_redundant": len(top_k & redundant),
        "selected_noise": len(top_k & noise),
    }


def run_pi(rf: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=RANDOM_SEED)
    return result.importances_mean


def run_shap(rf: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    import shap
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X)
    # shap 0.40-: list [class0_array, class1_array], each (n, p)
    # shap 0.41+: ndarray (n, p, n_classes)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return np.abs(sv).mean(axis=0)


def run_mi(X: np.ndarray, y: np.ndarray, discrete: bool = False) -> np.ndarray:
    return mutual_info_classif(X, y, discrete_features=discrete, random_state=RANDOM_SEED)


def run_mrmr(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> np.ndarray:
    from mrmr import mrmr_classif
    p = len(feature_names)
    ranked = mrmr_classif(
        pd.DataFrame(X, columns=feature_names),
        pd.Series(y),
        K=p,
    )
    # Convert rank to score: rank-1 feature gets score p, last gets score 1
    rank_score = {feat: p - i for i, feat in enumerate(ranked)}
    # Features not ranked by mRMR (MI ≈ 0 after stop) get score 0
    return np.array([rank_score.get(f, 0.0) for f in feature_names], dtype=float)


def build_comparison_table(
    feature_names: list[str],
    ground_truth: dict[str, str],
    scores: dict[str, np.ndarray],
    k: int,
) -> pd.DataFrame:
    """Build wide comparison table with one row per feature plus metric rows."""
    rows = []
    for i, feat in enumerate(feature_names):
        row = {"feature": feat, "role": ground_truth.get(feat, "unknown")}
        for method, arr in scores.items():
            row[method] = round(float(arr[i]), 5)
        rows.append(row)
    df = pd.DataFrame(rows)

    # One metrics row per metric key
    metric_keys = ["Precision@k", "Recall@k", "F1@k", "selected_redundant", "selected_noise"]
    metric_rows = []
    for mk in metric_keys:
        row = {"feature": mk, "role": "---"}
        for method, arr in scores.items():
            m = topk_metrics(arr, feature_names, ground_truth, k)
            row[method] = m[mk]
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    return pd.concat([df, metrics_df], ignore_index=True)


# ---------------------------------------------------------------------------
# MONK-1
# ---------------------------------------------------------------------------

def run_monks1() -> pd.DataFrame:
    print("=== MONK-1 ===")
    from load_datasets import load_monks1_full
    X, y, feature_names, ground_truth = load_monks1_full()

    # NS_order1: from existing order-1 CSV
    o1 = pd.read_csv(data_file("null_swap_order1_monks1.csv"))
    ns_order1 = o1.set_index("feature")["score_order1"].reindex(feature_names).values

    # NS_max: max over contexts (rows) per target (column) in order-2 matrix
    mat = pd.read_csv(data_file("null_swap_order2_matrix_monks1.csv"), index_col=0)
    ns_max = mat.max(axis=0).reindex(feature_names).values

    # External baselines
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    rf.fit(X, y)

    pi = run_pi(rf, X, y)
    shap_s = run_shap(rf, X)
    mi = run_mi(X, y, discrete=True)
    mrmr = run_mrmr(X, y, feature_names)

    scores = {
        "PI": pi,
        "SHAP": shap_s,
        "MI": mi,
        "mRMR": mrmr,
        "NS_order1": ns_order1,
        "NS_max": ns_max,
    }

    table = build_comparison_table(feature_names, ground_truth, scores, K_INFORMATIVE)

    out = _DATA_DIR / "exp5_monks1_comparison.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(table.to_string())
    return table


# ---------------------------------------------------------------------------
# make_classification
# ---------------------------------------------------------------------------

def generate_makeclf_dataset() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=16384,
        n_features=30,
        n_informative=3,
        n_redundant=7,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        shuffle=False,
        random_state=42,
    )
    feature_names = [f"f{i:02d}" for i in range(30)]
    ground_truth = {
        f: ("informative" if i < 3 else "redundant" if i < 10 else "noise")
        for i, f in enumerate(feature_names)
    }
    return X, y, feature_names, ground_truth


def run_makeclf() -> pd.DataFrame:
    print("\n=== make_classification (n=16384) ===")
    X, y, feature_names, ground_truth = generate_makeclf_dataset()

    # NS_order1: filter order==1, n_samples==16384, mean delta per feature
    raw = pd.read_csv(data_file("null_swap_order3_logreg.csv"))
    n16 = raw[raw["n_samples"] == 16384]
    ns_order1 = (
        n16[n16["order"] == 1]
        .groupby("target_feature")["delta"]
        .mean()
        .reindex(feature_names)
        .values
    )

    # NS_max: mean delta per (target, context) pair, then max over contexts
    ctx_mean = (
        n16.groupby(["target_feature", "context_features"])["delta"]
        .mean()
        .reset_index()
    )
    ns_max = (
        ctx_mean.groupby("target_feature")["delta"]
        .max()
        .reindex(feature_names)
        .values
    )

    # External baselines (RF backbone)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    rf.fit(X, y)

    pi = run_pi(rf, X, y)
    shap_s = run_shap(rf, X)
    mi = run_mi(X, y, discrete=False)
    mrmr = run_mrmr(X, y, feature_names)

    scores = {
        "PI": pi,
        "SHAP": shap_s,
        "MI": mi,
        "mRMR": mrmr,
        "NS_order1": ns_order1,
        "NS_max": ns_max,
    }

    table = build_comparison_table(feature_names, ground_truth, scores, K_INFORMATIVE)

    out = _DATA_DIR / "exp5_makeclf_comparison.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(table.to_string())
    return table


if __name__ == "__main__":
    run_monks1()
    run_makeclf()
