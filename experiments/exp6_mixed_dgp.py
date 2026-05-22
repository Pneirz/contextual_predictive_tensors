"""Experiment 6: Mixed DGP with matched-backbone threshold analysis.

DGP (p=15, n=4096):
  X1          : marginal singleton, X1 ~ N(0,1)
  X2, X3      : complementary pair (marginally uninformative)
  X4, X5      : redundant proxies of X1
  X6 -- X15   : pure noise

  Y = 1[1.0*X1 + 2.0*X2*X3 + e > 0],  e ~ N(0,1)

All model-based methods (null-swap, PI, SHAP) use RandomForestClassifier
so the comparison isolates summary type rather than estimator capacity.
MI and mRMR are model-free filter baselines.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from exp5_sota_comparison import (
    K_INFORMATIVE,
    RANDOM_SEED,
    build_comparison_table,
    run_mi,
    run_mrmr,
    run_pi,
    run_shap,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_FIG_DIR = _REPO_ROOT / "figures"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_FIG_DIR.mkdir(parents=True, exist_ok=True)

N = 4096
BETA1 = 1.0
BETA23 = 2.0
N_JOBS = int(os.environ.get("NULL_SWAP_N_JOBS", "-1"))
FEATURE_NAMES = [f"X{i+1}" for i in range(15)]
GROUND_TRUTH = {
    "X1": "informative",
    "X2": "informative",
    "X3": "informative",
    "X4": "redundant",
    "X5": "redundant",
    **{f"X{i}": "noise" for i in range(6, 16)},
}
INFORMATIVE_IDX = [0, 1, 2]
REDUNDANT_IDX = [3, 4]
NOISE_IDX = list(range(5, 15))

METHOD_COLORS = {
    "PI": "#e41a1c",
    "SHAP": "#ff7f00",
    "MI": "#984ea3",
    "mRMR": "#4daf4a",
    "NS_order1": "#377eb8",
    "NS_max": "#000000",
}


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def generate_mixed_dgp(seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal(N)
    X2 = rng.standard_normal(N)
    X3 = rng.standard_normal(N)
    X4 = X1 + rng.standard_normal(N) * 0.1
    X5 = X1 + rng.standard_normal(N) * 0.1
    noise_cols = rng.standard_normal((N, 10))
    e = rng.standard_normal(N)
    logit = BETA1 * X1 + BETA23 * X2 * X3 + e
    y = (logit > 0).astype(int)
    X = np.column_stack([X1, X2, X3, X4, X5, noise_cols])
    return X, y


def verify_calibration(X: np.ndarray, y: np.ndarray) -> None:
    mi = mutual_info_classif(X[:, :3], y, random_state=RANDOM_SEED, n_neighbors=15)
    print(f"Calibration: I(Y;X1)={mi[0]:.4f}  I(Y;X2)={mi[1]:.4f}  I(Y;X3)={mi[2]:.4f}")
    assert mi[0] > 0.05, f"X1 should be marginally informative, got {mi[0]:.4f}"
    assert mi[1] < 0.02, f"X2 should be near-zero marginal, got {mi[1]:.4f}"
    assert mi[2] < 0.02, f"X3 should be near-zero marginal, got {mi[2]:.4f}"
    print("Calibration OK.")


# ---------------------------------------------------------------------------
# Null-swap with RF backbone
# ---------------------------------------------------------------------------

def make_rf_estimator(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=50, random_state=random_state)


def run_nullswap(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    from null_swap_core import compute_order_scores, explode_raw_deltas

    all_records = []
    all_raw = []
    for order in [1, 2, 3]:
        print(f"  null-swap order {order}...")
        _, records = compute_order_scores(
            X, y,
            feature_names=FEATURE_NAMES,
            order=order,
            estimator_factory=make_rf_estimator,
            n_repeats=5,
            n_folds=5,
            random_seed=RANDOM_SEED,
            n_jobs=N_JOBS,
            store_raw=True,
        )
        records = records.copy()
        records["order"] = order
        all_records.append(records)
        raw = explode_raw_deltas(records, n_repeats=5, n_folds=5)
        raw["order"] = order
        all_raw.append(raw)

    raw_records = pd.concat(all_raw, ignore_index=True)
    raw_out = _DATA_DIR / "exp6_mixed_dgp_ns_raw.csv"
    raw_records.to_csv(raw_out, index=False)
    print(f"Saved: {raw_out}")
    return pd.concat(all_records, ignore_index=True)


# ---------------------------------------------------------------------------
# Score table
# ---------------------------------------------------------------------------

def compute_ns_scores(ns_records: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ns_o1 = (
        ns_records[ns_records["order"] == 1]
        .groupby("target_feature")["delta"]
        .mean()
        .reindex(FEATURE_NAMES)
        .values
    )
    ctx_mean = (
        ns_records.groupby(["target_feature", "context_features"])["delta"]
        .mean()
        .reset_index()
    )
    ns_max = (
        ctx_mean.groupby("target_feature")["delta"]
        .max()
        .reindex(FEATURE_NAMES)
        .values
    )
    return ns_o1, ns_max


def run_exp6() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    print("=== Exp 6: Mixed DGP ===")
    X, y = generate_mixed_dgp()
    verify_calibration(X, y)

    ns_records = run_nullswap(X, y)
    ns_order1, ns_max = compute_ns_scores(ns_records)

    # RF on all features for PI and SHAP (same class as null-swap backbone)
    rf_full = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_full.fit(X, y)

    pi = run_pi(rf_full, X, y)
    shap_s = run_shap(rf_full, X)
    mi = run_mi(X, y, discrete=False)
    mrmr = run_mrmr(X, y, FEATURE_NAMES)

    scores = {
        "PI": pi,
        "SHAP": shap_s,
        "MI": mi,
        "mRMR": mrmr,
        "NS_order1": ns_order1,
        "NS_max": ns_max,
    }

    table = build_comparison_table(FEATURE_NAMES, GROUND_TRUTH, scores, K_INFORMATIVE)
    out = _DATA_DIR / "exp6_mixed_dgp_scores.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(table.to_string())
    return table, ns_records, scores


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(scores_dict: dict[str, np.ndarray], n_steps: int = 100) -> pd.DataFrame:
    records = []
    for method, scores in scores_dict.items():
        s_min, s_max = float(scores.min()), float(scores.max())
        thresholds = np.linspace(s_min, s_max, n_steps)
        for thresh in thresholds:
            selected = set(int(i) for i in np.where(scores > thresh)[0])
            recall_inf = len(selected & set(INFORMATIVE_IDX)) / len(INFORMATIVE_IDX)
            fp_noise = len(selected & set(NOISE_IDX)) / len(NOISE_IDX)
            sel_red = len(selected & set(REDUNDANT_IDX)) / len(REDUNDANT_IDX)
            records.append({
                "method": method,
                "threshold": round(thresh, 6),
                "recall_informative": round(recall_inf, 4),
                "fp_noise_rate": round(fp_noise, 4),
                "selected_redundant_rate": round(sel_red, 4),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_threshold_sweep(sweep: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for method, grp in sweep.groupby("method"):
        color = METHOD_COLORS.get(method, "grey")
        lw = 2.5 if method.startswith("NS") else 1.2
        ls = "-" if method == "NS_max" else ("--" if method == "NS_order1" else "-")

        axes[0].plot(
            grp["fp_noise_rate"], grp["recall_informative"],
            label=method, color=color, lw=lw, ls=ls,
        )
        axes[1].plot(
            grp["selected_redundant_rate"], grp["recall_informative"],
            label=method, color=color, lw=lw, ls=ls,
        )

    for ax, xlabel in zip(axes, ["FP-noise rate", "Redundant selection rate"]):
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Recall (informative)", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_title("Pure-noise FP rate vs informative recall")
    axes[1].set_title("Redundant selection rate vs informative recall")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    score_table, ns_records, scores = run_exp6()

    sweep = threshold_sweep(scores)
    sweep_out = _DATA_DIR / "exp6_threshold_sweep.csv"
    sweep.to_csv(sweep_out, index=False)
    print(f"Saved: {sweep_out}")

    fig_out = _FIG_DIR / "exp6_threshold_sweep.png"
    plot_threshold_sweep(sweep, fig_out)
