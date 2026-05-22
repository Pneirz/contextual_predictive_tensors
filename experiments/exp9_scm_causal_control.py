"""Experiment 9: SCM causal-control, non-collapse of C, I, and P.

Demonstrates concretely that three notions of variable relevance do not coincide:

  C  : causal relevance, does do(Xk=v) change P(Y)?
  I  : informational relevance, is I(Y; Xk | context) > 0?
  P  : operational predictivity, does null-swap show a positive delta?

The experiment now runs multiple observational/interventional seeds and stores
both seed-level metric summaries and null-swap repeat/fold raw records. This
supports uncertainty intervals over the constructed SCM example.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

N_OBS = 4096
N_INT = 50_000
N_SEEDS = 20
RANDOM_SEED = 42
N_JOBS = -1
P = 8
FEATURE_NAMES = [f"X{i + 1}" for i in range(P)]

ROLES = {
    "X1": "confounder_proxy",
    "X2": "confounder_proxy",
    "X3": "true_cause",
    "X4": "redundant_proxy",
    "X5": "noise",
    "X6": "noise",
    "X7": "noise",
    "X8": "noise",
}

CAUSAL_SET = {"X3"}
INFO_SET = {"X1", "X2", "X3", "X4"}


def sample_observational(n: int, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Sample from the observational distribution of the SCM."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(n)
    X1 = 0.8 * U + rng.standard_normal(n) * 0.6
    X2 = 0.8 * U + rng.standard_normal(n) * 0.6
    X3 = rng.standard_normal(n)
    X4 = X3 + rng.standard_normal(n) * 0.1
    X_noise = rng.standard_normal((n, 4))
    X = np.column_stack([X1, X2, X3, X4, X_noise])
    eps_Y = rng.standard_normal(n)
    logit = X3 + U + eps_Y
    y = (logit > 0).astype(int)
    return X, y


def interventional_ate(
    k: int,
    v_hi: float,
    v_lo: float,
    n: int = N_INT,
    seed: int = RANDOM_SEED,
) -> float:
    """Estimate ATE = E[Y|do(Xk=v_hi)] - E[Y|do(Xk=v_lo)]."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(n)
    eps_Y = rng.standard_normal(n)

    def _y_under_do(v: float) -> np.ndarray:
        if k == 2:
            x3_val = np.full(n, v)
        else:
            x3_val = rng.standard_normal(n)
        logit = x3_val + U + eps_Y
        return (logit > 0).astype(float)

    y_hi = _y_under_do(v_hi)
    y_lo = _y_under_do(v_lo)
    return float(y_hi.mean() - y_lo.mean())


def compute_all_ates_for_seed(
    seed: int,
    v_hi: float = 1.0,
    v_lo: float = -1.0,
) -> pd.Series:
    """Compute ATE for each observed variable using seed-specific streams."""
    ates = {}
    for idx, name in enumerate(FEATURE_NAMES):
        ates[name] = interventional_ate(
            k=idx,
            v_hi=v_hi,
            v_lo=v_lo,
            seed=seed + idx * 10_000,
        )
    return pd.Series(ates)


def run_nullswap_order1(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return order-1 null-swap scores and repeat/fold raw records."""
    from null_swap_core import compute_order_scores, explode_raw_deltas, make_logreg_estimator

    scores, records = compute_order_scores(
        X,
        y,
        feature_names=FEATURE_NAMES,
        order=1,
        estimator_factory=make_logreg_estimator,
        n_repeats=5,
        n_folds=5,
        random_seed=seed,
        n_jobs=N_JOBS,
        store_raw=True,
    )
    raw = explode_raw_deltas(records, n_repeats=5, n_folds=5)
    return scores, raw


def compute_mi(X: np.ndarray, y: np.ndarray, seed: int) -> pd.Series:
    """Estimate marginal mutual information with seed-controlled kNN MI."""
    mi = mutual_info_classif(X, y, random_state=seed, n_neighbors=15)
    return pd.Series(mi, index=FEATURE_NAMES)


def run_one_seed(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all SCM measurements for one seed."""
    X, y = sample_observational(N_OBS, seed=seed)
    print(f"    n={N_OBS}, p={P}, class balance: {y.mean():.3f}")

    print("    Computing ATEs via interventional simulation (do-calculus)...")
    ates = compute_all_ates_for_seed(seed=seed)

    print("    Computing marginal MI (I proxy)...")
    mi = compute_mi(X, y, seed=seed)

    print("    Running null-swap order 1 (P)...")
    ns_scores, ns_raw = run_nullswap_order1(X, y, seed=seed)
    ns_raw["seed"] = seed

    rows = []
    for feature in FEATURE_NAMES:
        rows.append({
            "seed": seed,
            "feature": feature,
            "role": ROLES[feature],
            "in_C": feature in CAUSAL_SET,
            "in_I_expected": feature in INFO_SET,
            "ATE_do": float(ates[feature]),
            "MI_marginal": float(mi[feature]),
            "NS_delta_order1": float(ns_scores[feature]),
        })
    return pd.DataFrame(rows), ns_raw


def summarize_seed_results(seed_result: pd.DataFrame) -> pd.DataFrame:
    """Build the backward-compatible mean table used by the manuscript."""
    result = (
        seed_result
        .groupby(["feature", "role", "in_C", "in_I_expected"], as_index=False)
        [["ATE_do", "MI_marginal", "NS_delta_order1"]]
        .mean()
    )
    result[["ATE_do", "MI_marginal", "NS_delta_order1"]] = result[
        ["ATE_do", "MI_marginal", "NS_delta_order1"]
    ].round(4)
    return result


def print_membership_summary(result: pd.DataFrame) -> None:
    """Print empirical C/I/P memberships using fixed thresholds."""
    print("\n-- Category membership (thresholds: MI > 0.02, NS > 0.01, ATE abs > 0.05) --")
    thr_mi = 0.02
    thr_ns = 0.01
    thr_ate = 0.05

    for _, row in result.iterrows():
        in_I_empirical = row["MI_marginal"] > thr_mi
        in_P_empirical = row["NS_delta_order1"] > thr_ns
        in_C_empirical = abs(row["ATE_do"]) > thr_ate
        cats = []
        if in_C_empirical:
            cats.append("C")
        if in_I_empirical:
            cats.append("I")
        if in_P_empirical:
            cats.append("P")
        membership = " & ".join(cats) if cats else "(none)"
        print(
            f"  {row['feature']} ({row['role']:<20}): "
            f"ATE={row['ATE_do']:+.3f}  MI={row['MI_marginal']:.3f}  "
            f"NS={row['NS_delta_order1']:.3f}  -> {membership}"
        )


def run_exp9() -> pd.DataFrame:
    print("=== Exp 9: SCM causal-control - non-collapse of C, I, P ===")
    seed_tables = []
    ns_raw_tables = []

    for seed_ix in range(N_SEEDS):
        seed = RANDOM_SEED + seed_ix
        print(f"\n  seed {seed_ix + 1}/{N_SEEDS}: {seed}")
        seed_table, ns_raw = run_one_seed(seed)
        seed_tables.append(seed_table)
        ns_raw_tables.append(ns_raw)

    seed_result = pd.concat(seed_tables, ignore_index=True)
    seed_out = _DATA_DIR / "exp9_scm_causal_control_seed_summary.csv"
    seed_result.to_csv(seed_out, index=False)
    print(f"\nSaved: {seed_out}")

    ns_raw_result = pd.concat(ns_raw_tables, ignore_index=True)
    ns_raw_out = _DATA_DIR / "exp9_scm_causal_control_ns_raw.csv"
    ns_raw_result.to_csv(ns_raw_out, index=False)
    print(f"Saved: {ns_raw_out}")

    result = summarize_seed_results(seed_result)
    out = _DATA_DIR / "exp9_scm_causal_control.csv"
    result.to_csv(out, index=False)
    print(f"Saved: {out}")

    print("\n-- C / I / P comparison table --")
    print(result.to_string(index=False))
    print_membership_summary(result)
    return result


if __name__ == "__main__":
    run_exp9()
