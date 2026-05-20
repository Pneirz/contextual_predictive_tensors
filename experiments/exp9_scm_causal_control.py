"""Experiment 9: SCM causal-control — non-collapse of C, I, and P.

Demonstrates concretely that three notions of variable relevance do not coincide:

  C  : causal relevance — does do(Xk=v) change P(Y)?
  I  : informational relevance — is I(Y; Xk | context) > 0?
  P  : operational predictivity — does null-swap show a positive delta?

SCM design (p=8 observable variables, 1 latent confounder U):

  U   ~ N(0, 1)                             latent confounder (not in X)
  X1  = 0.8*U + N(0, 0.36)                  confounder proxy     C: NO  I: YES  P: YES
  X2  = 0.8*U + N(0, 0.36)                  confounder proxy     C: NO  I: YES  P: YES
  X3  ~ N(0, 1)                             true direct cause    C: YES I: YES  P: YES
  X4  = X3 + N(0, 0.01)                     redundant proxy      C: NO  I: YES  P: YES
  X5..X8 ~ N(0, 1)                          pure noise           C: NO  I: NO   P: NO

  Y   = 1[ 1.0*X3 + 1.0*U + N(0,1) > 0 ]

Interventional ATEs are computed by simulation (cut incoming edges):
  ATE(Xk) = E[Y | do(Xk=+1)] - E[Y | do(Xk=-1)]

Null-swap scores (P) are order-1 deltas from the empirical contrast.
Marginal MI (I proxy) is estimated via k-NN mutual information.

Key predicted pattern:
  | Var  | Role              | do-ATE | MI   | NS delta |
  |------|-------------------|--------|------|----------|
  | X1   | confounder proxy  | ~0     | >0   | >0       |
  | X2   | confounder proxy  | ~0     | >0   | >0       |
  | X3   | true cause        | >0     | >0   | >0       |
  | X4   | redundant proxy   | ~0     | >0   | >0       |
  | X5-8 | noise             | ~0     | ~0   | ~0       |

The table exposes C != I != P: methods in P (and I) agree on four informative
variables, but only do-calculus identifies X3 as the unique causal variable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

N_OBS = 4096          # observational sample size
N_INT = 50_000        # samples per interventional estimate (large for precision)
RANDOM_SEED = 42
P = 8
FEATURE_NAMES = [f"X{i+1}" for i in range(P)]

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

CAUSAL_SET = {"X3"}    # variables in C
INFO_SET = {"X1", "X2", "X3", "X4"}   # variables in I (expected)


# ---------------------------------------------------------------------------
# SCM data generation
# ---------------------------------------------------------------------------

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
    logit = 1.0 * X3 + 1.0 * U + eps_Y
    y = (logit > 0).astype(int)
    return X, y


def interventional_ate(
    k: int,
    v_hi: float,
    v_lo: float,
    n: int = N_INT,
    seed: int = RANDOM_SEED,
) -> float:
    """Estimate ATE = E[Y|do(Xk=v_hi)] - E[Y|do(Xk=v_lo)] by simulation.

    Implements do(Xk=v) by setting Xk=v and cutting its incoming edges.
    For X1, X2, X4 the structural equation is replaced by X_k := v.
    For X3 (true cause) similarly, but Y = 1[1.0*v + 1.0*U + eps > 0].
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(n)
    eps_Y = rng.standard_normal(n)

    def _y_under_do(v: float) -> np.ndarray:
        # Set Xk=v for all units; U and eps_Y are shared across interventions
        if k == 2:   # X3 (0-indexed): the true cause
            x3_val = np.full(n, v)
        else:
            x3_val = rng.standard_normal(n)  # X3 evolves freely
        logit = 1.0 * x3_val + 1.0 * U + eps_Y
        return (logit > 0).astype(float)

    # For X3, do(X3=v_hi) vs do(X3=v_lo): different x3 values, same U and eps_Y
    if k == 2:
        y_hi = _y_under_do(v_hi)
        y_lo = _y_under_do(v_lo)
    else:
        # For all other Xk: setting Xk does not enter Y's equation, so
        # Y = 1[X3 + U + eps > 0] regardless.  ATE = 0 by construction.
        y_hi = _y_under_do(v_hi)
        y_lo = _y_under_do(v_lo)

    return float(y_hi.mean() - y_lo.mean())


def compute_all_ates(v_hi: float = 1.0, v_lo: float = -1.0) -> pd.Series:
    """Compute ATE for each observed variable via simulation."""
    ates = {}
    for idx, name in enumerate(FEATURE_NAMES):
        ate = interventional_ate(k=idx, v_hi=v_hi, v_lo=v_lo)
        ates[name] = ate
    return pd.Series(ates).round(4)


# ---------------------------------------------------------------------------
# Null-swap order-1 (P metric)
# ---------------------------------------------------------------------------

def run_nullswap_order1(X: np.ndarray, y: np.ndarray) -> pd.Series:
    from null_swap_core import compute_order_scores, make_logreg_estimator

    scores, _ = compute_order_scores(
        X,
        y,
        feature_names=FEATURE_NAMES,
        order=1,
        estimator_factory=make_logreg_estimator,
        n_repeats=5,
        n_folds=5,
        random_seed=RANDOM_SEED,
        n_jobs=-1,
    )
    return scores.round(4)


# ---------------------------------------------------------------------------
# Marginal MI (I proxy)
# ---------------------------------------------------------------------------

def compute_mi(X: np.ndarray, y: np.ndarray) -> pd.Series:
    mi = mutual_info_classif(X, y, random_state=RANDOM_SEED, n_neighbors=15)
    return pd.Series(mi, index=FEATURE_NAMES).round(4)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp9() -> pd.DataFrame:
    print("=== Exp 9: SCM causal-control — non-collapse of C, I, P ===")
    X, y = sample_observational(N_OBS)
    print(f"  n={N_OBS}, p={P}, class balance: {y.mean():.3f}")

    print("  Computing ATEs via interventional simulation (do-calculus)...")
    ates = compute_all_ates()

    print("  Computing marginal MI (I proxy)...")
    mi = compute_mi(X, y)

    print("  Running null-swap order 1 (P)...")
    ns_scores = run_nullswap_order1(X, y)

    # Assemble results table
    result = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "role": [ROLES[f] for f in FEATURE_NAMES],
        "in_C": [f in CAUSAL_SET for f in FEATURE_NAMES],
        "in_I_expected": [f in INFO_SET for f in FEATURE_NAMES],
        "ATE_do": [ates[f] for f in FEATURE_NAMES],
        "MI_marginal": [mi[f] for f in FEATURE_NAMES],
        "NS_delta_order1": [ns_scores[f] for f in FEATURE_NAMES],
    })

    out = _DATA_DIR / "exp9_scm_causal_control.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    print("\n-- C / I / P comparison table --")
    print(result.to_string(index=False))

    # Summary: classify each variable into the three categories
    print("\n-- Category membership (empirical thresholds: MI > 0.02, NS > 0.01, ATE abs > 0.05) --")
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

    return result


if __name__ == "__main__":
    run_exp9()
