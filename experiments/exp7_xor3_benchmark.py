"""Experiment 7: Pure XOR3 benchmark — third-order contextual complementarity.

DGP (p=10, n=4096):
  X1, X2, X3  : XOR3-informative (sign-thresholded Gaussians)
  X4 -- X10   : pure noise

  B_k = 1[X_k > 0]
  Y   = (B_1 + B_2 + B_3) mod 2

Information-theoretic properties (exact):
  I(Y; X_k)              = 0   for k in {1, 2, 3}  (no marginal MI)
  I(Y; X_k | X_l)        = 0   for any pair {k,l} in {1,2,3}  (no pairwise MI)
  I(Y; X_k | X_l, X_m)   = log(2) for {k,l,m} = {1,2,3}  (pure third-order CMI)

Expected null-swap behaviour:
  Order 1 : delta ≈ 0 for X1, X2, X3
  Order 2 : delta ≈ 0 for X1, X2, X3 regardless of context
  Order 3 : delta >> 0 for X_k with context {X_l, X_m}, {k,l,m} = {1,2,3}
             delta ≈ 0 for noise features at all orders
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

N = 4096
RANDOM_SEED = 42
P = 10
FEATURE_NAMES = [f"X{i+1}" for i in range(P)]
INFORMATIVE_IDX = [0, 1, 2]
NOISE_IDX = list(range(3, P))
TRUE_CMI = float(np.log(2))  # I(Y;Xk|Xl,Xm) = log(2) nats for pure XOR3


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def generate_xor3(seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_inf = rng.standard_normal((N, 3))
    X_noise = rng.standard_normal((N, P - 3))
    X = np.column_stack([X_inf, X_noise])
    B = (X_inf > 0).astype(int)
    y = (B[:, 0] ^ B[:, 1] ^ B[:, 2]).astype(int)
    return X, y


def verify_calibration(X: np.ndarray, y: np.ndarray) -> None:
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X[:, :3], y, random_state=RANDOM_SEED, n_neighbors=15)
    print(
        f"Calibration: I(Y;X1)={mi[0]:.4f}  "
        f"I(Y;X2)={mi[1]:.4f}  I(Y;X3)={mi[2]:.4f}"
    )
    assert all(m < 0.02 for m in mi), (
        f"XOR3 informative features should have near-zero marginal MI, got {mi}"
    )
    print("Calibration OK.")


# ---------------------------------------------------------------------------
# Null-swap runs
# ---------------------------------------------------------------------------

def run_nullswap_orders(
    X: np.ndarray,
    y: np.ndarray,
    orders: list[int],
) -> pd.DataFrame:
    from null_swap_core import compute_order_scores, make_decision_tree_estimator

    all_records = []
    for order in orders:
        print(f"  null-swap order {order}...")
        _, records = compute_order_scores(
            X,
            y,
            feature_names=FEATURE_NAMES,
            order=order,
            estimator_factory=make_decision_tree_estimator,
            n_repeats=5,
            n_folds=5,
            random_seed=RANDOM_SEED,
            n_jobs=-1,
        )
        records = records.copy()
        records["order"] = order
        all_records.append(records)

    return pd.concat(all_records, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def order1_summary(records: pd.DataFrame) -> pd.DataFrame:
    o1 = records[records["order"] == 1].copy()
    agg = o1.groupby("target_feature")["delta"].agg(["mean", "std"]).reset_index()
    agg.columns = ["feature", "delta_mean", "delta_std"]
    agg["role"] = agg["feature"].map(
        lambda f: "informative" if f in {"X1", "X2", "X3"} else "noise"
    )
    return agg.round(4)


def order2_inf_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Order-2 deltas restricted to the three informative features."""
    o2 = records[records["order"] == 2].copy()
    inf_feats = {"X1", "X2", "X3"}
    mask = o2["target_feature"].isin(inf_feats) & o2["context_features"].map(
        lambda c: all(f in inf_feats for f in c)
    )
    sub = o2[mask].copy()
    sub["context_str"] = sub["context_features"].map(lambda c: str(c))
    agg = sub.groupby(["context_str", "target_feature"])["delta"].agg(
        ["mean", "std"]
    ).reset_index()
    agg.columns = ["context", "target", "delta_mean", "delta_std"]
    return agg.round(4)


def order3_key_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Order-3 deltas for {X1,X2,X3} permutations and representative noise."""
    o3 = records[records["order"] == 3].copy()
    inf_feats = {"X1", "X2", "X3"}

    # Triplets where target and both context are from the informative set
    mask_inf = o3["target_feature"].isin(inf_feats) & o3["context_features"].map(
        lambda c: all(f in inf_feats for f in c)
    )
    sub_inf = o3[mask_inf].copy()

    # One noise representative per order-3 context containing two informatives
    mask_noise = (~o3["target_feature"].isin(inf_feats)) & o3["context_features"].map(
        lambda c: set(c) == {"X1", "X2"}  # fixed context for comparison
    )
    sub_noise = o3[mask_noise].head(3).copy()

    sub = pd.concat([sub_inf, sub_noise], ignore_index=True)
    sub["context_str"] = sub["context_features"].map(lambda c: str(c))
    agg = sub.groupby(["context_str", "target_feature"])["delta"].agg(
        ["mean", "std"]
    ).reset_index()
    agg.columns = ["context", "target", "delta_mean", "delta_std"]
    agg = agg.sort_values(["context", "target"]).reset_index(drop=True)
    return agg.round(4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_exp7() -> None:
    print("=== Exp 7: Pure XOR3 benchmark ===")
    X, y = generate_xor3()
    verify_calibration(X, y)

    print(f"  n={N}, p={P}, class balance: {y.mean():.3f}")
    print(f"  True CMI I(Y;Xk|Xl,Xm) = log(2) = {TRUE_CMI:.4f} nats")

    records = run_nullswap_orders(X, y, orders=[1, 2, 3])

    # Save full records
    out_full = _DATA_DIR / "exp7_xor3_records.csv"
    records.to_csv(out_full, index=False)
    print(f"Saved: {out_full}")

    # Order-1 summary
    s1 = order1_summary(records)
    out_s1 = _DATA_DIR / "exp7_xor3_order1.csv"
    s1.to_csv(out_s1, index=False)
    print("\n-- Order-1 deltas --")
    print(s1.to_string(index=False))

    # Order-2 informative-only summary
    s2 = order2_inf_summary(records)
    out_s2 = _DATA_DIR / "exp7_xor3_order2_inf.csv"
    s2.to_csv(out_s2, index=False)
    print("\n-- Order-2 deltas (informative variables only) --")
    print(s2.to_string(index=False))

    # Order-3 key summary
    s3 = order3_key_summary(records)
    out_s3 = _DATA_DIR / "exp7_xor3_order3_key.csv"
    s3.to_csv(out_s3, index=False)
    print("\n-- Order-3 key deltas --")
    print(s3.to_string(index=False))
    print(f"\n  Reference: true CMI = {TRUE_CMI:.4f} nats")

    # Verify XOR3 structure
    xor3_triplets = s3[
        s3["target"].isin({"X1", "X2", "X3"})
        & s3["context"].apply(lambda c: "X1" in c and "X2" in c or
                                         "X1" in c and "X3" in c or
                                         "X2" in c and "X3" in c)
    ]
    if not xor3_triplets.empty:
        min_xor3 = xor3_triplets["delta_mean"].min()
        print(
            f"\n  XOR3 triplet deltas: min={min_xor3:.4f} "
            f"(expected >> 0)"
        )
        if min_xor3 > 0.02:
            print("  Structure check: XOR3 third-order signal detected. OK.")
        else:
            print("  WARNING: XOR3 signal weaker than expected.")


if __name__ == "__main__":
    run_exp7()
