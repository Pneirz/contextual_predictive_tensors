"""Experiment 8: Log-loss Bayes-consistency convergence.

Objective: show that the null-swap contrast under log-loss converges toward
conditional mutual information (CMI) as sample size grows, while the RMSE-based
contrast converges to a different limit.

DGP: binary XOR2
  X1, X2 ~ Bernoulli(0.5), independent
  Y       = X1 XOR X2  (= (X1 + X2) mod 2)
  X3, X4  : pure noise (Bernoulli(0.5))

True CMI:
  I(Y ; X1 | X2) = H(Y | X2) - H(Y | X1, X2) = log(2) - 0 = log(2) nats

Analytical limits:
  - Under RMSE + Bayes-consistent estimator:
      Delta_RMSE[{X2}, X1] -> RMSE(null) - RMSE(orig)
                             = 0.5 - 0 = 0.5
  - Under log-loss + Bayes-consistent estimator:
      Delta_logloss[{X2}, X1] -> log-loss(null) - log-loss(orig)
                               = log(2) - 0 = 0.6931...

The experiment reports both deltas at increasing n and compares them to:
  - True CMI = log(2) ~= 0.6931
  - RMSE analytical limit = 0.5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
P = 4
FEATURE_NAMES = ["X1", "X2", "X3", "X4"]
TRUE_CMI = float(np.log(2))          # log(2) nats ~= 0.6931
RMSE_LIMIT = 0.5                     # analytical RMSE limit for XOR2
N_REPEATS = 10
N_FOLDS = 5
SAMPLE_SIZES = [200, 400, 800, 1600, 3200, 6400, 12800]


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def generate_xor2(n: int, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, P)).astype(float)
    y = (X[:, 0].astype(int) ^ X[:, 1].astype(int)).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Core delta computation with pluggable loss
# ---------------------------------------------------------------------------

def nullswap_deltas_with_loss(
    X: np.ndarray,
    y: np.ndarray,
    context_idx: int,
    target_idx: int,
    loss: str,
    calibrate: bool,
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> list[dict[str, float | int]]:
    """Return repeat/fold null-swap deltas.

    loss : "rmse" or "logloss"
    calibrate : if True, wrap DT in isotonic CalibratedClassifierCV
    """
    all_cols = [context_idx, target_idx]
    target_col_pos = 1  # target is always at position 1 in the 2-col subset

    rows: list[dict[str, float | int]] = []
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed + repeat)
        for fold_num, (tr, te) in enumerate(kf.split(X)):
            X_tr = X[tr][:, all_cols]
            X_te = X[te][:, all_cols]
            y_tr, y_te = y[tr], y[te]

            base = DecisionTreeClassifier(random_state=random_seed + repeat)
            if calibrate:
                model = CalibratedClassifierCV(base, method="isotonic", cv=3)
            else:
                model = base
            model.fit(X_tr, y_tr)
            p_orig = model.predict_proba(X_te)[:, 1]

            rng = np.random.default_rng(
                random_seed + repeat * n_folds * 100 + fold_num
            )
            X_tr_null = X_tr.copy()
            X_tr_null[:, target_col_pos] = rng.permutation(
                X_tr_null[:, target_col_pos]
            )
            base_null = DecisionTreeClassifier(random_state=random_seed + repeat)
            if calibrate:
                model_null = CalibratedClassifierCV(
                    base_null, method="isotonic", cv=3
                )
            else:
                model_null = base_null
            model_null.fit(X_tr_null, y_tr)
            p_null = model_null.predict_proba(X_te)[:, 1]

            p_orig = np.clip(p_orig, 1e-7, 1 - 1e-7)
            p_null = np.clip(p_null, 1e-7, 1 - 1e-7)

            if loss == "rmse":
                l_orig = root_mean_squared_error(y_te, p_orig)
                l_null = root_mean_squared_error(y_te, p_null)
            else:
                l_orig = log_loss(y_te, p_orig)
                l_null = log_loss(y_te, p_null)

            rows.append({
                "repeat": repeat,
                "fold": fold_num,
                "delta": float(l_null - l_orig),
            })

    return rows


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw Exp 8 fold deltas while preserving old output columns."""
    return (
        raw.groupby(["n", "protocol", "loss", "calibrated"], as_index=False)["delta"]
        .agg(delta_mean="mean", delta_std="std")
        .assign(
            delta_mean=lambda df: df["delta_mean"].round(5),
            delta_std=lambda df: df["delta_std"].round(5),
        )
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp8() -> pd.DataFrame:
    print("=== Exp 8: Log-loss Bayes-consistency convergence ===")
    print(f"  True CMI = log(2) = {TRUE_CMI:.4f} nats")
    print(f"  RMSE analytical limit = {RMSE_LIMIT:.4f}")
    print(f"  Contrast: Delta[{{X2}}, X1]  (order 2)")

    # X1 is target_idx=0, X2 is context_idx=1 in FEATURE_NAMES
    CONTEXT_IDX = 1  # X2
    TARGET_IDX = 0   # X1

    protocols = [
        ("DT + RMSE", "rmse", False),
        ("DT + log-loss", "logloss", False),
        ("DT + log-loss (calibrated)", "logloss", True),
    ]

    raw_rows = []
    for n in SAMPLE_SIZES:
        print(f"\n  n = {n}")
        X, y = generate_xor2(n)
        for label, loss, calibrate in protocols:
            rows = nullswap_deltas_with_loss(
                X, y,
                context_idx=CONTEXT_IDX,
                target_idx=TARGET_IDX,
                loss=loss,
                calibrate=calibrate,
            )
            for row in rows:
                raw_rows.append({
                    "n": n,
                    "protocol": label,
                    "loss": loss,
                    "calibrated": calibrate,
                    "context_feature": FEATURE_NAMES[CONTEXT_IDX],
                    "target_feature": FEATURE_NAMES[TARGET_IDX],
                    **row,
                })
            arr = np.array([row["delta"] for row in rows], dtype=float)
            mean_d = float(arr.mean())
            std_d = float(arr.std(ddof=1))
            ref = TRUE_CMI if loss == "logloss" else RMSE_LIMIT
            print(
                f"    {label:<35}: "
                f"delta={mean_d:.4f} +/- {std_d:.4f}  "
                f"(ref={ref:.4f})"
            )

    raw = pd.DataFrame(raw_rows)
    raw_out = _DATA_DIR / "exp8_logloss_consistency_raw.csv"
    raw.to_csv(raw_out, index=False)
    print(f"\nSaved: {raw_out}")

    df = summarize_raw(raw)
    out = _DATA_DIR / "exp8_logloss_consistency.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

    # Print pivot summary
    print("\n-- Convergence toward true CMI (log-loss, calibrated) --")
    sub = df[df["protocol"] == "DT + log-loss (calibrated)"][
        ["n", "delta_mean", "delta_std"]
    ]
    sub = sub.rename(columns={"delta_mean": "delta", "delta_std": "std"})
    sub["true_cmi"] = round(TRUE_CMI, 4)
    sub["gap"] = (sub["delta"] - TRUE_CMI).round(4)
    print(sub.to_string(index=False))

    print("\n-- Convergence of RMSE contrast to 0.5 limit --")
    sub2 = df[df["protocol"] == "DT + RMSE"][["n", "delta_mean", "delta_std"]]
    sub2 = sub2.rename(columns={"delta_mean": "delta", "delta_std": "std"})
    sub2["rmse_limit"] = RMSE_LIMIT
    sub2["gap"] = (sub2["delta"] - RMSE_LIMIT).round(4)
    print(sub2.to_string(index=False))

    return df


if __name__ == "__main__":
    run_exp8()
