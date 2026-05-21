"""Outer-seed uncertainty checks for prespecified DGP-level claims.

This script complements the full tensor runs. It reruns only the tensor entries
used to support DGP-level claims, varying the dataset seed and preserving
repeat/fold raw deltas for seed-level bootstrap summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from exp6_mixed_dgp import (
    FEATURE_NAMES as MIXED_FEATURE_NAMES,
    generate_mixed_dgp,
    make_rf_estimator,
)
from exp7_xor3_benchmark import (
    FEATURE_NAMES as XOR3_FEATURE_NAMES,
    N as XOR3_N,
    generate_xor3,
)
from exp8_logloss_consistency import (
    FEATURE_NAMES as XOR2_FEATURE_NAMES,
    generate_xor2,
    nullswap_deltas_with_loss,
)
from null_swap_core import make_decision_tree_estimator, nullswap_delta


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 42
N_REPEATS = 5
N_FOLDS = 5
MIXED_N = 4096


def _append_nullswap_rows(
    rows: list[dict],
    *,
    experiment: str,
    claim: str,
    seed_ix: int,
    seed: int,
    order: int,
    context_indices: Iterable[int],
    target_idx: int,
    feature_names: list[str],
    raw_deltas: list[float],
    n: int,
    estimator: str,
    loss: str = "rmse",
) -> None:
    context_indices = tuple(context_indices)
    context_features = tuple(feature_names[idx] for idx in context_indices)
    for i, delta in enumerate(raw_deltas):
        rows.append({
            "experiment": experiment,
            "claim": claim,
            "seed_ix": seed_ix,
            "seed": seed,
            "n": n,
            "order": order,
            "estimator": estimator,
            "loss": loss,
            "target_feature": feature_names[target_idx],
            "context_features": context_features,
            "repeat": i // N_FOLDS,
            "fold": i % N_FOLDS,
            "delta": float(delta),
        })


def run_xor3_outer(rows: list[dict], seeds: list[int]) -> None:
    entries = [
        ("xor3_order1_zero", 1, (), 0),
        ("xor3_order1_zero", 1, (), 1),
        ("xor3_order1_zero", 1, (), 2),
        ("xor3_order2_zero", 2, (1,), 0),
        ("xor3_order2_zero", 2, (2,), 0),
        ("xor3_order2_zero", 2, (0,), 1),
        ("xor3_order2_zero", 2, (2,), 1),
        ("xor3_order2_zero", 2, (0,), 2),
        ("xor3_order2_zero", 2, (1,), 2),
        ("xor3_order3_signal", 3, (0, 1), 2),
        ("xor3_order3_signal", 3, (0, 2), 1),
        ("xor3_order3_signal", 3, (1, 2), 0),
    ]
    for seed_ix, seed in enumerate(seeds):
        print(f"[xor3] seed {seed_ix + 1}/{len(seeds)}: {seed}")
        X, y = generate_xor3(seed=seed)
        for claim, order, context, target_idx in entries:
            raw = nullswap_delta(
                X=X,
                y=y,
                context_indices=context,
                target_idx=target_idx,
                estimator_factory=make_decision_tree_estimator,
                n_repeats=N_REPEATS,
                n_folds=N_FOLDS,
                random_seed=seed,
                return_all=True,
            )
            _append_nullswap_rows(
                rows,
                experiment="xor3",
                claim=claim,
                seed_ix=seed_ix,
                seed=seed,
                order=order,
                context_indices=context,
                target_idx=target_idx,
                feature_names=XOR3_FEATURE_NAMES,
                raw_deltas=raw,
                n=XOR3_N,
                estimator="dt",
            )


def run_logloss_outer(rows: list[dict], seeds: list[int], sample_sizes: list[int]) -> None:
    protocols = [
        ("DT + RMSE", "rmse", False),
        ("DT + log-loss", "logloss", False),
        ("DT + log-loss (calibrated)", "logloss", True),
    ]
    context_idx = 1  # X2
    target_idx = 0   # X1
    for seed_ix, seed in enumerate(seeds):
        print(f"[logloss] seed {seed_ix + 1}/{len(seeds)}: {seed}")
        for n in sample_sizes:
            X, y = generate_xor2(n=n, seed=seed)
            for protocol, loss, calibrate in protocols:
                raw = nullswap_deltas_with_loss(
                    X=X,
                    y=y,
                    context_idx=context_idx,
                    target_idx=target_idx,
                    loss=loss,
                    calibrate=calibrate,
                    n_repeats=N_REPEATS,
                    n_folds=N_FOLDS,
                    random_seed=seed,
                )
                for row in raw:
                    rows.append({
                        "experiment": "xor2_logloss",
                        "claim": "loss_specific_limit",
                        "seed_ix": seed_ix,
                        "seed": seed,
                        "n": n,
                        "order": 2,
                        "estimator": "dt",
                        "protocol": protocol,
                        "loss": loss,
                        "calibrated": calibrate,
                        "target_feature": XOR2_FEATURE_NAMES[target_idx],
                        "context_feature": XOR2_FEATURE_NAMES[context_idx],
                        "context_features": (XOR2_FEATURE_NAMES[context_idx],),
                        **row,
                    })


def run_mixed_outer(rows: list[dict], seeds: list[int]) -> None:
    entries = [
        ("mixed_singleton_signal", 1, (), 0),
        ("mixed_redundant_proxy", 1, (), 3),
        ("mixed_redundant_proxy", 1, (), 4),
        ("mixed_pair_complement", 2, (2,), 1),
        ("mixed_pair_complement", 2, (1,), 2),
        ("mixed_noise_control", 1, (), 5),
        ("mixed_noise_control", 1, (), 6),
    ]
    for seed_ix, seed in enumerate(seeds):
        print(f"[mixed] seed {seed_ix + 1}/{len(seeds)}: {seed}")
        X, y = generate_mixed_dgp(seed=seed)
        for claim, order, context, target_idx in entries:
            raw = nullswap_delta(
                X=X,
                y=y,
                context_indices=context,
                target_idx=target_idx,
                estimator_factory=make_rf_estimator,
                n_repeats=N_REPEATS,
                n_folds=N_FOLDS,
                random_seed=seed,
                return_all=True,
            )
            _append_nullswap_rows(
                rows,
                experiment="mixed_dgp",
                claim=claim,
                seed_ix=seed_ix,
                seed=seed,
                order=order,
                context_indices=context,
                target_idx=target_idx,
                feature_names=MIXED_FEATURE_NAMES,
                raw_deltas=raw,
                n=MIXED_N,
                estimator="rf",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--logloss-sizes",
        type=int,
        nargs="+",
        default=[800, 3200, 12800],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "outer_seed_uq_raw.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seed + i for i in range(args.n_seeds)]
    rows: list[dict] = []
    run_xor3_outer(rows, seeds)
    run_logloss_outer(rows, seeds, args.logloss_sizes)
    run_mixed_outer(rows, seeds)
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Rows: {len(out):,}")


if __name__ == "__main__":
    main()
