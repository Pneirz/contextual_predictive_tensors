"""Resumable outer-seed UQ runner for the prespecified DGP checks.

Unlike outer_seed_uq.py, this script writes one CSV chunk per seed before
moving to the next seed. It is intended for long reruns where interruption is
more likely than logic failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from outer_seed_uq import (
    BASE_SEED,
    DATA_DIR,
    run_logloss_outer,
    run_mixed_outer,
    run_xor3_outer,
)


DEFAULT_CHUNK_DIR = DATA_DIR / "outer_seed_uq_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["xor3", "logloss", "mixed"],
        default=["xor3", "logloss", "mixed"],
    )
    parser.add_argument(
        "--logloss-sizes",
        type=int,
        nargs="+",
        default=[800, 3200, 12800],
    )
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=DATA_DIR / "outer_seed_uq_raw_long.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute chunks even if their seed CSV already exists.",
    )
    return parser.parse_args()


def run_one_seed(seed_ix: int, seed: int, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    if "xor3" in args.experiments:
        run_xor3_outer(rows, [seed])
    if "logloss" in args.experiments:
        run_logloss_outer(rows, [seed], args.logloss_sizes)
    if "mixed" in args.experiments:
        run_mixed_outer(rows, [seed])
    out = pd.DataFrame(rows)
    out["seed_ix"] = seed_ix
    return out


def combine_chunks(chunk_dir: Path, combined_output: Path) -> pd.DataFrame:
    chunks = sorted(chunk_dir.glob("seed_*.csv"))
    if not chunks:
        raise SystemExit(f"No chunks found in {chunk_dir}")
    df = pd.concat((pd.read_csv(path) for path in chunks), ignore_index=True)
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(combined_output, index=False)
    print(f"Combined {len(chunks)} chunks into {combined_output}")
    print(f"Rows: {len(df):,}")
    return df


def main() -> None:
    args = parse_args()
    args.chunk_dir.mkdir(parents=True, exist_ok=True)

    for seed_ix in range(args.n_seeds):
        seed = args.seed + seed_ix
        chunk_path = args.chunk_dir / f"seed_{seed:05d}.csv"
        if chunk_path.exists() and not args.force:
            print(f"[skip] seed_ix={seed_ix} seed={seed}: {chunk_path}")
            continue

        print(f"[run] seed_ix={seed_ix} seed={seed}")
        tmp_path = chunk_path.with_suffix(".tmp.csv")
        seed_df = run_one_seed(seed_ix, seed, args)
        seed_df.to_csv(tmp_path, index=False)
        tmp_path.replace(chunk_path)
        print(f"[saved] {chunk_path} rows={len(seed_df):,}")

    combine_chunks(args.chunk_dir, args.combined_output)


if __name__ == "__main__":
    main()
