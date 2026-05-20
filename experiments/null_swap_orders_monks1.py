"""Null-swap validation on the classic MONK-1 benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from null_swap_core import (
    build_order2_matrix,
    compute_order_scores,
    make_decision_tree_estimator,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_FIG_DIR = _REPO_ROOT / "figures"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_FIG_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = 5
N_FOLDS = 5
RANDOM_SEED = 42


def plot_score_distributions(
    order1: pd.Series,
    order2: pd.Series,
    ground_truth: dict[str, str],
    out_path: Path,
) -> None:
    """Plot order-1 and order-2 target scores grouped by feature role."""
    import matplotlib.pyplot as plt

    groups = ["informative", "noise"]
    colors = {"informative": "#2166ac", "noise": "#d1d1d1"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, (scores, title) in zip(axes, [(order1, "Order 1"), (order2, "Order 2")]):
        data_by_group = [
            [scores[feat] for feat in scores.index if ground_truth.get(feat) == group]
            for group in groups
        ]
        bp = ax.boxplot(data_by_group, patch_artist=True, tick_labels=groups)
        for patch, group in zip(bp["boxes"], groups):
            patch.set_facecolor(colors[group])
        ax.set_title(title)
        ax.set_ylabel("Target score (max Delta)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.suptitle("Null-swap target scores by group - MONK-1", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> None:
    """Run order-1 and order-2 null-swap validation on MONK-1."""
    from load_datasets import load_monks1_full

    print("Loading MONK-1...")
    X, y, feature_names, ground_truth = load_monks1_full()
    print(f"  Shape: {X.shape}")

    print("\nRunning order 1...")
    scores_o1, _ = compute_order_scores(
        X=X,
        y=y,
        feature_names=feature_names,
        order=1,
        estimator_factory=make_decision_tree_estimator,
        n_repeats=N_REPEATS,
        n_folds=N_FOLDS,
        random_seed=RANDOM_SEED,
        n_jobs=1,
    )

    print("\nRunning order 2...")
    scores_o2, records_o2 = compute_order_scores(
        X=X,
        y=y,
        feature_names=feature_names,
        order=2,
        estimator_factory=make_decision_tree_estimator,
        n_repeats=N_REPEATS,
        n_folds=N_FOLDS,
        random_seed=RANDOM_SEED,
        n_jobs=1,
    )
    matrix_o2 = build_order2_matrix(records_o2, feature_names)

    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "ground_truth": [ground_truth[name] for name in feature_names],
            "score_order1": scores_o1.values,
            "score_order2": scores_o2.values,
            "order2_minus_order1": scores_o2.values - scores_o1.values,
        }
    )
    summary.to_csv(_DATA_DIR / "null_swap_summary_monks1.csv", index=False)
    print("  Saved: data/null_swap_summary_monks1.csv")

    pd.DataFrame(
        {
            "feature": feature_names,
            "ground_truth": [ground_truth[name] for name in feature_names],
            "score_order1": scores_o1.values,
        }
    ).to_csv(_DATA_DIR / "null_swap_order1_monks1.csv", index=False)
    print("  Saved: data/null_swap_order1_monks1.csv")

    pd.DataFrame(
        {
            "feature": feature_names,
            "ground_truth": [ground_truth[name] for name in feature_names],
            "score_order2": scores_o2.values,
        }
    ).to_csv(_DATA_DIR / "null_swap_order2_monks1.csv", index=False)
    print("  Saved: data/null_swap_order2_monks1.csv")

    matrix_o2.to_csv(_DATA_DIR / "null_swap_order2_matrix_monks1.csv")
    print("  Saved: data/null_swap_order2_matrix_monks1.csv")

    plot_score_distributions(
        scores_o1,
        scores_o2,
        ground_truth,
        _FIG_DIR / "null_swap_orders_monks1.png",
    )
    print("  Saved: figures/null_swap_orders_monks1.png")

    print("\n=== MONK-1 summary ===")
    print(summary.sort_values("score_order2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run_experiment()
