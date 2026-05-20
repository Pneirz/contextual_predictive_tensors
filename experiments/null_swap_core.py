from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

EstimatorFactory = Callable[[int], object]

DEFAULT_N_REPEATS = 5
DEFAULT_N_FOLDS = 5
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_EVALUATIONS = 2_000_000


def make_decision_tree_estimator(random_state: int):
    """Default estimator for low-order null-swap contrasts."""
    return DecisionTreeClassifier(random_state=random_state)


def make_logreg_estimator(random_state: int):
    """Standardized logistic-regression baseline for low-order contrasts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def make_extratree_estimator(random_state: int):
    """ExtraTreeClassifier for fast null-swap contrasts on small feature subsets.

    Faster than DecisionTreeClassifier because it skips exhaustive split search,
    using random splits instead. For 3-feature subsets and ~2000 samples this is
    the fastest CPU option (~6ms/eval at n_jobs=20 vs ~4000ms for XGBoost GPU).
    max_depth=6 is sufficient for order-3 contrasts (3 features).
    """
    from sklearn.tree import ExtraTreeClassifier

    return ExtraTreeClassifier(max_depth=6, random_state=random_state)


def make_xgboost_gpu_estimator(random_state: int):
    """XGBoost estimator with GPU acceleration for null-swap contrasts.

    Tuned for small feature subsets (order-3 = 3 features) and ~2000 samples.
    n_estimators=50, max_depth=3 approximates a single unpruned Decision Tree
    on 3 features.
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )


def verify_xgboost_gpu() -> bool:
    """Return True if XGBoost can train on CUDA. Raises on import failure."""
    import json

    import numpy as np
    from xgboost import XGBClassifier

    clf = XGBClassifier(n_estimators=1, tree_method="hist", device="cuda", verbosity=0)
    rng = np.random.default_rng(0)
    clf.fit(rng.standard_normal((100, 3)), rng.integers(0, 2, 100))
    config = json.loads(clf.get_booster().save_config())
    device = config.get("learner", {}).get("generic_param", {}).get("device", "")
    return "cuda" in device


def nullswap_delta(
    X: np.ndarray,
    y: np.ndarray,
    context_indices: Sequence[int],
    target_idx: int,
    estimator_factory: EstimatorFactory = make_decision_tree_estimator,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_folds: int = DEFAULT_N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    return_all: bool = False,
) -> float | list[float]:
    """Compute the null-swap contrast for one ordered (context, target) tuple.

    return_all=False (default): returns the mean over all repeats × folds.
    return_all=True: returns all n_repeats × n_folds individual fold deltas.
    """
    all_cols = list(context_indices) + [target_idx]
    target_col_pos = len(context_indices)

    all_fold_deltas: list[float] = []
    repeat_means: list[float] = []
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed + repeat)
        fold_deltas: list[float] = []

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_tr = X[train_idx][:, all_cols]
            X_te = X[test_idx][:, all_cols]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = estimator_factory(random_seed + repeat)
            model.fit(X_tr, y_tr)
            rmse_orig = root_mean_squared_error(y_te, model.predict_proba(X_te)[:, 1])

            rng = np.random.default_rng(random_seed + repeat * n_folds * 100 + fold_num)
            X_tr_null = X_tr.copy()
            X_tr_null[:, target_col_pos] = rng.permutation(X_tr_null[:, target_col_pos])

            model_null = estimator_factory(random_seed + repeat)
            model_null.fit(X_tr_null, y_tr)
            rmse_null = root_mean_squared_error(
                y_te, model_null.predict_proba(X_te)[:, 1]
            )
            fold_deltas.append(rmse_null - rmse_orig)
            all_fold_deltas.append(rmse_null - rmse_orig)

        repeat_means.append(float(np.mean(fold_deltas)))

    if return_all:
        return all_fold_deltas
    return float(np.mean(repeat_means))


def build_order_tasks(
    feature_indices: Sequence[int],
    order: int,
    max_contexts_per_target: int | None = None,
    rng_seed: int = DEFAULT_RANDOM_SEED,
    max_evaluations: int | None = DEFAULT_MAX_EVALUATIONS,
) -> list[tuple[list[int], int]]:
    """Enumerate or subsample null-swap tasks for a fixed order."""
    if order < 1:
        raise ValueError("order must be >= 1")

    feature_indices = list(feature_indices)
    context_size = order - 1
    if context_size >= len(feature_indices) and feature_indices:
        raise ValueError("order is too large for the selected feature set")

    if not feature_indices:
        return []

    contexts_per_target = comb(len(feature_indices) - 1, context_size)
    if max_contexts_per_target is not None:
        contexts_per_target = min(contexts_per_target, max_contexts_per_target)

    total_evaluations = len(feature_indices) * contexts_per_target
    if max_evaluations is not None and total_evaluations > max_evaluations:
        raise ValueError(
            "estimated evaluation count is too large; use pre-screening or "
            "max_contexts_per_target to keep the run tractable"
        )

    rng = np.random.default_rng(rng_seed)
    tasks: list[tuple[list[int], int]] = []

    for target_idx in feature_indices:
        if context_size == 0:
            tasks.append(([], target_idx))
            continue

        candidates = [idx for idx in feature_indices if idx != target_idx]
        all_contexts = list(combinations(candidates, context_size))
        if max_contexts_per_target is None or len(all_contexts) <= max_contexts_per_target:
            chosen_contexts = all_contexts
        else:
            chosen_rows = rng.choice(
                len(all_contexts), size=max_contexts_per_target, replace=False
            )
            chosen_contexts = [all_contexts[row] for row in chosen_rows]

        tasks.extend((list(context), target_idx) for context in chosen_contexts)

    return tasks


def compute_order_scores(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    order: int,
    estimator_factory: EstimatorFactory = make_decision_tree_estimator,
    candidate_indices: Sequence[int] | None = None,
    max_contexts_per_target: int | None = None,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_folds: int = DEFAULT_N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_jobs: int = -1,
    backend: str = "loky",
    batch_size: int | str = "auto",
    max_evaluations: int | None = DEFAULT_MAX_EVALUATIONS,
    aggregation: str = "mean",
    store_raw: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute null-swap scores for a fixed order over a feature subset.

    aggregation: how to reduce per-context deltas to a single score per target.
      "mean" — average delta across all contexts (default).
      "max"  — maximum delta across all contexts.
    store_raw: if True, records DataFrame includes a 'raw_deltas' column with
      all n_repeats*n_folds individual fold deltas per (S,j) combination.
    """
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X.shape[1]")

    eval_indices = (
        list(range(X.shape[1])) if candidate_indices is None else list(candidate_indices)
    )
    tasks = build_order_tasks(
        eval_indices,
        order=order,
        max_contexts_per_target=max_contexts_per_target,
        rng_seed=random_seed,
        max_evaluations=max_evaluations,
    )

    print(f"  Order {order}: {len(tasks)} evaluations")

    results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
        delayed(nullswap_delta)(
            X=X,
            y=y,
            context_indices=context,
            target_idx=target_idx,
            estimator_factory=estimator_factory,
            n_repeats=n_repeats,
            n_folds=n_folds,
            random_seed=random_seed,
            return_all=store_raw,
        )
        for context, target_idx in tasks
    )

    if store_raw:
        raw_deltas = results
        deltas = [float(np.mean(r)) for r in raw_deltas]
    else:
        deltas = results
        raw_deltas = None

    rec_dict: dict = {
        "order": order,
        "context_indices": [tuple(context) for context, _ in tasks],
        "target_idx": [target_idx for _, target_idx in tasks],
        "delta": deltas,
    }
    if store_raw:
        rec_dict["raw_deltas"] = raw_deltas

    records = pd.DataFrame(rec_dict)
    if not records.empty:
        records["context_size"] = records["context_indices"].map(len)
        records["target_feature"] = records["target_idx"].map(lambda idx: feature_names[idx])
        records["context_features"] = records["context_indices"].map(
            lambda context: tuple(feature_names[idx] for idx in context)
        )

    scores = pd.Series(np.nan, index=feature_names, name=f"score_order{order}")
    if not records.empty:
        if aggregation == "mean":
            agg_by_target = records.groupby("target_idx")["delta"].mean()
        elif aggregation == "max":
            agg_by_target = records.groupby("target_idx")["delta"].max()
        else:
            raise ValueError(f"aggregation must be 'mean' or 'max', got {aggregation!r}")
        for target_idx, delta in agg_by_target.items():
            scores.iloc[int(target_idx)] = float(delta)

    return scores, records


def build_order2_matrix(records: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Convert order-2 task records into the usual context-target matrix."""
    matrix = pd.DataFrame(np.nan, index=feature_names, columns=feature_names)
    if records.empty:
        return matrix

    for row in records.itertuples(index=False):
        context = row.context_indices
        if len(context) != 1:
            raise ValueError("order-2 matrix requires records with exactly one context feature")
        matrix.iat[context[0], row.target_idx] = row.delta

    return matrix
