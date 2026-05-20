# load_datasets.py
# Dataset loader for the null-swap MONK-1 experiment.

from __future__ import annotations

from itertools import product
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
_PROCESSED_DATA_DIR = _REPO_ROOT / "data" / "processed"
_WDBC_UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/17/"
    "breast+cancer+wisconsin+diagnostic.zip"
)
_WDBC_ARCHIVE_PATH = _RAW_DATA_DIR / "breast_cancer_wisconsin_diagnostic.zip"
_WDBC_EXTRACT_DIR = _RAW_DATA_DIR / "breast_cancer_wisconsin_diagnostic"
_WDBC_RAW_PATH = _WDBC_EXTRACT_DIR / "wdbc.data"
_WDBC_PROCESSED_PATH = _PROCESSED_DATA_DIR / "wdbc_full.csv"
_WDBC_FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


# =============================================================================
# MONK-1 (classic rule-learning benchmark; exact rule known)
# =============================================================================

def load_monks1_full(
    save_csv: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str]]:
    """Load the complete MONK-1 design space with the original deterministic rule.

    MONK-1 is a classic benchmark from rule learning. The target is:

        y = 1[(a1 == a2) or (a5 == 1)]

    We generate the full cartesian product locally rather than relying on the
    original train/test split files so the benchmark is self-contained and does
    not require network access.

    Returns
    -------
    X : ndarray of shape (432, 6)
    y : ndarray of shape (432,), values in {0, 1}
    feature_names : ["a1", "a2", "a3", "a4", "a5", "a6"]
    ground_truth : dict mapping feature name -> taxonomy category
        'informative' : a1, a2, a5
        'noise'       : a3, a4, a6
    """
    csv_path = _PROCESSED_DATA_DIR / "monks1_full.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        feature_specs = [
            ("a1", (1, 2, 3)),
            ("a2", (1, 2, 3)),
            ("a3", (1, 2)),
            ("a4", (1, 2, 3)),
            ("a5", (1, 2, 3, 4)),
            ("a6", (1, 2)),
        ]
        feature_names = [name for name, _ in feature_specs]

        rows: list[dict[str, int]] = []
        for values in product(*(levels for _, levels in feature_specs)):
            row = dict(zip(feature_names, values))
            row["target"] = int((row["a1"] == row["a2"]) or (row["a5"] == 1))
            rows.append(row)

        df = pd.DataFrame(rows, columns=feature_names + ["target"])
        if save_csv:
            _PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)

    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].values.astype(float)
    y = df["target"].values.astype(int)

    ground_truth = {
        "a1": "informative",
        "a2": "informative",
        "a3": "noise",
        "a4": "noise",
        "a5": "informative",
        "a6": "noise",
    }
    return X, y, feature_names, ground_truth


# =============================================================================
# Breast Cancer Wisconsin (Diagnostic) (UCI real dataset)
# =============================================================================

def ensure_wdbc_downloaded() -> Path:
    """Ensure the official UCI WDBC archive is available locally and extracted."""
    if _WDBC_RAW_PATH.exists():
        return _WDBC_RAW_PATH

    _RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _WDBC_ARCHIVE_PATH.exists():
        try:
            urlretrieve(_WDBC_UCI_ZIP_URL, _WDBC_ARCHIVE_PATH)
        except Exception as exc:  # pragma: no cover - network depends on environment
            raise RuntimeError(
                "Could not download the UCI Breast Cancer Wisconsin (Diagnostic) "
                "dataset. Download the archive manually into data/raw "
                "or provide network access, then retry."
            ) from exc

    _WDBC_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(_WDBC_ARCHIVE_PATH) as zip_file:
        zip_file.extractall(_WDBC_EXTRACT_DIR)

    if not _WDBC_RAW_PATH.exists():
        raise FileNotFoundError(
            "The WDBC archive was extracted, but wdbc.data was not found."
        )
    return _WDBC_RAW_PATH


def _load_wdbc_dataframe(raw_path: Path) -> pd.DataFrame:
    cols = ["id", "diagnosis", *_WDBC_FEATURE_NAMES]
    return pd.read_csv(raw_path, header=None, names=cols)


def load_breast_cancer_wisconsin_diagnostic(
    save_csv: bool = True,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], None]:
    """Load the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

    This dataset contains 569 breast mass aspirates described by 30 real-valued
    morphology features and a binary diagnosis target (`M` vs `B`).

    Returns
    -------
    X : ndarray of shape (569, 30)
    y : ndarray of shape (569,), values in {0, 1}
        1 = malignant, 0 = benign
    feature_names : list[str]
        Machine-friendly names derived from the UCI attribute descriptions.
    ground_truth : None
        No per-feature ground truth is available for this real dataset.
    """
    if _WDBC_PROCESSED_PATH.exists():
        df = pd.read_csv(_WDBC_PROCESSED_PATH)
    else:
        if _WDBC_RAW_PATH.exists():
            raw_path = _WDBC_RAW_PATH
        elif download_if_missing:
            raw_path = ensure_wdbc_downloaded()
        else:
            raise FileNotFoundError(
                "WDBC raw data not found. Run ensure_wdbc_downloaded() first "
                "or enable download_if_missing."
            )

        raw_df = _load_wdbc_dataframe(raw_path)
        df = raw_df.drop(columns=["id", "diagnosis"]).copy()
        df["target"] = (raw_df["diagnosis"] == "M").astype(int)
        if save_csv:
            _PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(_WDBC_PROCESSED_PATH, index=False)

    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].values.astype(float)
    y = df["target"].values.astype(int)
    return X, y, feature_names, None


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    X, y, fn, gt = load_monks1_full()
    counts = pd.Series(gt).value_counts()
    print(f"Shape: {X.shape}, y mean: {y.mean():.3f}")
    print(f"Taxonomy counts: {counts.to_dict()}")
