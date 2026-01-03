# src/eval_utils.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# ============================================================
# filesystem
# ============================================================

def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, d: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2) + "\n")


# ============================================================
# X / y loading + preprocessing
# ============================================================

def read_design_matrix(path: Path) -> pd.DataFrame:
    """
    Load X from:
      - .csv (header expected)
      - .npy (2D array)
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            raise ValueError(f"No columns found in CSV: {path}")
        return df

    if suf == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        cols = [f"Dim{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    raise ValueError(f"Unsupported X format {path.suffix}; use .csv or .npy")


def _coerce_binary_target(y_raw: pd.Series, *, name: str) -> pd.Series:
    """
    Coerce target to {0,1}. Supports common strings like YES/NO, TRUE/FALSE.
    """
    if y_raw.dtype == object:
        s = y_raw.astype(str).str.strip().str.upper()
        mapping = {"YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
        y = s.map(mapping)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    if y.isna().any():
        bad = y_raw[y.isna()].head(10).tolist()
        raise ValueError(
            "Target contains values that could not be coerced to 0/1. "
            f"Examples: {bad}"
        )

    y = y.astype(int)
    uniq = sorted(y.unique().tolist())
    if uniq not in ([0, 1], [0], [1]):
        raise ValueError(f"Target must be binary 0/1; got unique values {uniq}")
    return y.rename(name)


def read_target_csv(path: Path, *, target_col: str) -> pd.Series:
    """
    Load target column from CSV and coerce to {0,1}.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {path}")
    return _coerce_binary_target(df[target_col], name=target_col)


def split_raw_csv(
    raw_csv: Path,
    *,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Given a single raw CSV that contains both X and y, split into (X, y).
    - If feature_cols is provided, only those columns are used for X.
    - Else X = all columns except drop_cols and target_col.
    """
    raw_csv = raw_csv.resolve()
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)

    df = pd.read_csv(raw_csv)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {raw_csv}")

    y = _coerce_binary_target(df[target_col], name=target_col)

    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Requested feature columns missing from raw CSV: {missing}")
        X = df[feature_cols].copy()
    else:
        drop = set(drop_cols or [])
        drop.add(target_col)
        X = df[[c for c in df.columns if c not in drop]].copy()

    return X, y


def coerce_X_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric, error if any NaNs appear after coercion.
    """
    Xn = X.copy()
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")

    if Xn.isna().any().any():
        bad_cols = Xn.columns[Xn.isna().any()].tolist()
        raise ValueError(
            "X contains NaNs after numeric coercion. "
            f"Problem columns (examples): {bad_cols[:20]}"
        )
    return Xn


def align_xy_dropna(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Ensure X and y are aligned by row order, then drop rows with any NA in X or y.
    Returns:
      - X2, y2 (reset index)
      - orig_row_index: original row indices BEFORE dropna (0..n-1)
    """
    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch: X has {len(X)} rows, y has {len(y)} rows. "
            "Make sure both were saved in the same order."
        )

    orig_idx = np.arange(len(y), dtype=int)
    keep = ~(X.isna().any(axis=1) | y.isna())
    X2 = X.loc[keep].reset_index(drop=True)
    y2 = y.loc[keep].reset_index(drop=True)
    orig_idx2 = orig_idx[keep.values]
    return X2, y2, orig_idx2


# ============================================================
# metrics + plots (shared)
# ============================================================

def compute_test_metrics(*, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (probs >= float(threshold)).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def plot_and_save_confusion_matrix(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    out_path: Path,
    title: str,
) -> None:
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (probs >= float(threshold)).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    disp.plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(
        f"{title}\n"
        f"thr={threshold:.3g} | acc={acc:.3f} | bal_acc={bal_acc:.3f}"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_and_save_roc(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    RocCurveDisplay.from_predictions(
        y_true,
        probs,
        ax=ax,
        name="Model",
    )

    # identity / chance line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def build_predictions_all_df(
    *,
    y_true: pd.Series,
    probs_all: np.ndarray,
    threshold: float,
    orig_row_index: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """
    For holdout-like splits: label rows as train/test based on test_idx.
    """
    n = len(y_true)
    probs_all = np.asarray(probs_all).ravel()
    y_pred = (probs_all >= float(threshold)).astype(int)

    split = np.array(["train"] * n, dtype=object)
    split[np.asarray(test_idx, dtype=int)] = "test"

    return (
        pd.DataFrame(
            {
                "row_index": np.arange(n, dtype=int),
                "orig_row_index": orig_row_index.astype(int),
                "split": split,
                "y_true": y_true.values.astype(int),
                "prob": probs_all.astype(float),
                "y_pred": y_pred.astype(int),
            }
        )
        .sort_values("row_index", kind="mergesort")
        .reset_index(drop=True)
    )
