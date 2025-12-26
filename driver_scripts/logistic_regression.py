#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
)

"""
Examples:

# 1) PCA / embedding space mode (your current workflow)
python driver_scripts/logistic_regression.py \
  --mode space \
  --x analysis/pca/pca_space.npy \
  --y data/qualification_target.csv \
  --space-name pca

# 2) Raw CSV mode (target column included in same CSV)
python driver_scripts/logistic_regression.py \
  --mode raw \
  --raw-csv data/raw_features_plus_target.csv \
  --target-col "Qualified Municipality" \
  --space-name raw

# Optional: drop ID-like columns in raw mode
python driver_scripts/logistic_regression.py \
  --mode raw \
  --raw-csv data/TSM_processed.csv \
  --target-col "Qualified Municipality" \
  --drop-cols Municipality Housing\ Region
"""


# ----------------------------
# IO helpers
# ----------------------------
def read_design_matrix(path: Path) -> pd.DataFrame:
    """
    Read feature/embedding space from .csv or .npy.
    - .csv: returns DataFrame with existing column names
    - .npy: returns DataFrame with columns Dim1..DimK
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            raise ValueError(f"No columns found in CSV: {path}")
        return df

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        cols = [f"Dim{i}" for i in range(1, arr.shape[1] + 1)]
        return pd.DataFrame(arr, columns=cols)

    raise ValueError(f"Unsupported X format {path.suffix}; use .csv or .npy")


def coerce_binary_target(y_raw: pd.Series, *, name: str) -> pd.Series:
    """
    Coerce common target encodings to binary {0,1}.
    Accepts YES/NO, True/False, 1/0.
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


def read_target(path: Path, target_col: str) -> pd.Series:
    """
    Read target from CSV and coerce to binary {0,1}.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {path}")

    return coerce_binary_target(df[target_col], name=target_col)


def align_xy(X: pd.DataFrame, y: pd.Series, *, drop_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align X and y by row order. If lengths differ, hard error.
    """
    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch: X has {len(X)} rows, y has {len(y)} rows. "
            "Make sure both were generated from the same filtered rows and saved in the same order."
        )

    X2 = X.copy()
    y2 = y.copy()

    if drop_na:
        keep = ~(X2.isna().any(axis=1) | y2.isna())
        X2 = X2.loc[keep].reset_index(drop=True)
        y2 = y2.loc[keep].reset_index(drop=True)

    return X2, y2


def split_raw_csv(
    raw_csv: Path,
    *,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a single raw CSV containing both features and target.
    You can either:
      - pass feature_cols to explicitly select features, OR
      - (default) use all columns except target_col and drop_cols.
    """
    raw_csv = raw_csv.resolve()
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)

    df = pd.read_csv(raw_csv)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {raw_csv}")

    y = coerce_binary_target(df[target_col], name=target_col)

    if feature_cols is not None and len(feature_cols) > 0:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Requested feature columns missing from raw CSV: {missing}")
        X = df[feature_cols].copy()
    else:
        drop = set(drop_cols or [])
        drop.add(target_col)
        X = df[[c for c in df.columns if c not in drop]].copy()

    return X, y


# ----------------------------
# Modeling + outputs
# ----------------------------
def fit_logit_statsmodels(X: pd.DataFrame, y: pd.Series, *, add_intercept: bool = True):
    """
    Fit statsmodels Logit and return fitted result and the design matrix used for fitting/prediction.
    """
    X_mat = X.copy()

    # Coerce all columns to numeric
    for c in X_mat.columns:
        X_mat[c] = pd.to_numeric(X_mat[c], errors="coerce")

    if X_mat.isna().any().any():
        # tell user which columns are problematic
        bad_cols = X_mat.columns[X_mat.isna().any()].tolist()
        raise ValueError(
            "X contains NaNs after numeric coercion. "
            f"Problem columns (examples): {bad_cols[:20]}"
        )

    if add_intercept:
        X_mat = sm.add_constant(X_mat, has_constant="add")

    model = sm.Logit(y, X_mat)
    res = model.fit(disp=False)
    return res, X_mat


def save_statsmodels_summary(result, out_path: Path) -> None:
    out_path.write_text(result.summary2().as_text())


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    threshold: float,
    accuracy: float,
    balanced_accuracy: float,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NO", "YES"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(
        "Confusion Matrix "
        f"(thr={threshold:g})\n"
        f"acc={accuracy:.3f}  bal_acc={balanced_accuracy:.3f}"
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_and_save_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
    *,
    title: str = "ROC Curve",
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

    return float(roc_auc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run statsmodels logistic regression on either (a) a reduced/embedded space (PCA/PLS/etc) "
            "or (b) a single raw CSV that contains both features and the target."
        )
    )

    # Mode switch
    p.add_argument(
        "--mode",
        choices=["space", "raw"],
        default="space",
        help="space: X from .npy/.csv + y from separate CSV. raw: X+y from one raw CSV.",
    )

    # space-mode inputs
    p.add_argument(
        "--x",
        type=Path,
        default=Path("analysis/pca/pca_space.npy"),
        help="(space mode) Path to feature space file (.npy or .csv).",
    )
    p.add_argument(
        "--y",
        type=Path,
        default=Path("data/qualification_target.csv"),
        help="(space mode) Path to CSV containing target labels.",
    )

    # raw-mode input
    p.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="(raw mode) Path to raw CSV containing BOTH features and the target column.",
    )
    p.add_argument(
        "--drop-cols",
        nargs="*",
        default=None,
        help="(raw mode) Columns to drop from features (e.g., Municipality, Housing Region).",
    )
    p.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        help="(raw mode) Explicit feature columns to use. If provided, drop-cols is ignored.",
    )

    # target column (used in BOTH modes)
    p.add_argument(
        "--target-col",
        type=str,
        default="Qualified Municipality",
        help="Target column name (in y CSV for space-mode, or in raw CSV for raw-mode).",
    )

    # Output organization
    p.add_argument(
        "--space-name",
        type=str,
        default="pca",
        help="Name of the space folder under analysis/ (e.g., pca, pls, raw). Used for output path.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("analysis"),
        help="Root analysis directory (default: analysis).",
    )

    # Modeling knobs
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predicting YES (default: 0.5).",
    )
    p.add_argument(
        "--no-intercept",
        action="store_true",
        help="Disable intercept term in logistic regression.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = (args.out_root / args.space_name / "logistic_regression").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load X,y
    if args.mode == "space":
        X = read_design_matrix(args.x)
        y = read_target(args.y, args.target_col)
        X, y = align_xy(X, y, drop_na=True)

    else:  # raw
        if args.raw_csv is None:
            raise ValueError("--raw-csv is required when --mode raw")
        X, y = split_raw_csv(
            args.raw_csv,
            target_col=args.target_col,
            drop_cols=args.drop_cols,
            feature_cols=args.feature_cols,
        )
        X, y = align_xy(X, y, drop_na=True)

    # Fit model
    res, X_mat = fit_logit_statsmodels(X, y, add_intercept=(not args.no_intercept))

    # Predict
    probs = res.predict(X_mat.values)
    preds = (probs >= args.threshold).astype(int)

    # Metrics
    acc = float(accuracy_score(y.values, preds))
    bal_acc = float(balanced_accuracy_score(y.values, preds))
    auc_val = float(roc_auc_score(y.values, probs))

    # Save model output
    save_statsmodels_summary(res, out_dir / "statsmodels_logit_summary.txt")

    # Coeff table
    terms = (["const"] + list(X.columns)) if (not args.no_intercept) else list(X.columns)
    coef_df = pd.DataFrame(
        {
            "term": terms,
            "coef": res.params,
            "std_err": res.bse,
            "z": res.tvalues,
            "p_value": res.pvalues,
        }
    )
    coef_df.to_csv(out_dir / "coefficients.csv", index=False)

    # Plots
    plot_and_save_confusion_matrix(
        y_true=y.values,
        y_pred=preds,
        out_path=out_dir / "confusion_matrix.png",
        threshold=args.threshold,
        accuracy=acc,
        balanced_accuracy=bal_acc,
    )

    plot_and_save_roc(
        y_true=y.values,
        y_score=probs,
        out_path=out_dir / "roc_curve.png",
        title="ROC Curve (statsmodels Logit)",
    )

    # Metrics text
    (out_dir / "metrics.txt").write_text(
        f"mode: {args.mode}\n"
        f"AUC: {auc_val:.6f}\n"
        f"accuracy: {acc:.6f}\n"
        f"balanced_accuracy: {bal_acc:.6f}\n"
        f"threshold: {args.threshold:g}\n"
        f"n: {len(y)}\n"
        f"p: {X.shape[1]}\n"
    )

    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[OK] AUC={auc_val:.4f}  acc={acc:.4f}  bal_acc={bal_acc:.4f}")
    print(f"[OK] n={len(y)}  p={X.shape[1]}  threshold={args.threshold:g}  mode={args.mode}")


if __name__ == "__main__":
    main()
