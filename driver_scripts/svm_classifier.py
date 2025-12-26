#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
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
CV-FREE SVM driver (holdout only).

What it does:
  - One stratified train/test split
  - Fit SVM on train
  - Evaluate once on test
  - Save confusion matrix + ROC + metrics to:
      analysis/<space-name>/svm/holdout/

Run like your logistic regression:
python driver_scripts/svm.py \
  --mode space \
  --x data/features.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name raw_features \
  --test-size 0.25 \
  --random-state 0
"""

# ----------------------------
# IO helpers
# ----------------------------
def read_design_matrix(path: Path) -> pd.DataFrame:
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
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {path}")

    return coerce_binary_target(df[target_col], name=target_col)


def align_xy(X: pd.DataFrame, y: pd.Series, *, drop_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
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


def coerce_X_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
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


# ----------------------------
# Plots
# ----------------------------
def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    threshold: float,
    accuracy: float,
    balanced_accuracy: float,
    title_prefix: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NO", "YES"])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(
        f"{title_prefix} (thr={threshold:g})\n"
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


# ----------------------------
# Model helpers
# ----------------------------
def _parse_gamma(gamma_str: str) -> Any:
    if gamma_str in {"scale", "auto"}:
        return gamma_str
    try:
        return float(gamma_str)
    except ValueError:
        raise ValueError("--gamma must be 'scale', 'auto', or a float")


def _build_model(args: argparse.Namespace) -> Pipeline:
    cw = None if args.class_weight == "none" else "balanced"

    if args.svm_type == "linear_svc":
        clf = LinearSVC(
            C=float(args.C),
            class_weight=cw,
            random_state=int(args.random_state),
            max_iter=int(args.max_iter),
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    svc = SVC(
        kernel=str(args.kernel),
        C=float(args.C),
        gamma=args.gamma,
        degree=int(args.degree),
        class_weight=cw,
        probability=bool(args.probability),
        random_state=int(args.random_state),
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", svc)])


def _get_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Return a 1D score array where larger = more likely class 1.
    Uses decision_function if available; otherwise uses predict_proba[:,1].
    """
    clf = model.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s).ravel()
    if hasattr(clf, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        return np.asarray(p).ravel()
    raise RuntimeError("Model has neither decision_function nor predict_proba; cannot score.")


def _scores_to_preds(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str,
    grid_size: int,
) -> Tuple[float, float]:
    if metric not in {"balanced_accuracy", "accuracy"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    lo = float(np.nanmin(scores))
    hi = float(np.nanmax(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        thr = 0.0
        preds = _scores_to_preds(scores, thr)
        val = balanced_accuracy_score(y_true, preds) if metric == "balanced_accuracy" else accuracy_score(y_true, preds)
        return float(thr), float(val)

    thresholds = np.linspace(lo, hi, num=int(grid_size))
    best_thr = float(thresholds[0])
    best_val = -np.inf
    for thr in thresholds:
        preds = _scores_to_preds(scores, float(thr))
        v = balanced_accuracy_score(y_true, preds) if metric == "balanced_accuracy" else accuracy_score(y_true, preds)
        if v > best_val:
            best_val = float(v)
            best_thr = float(thr)
    return float(best_thr), float(best_val)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _write_run_config(args: argparse.Namespace, out_path: Path) -> None:
    _write_json(
        out_path,
        {
            "model": "svm",
            "svm_type": args.svm_type,
            "kernel": args.kernel,
            "C": float(args.C),
            "gamma": args.gamma if isinstance(args.gamma, str) else float(args.gamma),
            "degree": int(args.degree),
            "probability": bool(args.probability),
            "max_iter": int(args.max_iter),
            "class_weight": args.class_weight,
            "threshold": float(args.threshold),
            "optimize_threshold": bool(args.optimize_threshold),
            "threshold_metric": args.threshold_metric,
            "threshold_grid_size": int(args.threshold_grid_size),
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "mode": args.mode,
            "space_name": args.space_name,
            "x": str(args.x) if args.x is not None else None,
            "y": str(args.y) if args.y is not None else None,
            "raw_csv": str(args.raw_csv) if args.raw_csv is not None else None,
            "target_col": args.target_col,
        },
    )


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run an SVM classifier (with scaling) on either (a) X (.npy/.csv) + y (CSV), "
            "or (b) a raw CSV containing both features and target.\n"
            "CV-FREE: always does a single holdout train/test split."
        )
    )

    p.add_argument("--mode", choices=["space", "raw"], default="space")

    # space-mode inputs
    p.add_argument("--x", type=Path, default=Path("analysis/pca/pca_space.npy"))
    p.add_argument("--y", type=Path, default=Path("data/qualification_target.csv"))

    # raw-mode input
    p.add_argument("--raw-csv", type=Path, default=None)
    p.add_argument("--drop-cols", nargs="*", default=None)
    p.add_argument("--feature-cols", nargs="*", default=None)

    p.add_argument("--target-col", type=str, default="Qualified Municipality")

    # Holdout split
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=0)

    # Output
    p.add_argument("--space-name", type=str, default="pca")
    p.add_argument("--out-root", type=Path, default=Path("analysis"))

    # SVM config
    p.add_argument("--svm-type", choices=["svc", "linear_svc"], default="svc")
    p.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="linear")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", type=str, default="scale")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--probability", action="store_true", help="If set, SVC fits probability estimates (slower).")
    p.add_argument("--max-iter", type=int, default=10000, help="For LinearSVC max_iter.")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")

    # Thresholding
    p.add_argument("--threshold", type=float, default=0.0, help="Threshold on SVM score (default: 0.0).")
    p.add_argument("--optimize-threshold", action="store_true")
    p.add_argument("--threshold-metric", choices=["balanced_accuracy", "accuracy"], default="balanced_accuracy")
    p.add_argument("--threshold-grid-size", type=int, default=200)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.gamma = _parse_gamma(args.gamma)

    out_dir = (args.out_root / args.space_name / "svm" / "holdout").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load X,y
    if args.mode == "space":
        X = read_design_matrix(args.x)
        y = read_target(args.y, args.target_col)
        X, y = align_xy(X, y, drop_na=True)
    else:
        if args.raw_csv is None:
            raise ValueError("--raw-csv is required when --mode raw")
        X, y = split_raw_csv(
            args.raw_csv,
            target_col=args.target_col,
            drop_cols=args.drop_cols,
            feature_cols=args.feature_cols,
        )
        X, y = align_xy(X, y, drop_na=True)

    X = coerce_X_to_numeric(X)

    uniq = np.unique(y.values)
    if len(uniq) != 2:
        raise ValueError(f"Target must have two classes; got {uniq.tolist()}.")

    # Save config snapshot
    _write_run_config(args, out_dir / "run_config.json")

    # Holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=y,
    )

    model = _build_model(args)
    model.fit(X_train, y_train)

    test_scores = _get_scores(model, X_test)

    thr_used = float(args.threshold)
    if args.optimize_threshold:
        # Note: without CV, "cleanest" is to optimize on TRAIN in-sample scores only.
        train_scores = _get_scores(model, X_train)
        thr_used, thr_val = find_best_threshold(
            y_train.values,
            train_scores,
            metric=args.threshold_metric,
            grid_size=int(args.threshold_grid_size),
        )
        (out_dir / "threshold_optimized_train_only.txt").write_text(
            "\n".join(
                [
                    f"optimized_threshold: {thr_used}",
                    f"optimized_metric: {args.threshold_metric}",
                    f"optimized_metric_value_train_in_sample: {thr_val}",
                    "note: optimized on training in-sample scores (no CV).",
                ]
            )
            + "\n"
        )

    test_pred = _scores_to_preds(test_scores, thr_used)

    acc = float(accuracy_score(y_test.values, test_pred))
    bal = float(balanced_accuracy_score(y_test.values, test_pred))
    auc_val = float(roc_auc_score(y_test.values, test_scores))

    plot_and_save_confusion_matrix(
        y_true=y_test.values,
        y_pred=test_pred,
        out_path=out_dir / "confusion_matrix_test.png",
        threshold=thr_used,
        accuracy=acc,
        balanced_accuracy=bal,
        title_prefix="SVM Confusion Matrix (holdout test)",
    )
    plot_and_save_roc(
        y_true=y_test.values,
        y_score=test_scores,
        out_path=out_dir / "roc_curve_test.png",
        title="SVM ROC (holdout test)",
    )

    (out_dir / "metrics_test.txt").write_text(
        "\n".join(
            [
                "eval: holdout",
                f"n_total: {len(y)}",
                f"n_train: {len(y_train)}",
                f"n_test: {len(y_test)}",
                "",
                f"threshold_used: {thr_used}",
                f"TEST_auc: {auc_val:.6f}",
                f"TEST_accuracy: {acc:.6f}",
                f"TEST_balanced_accuracy: {bal:.6f}",
            ]
        )
        + "\n"
    )

    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[OK] TEST AUC={auc_val:.4f}  acc={acc:.4f}  bal_acc={bal:.4f}  thr={thr_used:.4g}")


if __name__ == "__main__":
    main()
