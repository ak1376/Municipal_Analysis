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
Holdout-only SVM driver (NO CV), with "predictions for ALL datapoints" that
preserve the ORIGINAL row order from your input files.

Key outputs (mirrors your logistic regression behavior):
  analysis/<space-name>/svm/holdout/
    - metrics_test.txt
    - confusion_matrix_test.png
    - roc_curve_test.png
    - run_config.json
    - predictions_all_in_original_order.csv

About row order / shuffling:
- We NEVER shuffle the dataset itself.
- We create train/test SPLITS by sampling indices, but we store predictions back into
  an array aligned to the original row order (0..N-1 after optional NA dropping).
- The output `predictions_all_in_original_order.csv` is sorted by row_index ascending,
  so row_index i corresponds to row i of the *post-dropna aligned* X/y.

If you want to keep a mapping to the *pre-dropna* original row indices, we also store
`orig_row_index` (the row number in the original loaded file(s) before dropping NA).
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


def align_xy_preserve_original_index(
    X: pd.DataFrame, y: pd.Series, *, drop_na: bool = True
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Align X and y by row order, optionally drop NA rows, and return:
      - X_aligned
      - y_aligned
      - orig_row_index: mapping back to the original row indices BEFORE drop_na
    """
    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch: X has {len(X)} rows, y has {len(y)} rows. "
            "Make sure both were saved in the same order."
        )

    orig_idx = np.arange(len(y), dtype=int)

    if drop_na:
        keep = ~(X.isna().any(axis=1) | y.isna())
        X2 = X.loc[keep].reset_index(drop=True)
        y2 = y.loc[keep].reset_index(drop=True)
        orig_idx2 = orig_idx[keep.values]  # keep.values is boolean array
        return X2, y2, orig_idx2

    return X.reset_index(drop=True), y.reset_index(drop=True), orig_idx


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


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Holdout-only SVM (no CV). Fits on train split, evaluates on test split.\n"
            "Also saves predictions for all datapoints in ORIGINAL row order."
        )
    )

    p.add_argument("--mode", choices=["space", "raw"], default="space")

    # space-mode inputs (match your logistic_regression call style)
    p.add_argument("--x", type=Path, default=Path("data/features.csv"))
    p.add_argument("--y", type=Path, default=Path("data/qualification_target.csv"))

    # raw-mode input
    p.add_argument("--raw-csv", type=Path, default=None)
    p.add_argument("--drop-cols", nargs="*", default=None)
    p.add_argument("--feature-cols", nargs="*", default=None)

    p.add_argument("--target-col", type=str, default="Qualified Municipality")

    # Split params (match your logistic call)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=0)

    # Output organization
    p.add_argument("--space-name", type=str, default="raw_features")
    p.add_argument("--out-root", type=Path, default=Path("analysis"))

    # SVM config
    p.add_argument("--svm-type", choices=["svc", "linear_svc"], default="svc")
    p.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="linear")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", type=str, default="scale")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--probability", action="store_true", help="For SVC only; slower but enables predict_proba.")
    p.add_argument("--max-iter", type=int, default=10000, help="For LinearSVC max_iter.")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")

    # Thresholding + optional optimization (train-only, no CV)
    p.add_argument("--threshold", type=float, default=0.0, help="Threshold on SVM score (default: 0.0).")
    p.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="If set, choose threshold to maximize a metric using TRAIN in-sample scores (no CV).",
    )
    p.add_argument("--threshold-metric", choices=["balanced_accuracy", "accuracy"], default="balanced_accuracy")
    p.add_argument("--threshold-grid-size", type=int, default=200)

    # Optional refit on full dataset after evaluation (for a production model)
    p.add_argument(
        "--refit-full",
        action="store_true",
        help="After evaluating on test, refit the same model on ALL data and save predictions_all_refit_full.csv",
    )

    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    args.gamma = _parse_gamma(args.gamma)

    out_dir = (args.out_root / args.space_name / "svm" / "holdout").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load X,y
    if args.mode == "space":
        X_raw = read_design_matrix(args.x)
        y_raw = read_target(args.y, args.target_col)
        X, y, orig_row_index = align_xy_preserve_original_index(X_raw, y_raw, drop_na=True)
    else:
        if args.raw_csv is None:
            raise ValueError("--raw-csv is required when --mode raw")
        X_raw, y_raw = split_raw_csv(
            args.raw_csv,
            target_col=args.target_col,
            drop_cols=args.drop_cols,
            feature_cols=args.feature_cols,
        )
        X, y, orig_row_index = align_xy_preserve_original_index(X_raw, y_raw, drop_na=True)

    X = coerce_X_to_numeric(X)

    uniq = np.unique(y.values)
    if len(uniq) != 2:
        raise ValueError(f"Target must have two classes; got {uniq.tolist()}.")

    n = len(y)
    # IMPORTANT: we split on INDICES, not by shuffling rows
    idx = np.arange(n, dtype=int)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        shuffle=True,
        stratify=y.values,
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Fit on train
    model = _build_model(args)
    model.fit(X_train, y_train)

    # Threshold (either fixed or optimized on TRAIN in-sample)
    thr_used = float(args.threshold)
    if args.optimize_threshold:
        train_scores = _get_scores(model, X_train)
        thr_used, thr_val = find_best_threshold(
            y_true=y_train.values,
            scores=train_scores,
            metric=args.threshold_metric,
            grid_size=int(args.threshold_grid_size),
        )
        (out_dir / "threshold_optimized_train_only.txt").write_text(
            "\n".join(
                [
                    f"optimized_threshold: {thr_used}",
                    f"optimized_metric: {args.threshold_metric}",
                    f"optimized_metric_value_train_in_sample: {thr_val}",
                    "note: optimized on TRAIN in-sample scores (no CV).",
                ]
            )
            + "\n"
        )

    # Evaluate on test
    test_scores = _get_scores(model, X_test)
    test_pred = _scores_to_preds(test_scores, thr_used)

    test_acc = float(accuracy_score(y_test.values, test_pred))
    test_bal = float(balanced_accuracy_score(y_test.values, test_pred))
    test_auc = float(roc_auc_score(y_test.values, test_scores))

    # Save plots + metrics
    plot_and_save_confusion_matrix(
        y_true=y_test.values,
        y_pred=test_pred,
        out_path=out_dir / "confusion_matrix_test.png",
        threshold=thr_used,
        accuracy=test_acc,
        balanced_accuracy=test_bal,
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
                "eval: holdout_only",
                f"mode: {args.mode}",
                f"space_name: {args.space_name}",
                "",
                f"n_total_after_dropna: {n}",
                f"n_train: {len(train_idx)}",
                f"n_test: {len(test_idx)}",
                "",
                f"svm_type: {args.svm_type}",
                f"kernel: {args.kernel}",
                f"C: {float(args.C)}",
                f"gamma: {args.gamma}",
                f"degree: {int(args.degree)}",
                f"class_weight: {args.class_weight}",
                f"probability: {bool(args.probability)}",
                "",
                f"threshold_used: {thr_used}",
                "",
                f"TEST_auc: {test_auc:.6f}",
                f"TEST_accuracy: {test_acc:.6f}",
                f"TEST_balanced_accuracy: {test_bal:.6f}",
            ]
        )
        + "\n"
    )

    _write_json(
        out_dir / "run_config.json",
        {
            "model": "svm",
            "eval": "holdout_only",
            "mode": args.mode,
            "space_name": args.space_name,
            "x_path": str(args.x) if args.mode == "space" else None,
            "y_path": str(args.y) if args.mode == "space" else None,
            "raw_csv": str(args.raw_csv) if args.mode == "raw" else None,
            "target_col": args.target_col,
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "svm_type": args.svm_type,
            "kernel": args.kernel,
            "C": float(args.C),
            "gamma": args.gamma if isinstance(args.gamma, str) else float(args.gamma),
            "degree": int(args.degree),
            "probability": bool(args.probability),
            "max_iter": int(args.max_iter),
            "class_weight": args.class_weight,
            "threshold_used": float(thr_used),
            "optimize_threshold": bool(args.optimize_threshold),
            "threshold_metric": args.threshold_metric,
            "threshold_grid_size": int(args.threshold_grid_size),
            "refit_full": bool(args.refit_full),
            "notes": [
                "Rows are NOT reordered. Train/test split is done by sampling indices.",
                "predictions_all_in_original_order.csv is sorted by row_index and aligns to post-dropna aligned X/y.",
                "orig_row_index maps back to the original file row number before dropna.",
            ],
        },
    )

    # ----------------------------
    # Predictions for ALL datapoints (in original row order)
    # ----------------------------
    # Compute scores for all rows using the TRAIN-fit model
    all_scores = _get_scores(model, X)
    all_pred = _scores_to_preds(all_scores, thr_used)

    split = np.array(["train"] * n, dtype=object)
    split[test_idx] = "test"

    # Assemble and SORT by row_index so it matches original row order
    pred_all = pd.DataFrame(
        {
            # row index in the post-dropna aligned dataset
            "row_index": np.arange(n, dtype=int),
            # row index in the original input files before dropping NA
            "orig_row_index": orig_row_index.astype(int),
            "split": split,
            "y_true": y.values.astype(int),
            "score": all_scores.astype(float),
            "y_pred": all_pred.astype(int),
        }
    ).sort_values("row_index", kind="mergesort")  # stable sort

    pred_all.to_csv(out_dir / "predictions_all_in_original_order.csv", index=False)

    (out_dir / "note_predictions_all.txt").write_text(
        "\n".join(
            [
                "predictions_all_in_original_order.csv notes:",
                "- File is sorted by row_index ascending.",
                "- row_index refers to the aligned (post-dropna) X/y row number used for modeling.",
                "- orig_row_index refers to the original row number in the loaded input file(s) before dropna.",
                "- 'score' is decision_function (preferred) or predict_proba[:,1] if decision_function unavailable.",
                "- Model used here is fit on TRAIN ONLY.",
                "- split='test' rows are out-of-sample; split='train' rows are in-sample.",
            ]
        )
        + "\n"
    )

    # Optional: refit on full data and save full-data predictions (production-style)
    if args.refit_full:
        model_full = _build_model(args)
        model_full.fit(X, y)
        full_scores = _get_scores(model_full, X)
        full_pred = _scores_to_preds(full_scores, thr_used)

        pred_full = pd.DataFrame(
            {
                "row_index": np.arange(n, dtype=int),
                "orig_row_index": orig_row_index.astype(int),
                "y_true": y.values.astype(int),
                "score": full_scores.astype(float),
                "y_pred": full_pred.astype(int),
            }
        ).sort_values("row_index", kind="mergesort")

        pred_full.to_csv(out_dir / "predictions_all_refit_full.csv", index=False)

        (out_dir / "note_predictions_all_refit_full.txt").write_text(
            "\n".join(
                [
                    "predictions_all_refit_full.csv notes:",
                    "- Model was refit on ALL data after reporting holdout test metrics.",
                    "- These are NOT out-of-sample predictions (fit-on-all).",
                    "- File is sorted by row_index to match aligned X/y order.",
                ]
            )
            + "\n"
        )

    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[OK] TEST AUC={test_auc:.4f}  acc={test_acc:.4f}  bal_acc={test_bal:.4f}  thr={thr_used:.4g}")


if __name__ == "__main__":
    main()
