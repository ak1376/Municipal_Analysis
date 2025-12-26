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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
Holdout-only Decision Tree driver (NO CV),
with "predictions for ALL datapoints" that preserve the ORIGINAL row order
from your input files (mirrors your Logistic Regression + SVM scripts).

Key outputs:
  analysis/<space-name>/decision_tree/holdout/
    - metrics_test.txt
    - confusion_matrix_test.png
    - roc_curve_test.png
    - run_config.json
    - feature_importances.csv
    - tree_plot.png
    - tree_rules.txt (unless --no-rules)
    - predictions_all_in_original_order.csv
    - note_predictions_all.txt

Row order / shuffling:
- We NEVER shuffle the dataset itself.
- We create train/test SPLITS by sampling indices, but we store predictions back into
  an array aligned to the original row order (0..N-1 after optional NA dropping).
- predictions_all_in_original_order.csv is sorted by row_index ascending, so row_index i
  corresponds to row i of the *post-dropna aligned* X/y.
- orig_row_index maps back to the original loaded file row number before dropna.
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
    """Read target from CSV and coerce to binary {0,1}."""
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
            "Make sure both were generated from the same filtered rows and saved in the same order."
        )

    orig_idx = np.arange(len(y), dtype=int)

    if drop_na:
        keep = ~(X.isna().any(axis=1) | y.isna())
        X2 = X.loc[keep].reset_index(drop=True)
        y2 = y.loc[keep].reset_index(drop=True)
        orig_idx2 = orig_idx[keep.values]
        return X2, y2, orig_idx2

    return X.reset_index(drop=True), y.reset_index(drop=True), orig_idx


# ----------------------------
# Plots / outputs
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

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
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


def plot_and_save_tree_readable(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    out_path: Path,
    *,
    max_depth_plot: int = 3,
    class_names: Tuple[str, str] = ("NO", "YES"),
    fontsize: int = 12,
) -> None:
    n_nodes = clf.tree_.node_count
    w = int(min(60, max(18, 8 + 2 * max_depth_plot)))
    h = int(min(30, max(10, 6 + 2 * max_depth_plot)))

    fig, ax = plt.subplots(figsize=(w, h))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=list(class_names),
        filled=True,
        rounded=True,
        max_depth=max_depth_plot,
        proportion=False,
        impurity=True,
        label="all",
        precision=3,
        fontsize=fontsize,
        ax=ax,
    )
    ax.set_title(f"Decision Tree (shown up to depth={max_depth_plot})  |  nodes={n_nodes}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_tree_rules_text(clf: DecisionTreeClassifier, feature_names: List[str], out_path: Path) -> None:
    rules = export_text(clf, feature_names=feature_names, decimals=3)
    out_path.write_text(rules + "\n")


def save_feature_importances(clf: DecisionTreeClassifier, feature_names: List[str], out_path: Path) -> None:
    imp = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp.to_csv(out_path, index=False)


# ----------------------------
# Thresholding
# ----------------------------
def _probs_to_preds(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(int)


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    metric: str,
    grid_size: int,
) -> Tuple[float, float]:
    """
    Optimize threshold on provided (y_true, probs).
    metric: "balanced_accuracy" or "accuracy"
    Returns (best_threshold, best_metric_value)
    """
    if metric not in {"balanced_accuracy", "accuracy"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    lo = float(np.nanmin(probs))
    hi = float(np.nanmax(probs))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        thr = 0.5
        preds = _probs_to_preds(probs, thr)
        val = balanced_accuracy_score(y_true, preds) if metric == "balanced_accuracy" else accuracy_score(y_true, preds)
        return float(thr), float(val)

    thresholds = np.linspace(lo, hi, num=int(grid_size))
    best_thr = float(thresholds[0])
    best_val = -np.inf

    for thr in thresholds:
        preds = _probs_to_preds(probs, float(thr))
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
            "Decision tree driver (HOLDOUT ONLY; no CV).\n"
            "Fit on train; evaluate once on test; save predictions for ALL datapoints in original row order."
        )
    )

    p.add_argument("--mode", choices=["space", "raw"], default="space")

    # space-mode inputs
    p.add_argument("--x", type=Path, default=Path("data/features.csv"))
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
    p.add_argument("--space-name", type=str, default="raw_features")
    p.add_argument("--out-root", type=Path, default=Path("analysis"))

    # Modeling knobs
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")

    # Optional threshold optimization (train-only)
    p.add_argument("--optimize-threshold", action="store_true")
    p.add_argument("--threshold-metric", choices=["balanced_accuracy", "accuracy"], default="balanced_accuracy")
    p.add_argument("--threshold-grid-size", type=int, default=200)

    # Visualization controls
    p.add_argument("--max-depth-plot", type=int, default=3)
    p.add_argument("--tree-fontsize", type=int, default=12)
    p.add_argument("--no-rules", action="store_true")

    # Match logistic/SVM default behavior: drop NaNs during align
    p.add_argument("--drop-na", action="store_true", help="Drop rows with NaNs before splitting (default: False).")

    return p.parse_args()


def _build_clf(args: argparse.Namespace) -> DecisionTreeClassifier:
    cw = None if args.class_weight == "none" else "balanced"
    return DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=int(args.min_samples_leaf),
        class_weight=cw,
        random_state=int(args.random_state),
    )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()

    hold_dir = (args.out_root / args.space_name / "decision_tree" / "holdout").resolve()
    hold_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load X,y
    # ----------------------------
    if args.mode == "space":
        X_raw = read_design_matrix(args.x)
        y_raw = read_target(args.y, args.target_col)
        X, y, orig_row_index = align_xy_preserve_original_index(X_raw, y_raw, drop_na=bool(args.drop_na))
    else:
        if args.raw_csv is None:
            raise ValueError("--raw-csv is required when --mode raw")
        X_raw, y_raw = split_raw_csv(
            args.raw_csv,
            target_col=args.target_col,
            drop_cols=args.drop_cols,
            feature_cols=args.feature_cols,
        )
        X, y, orig_row_index = align_xy_preserve_original_index(X_raw, y_raw, drop_na=bool(args.drop_na))

    X = coerce_X_to_numeric(X)

    uniq = np.unique(y.values)
    if len(uniq) != 2:
        raise ValueError(f"Target must have two classes; got {uniq.tolist()}.")

    feature_names = list(X.columns)
    n = len(y)

    # ----------------------------
    # Holdout split on INDICES (to preserve original order later)
    # ----------------------------
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

    # ----------------------------
    # Fit on train
    # ----------------------------
    clf = _build_clf(args)
    clf.fit(X_train, y_train)

    # ----------------------------
    # Threshold (optionally optimize on TRAIN probs)
    # ----------------------------
    thr_used = float(args.threshold)

    if args.optimize_threshold:
        train_prob = clf.predict_proba(X_train)[:, 1]
        thr_used, thr_val = find_best_threshold(
            y_true=y_train.values,
            probs=train_prob,
            metric=args.threshold_metric,
            grid_size=int(args.threshold_grid_size),
        )
        (hold_dir / "threshold_optimized_train_only.txt").write_text(
            "\n".join(
                [
                    f"optimized_threshold: {thr_used}",
                    f"optimized_metric: {args.threshold_metric}",
                    f"optimized_metric_value_train_only: {thr_val}",
                    "note: optimized on TRAIN probabilities only (no test leakage).",
                ]
            )
            + "\n"
        )

    # ----------------------------
    # Evaluate on test
    # ----------------------------
    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = _probs_to_preds(test_prob, thr_used)

    test_acc = float(accuracy_score(y_test.values, test_pred))
    test_bal = float(balanced_accuracy_score(y_test.values, test_pred))
    test_auc = float(roc_auc_score(y_test.values, test_prob))

    # ----------------------------
    # Save standard outputs
    # ----------------------------
    _write_json(
        hold_dir / "run_config.json",
        {
            "model": "decision_tree",
            "eval": "holdout_only",
            "mode": args.mode,
            "space_name": args.space_name,
            "x_path": str(args.x) if args.mode == "space" else None,
            "y_path": str(args.y) if args.mode == "space" else None,
            "raw_csv": str(args.raw_csv) if args.mode == "raw" else None,
            "target_col": args.target_col,
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "threshold_used": float(thr_used),
            "optimize_threshold": bool(args.optimize_threshold),
            "threshold_metric": args.threshold_metric,
            "threshold_grid_size": int(args.threshold_grid_size),
            "max_depth": args.max_depth,
            "min_samples_leaf": int(args.min_samples_leaf),
            "class_weight": args.class_weight,
            "drop_na": bool(args.drop_na),
            "notes": [
                "Rows are NOT reordered. Train/test split is done by sampling indices.",
                "predictions_all_in_original_order.csv is sorted by row_index and aligns to post-dropna aligned X/y.",
                "orig_row_index maps back to the original file row number before dropna.",
            ],
        },
    )

    (hold_dir / "metrics_test.txt").write_text(
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
                f"threshold_used: {thr_used}",
                "",
                f"TEST_auc: {test_auc:.6f}",
                f"TEST_accuracy: {test_acc:.6f}",
                f"TEST_balanced_accuracy: {test_bal:.6f}",
                f"tree_nodes: {clf.tree_.node_count}",
            ]
        )
        + "\n"
    )

    plot_and_save_confusion_matrix(
        y_true=y_test.values,
        y_pred=test_pred,
        out_path=hold_dir / "confusion_matrix_test.png",
        threshold=thr_used,
        accuracy=test_acc,
        balanced_accuracy=test_bal,
        title_prefix="Decision Tree Confusion Matrix (holdout test)",
    )
    plot_and_save_roc(
        y_true=y_test.values,
        y_score=test_prob,
        out_path=hold_dir / "roc_curve_test.png",
        title="Decision Tree ROC (holdout test)",
    )

    save_feature_importances(clf, feature_names, hold_dir / "feature_importances.csv")
    plot_and_save_tree_readable(
        clf,
        feature_names=feature_names,
        out_path=hold_dir / "tree_plot.png",
        max_depth_plot=int(args.max_depth_plot),
        fontsize=int(args.tree_fontsize),
    )
    if not args.no_rules:
        save_tree_rules_text(clf, feature_names, hold_dir / "tree_rules.txt")

    # ----------------------------
    # Predictions for ALL datapoints (in original row order)
    # ----------------------------
    all_prob = clf.predict_proba(X)[:, 1]
    all_pred = _probs_to_preds(all_prob, thr_used)

    split = np.array(["train"] * n, dtype=object)
    split[test_idx] = "test"

    pred_all = pd.DataFrame(
        {
            "row_index": np.arange(n, dtype=int),
            "orig_row_index": orig_row_index.astype(int),
            "split": split,
            "y_true": y.values.astype(int),
            "prob": all_prob.astype(float),
            "y_pred": all_pred.astype(int),
        }
    ).sort_values("row_index", kind="mergesort")

    pred_all.to_csv(hold_dir / "predictions_all_in_original_order.csv", index=False)

    (hold_dir / "note_predictions_all.txt").write_text(
        "\n".join(
            [
                "predictions_all_in_original_order.csv notes:",
                "- File is sorted by row_index ascending.",
                "- row_index refers to the aligned (post-dropna) X/y row number used for modeling.",
                "- orig_row_index refers to the original row number in the loaded input file(s) before dropna.",
                "- 'prob' is the predicted probability for class 1 from the TRAIN-fit model.",
                "- Model used here is fit on TRAIN ONLY.",
                "- split='test' rows are out-of-sample; split='train' rows are in-sample.",
            ]
        )
        + "\n"
    )

    print(f"[OK] Wrote outputs to: {hold_dir}")
    print(f"[OK] TEST AUC={test_auc:.4f}  acc={test_acc:.4f}  bal_acc={test_bal:.4f}  thr={thr_used:.4g}")


if __name__ == "__main__":
    main()
