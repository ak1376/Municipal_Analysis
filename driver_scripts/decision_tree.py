#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
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
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {path}")

    return coerce_binary_target(df[target_col], name=target_col)


def align_xy(
    X: pd.DataFrame, y: pd.Series, *, drop_na: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
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
    """
    Coerce all columns to numeric (hard error if coercion introduces NaNs).
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

    fig, ax = plt.subplots(figsize=(6, 6))
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
    """
    Make a readable tree plot (cropped to max_depth_plot).
    """
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


def save_tree_rules_text(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    out_path: Path,
) -> None:
    rules = export_text(clf, feature_names=feature_names, decimals=3)
    out_path.write_text(rules)


def save_feature_importances(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    out_path: Path,
) -> None:
    imp = pd.DataFrame(
        {"feature": feature_names, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp.to_csv(out_path, index=False)


# ----------------------------
# CV helpers
# ----------------------------
def _build_clf(args: argparse.Namespace) -> DecisionTreeClassifier:
    cw = None if args.class_weight == "none" else "balanced"
    return DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=cw,
        random_state=args.random_state,
    )


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Stratified K-fold CV with out-of-fold (OOF) predicted probabilities.
    Saves:
      - fold_metrics.csv
      - oof_predictions.csv
      - aggregate ROC + confusion matrix from OOF
      - summary_metrics.txt
    Returns summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose K that works even for small class counts
    uniq, counts = np.unique(y.values, return_counts=True)
    if len(uniq) == 2:
        min_class = int(counts.min())
        # user can request larger, but cap by minority class
        k = min(args.cv_folds, max(2, min_class))
        if k < args.cv_folds:
            print(f"[WARN] Reducing --cv-folds to {k} due to small minority class (min_class={min_class}).")
    else:
        # degenerate case: only one class -> no ROC/AUC meaningful
        k = 2

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.random_state)

    oof_prob = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)
    oof_pred = np.full(shape=(len(y),), fill_value=-1, dtype=int)

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        clf = _build_clf(args)
        clf.fit(X_tr, y_tr)

        # predict_proba exists for sklearn trees
        prob = clf.predict_proba(X_va)[:, 1]
        pred = (prob >= args.threshold).astype(int)

        oof_prob[va_idx] = prob
        oof_pred[va_idx] = pred

        acc = float(accuracy_score(y_va.values, pred))
        bal = float(balanced_accuracy_score(y_va.values, pred))
        # fold-level AUC only defined if both classes present in the fold
        try:
            auc_fold = float(roc_auc_score(y_va.values, prob))
        except ValueError:
            auc_fold = float("nan")

        rows.append(
            {
                "fold": fold,
                "n_train": len(tr_idx),
                "n_val": len(va_idx),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "auc": auc_fold,
                "tree_nodes": int(clf.tree_.node_count),
            }
        )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    # Save OOF predictions
    oof_df = pd.DataFrame(
        {
            "y_true": y.values.astype(int),
            "oof_prob": oof_prob,
            "oof_pred": oof_pred,
        }
    )
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    summary: Dict[str, Any] = {
        "cv_folds_used": int(k),
        "fold_metrics": fold_df,
    }

    # Aggregate metrics from OOF (most useful single estimate)
    if len(np.unique(y.values)) == 2 and np.isfinite(oof_prob).all():
        oof_acc = float(accuracy_score(y.values, oof_pred))
        oof_bal = float(balanced_accuracy_score(y.values, oof_pred))
        oof_auc = float(roc_auc_score(y.values, oof_prob))

        summary.update({"oof_accuracy": oof_acc, "oof_balanced_accuracy": oof_bal, "oof_auc": oof_auc})

        # Plots from OOF
        plot_and_save_confusion_matrix(
            y_true=y.values,
            y_pred=oof_pred,
            out_path=out_dir / "confusion_matrix_oof.png",
            threshold=args.threshold,
            accuracy=oof_acc,
            balanced_accuracy=oof_bal,
            title_prefix="Decision Tree Confusion Matrix (OOF CV)",
        )
        plot_and_save_roc(
            y_true=y.values,
            y_score=oof_prob,
            out_path=out_dir / "roc_curve_oof.png",
            title="Decision Tree ROC (OOF CV)",
        )

    # Summary text
    mean_auc = float(np.nanmean(fold_df["auc"].values))
    std_auc = float(np.nanstd(fold_df["auc"].values, ddof=1)) if len(fold_df) > 1 else float("nan")
    mean_acc = float(np.nanmean(fold_df["accuracy"].values))
    std_acc = float(np.nanstd(fold_df["accuracy"].values, ddof=1)) if len(fold_df) > 1 else float("nan")
    mean_bal = float(np.nanmean(fold_df["balanced_accuracy"].values))
    std_bal = float(np.nanstd(fold_df["balanced_accuracy"].values, ddof=1)) if len(fold_df) > 1 else float("nan")

    txt = [
        f"cv_folds_requested: {args.cv_folds}",
        f"cv_folds_used: {k}",
        f"threshold: {args.threshold}",
        f"class_weight: {args.class_weight}",
        f"max_depth: {args.max_depth}",
        f"min_samples_leaf: {args.min_samples_leaf}",
        "",
        f"fold_mean_auc: {mean_auc:.6f}",
        f"fold_std_auc:  {std_auc:.6f}",
        f"fold_mean_accuracy: {mean_acc:.6f}",
        f"fold_std_accuracy:  {std_acc:.6f}",
        f"fold_mean_balanced_accuracy: {mean_bal:.6f}",
        f"fold_std_balanced_accuracy:  {std_bal:.6f}",
    ]
    if "oof_auc" in summary:
        txt += [
            "",
            f"oof_auc: {summary['oof_auc']:.6f}",
            f"oof_accuracy: {summary['oof_accuracy']:.6f}",
            f"oof_balanced_accuracy: {summary['oof_balanced_accuracy']:.6f}",
        ]
    (out_dir / "summary_metrics.txt").write_text("\n".join(txt) + "\n")

    return summary


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit a sklearn DecisionTreeClassifier on either "
            "(a) separate X + y files, or (b) one raw CSV containing X+y, "
            "and save plots + rules + importances. Supports CV for small datasets."
        )
    )

    p.add_argument(
        "--mode",
        choices=["space", "raw"],
        default="space",
        help="space: X from .npy/.csv + y from separate CSV. raw: X+y from one raw CSV.",
    )

    # space-mode inputs
    p.add_argument("--x", type=Path, default=None, help="(space mode) Path to X (.csv or .npy).")
    p.add_argument("--y", type=Path, default=None, help="(space mode) Path to y CSV.")

    # raw-mode input
    p.add_argument("--raw-csv", type=Path, default=None, help="(raw mode) Path to raw CSV containing BOTH X and y.")
    p.add_argument("--drop-cols", nargs="*", default=None, help="(raw mode) Columns to drop from features.")
    p.add_argument("--feature-cols", nargs="*", default=None, help="(raw mode) Explicit feature columns to use.")

    p.add_argument(
        "--target-col",
        type=str,
        default="Qualified Municipality",
        help="Target column name (in y CSV for space-mode, or in raw CSV for raw-mode).",
    )

    # CV / splitting
    p.add_argument(
        "--eval",
        choices=["holdout", "cv", "cv+holdout"],
        default="cv+holdout",
        help=(
            "holdout: single train/test split only. "
            "cv: cross-validation on all data (no test set). "
            "cv+holdout: keep a test set, CV on remaining train data (recommended)."
        ),
    )
    p.add_argument("--test-size", type=float, default=0.25, help="Holdout test fraction (default: 0.25).")
    p.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5).")
    p.add_argument("--random-state", type=int, default=0, help="Random seed for split/model.")

    p.add_argument("--threshold", type=float, default=0.5, help="Threshold on predicted prob for YES (default: 0.5).")
    p.add_argument("--max-depth", type=int, default=None, help="Decision tree max depth (default: None).")
    p.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples per leaf (default: 1).")
    p.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Use class_weight='balanced' or none (default: balanced).",
    )

    p.add_argument("--out-root", type=Path, default=Path("analysis"), help="Root analysis dir (default: analysis).")
    p.add_argument("--space-name", type=str, default="raw", help="Folder under out-root (e.g., raw, pca, pls).")

    # Visualization controls
    p.add_argument("--max-depth-plot", type=int, default=3, help="Max depth to visualize (default: 3).")
    p.add_argument("--tree-fontsize", type=int, default=12, help="Font size for tree plot (default: 12).")
    p.add_argument("--no-rules", action="store_true", help="Disable writing a text rules file (export_text).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = (args.out_root / args.space_name / "decision_tree").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load X,y
    if args.mode == "space":
        if args.x is None or args.y is None:
            raise ValueError("--x and --y are required when --mode space")
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

    # Ensure numeric
    X = coerce_X_to_numeric(X)
    feature_names = list(X.columns)

    # Make sure binary target is usable
    uniq = np.unique(y.values)
    if len(uniq) < 2:
        raise ValueError(
            f"Target has only one class present: {uniq.tolist()}. "
            "ROC/AUC and stratified CV are undefined. Check your filtering."
        )

    # ----------------------------
    # Evaluation modes
    # ----------------------------
    if args.eval == "holdout":
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )

        clf = _build_clf(args)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= args.threshold).astype(int)

        acc = float(accuracy_score(y_test.values, preds))
        bal_acc = float(balanced_accuracy_score(y_test.values, preds))
        auc_val = float(roc_auc_score(y_test.values, probs))

        (out_dir / "metrics.txt").write_text(
            f"eval: holdout\n"
            f"mode: {args.mode}\n"
            f"space_name: {args.space_name}\n"
            f"test_size: {args.test_size}\n"
            f"random_state: {args.random_state}\n"
            f"threshold: {args.threshold}\n"
            f"class_weight: {args.class_weight}\n"
            f"max_depth: {args.max_depth}\n"
            f"min_samples_leaf: {args.min_samples_leaf}\n"
            f"\n"
            f"AUC: {auc_val:.6f}\n"
            f"accuracy: {acc:.6f}\n"
            f"balanced_accuracy: {bal_acc:.6f}\n"
            f"n_total: {len(y)}\n"
            f"n_test: {len(y_test)}\n"
            f"p: {X.shape[1]}\n"
            f"tree_nodes: {clf.tree_.node_count}\n"
        )

        plot_and_save_confusion_matrix(
            y_true=y_test.values,
            y_pred=preds,
            out_path=out_dir / "confusion_matrix.png",
            threshold=args.threshold,
            accuracy=acc,
            balanced_accuracy=bal_acc,
            title_prefix="Decision Tree Confusion Matrix (test set)",
        )

        plot_and_save_roc(
            y_true=y_test.values,
            y_score=probs,
            out_path=out_dir / "roc_curve.png",
            title="Decision Tree ROC (test set)",
        )

        save_feature_importances(clf, feature_names, out_dir / "feature_importances.csv")
        plot_and_save_tree_readable(
            clf,
            feature_names=feature_names,
            out_path=out_dir / "tree_plot.png",
            max_depth_plot=args.max_depth_plot,
            fontsize=args.tree_fontsize,
        )
        if not args.no_rules:
            save_tree_rules_text(clf, feature_names, out_dir / "tree_rules.txt")

        print(f"[OK] Wrote outputs to: {out_dir}")
        print(f"[OK] AUC={auc_val:.4f}  acc={acc:.4f}  bal_acc={bal_acc:.4f}  (holdout test)")
        return

    if args.eval == "cv":
        cv_dir = out_dir / "cv"
        summary = run_cv(X, y, args=args, out_dir=cv_dir)

        # Also fit on ALL data for interpretability artifacts
        clf_full = _build_clf(args)
        clf_full.fit(X, y)
        save_feature_importances(clf_full, feature_names, out_dir / "feature_importances_full.csv")
        plot_and_save_tree_readable(
            clf_full,
            feature_names=feature_names,
            out_path=out_dir / "tree_plot_full.png",
            max_depth_plot=args.max_depth_plot,
            fontsize=args.tree_fontsize,
        )
        if not args.no_rules:
            save_tree_rules_text(clf_full, feature_names, out_dir / "tree_rules_full.txt")

        print(f"[OK] Wrote CV outputs to: {cv_dir}")
        if "oof_auc" in summary:
            print(f"[OK] OOF AUC={summary['oof_auc']:.4f}  OOF acc={summary['oof_accuracy']:.4f}")
        return

    # eval == "cv+holdout" (recommended)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # CV on training portion
    cv_dir = out_dir / "cv_train"
    summary = run_cv(X_train.reset_index(drop=True), y_train.reset_index(drop=True), args=args, out_dir=cv_dir)

    # Fit final model on all training data, evaluate once on held-out test set
    clf = _build_clf(args)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    acc = float(accuracy_score(y_test.values, preds))
    bal_acc = float(balanced_accuracy_score(y_test.values, preds))
    auc_val = float(roc_auc_score(y_test.values, probs))

    # Save holdout test artifacts under test/
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    (test_dir / "metrics_test.txt").write_text(
        f"eval: cv+holdout\n"
        f"mode: {args.mode}\n"
        f"space_name: {args.space_name}\n"
        f"test_size: {args.test_size}\n"
        f"cv_folds_requested: {args.cv_folds}\n"
        f"random_state: {args.random_state}\n"
        f"threshold: {args.threshold}\n"
        f"class_weight: {args.class_weight}\n"
        f"max_depth: {args.max_depth}\n"
        f"min_samples_leaf: {args.min_samples_leaf}\n"
        f"\n"
        f"TEST_AUC: {auc_val:.6f}\n"
        f"TEST_accuracy: {acc:.6f}\n"
        f"TEST_balanced_accuracy: {bal_acc:.6f}\n"
        f"n_total: {len(y)}\n"
        f"n_train: {len(y_train)}\n"
        f"n_test: {len(y_test)}\n"
        f"p: {X.shape[1]}\n"
        f"tree_nodes: {clf.tree_.node_count}\n"
    )

    plot_and_save_confusion_matrix(
        y_true=y_test.values,
        y_pred=preds,
        out_path=test_dir / "confusion_matrix_test.png",
        threshold=args.threshold,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        title_prefix="Decision Tree Confusion Matrix (holdout test)",
    )
    plot_and_save_roc(
        y_true=y_test.values,
        y_score=probs,
        out_path=test_dir / "roc_curve_test.png",
        title="Decision Tree ROC (holdout test)",
    )

    # Interpretability outputs (final model)
    save_feature_importances(clf, feature_names, out_dir / "feature_importances_final.csv")
    plot_and_save_tree_readable(
        clf,
        feature_names=feature_names,
        out_path=out_dir / "tree_plot_final.png",
        max_depth_plot=args.max_depth_plot,
        fontsize=args.tree_fontsize,
    )
    if not args.no_rules:
        save_tree_rules_text(clf, feature_names, out_dir / "tree_rules_final.txt")

    # Top-level summary file that links CV + test
    lines = [
        f"eval: cv+holdout",
        f"cv_dir: {cv_dir}",
        f"test_dir: {test_dir}",
        "",
        "CV on TRAIN portion:",
    ]
    if "oof_auc" in summary:
        lines += [
            f"  OOF_AUC(train): {summary['oof_auc']:.6f}",
            f"  OOF_accuracy(train): {summary['oof_accuracy']:.6f}",
            f"  OOF_balanced_accuracy(train): {summary['oof_balanced_accuracy']:.6f}",
        ]
    lines += [
        "",
        "Holdout TEST:",
        f"  TEST_AUC: {auc_val:.6f}",
        f"  TEST_accuracy: {acc:.6f}",
        f"  TEST_balanced_accuracy: {bal_acc:.6f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    print(f"[OK] Wrote CV outputs to: {cv_dir}")
    print(f"[OK] Wrote TEST outputs to: {test_dir}")
    print(f"[OK] TEST AUC={auc_val:.4f}  acc={acc:.4f}  bal_acc={bal_acc:.4f}")


if __name__ == "__main__":
    main()
