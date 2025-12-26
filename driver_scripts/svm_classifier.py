#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
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


# ----------------------------
# Model helpers
# ----------------------------
def _build_model(args: argparse.Namespace) -> Pipeline:
    cw = None if args.class_weight == "none" else "balanced"

    if args.svm_type == "linear_svc":
        # LinearSVC: fast + good baseline; uses decision_function; no predict_proba.
        clf = LinearSVC(
            C=args.C,
            class_weight=cw,
            random_state=args.random_state,
            max_iter=args.max_iter,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # SVC supports kernels and probability if desired (probability=True is slower).
    svc = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        degree=args.degree,
        class_weight=cw,
        probability=args.probability,  # only used if you want calibrated-ish probs
        random_state=args.random_state,
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


# ----------------------------
# Threshold optimization
# ----------------------------
def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str,
    grid_size: int,
) -> Tuple[float, float]:
    """
    Choose threshold that maximizes chosen metric on provided (y_true, scores).
    metric: "balanced_accuracy" or "accuracy"
    Returns (best_threshold, best_metric_value)
    """
    if metric not in {"balanced_accuracy", "accuracy"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    lo = float(np.nanmin(scores))
    hi = float(np.nanmax(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        # degenerate; fall back
        thr = 0.0
        preds = _scores_to_preds(scores, thr)
        val = balanced_accuracy_score(y_true, preds) if metric == "balanced_accuracy" else accuracy_score(y_true, preds)
        return float(thr), float(val)

    thresholds = np.linspace(lo, hi, num=int(grid_size))
    best_thr = thresholds[0]
    best_val = -np.inf
    for thr in thresholds:
        preds = _scores_to_preds(scores, float(thr))
        if metric == "balanced_accuracy":
            v = balanced_accuracy_score(y_true, preds)
        else:
            v = accuracy_score(y_true, preds)
        if v > best_val:
            best_val = v
            best_thr = float(thr)
    return float(best_thr), float(best_val)


# ----------------------------
# CV
# ----------------------------
def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    uniq, counts = np.unique(y.values, return_counts=True)
    if len(uniq) != 2:
        raise ValueError("Binary target required for CV/AUC.")
    min_class = int(counts.min())
    k = min(args.cv_folds, max(2, min_class))
    if k < args.cv_folds:
        print(f"[WARN] Reducing --cv-folds to {k} due to small minority class (min_class={min_class}).")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.random_state)

    oof_score = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)
    oof_pred = np.full(shape=(len(y),), fill_value=-1, dtype=int)

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        model = _build_model(args)
        model.fit(X_tr, y_tr)

        scores = _get_scores(model, X_va)
        preds = _scores_to_preds(scores, args.threshold)

        oof_score[va_idx] = scores
        oof_pred[va_idx] = preds

        acc = float(accuracy_score(y_va.values, preds))
        bal = float(balanced_accuracy_score(y_va.values, preds))
        auc_fold = float(roc_auc_score(y_va.values, scores))

        rows.append(
            {
                "fold": fold,
                "n_train": len(tr_idx),
                "n_val": len(va_idx),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "auc": auc_fold,
            }
        )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    oof_df = pd.DataFrame({"y_true": y.values.astype(int), "oof_score": oof_score, "oof_pred": oof_pred})
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    oof_acc = float(accuracy_score(y.values, oof_pred))
    oof_bal = float(balanced_accuracy_score(y.values, oof_pred))
    oof_auc = float(roc_auc_score(y.values, oof_score))

    plot_and_save_confusion_matrix(
        y_true=y.values,
        y_pred=oof_pred,
        out_path=out_dir / "confusion_matrix_oof.png",
        threshold=args.threshold,
        accuracy=oof_acc,
        balanced_accuracy=oof_bal,
        title_prefix="SVM Confusion Matrix (OOF CV)",
    )
    plot_and_save_roc(
        y_true=y.values,
        y_score=oof_score,
        out_path=out_dir / "roc_curve_oof.png",
        title="SVM ROC (OOF CV)",
    )

    mean_auc = float(np.mean(fold_df["auc"].values))
    std_auc = float(np.std(fold_df["auc"].values, ddof=1)) if len(fold_df) > 1 else float("nan")
    mean_acc = float(np.mean(fold_df["accuracy"].values))
    std_acc = float(np.std(fold_df["accuracy"].values, ddof=1)) if len(fold_df) > 1 else float("nan")
    mean_bal = float(np.mean(fold_df["balanced_accuracy"].values))
    std_bal = float(np.std(fold_df["balanced_accuracy"].values, ddof=1)) if len(fold_df) > 1 else float("nan")

    (out_dir / "summary_metrics.txt").write_text(
        "\n".join(
            [
                f"svm_type: {args.svm_type}",
                f"kernel: {args.kernel}",
                f"C: {args.C}",
                f"gamma: {args.gamma}",
                f"degree: {args.degree}",
                f"class_weight: {args.class_weight}",
                "",
                f"cv_folds_requested: {args.cv_folds}",
                f"cv_folds_used: {k}",
                f"threshold: {args.threshold}",
                "",
                f"fold_mean_auc: {mean_auc:.6f}",
                f"fold_std_auc:  {std_auc:.6f}",
                f"fold_mean_accuracy: {mean_acc:.6f}",
                f"fold_std_accuracy:  {std_acc:.6f}",
                f"fold_mean_balanced_accuracy: {mean_bal:.6f}",
                f"fold_std_balanced_accuracy:  {std_bal:.6f}",
                "",
                f"oof_auc: {oof_auc:.6f}",
                f"oof_accuracy: {oof_acc:.6f}",
                f"oof_balanced_accuracy: {oof_bal:.6f}",
            ]
        )
        + "\n"
    )

    return {
        "cv_folds_used": int(k),
        "fold_metrics": fold_df,
        "oof_auc": oof_auc,
        "oof_accuracy": oof_acc,
        "oof_balanced_accuracy": oof_bal,
        "oof_score": oof_score,
        "oof_pred": oof_pred,
    }


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit an SVM classifier (with scaling) on either "
            "(a) separate X + y files, or (b) one raw CSV containing X+y, "
            "and save plots + metrics. Supports CV and CV+holdout."
        )
    )

    p.add_argument("--mode", choices=["space", "raw"], default="space")
    p.add_argument("--x", type=Path, default=None, help="(space mode) Path to X (.csv or .npy).")
    p.add_argument("--y", type=Path, default=None, help="(space mode) Path to y CSV.")
    p.add_argument("--raw-csv", type=Path, default=None, help="(raw mode) Raw CSV with X+y.")
    p.add_argument("--drop-cols", nargs="*", default=None, help="(raw mode) Columns to drop from features.")
    p.add_argument("--feature-cols", nargs="*", default=None, help="(raw mode) Explicit feature columns to use.")
    p.add_argument("--target-col", type=str, default="Qualified Municipality")

    p.add_argument(
        "--eval",
        choices=["holdout", "cv", "cv+holdout"],
        default="cv+holdout",
        help="holdout: single split. cv: CV on all data. cv+holdout: keep a test set, CV on train.",
    )
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=0)

    # SVM config
    p.add_argument(
        "--svm-type",
        choices=["svc", "linear_svc"],
        default="svc",
        help="svc: SVC with kernel. linear_svc: LinearSVC baseline (fast, no predict_proba).",
    )
    p.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="linear")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", type=str, default="scale", help="For SVC only: 'scale', 'auto', or a float as string.")
    p.add_argument("--degree", type=int, default=3, help="For poly kernel.")
    p.add_argument("--probability", action="store_true", help="If set, SVC will fit probability estimates (slower).")
    p.add_argument("--max-iter", type=int, default=10000, help="For LinearSVC max_iter.")

    p.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Use class_weight='balanced' or none.",
    )

    # Thresholding + optional optimization
    p.add_argument("--threshold", type=float, default=0.0, help="Threshold on SVM score (default: 0.0).")
    p.add_argument(
        "--optimize-threshold",
        action="store_true",
        help=(
            "If set, choose threshold to maximize a metric using CV OOF scores on TRAIN only "
            "(in cv+holdout) or on all data (in cv)."
        ),
    )
    p.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "accuracy"],
        default="balanced_accuracy",
        help="Metric to optimize threshold for.",
    )
    p.add_argument("--threshold-grid-size", type=int, default=200)

    p.add_argument("--out-root", type=Path, default=Path("analysis"))
    p.add_argument("--space-name", type=str, default="raw")

    return p.parse_args()


def _parse_gamma(gamma_str: str) -> Any:
    if gamma_str in {"scale", "auto"}:
        return gamma_str
    try:
        return float(gamma_str)
    except ValueError:
        raise ValueError("--gamma must be 'scale', 'auto', or a float")


def main() -> None:
    args = parse_args()
    args.gamma = _parse_gamma(args.gamma)

    out_dir = (args.out_root / args.space_name / "svm").resolve()
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

    X = coerce_X_to_numeric(X)

    uniq = np.unique(y.values)
    if len(uniq) != 2:
        raise ValueError(f"Target must have two classes; got {uniq.tolist()}.")

    # ----------------------------
    # eval modes
    # ----------------------------
    if args.eval == "holdout":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        model = _build_model(args)
        model.fit(X_train, y_train)

        test_scores = _get_scores(model, X_test)

        thr_used = float(args.threshold)
        if args.optimize_threshold:
            # NOTE: in pure holdout mode, "kosher" threshold optimization is ambiguous;
            # we'll optimize on training via an inner CV-style split? We avoid that here:
            # We'll just optimize on training *in-sample* scores (not ideal).
            # Better: use cv+holdout.
            train_scores = _get_scores(model, X_train)
            thr_used, thr_val = find_best_threshold(
                y_train.values, train_scores, metric=args.threshold_metric, grid_size=args.threshold_grid_size
            )
            (out_dir / "threshold_optimized_train_only.txt").write_text(
                f"optimized_threshold: {thr_used}\noptimized_metric: {args.threshold_metric}\n"
                f"optimized_metric_value_train: {thr_val}\n"
            )

        test_pred = _scores_to_preds(test_scores, thr_used)

        acc = float(accuracy_score(y_test.values, test_pred))
        bal = float(balanced_accuracy_score(y_test.values, test_pred))
        auc_val = float(roc_auc_score(y_test.values, test_scores))

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

        print(f"[OK] Wrote outputs to: {out_dir}")
        print(f"[OK] TEST AUC={auc_val:.4f}  acc={acc:.4f}  bal_acc={bal:.4f}  thr={thr_used:.4g}")
        return

    if args.eval == "cv":
        cv_dir = out_dir / "cv"
        summary = run_cv(X, y, args=args, out_dir=cv_dir)

        thr_used = float(args.threshold)
        if args.optimize_threshold:
            thr_used, thr_val = find_best_threshold(
                y.values, summary["oof_score"], metric=args.threshold_metric, grid_size=args.threshold_grid_size
            )
            oof_pred_opt = _scores_to_preds(summary["oof_score"], thr_used)
            oof_acc_opt = float(accuracy_score(y.values, oof_pred_opt))
            oof_bal_opt = float(balanced_accuracy_score(y.values, oof_pred_opt))

            (cv_dir / "threshold_optimized_oof.txt").write_text(
                "\n".join(
                    [
                        f"optimized_threshold: {thr_used}",
                        f"optimized_metric: {args.threshold_metric}",
                        f"optimized_metric_value_oof: {thr_val}",
                        f"oof_accuracy_at_opt_threshold: {oof_acc_opt}",
                        f"oof_balanced_accuracy_at_opt_threshold: {oof_bal_opt}",
                    ]
                )
                + "\n"
            )

        print(f"[OK] Wrote CV outputs to: {cv_dir}")
        print(f"[OK] OOF AUC={summary['oof_auc']:.4f}  OOF bal_acc={summary['oof_balanced_accuracy']:.4f}")
        return

    # eval == cv+holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # CV on train portion
    cv_dir = out_dir / "cv_train"
    summary = run_cv(X_train.reset_index(drop=True), y_train.reset_index(drop=True), args=args, out_dir=cv_dir)

    thr_used = float(args.threshold)
    if args.optimize_threshold:
        thr_used, thr_val = find_best_threshold(
            y_train.values,
            summary["oof_score"],
            metric=args.threshold_metric,
            grid_size=args.threshold_grid_size,
        )
        (cv_dir / "threshold_optimized_oof_train.txt").write_text(
            "\n".join(
                [
                    f"optimized_threshold: {thr_used}",
                    f"optimized_metric: {args.threshold_metric}",
                    f"optimized_metric_value_oof_train: {thr_val}",
                    "note: threshold optimized using OOF scores on TRAIN split only (no test leakage).",
                ]
            )
            + "\n"
        )

    # Fit final model on full train, evaluate once on test
    model = _build_model(args)
    model.fit(X_train, y_train)

    test_scores = _get_scores(model, X_test)
    test_pred = _scores_to_preds(test_scores, thr_used)

    acc = float(accuracy_score(y_test.values, test_pred))
    bal = float(balanced_accuracy_score(y_test.values, test_pred))
    auc_val = float(roc_auc_score(y_test.values, test_scores))

    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    (test_dir / "metrics_test.txt").write_text(
        "\n".join(
            [
                "eval: cv+holdout",
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

    plot_and_save_confusion_matrix(
        y_true=y_test.values,
        y_pred=test_pred,
        out_path=test_dir / "confusion_matrix_test.png",
        threshold=thr_used,
        accuracy=acc,
        balanced_accuracy=bal,
        title_prefix="SVM Confusion Matrix (holdout test)",
    )
    plot_and_save_roc(
        y_true=y_test.values,
        y_score=test_scores,
        out_path=test_dir / "roc_curve_test.png",
        title="SVM ROC (holdout test)",
    )

    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                "eval: cv+holdout",
                f"cv_dir: {cv_dir}",
                f"test_dir: {test_dir}",
                "",
                "CV on TRAIN portion (OOF):",
                f"  OOF_auc(train): {summary['oof_auc']:.6f}",
                f"  OOF_accuracy(train): {summary['oof_accuracy']:.6f}",
                f"  OOF_balanced_accuracy(train): {summary['oof_balanced_accuracy']:.6f}",
                "",
                "Holdout TEST:",
                f"  TEST_auc: {auc_val:.6f}",
                f"  TEST_accuracy: {acc:.6f}",
                f"  TEST_balanced_accuracy: {bal:.6f}",
                "",
                f"threshold_used: {thr_used}",
                f"threshold_optimized: {bool(args.optimize_threshold)}",
                f"threshold_metric: {args.threshold_metric}",
            ]
        )
        + "\n"
    )

    print(f"[OK] Wrote CV outputs to: {cv_dir}")
    print(f"[OK] Wrote TEST outputs to: {test_dir}")
    print(f"[OK] TEST AUC={auc_val:.4f}  acc={acc:.4f}  bal_acc={bal:.4f}  thr={thr_used:.4g}")


if __name__ == "__main__":
    main()
