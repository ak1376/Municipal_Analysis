# src/logistic_regression_helpers.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# Model construction
# ============================================================

def make_sklearn_logit_pipeline(
    *,
    fit_intercept: bool = True,
    C: float = 1.0,
    penalty: str = "l2",
    class_weight: Optional[str] = None,
    max_iter: int = 10_000,
) -> Pipeline:
    """
    StandardScaler + LogisticRegression pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    solver="lbfgs",
                    penalty=penalty,
                    C=float(C),
                    fit_intercept=bool(fit_intercept),
                    class_weight=class_weight,
                    max_iter=int(max_iter),
                ),
            ),
        ]
    )


# ============================================================
# Saving helpers
# ============================================================

def save_sklearn_model(path: Path, model: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_sklearn_logit_coefficients_csv(
    *,
    model: Pipeline,
    feature_cols: List[str],
    out_path: Path,
) -> None:
    """
    Save coefficients from a fitted sklearn LogisticRegression pipeline.

    NOTE:
    - Coefficients are in *scaled feature space* because we use StandardScaler.
    """
    if not hasattr(model, "named_steps") or "logit" not in model.named_steps:
        raise KeyError("Expected a sklearn Pipeline with a 'logit' step.")

    clf = model.named_steps["logit"]
    if not hasattr(clf, "coef_"):
        raise TypeError("Model not fitted or has no coef_.")

    coef = np.asarray(clf.coef_).ravel()
    if len(coef) != len(feature_cols):
        raise ValueError(f"coef length {len(coef)} != n_features {len(feature_cols)}")

    rows = [{"term": c, "coef": float(w)} for c, w in zip(feature_cols, coef)]
    if getattr(clf, "fit_intercept", False):
        rows.insert(0, {"term": "intercept", "coef": float(np.asarray(clf.intercept_).ravel()[0])})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================
# Prediction table builders (logistic-specific wrappers)
# ============================================================

def build_predictions_df_holdout(
    *,
    y_true: pd.Series,
    probs_all: np.ndarray,
    threshold: float,
    orig_row_index: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Build predictions_all_in_original_order.csv for holdout.
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


def build_predictions_df_loocv(
    *,
    y_true: pd.Series,
    probs_all: np.ndarray,
    threshold: float,
    orig_row_index: np.ndarray,
) -> pd.DataFrame:
    """
    Build predictions_all_in_original_order.csv for LOOCV.
    Every row is out-of-sample.
    """
    n = len(y_true)
    probs_all = np.asarray(probs_all).ravel()
    y_pred = (probs_all >= float(threshold)).astype(int)

    return (
        pd.DataFrame(
            {
                "row_index": np.arange(n, dtype=int),
                "orig_row_index": orig_row_index.astype(int),
                "split": np.array(["loocv"] * n, dtype=object),
                "y_true": y_true.values.astype(int),
                "prob": probs_all.astype(float),
                "y_pred": y_pred.astype(int),
            }
        )
        .sort_values("row_index", kind="mergesort")
        .reset_index(drop=True)
    )
