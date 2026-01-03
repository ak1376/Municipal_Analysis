# src/loocv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
)


@dataclass(frozen=True)
class LOOCVResult:
    probs: Optional[np.ndarray]        # (n,) for binary, or (n, K) for multiclass
    preds: Optional[np.ndarray]        # (n,) hard predictions (if threshold/argmax applied)
    scores: Optional[np.ndarray]       # (n,) decision_function values if probs unavailable
    metrics: Dict[str, float]


def loocv_predict(
    estimator: BaseEstimator,
    X,
    y,
    *,
    method: str = "predict_proba",
    threshold: float = 0.5,
    pos_index: int = 1,
    n_jobs: int = -1,
) -> LOOCVResult:
    """
    General LOOCV prediction helper.

    - If method="predict_proba": returns probs (binary: prob of class pos_index)
    - If method="decision_function": returns scores
    - If method="predict": returns preds

    For binary classification, this also computes auc/accuracy/balanced_accuracy when possible.
    """
    loo = LeaveOneOut()

    y_arr = np.asarray(y).ravel()

    probs = None
    scores = None
    preds = None
    metrics: Dict[str, float] = {}

    if method == "predict_proba":
        proba = cross_val_predict(
            estimator,
            X,
            y_arr,
            cv=loo,
            method="predict_proba",
            n_jobs=n_jobs,
        )
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            probs = proba[:, pos_index].astype(float)
            preds = (probs >= float(threshold)).astype(int)

            # Metrics for binary targets only
            uniq = np.unique(y_arr)
            if set(uniq.tolist()) <= {0, 1} and len(uniq) == 2:
                metrics["auc"] = float(roc_auc_score(y_arr, probs))
                metrics["accuracy"] = float(accuracy_score(y_arr, preds))
                metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_arr, preds))
        else:
            # multiclass probability matrix
            probs = proba.astype(float)
            preds = np.argmax(probs, axis=1)

    elif method == "decision_function":
        scores = cross_val_predict(
            estimator,
            X,
            y_arr,
            cv=loo,
            method="decision_function",
            n_jobs=n_jobs,
        )
        scores = np.asarray(scores).ravel()

        uniq = np.unique(y_arr)
        if set(uniq.tolist()) <= {0, 1} and len(uniq) == 2:
            # AUC works with raw scores too
            metrics["auc"] = float(roc_auc_score(y_arr, scores))
            preds = (scores >= 0.0).astype(int)  # default threshold in score space
            metrics["accuracy"] = float(accuracy_score(y_arr, preds))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_arr, preds))

    elif method == "predict":
        preds = cross_val_predict(
            estimator,
            X,
            y_arr,
            cv=loo,
            method="predict",
            n_jobs=n_jobs,
        )
        preds = np.asarray(preds).ravel()

        uniq = np.unique(y_arr)
        if set(uniq.tolist()) <= {0, 1} and len(uniq) == 2:
            metrics["accuracy"] = float(accuracy_score(y_arr, preds))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_arr, preds))

    else:
        raise ValueError("method must be one of: predict_proba, decision_function, predict")

    return LOOCVResult(probs=probs, preds=preds, scores=scores, metrics=metrics)


def fit_final_model(estimator: BaseEstimator, X, y) -> BaseEstimator:
    """
    Fit a *fresh clone* of estimator on all data and return it.
    """
    est = clone(estimator)
    est.fit(X, np.asarray(y).ravel())
    return est
