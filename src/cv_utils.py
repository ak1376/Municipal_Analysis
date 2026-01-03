# src/cv_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneOut, cross_val_predict


@dataclass(frozen=True)
class CVPredictions:
    """
    Generic container for cross-validated predictions.
    - probs: (n,) for binary prob-of-positive OR (n, K) for multiclass
    - preds: (n,) predicted class labels
    """
    probs: Optional[np.ndarray]
    preds: Optional[np.ndarray]


def loocv_predict(
    estimator: BaseEstimator,
    X,
    y,
    *,
    method: str = "predict_proba",
    threshold: float = 0.5,
    pos_index: int = 1,
    n_jobs: int = -1,
) -> CVPredictions:
    """
    General LOOCV prediction for sklearn estimators.

    method:
      - "predict_proba": returns probs + preds
      - "predict": returns preds only

    For binary classification with predict_proba: returns probs=(n,) of class pos_index,
    preds=(n,) using threshold.
    """
    y_arr = np.asarray(y).ravel()
    loo = LeaveOneOut()

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

        # Binary: extract positive-class probability
        if proba.ndim == 2 and proba.shape[1] >= 2:
            probs = proba[:, pos_index].astype(float)
            preds = (probs >= float(threshold)).astype(int)
            return CVPredictions(probs=probs, preds=preds)

        # Multiclass: return full proba matrix; preds = argmax
        probs = proba.astype(float)
        preds = np.argmax(probs, axis=1)
        return CVPredictions(probs=probs, preds=preds)

    if method == "predict":
        preds = cross_val_predict(
            estimator,
            X,
            y_arr,
            cv=loo,
            method="predict",
            n_jobs=n_jobs,
        )
        preds = np.asarray(preds).ravel()
        return CVPredictions(probs=None, preds=preds)

    raise ValueError("method must be 'predict_proba' or 'predict'")


def fit_final_model(estimator: BaseEstimator, X, y) -> BaseEstimator:
    """
    Fit a fresh clone on all data and return it.
    """
    est = clone(estimator)
    est.fit(X, np.asarray(y).ravel())
    return est
