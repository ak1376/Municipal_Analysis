# src/svm_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


@dataclass(frozen=True)
class SVMParams:
    svm_type: str               # "svc" or "linear_svc"
    kernel: str                 # SVC only: "linear","rbf","poly","sigmoid"
    C: float
    gamma: Any                  # SVC only: "scale","auto", or float
    degree: int                 # SVC only
    probability: bool           # SVC only; enables predict_proba (slower)
    max_iter: int               # LinearSVC only
    class_weight: Optional[str] # None or "balanced"
    random_state: int


def parse_gamma(gamma_str: str) -> Any:
    if gamma_str in {"scale", "auto"}:
        return gamma_str
    try:
        return float(gamma_str)
    except ValueError as e:
        raise ValueError("--gamma must be 'scale', 'auto', or a float") from e


def build_svm_model(params: SVMParams) -> Pipeline:
    if params.svm_type == "linear_svc":
        clf = LinearSVC(
            C=float(params.C),
            class_weight=params.class_weight,
            random_state=int(params.random_state),
            max_iter=int(params.max_iter),
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # SVC
    clf = SVC(
        kernel=str(params.kernel),
        C=float(params.C),
        gamma=params.gamma,
        degree=int(params.degree),
        class_weight=params.class_weight,
        probability=bool(params.probability),
        random_state=int(params.random_state),
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def get_scores(model: Pipeline, X) -> np.ndarray:
    """
    Returns a 1D score where larger => more likely class 1.
    Uses decision_function if available, else predict_proba[:,1] if enabled.
    """
    clf = model.named_steps["clf"]

    if hasattr(clf, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()

    if hasattr(clf, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1]).ravel()

    raise RuntimeError("SVM has neither decision_function nor predict_proba; cannot score.")


def scores_to_preds(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores).ravel() >= float(threshold)).astype(int)
