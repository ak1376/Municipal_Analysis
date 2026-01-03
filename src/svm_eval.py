# src/svm_eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.base import clone

from src.svm_helpers import SVMParams, build_svm_model, get_scores, scores_to_preds
from src.eval_utils import (
    ensure_out_dir,
    write_json,
    compute_test_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_roc,
)


@dataclass(frozen=True)
class SVMEvalConfig:
    eval: str          # "holdout" | "loocv"
    out_dir: Path
    threshold: float   # threshold applied to *scores*
    test_size: float   # holdout only
    random_state: int  # holdout only


def _save_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _write_metrics(out_dir: Path, metrics: Dict[str, Any]) -> None:
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "metrics.txt").write_text("\n".join(f"{k}: {v}" for k, v in metrics.items()) + "\n")


def _loocv_scores(base_model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    loo = LeaveOneOut()
    n = len(y)
    scores_all = np.empty(n, dtype=float)

    Xv = X.values
    yv = y.values.astype(int)

    for tr_idx, te_idx in loo.split(Xv, yv):
        m = clone(base_model)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        s = get_scores(m, X.iloc[te_idx])
        scores_all[te_idx[0]] = float(np.asarray(s).ravel()[0])

    return scores_all


def run_svm_eval(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    orig_row_index: np.ndarray,
    mode: str,
    space_name: str,
    x_path: Optional[str],
    y_path: Optional[str],
    raw_csv: Optional[str],
    target_col: str,
    svm_params: SVMParams,
    cfg: SVMEvalConfig,
) -> None:
    out_dir = ensure_out_dir(cfg.out_dir)
    n = len(y)
    thr = float(cfg.threshold)

    base_model = build_svm_model(svm_params)

    if cfg.eval == "holdout":
        idx = np.arange(n, dtype=int)
        train_idx, test_idx = train_test_split(
            idx,
            test_size=float(cfg.test_size),
            random_state=int(cfg.random_state),
            shuffle=True,
            stratify=y.values,
        )

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = build_svm_model(svm_params)
        model.fit(X_train, y_train)

        test_scores = get_scores(model, X_test)
        metrics = compute_test_metrics(y_true=y_test.values, probs=test_scores, threshold=thr)

        plot_and_save_confusion_matrix(
            y_true=y_test.values,
            probs=test_scores,
            threshold=thr,
            out_path=out_dir / "confusion_matrix_test.png",
            title="SVM Confusion Matrix (holdout test)",
        )
        plot_and_save_roc(
            y_true=y_test.values,
            probs=test_scores,
            out_path=out_dir / "roc_curve_test.png",
            title="SVM ROC (holdout test)",
        )

        # predictions for all rows using the train-fit model
        all_scores = get_scores(model, X)
        all_pred = scores_to_preds(all_scores, thr)

        split = np.array(["train"] * n, dtype=object)
        split[np.asarray(test_idx, dtype=int)] = "test"

        pred_all = pd.DataFrame(
            {
                "row_index": np.arange(n, dtype=int),
                "orig_row_index": orig_row_index.astype(int),
                "split": split,
                "y_true": y.values.astype(int),
                "score": np.asarray(all_scores).astype(float),
                "y_pred": np.asarray(all_pred).astype(int),
            }
        ).sort_values("row_index", kind="mergesort")
        pred_all.to_csv(out_dir / "predictions_all_in_original_order.csv", index=False)

        _write_metrics(
            out_dir,
            {
                "eval": "holdout",
                "auc": float(metrics["auc"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "n": int(n),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "threshold": float(thr),
            },
        )

        # final model on all data (optional artifact; keeps consistent with your other models)
        final_model = build_svm_model(svm_params)
        final_model.fit(X, y)
        _save_model(out_dir / "final_model.joblib", final_model)

        write_json(
            out_dir / "run_config.json",
            {
                "model": "svm",
                "eval": "holdout",
                "mode": mode,
                "space_name": space_name,
                "x_path": x_path if mode == "space" else None,
                "y_path": y_path if mode == "space" else None,
                "raw_csv": raw_csv if mode == "raw" else None,
                "target_col": target_col,
                "test_size": float(cfg.test_size),
                "random_state": int(cfg.random_state),
                "threshold": float(thr),
                "svm_params": {
                    "svm_type": svm_params.svm_type,
                    "kernel": svm_params.kernel,
                    "C": float(svm_params.C),
                    "gamma": svm_params.gamma if isinstance(svm_params.gamma, str) else float(svm_params.gamma),
                    "degree": int(svm_params.degree),
                    "probability": bool(svm_params.probability),
                    "max_iter": int(svm_params.max_iter),
                    "class_weight": svm_params.class_weight,
                    "random_state": int(svm_params.random_state),
                },
                "notes": [
                    "score is decision_function if available, otherwise predict_proba[:,1].",
                    "predictions_all_in_original_order.csv uses the TRAIN-fit holdout model.",
                    "final_model.joblib is trained on ALL rows (post-dropna).",
                ],
            },
        )

        print(f"[OK] Wrote outputs to: {out_dir}")
        print(
            f"[OK] HOLDOUT TEST AUC={metrics['auc']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"thr={thr:.3g}"
        )
        return

    if cfg.eval == "loocv":
        scores_all = _loocv_scores(base_model, X, y)
        metrics = compute_test_metrics(y_true=y.values, probs=scores_all, threshold=thr)

        plot_and_save_confusion_matrix(
            y_true=y.values,
            probs=scores_all,
            threshold=thr,
            out_path=out_dir / "confusion_matrix_loocv.png",
            title="SVM Confusion Matrix (LOOCV)",
        )
        plot_and_save_roc(
            y_true=y.values,
            probs=scores_all,
            out_path=out_dir / "roc_curve_loocv.png",
            title="SVM ROC (LOOCV)",
        )

        y_pred = scores_to_preds(scores_all, thr)

        pred_all = pd.DataFrame(
            {
                "row_index": np.arange(n, dtype=int),
                "orig_row_index": orig_row_index.astype(int),
                "split": np.array(["loocv"] * n, dtype=object),
                "y_true": y.values.astype(int),
                "score": np.asarray(scores_all).astype(float),
                "y_pred": np.asarray(y_pred).astype(int),
            }
        ).sort_values("row_index", kind="mergesort")
        pred_all.to_csv(out_dir / "predictions_all_in_original_order.csv", index=False)

        _write_metrics(
            out_dir,
            {
                "eval": "loocv",
                "auc": float(metrics["auc"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "n": int(n),
                "threshold": float(thr),
            },
        )

        final_model = build_svm_model(svm_params)
        final_model.fit(X, y)
        _save_model(out_dir / "final_model.joblib", final_model)

        write_json(
            out_dir / "run_config.json",
            {
                "model": "svm",
                "eval": "loocv",
                "mode": mode,
                "space_name": space_name,
                "x_path": x_path if mode == "space" else None,
                "y_path": y_path if mode == "space" else None,
                "raw_csv": raw_csv if mode == "raw" else None,
                "target_col": target_col,
                "threshold": float(thr),
                "svm_params": {
                    "svm_type": svm_params.svm_type,
                    "kernel": svm_params.kernel,
                    "C": float(svm_params.C),
                    "gamma": svm_params.gamma if isinstance(svm_params.gamma, str) else float(svm_params.gamma),
                    "degree": int(svm_params.degree),
                    "probability": bool(svm_params.probability),
                    "max_iter": int(svm_params.max_iter),
                    "class_weight": svm_params.class_weight,
                    "random_state": int(svm_params.random_state),
                },
                "notes": [
                    "predictions_all_in_original_order.csv contains LOOCV out-of-sample predictions for every row.",
                    "final_model.joblib is trained on ALL rows (post-dropna).",
                ],
            },
        )

        print(f"[OK] Wrote outputs to: {out_dir}")
        print(
            f"[OK] LOOCV AUC={metrics['auc']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"thr={thr:.3g}"
        )
        return

    raise ValueError(f"Unknown eval: {cfg.eval!r} (expected 'holdout' or 'loocv').")
