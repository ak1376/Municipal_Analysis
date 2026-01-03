# src/logistic_regression_eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cv_utils import loocv_predict, fit_final_model
from src.eval_utils import (
    ensure_out_dir,
    write_json,
    compute_test_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_roc,
    build_predictions_all_df,
)

from src.logistic_regression_helpers import (
    make_sklearn_logit_pipeline,
    save_sklearn_model,
    save_sklearn_logit_coefficients_csv,
    build_predictions_df_holdout,
    build_predictions_df_loocv,
)


@dataclass(frozen=True)
class EvalConfig:
    eval: str                      # "holdout" or "loocv"
    out_dir: Path
    threshold: float
    add_intercept: bool
    test_size: float
    random_state: int


def _write_metrics_txt(out_dir: Path, header_lines: list[str], metrics: Dict[str, float], prefix: str) -> None:
    (out_dir / f"metrics_{prefix}.txt").write_text(
        "\n".join(
            header_lines
            + [
                "",
                f"{prefix.upper()}_auc: {metrics['auc']:.6f}",
                f"{prefix.upper()}_accuracy: {metrics['accuracy']:.6f}",
                f"{prefix.upper()}_balanced_accuracy: {metrics['balanced_accuracy']:.6f}",
            ]
        )
        + "\n"
    )


def run_logistic_regression_eval(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    orig_row_index: np.ndarray,
    feature_cols: list[str],
    mode: str,
    space_name: str,
    x_path: Optional[str],
    y_path: Optional[str],
    raw_csv: Optional[str],
    target_col: str,
    cfg: EvalConfig,
) -> None:
    """
    ALL evaluation logic for Logistic Regression lives here.
    Produces:
      - predictions_all_in_original_order.csv (holdout: in/out of sample; loocv: out-of-sample for all rows)
      - confusion matrix + roc
      - run_config.json
      - final_model.joblib (trained on ALL rows)
      - final_coefficients.csv
    """
    out_dir = ensure_out_dir(cfg.out_dir)

    # sklearn pipeline
    pipe = make_sklearn_logit_pipeline(fit_intercept=cfg.add_intercept)

    n = len(y)
    threshold = float(cfg.threshold)

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

        pipe.fit(X_train, y_train.values.ravel())

        test_probs = np.asarray(pipe.predict_proba(X_test)[:, 1]).ravel()
        metrics = compute_test_metrics(y_true=y_test.values, probs=test_probs, threshold=threshold)

        # plots on TEST only
        plot_and_save_confusion_matrix(
            y_true=y_test.values,
            probs=test_probs,
            threshold=threshold,
            out_path=out_dir / "confusion_matrix_test.png",
            title="Logit Confusion Matrix (holdout test)",
        )
        plot_and_save_roc(
            y_true=y_test.values,
            probs=test_probs,
            out_path=out_dir / "roc_curve_test.png",
            title="Logit ROC (holdout test)",
        )

        # predictions for ALL datapoints using the holdout-trained model
        all_probs = np.asarray(pipe.predict_proba(X)[:, 1]).ravel()
        pred_all = build_predictions_df_holdout(
            y_true=y,
            probs_all=all_probs,
            threshold=threshold,
            orig_row_index=orig_row_index,
            test_idx=test_idx,
        )
        pred_all.to_csv(out_dir / "predictions_all_in_original_order.csv", index=False)

        header = [
            "eval: holdout",
            f"mode: {mode}",
            f"space_name: {space_name}",
            "",
            f"n_total_after_dropna: {n}",
            f"n_train: {len(train_idx)}",
            f"n_test: {len(test_idx)}",
            "",
            f"add_intercept: {cfg.add_intercept}",
            f"threshold: {threshold}",
        ]
        _write_metrics_txt(out_dir, header, metrics, prefix="test")

        # Fit FINAL model on all data (separate from holdout fit)
        final_model = fit_final_model(pipe, X, y)
        save_sklearn_model(out_dir / "final_model.joblib", final_model)
        save_sklearn_logit_coefficients_csv(
            model=final_model,
            feature_cols=feature_cols,
            out_path=out_dir / "final_coefficients.csv",
        )

        write_json(
            out_dir / "run_config.json",
            {
                "model": "logistic_regression_sklearn",
                "eval": "holdout",
                "mode": mode,
                "space_name": space_name,
                "x_path": x_path if mode == "space" else None,
                "y_path": y_path if mode == "space" else None,
                "raw_csv": raw_csv if mode == "raw" else None,
                "target_col": target_col,
                "test_size": float(cfg.test_size),
                "random_state": int(cfg.random_state),
                "add_intercept": bool(cfg.add_intercept),
                "threshold": float(threshold),
                "notes": [
                    "predictions_all_in_original_order.csv uses the holdout-trained model: train rows are in-sample.",
                    "final_model.joblib is trained on ALL rows (post-dropna).",
                    "final_coefficients.csv is in StandardScaler feature space.",
                ],
            },
        )

        print(f"[OK] Wrote outputs to: {out_dir}")
        print(
            f"[OK] HOLDOUT TEST AUC={metrics['auc']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"thr={threshold:.3g}"
        )
        return

    if cfg.eval == "loocv":
        cvpred = loocv_predict(
            pipe,
            X,
            y,
            method="predict_proba",
            threshold=threshold,
            n_jobs=-1,
        )
        if cvpred.probs is None:
            raise RuntimeError("Expected LOOCV predict_proba to return probs for LogisticRegression.")

        probs_all = np.asarray(cvpred.probs).ravel()
        metrics = compute_test_metrics(y_true=y.values, probs=probs_all, threshold=threshold)

        # plots on LOOCV predictions across all rows
        plot_and_save_confusion_matrix(
            y_true=y.values,
            probs=probs_all,
            threshold=threshold,
            out_path=out_dir / "confusion_matrix_loocv.png",
            title="Logit Confusion Matrix (LOOCV)",
        )
        plot_and_save_roc(
            y_true=y.values,
            probs=probs_all,
            out_path=out_dir / "roc_curve_loocv.png",
            title="Logit ROC (LOOCV)",
        )

        pred_all = build_predictions_df_loocv(
            y_true=y,
            probs_all=probs_all,
            threshold=threshold,
            orig_row_index=orig_row_index,
        )
        pred_all.to_csv(out_dir / "predictions_all_in_original_order.csv", index=False)

        header = [
            "eval: loocv",
            f"mode: {mode}",
            f"space_name: {space_name}",
            "",
            f"n_total_after_dropna: {n}",
            "",
            f"add_intercept: {cfg.add_intercept}",
            f"threshold: {threshold}",
        ]
        _write_metrics_txt(out_dir, header, metrics, prefix="loocv")

        # Fit FINAL model on all data
        final_model = fit_final_model(pipe, X, y)
        save_sklearn_model(out_dir / "final_model.joblib", final_model)
        save_sklearn_logit_coefficients_csv(
            model=final_model,
            feature_cols=feature_cols,
            out_path=out_dir / "final_coefficients.csv",
        )

        write_json(
            out_dir / "run_config.json",
            {
                "model": "logistic_regression_sklearn",
                "eval": "loocv",
                "mode": mode,
                "space_name": space_name,
                "x_path": x_path if mode == "space" else None,
                "y_path": y_path if mode == "space" else None,
                "raw_csv": raw_csv if mode == "raw" else None,
                "target_col": target_col,
                "add_intercept": bool(cfg.add_intercept),
                "threshold": float(threshold),
                "notes": [
                    "predictions_all_in_original_order.csv contains LOOCV out-of-sample predictions for every row.",
                    "final_model.joblib is trained on ALL rows (post-dropna).",
                    "final_coefficients.csv is in StandardScaler feature space.",
                ],
            },
        )

        print(f"[OK] Wrote outputs to: {out_dir}")
        print(
            f"[OK] LOOCV AUC={metrics['auc']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"thr={threshold:.3g}"
        )
        return

    raise ValueError(f"Unknown eval: {cfg.eval!r}. Expected 'holdout' or 'loocv'.")
