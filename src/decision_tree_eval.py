# src/decision_tree_eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from src.cv_utils import loocv_predict, fit_final_model
from src.decision_tree_helpers import (
    TreeParams,
    build_decision_tree,
    probs_to_preds,
    save_feature_importances,
    plot_and_save_tree_readable,
    save_tree_rules_text,
)
from src.eval_utils import (
    ensure_out_dir,
    write_json,
    compute_test_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_roc,
    build_predictions_all_df,
)



@dataclass(frozen=True)
class DTEvalConfig:
    eval: str                 # "holdout" | "loocv" | "final"
    out_dir: Path
    threshold: float

    # holdout only
    test_size: float
    random_state: int

    # viz
    max_depth_plot: int
    tree_fontsize: int
    write_rules: bool


def _save_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _write_metrics(out_dir: Path, metrics: Dict[str, Any]) -> None:
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "metrics.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in metrics.items()) + "\n"
    )


def run_decision_tree_eval(
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
    tree_params: TreeParams,
    cfg: DTEvalConfig,
) -> None:
    out_dir = ensure_out_dir(cfg.out_dir)
    n = len(y)
    thr = float(cfg.threshold)

    # ============================================================
    # HOLDOUT
    # ============================================================
    if cfg.eval == "holdout":
        idx = np.arange(n)
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y.values,
        )

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        clf = build_decision_tree(tree_params)
        clf.fit(X_tr, y_tr)

        probs_te = clf.predict_proba(X_te)[:, 1]
        metrics = compute_test_metrics(
            y_true=y_te.values,
            probs=probs_te,
            threshold=thr,
        )



        plot_and_save_confusion_matrix(
            y_true=y_te.values,
            probs=probs_te,
            threshold=thr,
            out_path=out_dir / "confusion_matrix_test.png",
            title="Decision Tree (holdout)",
        )
        plot_and_save_roc(
            y_true=y_te.values,
            probs=probs_te,
            out_path=out_dir / "roc_curve_test.png",
            title="Decision Tree ROC (holdout)",
        )

        all_probs = clf.predict_proba(X)[:, 1]
        pred_all = build_predictions_all_df(
            y_true=y,
            probs_all=all_probs,
            threshold=thr,
            orig_row_index=orig_row_index,
            test_idx=te_idx,
        )
        pred_all.to_csv(out_dir / "predictions_all_in_original_order.csv", index=False)

        _write_metrics(
            out_dir,
            {
                "eval": "holdout",
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n": n,
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
                "threshold": thr,
            },
        )

        return

    # ============================================================
    # LOOCV  (THIS is what you compare across models)
    # ============================================================
    if cfg.eval == "loocv":
        base = build_decision_tree(tree_params)

        cv = loocv_predict(
            base,
            X,
            y,
            method="predict_proba",
            n_jobs=-1,
        )

        probs = np.asarray(cv.probs).ravel()
        metrics = compute_test_metrics(
            y_true=y.values,
            probs=probs,
            threshold=thr,
        )


        plot_and_save_confusion_matrix(
            y_true=y.values,
            probs=probs,
            threshold=thr,
            out_path=out_dir / "confusion_matrix_loocv.png",
            title="Decision Tree (LOOCV)",
        )
        plot_and_save_roc(
            y_true=y.values,
            probs=probs,
            out_path=out_dir / "roc_curve_loocv.png",
            title="Decision Tree ROC (LOOCV)",
        )

        preds = probs_to_preds(probs, thr)
        pd.DataFrame(
            {
                "row_index": np.arange(n),
                "orig_row_index": orig_row_index,
                "split": "loocv",
                "y_true": y.values,
                "prob": probs,
                "y_pred": preds,
            }
        ).to_csv(out_dir / "predictions_all_in_original_order.csv", index=False)

        _write_metrics(
            out_dir,
            {
                "eval": "loocv",
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n": n,
                "threshold": thr,
            },
        )

        return

    # ============================================================
    # FINAL  (fit once, after model selection)
    # ============================================================
    if cfg.eval == "final":
        clf = fit_final_model(build_decision_tree(tree_params), X, y)
        _save_model(out_dir / "final_model.joblib", clf)

        probs = clf.predict_proba(X)[:, 1]
        preds = probs_to_preds(probs, thr)

        pd.DataFrame(
            {
                "row_index": np.arange(n),
                "orig_row_index": orig_row_index,
                "split": "fit_all",
                "y_true": y.values,
                "prob": probs,
                "y_pred": preds,
            }
        ).to_csv(out_dir / "predictions_fit_all.csv", index=False)

        save_feature_importances(clf, feature_cols, out_dir / "feature_importances.csv")
        plot_and_save_tree_readable(
            clf,
            feature_names=feature_cols,
            out_path=out_dir / "tree_plot.png",
            max_depth_plot=cfg.max_depth_plot,
            fontsize=cfg.tree_fontsize,
        )
        if cfg.write_rules:
            save_tree_rules_text(clf, feature_cols, out_dir / "tree_rules.txt")

        write_json(
            out_dir / "run_config.json",
            {
                "model": "decision_tree",
                "eval": "final",
                "threshold": thr,
                "tree_params": tree_params.__dict__,
            },
        )

        return

    raise ValueError(cfg.eval)
