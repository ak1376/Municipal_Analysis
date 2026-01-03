#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np


def _add_repo_root_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_add_repo_root_to_syspath()

from src.eval_utils import (
    read_design_matrix,
    read_target_csv,
    split_raw_csv,
    align_xy_dropna,
    coerce_X_to_numeric,
    ensure_out_dir,
)
from src.svm_helpers import SVMParams, parse_gamma
from src.svm_eval import SVMEvalConfig, run_svm_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVM driver (simple).")

    p.add_argument("--mode", choices=["space", "raw"], default="space")
    p.add_argument("--eval", choices=["holdout", "loocv"], default="holdout")

    # space-mode
    p.add_argument("--x", type=Path, default=Path("data/features.csv"))
    p.add_argument("--y", type=Path, default=Path("data/qualification_target.csv"))
    p.add_argument("--target-col", type=str, default="Qualified Municipality")

    # raw-mode
    p.add_argument("--raw-csv", type=Path, default=None)
    p.add_argument("--drop-cols", nargs="*", default=None)
    p.add_argument("--feature-cols", nargs="*", default=None)

    # holdout split
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=0)

    # outputs
    p.add_argument("--space-name", type=str, default="raw_features")
    p.add_argument("--out-root", type=Path, default=Path("analysis"))

    # svm params
    p.add_argument("--svm-type", choices=["svc", "linear_svc"], default="svc")
    p.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="linear")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", type=str, default="scale")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--probability", action="store_true", help="SVC only; slower, enables predict_proba")
    p.add_argument("--max-iter", type=int, default=10000, help="LinearSVC max_iter")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")

    # threshold on SCORE
    # (0.0 is the natural threshold for decision_function)
    p.add_argument("--threshold", type=float, default=0.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    gamma = parse_gamma(args.gamma)

    out_dir = ensure_out_dir(args.out_root / args.space_name / "svm" / args.eval)

    # load X, y
    if args.mode == "space":
        X_raw = read_design_matrix(args.x)
        y_raw = read_target_csv(args.y, target_col=args.target_col)
        x_path, y_path, raw_csv = str(args.x), str(args.y), None
    else:
        if args.raw_csv is None:
            raise ValueError("--raw-csv is required when --mode raw")
        X_raw, y_raw = split_raw_csv(
            args.raw_csv,
            target_col=args.target_col,
            drop_cols=args.drop_cols,
            feature_cols=args.feature_cols,
        )
        x_path, y_path, raw_csv = None, None, str(args.raw_csv)

    X, y, orig_row_index = align_xy_dropna(X_raw, y_raw)
    X = coerce_X_to_numeric(X)

    uniq = np.unique(y.values)
    if len(uniq) != 2 or set(uniq.tolist()) != {0, 1}:
        raise ValueError(f"Target must be binary with classes {{0,1}}; got {uniq.tolist()}.")

    svm_params = SVMParams(
        svm_type=args.svm_type,
        kernel=args.kernel,
        C=float(args.C),
        gamma=gamma,
        degree=int(args.degree),
        probability=bool(args.probability),
        max_iter=int(args.max_iter),
        class_weight=None if args.class_weight == "none" else "balanced",
        random_state=int(args.random_state),
    )

    cfg = SVMEvalConfig(
        eval=args.eval,
        out_dir=out_dir,
        threshold=float(args.threshold),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    run_svm_eval(
        X=X,
        y=y,
        orig_row_index=orig_row_index,
        mode=args.mode,
        space_name=args.space_name,
        x_path=x_path,
        y_path=y_path,
        raw_csv=raw_csv,
        target_col=args.target_col,
        svm_params=svm_params,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
