#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _add_repo_root_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]  # .../modeling
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _add_repo_root_to_syspath()

from scripts.pca_decomposition import run_pca_auto, pair_plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PCA, save scree plot, and generate pairplots of PCs.")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repo root directory (default: inferred from this script).",
    )
    p.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help="Path to features.csv (default: <repo-root>/data/features.csv).",
    )
    p.add_argument(
        "--target-csv",
        type=Path,
        default=None,
        help="Path to qualification_target.csv (default: <repo-root>/data/qualification_target.csv).",
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="Qualified Municipality",
        help="Target column name in target CSV.",
    )
    p.add_argument(
        "--variance-threshold",
        type=float,
        default=0.90,
        help="Cumulative explained variance threshold for choosing k.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo-root>/analysis/pca).",
    )
    p.add_argument(
        "--no-pairplot",
        action="store_true",
        help="Disable pairplot generation (useful if k is large).",
    )
    p.add_argument(
        "--pairplot-max-pcs",
        type=int,
        default=None,
        help="Optional: cap PCs in pairplot (e.g., 8). If not set, uses all PCs (can be huge).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figures.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    features_csv = (args.features_csv or (repo_root / "data" / "features.csv")).resolve()
    target_csv = (args.target_csv or (repo_root / "data" / "qualification_target.csv")).resolve()

    out_dir = (args.out_dir or (repo_root / "analysis" / "pca")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(features_csv)
    y = pd.read_csv(target_csv)

    if args.target_col not in y.columns:
        raise KeyError(f"Target column {args.target_col!r} not found in {target_csv}. Columns: {list(y.columns)}")
    if len(X) != len(y):
        raise ValueError(f"Row mismatch: features has {len(X)} rows but target has {len(y)} rows")

    X_pca, pca, scaler, scree_df, scree_fig = run_pca_auto(
        X,
        variance_threshold=args.variance_threshold,
    )

    # Save PCA space
    np.save(out_dir / "pca_space.npy", X_pca.to_numpy())
    X_pca.to_csv(out_dir / "pca_space.csv", index=False)
    scree_df.to_csv(out_dir / "scree_table.csv", index=False)

    # Save scree plot
    if scree_fig is not None:
        scree_fig.savefig(out_dir / "scree_plot.png", dpi=args.dpi)

    # Attach target for hue
    X_pca[args.target_col] = y[args.target_col].to_numpy()

    pc_cols = [c for c in X_pca.columns if c.startswith("PC")]
    if args.pairplot_max_pcs is not None:
        pc_cols = pc_cols[: args.pairplot_max_pcs]

    if not args.no_pairplot:
        fig = pair_plot(
            X_pca,
            columns=pc_cols,
            hue=args.target_col,
        )
        fig.savefig(out_dir / "pairplots.png", dpi=args.dpi)

    print(f"[OK] features:  {features_csv}  shape={X.shape}")
    print(f"[OK] target:    {target_csv}    shape={y.shape}")
    print(f"[OK] k (PCs):    {len([c for c in X_pca.columns if c.startswith('PC')])}")
    print(f"[OK] out_dir:    {out_dir}")


if __name__ == "__main__":
    main()
