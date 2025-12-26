#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys
import numpy as np
import pandas as pd

def _add_repo_root_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]  # .../modeling
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _add_repo_root_to_syspath()

from src.pls_decomposition import run_pls_auto, pair_plot


def _read_features(path: Path) -> pd.DataFrame:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty features file: {path}")
    return df


def _read_targets(path: Path, target_col: str) -> pd.Series:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {path}")
    return df[target_col]


def _to_binary_target(y_raw: pd.Series) -> pd.Series:
    """
    Map YES/NO (or true/false, 1/0) -> {0,1} for PLS y.
    """
    if y_raw.dtype == object:
        s = y_raw.astype(str).str.strip().str.upper()
        mapping = {"YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
        y = s.map(mapping)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    if y.isna().any():
        bad = y_raw[y.isna()].head(10).tolist()
        raise ValueError(f"Target has values that can't be coerced to 0/1. Examples: {bad}")

    y = y.astype(int)
    uniq = sorted(y.unique().tolist())
    if uniq not in ([0, 1], [0], [1]):
        raise ValueError(f"Target must be binary 0/1 after mapping; got {uniq}")
    return y


def _align_xy(X: pd.DataFrame, y: pd.Series, *, drop_na: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch: X has {len(X)} rows, y has {len(y)} rows. "
            "Make sure they were saved from the same filtered rows and in the same order."
        )

    if not drop_na:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    mask = ~(X.isna().any(axis=1) | y.isna())
    X2 = X.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    return X2, y2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PLS decomposition (auto-select components) + save latent space, scree plot, and pairplot."
    )

    p.add_argument("--x", type=Path, default=Path("data/features.csv"),
                   help="Input features CSV (default: data/features.csv)")
    p.add_argument("--y", type=Path, default=Path("data/qualification_target.csv"),
                   help="Input target CSV (default: data/qualification_target.csv)")
    p.add_argument("--target-col", type=str, default="Qualified Municipality",
                   help="Target column name in --y CSV (default: 'Qualified Municipality')")

    p.add_argument("--variance-threshold", type=float, default=0.90,
                   help="Cumulative variance threshold for auto component selection (default: 0.90)")
    p.add_argument("--scale", action="store_true", default=True,
                   help="Scale X before PLS (default: on)")
    p.add_argument("--no-scale", action="store_false", dest="scale",
                   help="Disable scaling before PLS")

    p.add_argument("--plot-scree", action="store_true", default=True,
                   help="Save scree plot (default: on)")
    p.add_argument("--no-scree", action="store_false", dest="plot_scree",
                   help="Disable scree plot saving")

    p.add_argument("--pairplot", action="store_true", default=True,
                   help="Save pair plot of all PLS dims (default: on)")
    p.add_argument("--no-pairplot", action="store_false", dest="pairplot",
                   help="Disable pair plot saving")

    p.add_argument("--space-name", type=str, default="pls",
                   help="Output folder name under analysis/ (default: pls)")
    p.add_argument("--out-root", type=Path, default=Path("analysis"),
                   help="Root output directory (default: analysis)")

    p.add_argument("--pairplot-dpi", type=int, default=200,
                   help="DPI for saved pairplot PNG (default: 200)")
    p.add_argument("--scree-dpi", type=int, default=200,
                   help="DPI for saved scree PNG (default: 200)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = (args.out_root / args.space_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X = _read_features(args.x)
    y_raw = _read_targets(args.y, args.target_col)
    y_num = _to_binary_target(y_raw)

    # align + drop NAs once
    X, y_num = _align_xy(X, y_num, drop_na=True)
    # keep original YES/NO labels aligned too (for hue)
    y_lbl = y_raw.loc[y_num.index] if hasattr(y_num, "index") else y_raw.iloc[: len(y_num)]
    y_lbl = y_lbl.reset_index(drop=True)

    X_pls, pls, scaler, scree_df, scree_fig = run_pls_auto(
        X,
        y=y_num,
        variance_threshold=args.variance_threshold,
        scale=args.scale,
        plot_scree=args.plot_scree,
    )

    # Save latent space
    np.save(out_dir / "pls_space.npy", X_pls.to_numpy())
    X_pls.to_csv(out_dir / "pls_space.csv", index=False)

    # Save scree table
    scree_df.to_csv(out_dir / "scree_table.csv", index=False)

    # Save scree plot
    if args.plot_scree and scree_fig is not None:
        scree_fig.savefig(out_dir / "scree_plot.png", dpi=args.scree_dpi)

    # Pairplot of ALL PLS dims (warning: huge if k is large)
    if args.pairplot:
        pls_cols = [c for c in X_pls.columns if c.startswith("PLS")]
        plot_df = X_pls.copy()
        plot_df["Qualified Municipality"] = y_lbl.astype(str).to_numpy()

        fig = pair_plot(
            plot_df,
            columns=pls_cols,
            hue="Qualified Municipality",
        )
        fig.savefig(out_dir / "pairplots.png", dpi=args.pairplot_dpi)

    print(f"[OK] Wrote PLS outputs to: {out_dir}")
    print(f"[OK] X_pls shape: {X_pls.shape} | n={len(X)}")


if __name__ == "__main__":
    main()
