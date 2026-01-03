#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_repo_root_to_syspath() -> Path:
    """
    Ensure repo root (parent of driver_scripts/) is importable so `import scripts.*` works.
    Returns resolved repo root Path.
    """
    repo_root = Path(__file__).resolve().parents[1]  # .../modeling
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _add_repo_root_to_syspath()

# Now these imports work when run as: python driver_scripts/process.py
from src.create_map import plot_nj_qualification_map
from src.data_processing import read_tsm_excel, build_xy


TARGET_COL_DEFAULT = "Qualified Municipality"

FEATURE_COLS_DEFAULT = [
    "2015 Median Muni Household Income",
    "2023 Median Muni Household Income",
    "Muni Median Household Income Change (2015-2023)",
    "2015* Median Household Income for Housing Region",
    "2023 Median Household Income for Housing Region",
    "Region Median Household Income Change (2015-2023)",
    "Change in Household Income Relative to Region (2015-2023)",
    "Muni Poverty Rate 2023",
    "NJ Poverty Rate 2023",
    "Muni v. NJ Poverty Rate 2023",
    "Muni Unemployment Rate 2023",
    "NJ Unemployment Rate 2023",
    "Muni Labor Participation Rate 2023",
    "NJ Labor Participation Rate 2023",
    "Percent of Muni Homes Built Before 1960",
    "Percent of NJ Homes Built Before 1960",
    "Muni Percent of Vacant Units",
    "NJ Percent of Vacant Units",
    "Median Property Value 2015",
    "Median Property Value 2023",
    "Median Property Value Change (2015-2023)",
    "2018 Commercial Value",
    "2023 Commercial Value",
    "Change in Commercial Ratables 2023",
    "2024 Fiscal Stress Index",
    "2024 Fiscal Stress Rank",
    "2024 Fiscal Stress Score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process TSM.xlsx into features/target CSVs and (optionally) plot NJ qualification map."
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repo root directory (default: inferred from this script).",
    )
    p.add_argument(
        "--input-xlsx",
        type=Path,
        default=None,
        help="Path to TSM.xlsx (default: <repo-root>/data/TSM.xlsx).",
    )
    p.add_argument(
        "--header-row",
        type=int,
        default=2,
        help="Header row index for Excel reading (passed to read_tsm_excel).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write outputs (default: <repo-root>/data).",
    )
    p.add_argument(
        "--features-out",
        type=Path,
        default=None,
        help="Output CSV path for features (default: <output-dir>/features.csv).",
    )
    p.add_argument(
        "--target-out",
        type=Path,
        default=None,
        help="Output CSV path for target (default: <output-dir>/qualification_target.csv).",
    )
    p.add_argument(
        "--target-col",
        type=str,
        default=TARGET_COL_DEFAULT,
        help=f"Target column name (default: {TARGET_COL_DEFAULT!r}).",
    )
    p.add_argument(
        "--no-map",
        action="store_true",
        help="Disable plotting NJ qualification map.",
    )
    p.add_argument(
        "--map-out",
        type=Path,
        default=None,
        help="Optional: save map to this path instead of showing (depends on your plot function behavior).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    input_xlsx = (args.input_xlsx or (repo_root / "data" / "TSM.xlsx")).resolve()

    output_dir = (args.output_dir or (repo_root / "data")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_out = (args.features_out or (output_dir / "features.csv")).resolve()
    target_out = (args.target_out or (output_dir / "qualification_target.csv")).resolve()

    df = read_tsm_excel(input_xlsx, header_row=args.header_row)

    if not args.no_map:
        fig = plot_nj_qualification_map(df)
        # Only save if your function returns a matplotlib Figure; otherwise ignore.
        if args.map_out is not None and fig is not None:
            args.map_out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.map_out, dpi=200)

    X, y = build_xy(df, feature_cols=FEATURE_COLS_DEFAULT, target_col=args.target_col)

    X.to_csv(features_out, index=False)
    y.to_csv(target_out, index=False)

    print(f"[OK] input:      {input_xlsx}")
    print(f"[OK] features:   {features_out}  shape={X.shape}")
    print(f"[OK] target:     {target_out}    shape={y.shape}")


if __name__ == "__main__":
    main()
