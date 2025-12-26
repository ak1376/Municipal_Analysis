#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: name standardization
# -----------------------------
_SUFFIXES = ["Township", "Borough", "City", "Town", "Village"]


def canonical_muni_name(s: str) -> str:
    """
    Make municipality names comparable across your table and Census shapefiles.
    - lowercases
    - removes punctuation
    - removes common NJ municipal suffixes
    - normalizes a few special cases seen in NJ lists
    """
    if s is None:
        return ""
    s = str(s).strip().lower()

    # normalize punctuation/hyphens
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"[^\w\s\-]", "", s)  # keep letters/numbers/underscore/space/hyphen
    s = re.sub(r"\s+", " ", s).strip()

    # common special-case normalizations
    s = s.replace("hillsborough township", "hillsborough")
    s = s.replace("boonton town", "boonton")
    s = s.replace("clinton town", "clinton")
    s = s.replace("dover town", "dover")
    s = s.replace("guttenberg town", "guttenberg")
    s = s.replace("harrison town", "harrison")
    s = s.replace("kearny town", "kearny")
    s = s.replace("secaucus town", "secaucus")
    s = s.replace("west new york town", "west new york")

    # remove "city of X township" style (e.g., "City of Orange Township")
    s = s.replace("city of ", "")

    # strip suffixes at end
    for suf in _SUFFIXES:
        suf_l = suf.lower()
        s = re.sub(rf"\s+{suf_l}$", "", s).strip()

    # collapse remaining whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_yes_no(x) -> Optional[int]:
    """
    Return 1 for YES/True, 0 for NO/False, None if unknown.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"YES", "Y", "TRUE", "1"}:
        return 1
    if s in {"NO", "N", "FALSE", "0"}:
        return 0
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -----------------------------
# IO
# -----------------------------
def read_raw_table(path: Path, *, sheet: Optional[str], excel_header: int) -> pd.DataFrame:
    """
    Reads CSV or Excel and returns a single DataFrame (never a dict).
    - If Excel and sheet is None: uses first sheet (0).
    - If Excel and sheet is provided: can be name or integer string.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
        return normalize_columns(df)

    if suf in {".xlsx", ".xls"}:
        sheet_name: Any
        if sheet is None:
            sheet_name = 0
        else:
            # allow "0", "1", ... as well as actual sheet names
            try:
                sheet_name = int(sheet)
            except ValueError:
                sheet_name = sheet

        df = pd.read_excel(path, sheet_name=sheet_name, header=int(excel_header))
        # If user accidentally passed something that yields dict, pick the first sheet deterministically.
        if isinstance(df, dict):
            first_key = list(df.keys())[0]
            df = df[first_key]
        return normalize_columns(df)

    raise ValueError(f"Unsupported raw file type: {path.suffix} (expected .csv/.xlsx)")


def read_predictions(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    # required-ish columns
    for c in ["y_true", "y_pred"]:
        if c not in df.columns:
            raise KeyError(f"pred-csv missing required column {c!r}: {path}")

    # choose join index column
    if "orig_row_index" in df.columns:
        join_col = "orig_row_index"
    elif "row_index" in df.columns:
        join_col = "row_index"
    else:
        raise KeyError("pred-csv must contain 'orig_row_index' or 'row_index'.")

    df = df.copy()
    df[join_col] = pd.to_numeric(df[join_col], errors="raise").astype(int)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="raise").astype(int)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="raise").astype(int)
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    df["_join_col"] = join_col
    return df


# -----------------------------
# Geometries
# -----------------------------
def load_nj_muni_geometries() -> gpd.GeoDataFrame:
    """
    Load NJ municipal geometries from TIGER/Line.
    Internet required.
    """
    cousub_url = "https://www2.census.gov/geo/tiger/TIGER2023/COUSUB/tl_2023_34_cousub.zip"
    place_url = "https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_34_place.zip"

    g_cousub = gpd.read_file(cousub_url)[["NAME", "geometry"]].copy()
    g_cousub["muni_key"] = g_cousub["NAME"].apply(canonical_muni_name)

    g_place = gpd.read_file(place_url)[["NAME", "geometry"]].copy()
    g_place["muni_key"] = g_place["NAME"].apply(canonical_muni_name)

    g_all = pd.concat([g_cousub, g_place], ignore_index=True)
    g_all = g_all.drop_duplicates(subset=["muni_key"], keep="first")
    g_all = gpd.GeoDataFrame(g_all, geometry="geometry", crs=g_cousub.crs)
    return g_all


# -----------------------------
# Plotting
# -----------------------------
def plot_correctness_dots(
    ax: plt.Axes,
    g_pts: gpd.GeoDataFrame,
    *,
    title: str,
) -> None:
    # 1=correct, 0=incorrect
    correct = g_pts[g_pts["correct"] == 1]
    incorrect = g_pts[g_pts["correct"] == 0]

    # Correct -> green, Incorrect -> red
    if len(correct) > 0:
        correct.plot(ax=ax, color="green", markersize=18, label="Correct")
    if len(incorrect) > 0:
        incorrect.plot(ax=ax, color="red", markersize=18, label="Incorrect")

    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="lower left")


def plot_percent_correct_polygons(
    fig: plt.Figure,
    ax: plt.Axes,
    g_poly: gpd.GeoDataFrame,
    *,
    title: str,
) -> None:
    # continuous [0,1]
    # leave municipalities with no data as light gray
    base = g_poly.copy()
    has = base["pct_correct"].notna()

    if (~has).any():
        base.loc[~has].plot(ax=ax, color="lightgray", linewidth=0.2, edgecolor="white")

    mappable_ax = base.loc[has].plot(
        ax=ax,
        column="pct_correct",
        legend=True,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidth=0.2,
        edgecolor="white",
        legend_kwds={"label": "Percent correct", "shrink": 0.7},
    )

    ax.set_title(title)
    ax.set_axis_off()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot prediction correctness on an NJ municipality map.")

    p.add_argument("--raw", type=Path, required=True, help="Raw table (.csv or .xlsx).")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name or index (default: first sheet).")
    p.add_argument("--excel-header", type=int, default=0, help="Header row (0-indexed) for Excel. e.g. 2.")
    p.add_argument("--pred-csv", type=Path, required=True, help="predictions_all.csv from your model driver.")

    p.add_argument("--muni-col", type=str, required=True, help="Municipality column in raw table.")
    p.add_argument("--target-col", type=str, required=True, help="Target/label column in raw table (YES/NO).")

    p.add_argument("--mode", choices=["correctness", "percent_correct"], default="correctness")
    p.add_argument("--dot", action="store_true", help="Plot dots (representative points).")
    p.add_argument("--polygon", action="store_true", help="Plot polygons (fills municipalities).")

    p.add_argument("--model-label", type=str, default="", help="Used in title, e.g. 'svm' or 'logreg'.")
    p.add_argument("--out", type=Path, required=True)

    args = p.parse_args()

    # Defaults:
    # - correctness: dots by default
    # - percent_correct: polygons by default
    if not args.dot and not args.polygon:
        if args.mode == "correctness":
            args.dot = True
        else:
            args.polygon = True

    raw = read_raw_table(args.raw, sheet=args.sheet, excel_header=args.excel_header)

    # Validate columns
    for c in [args.muni_col, args.target_col]:
        if c not in raw.columns:
            raise KeyError(
                f"Column {c!r} not found in raw table: {args.raw}\n"
                f"Available columns (first 25): {list(raw.columns)[:25]}"
            )

    # Add stable row index for joining against predictions' orig_row_index
    raw = raw.copy()
    raw["orig_row_index"] = np.arange(len(raw), dtype=int)

    # Target -> 0/1
    raw["y_true_raw"] = raw[args.target_col].apply(normalize_yes_no)
    if raw["y_true_raw"].isna().any():
        # It's okay to have some missing, but warn in output
        n_missing = int(raw["y_true_raw"].isna().sum())
        print(f"[WARN] {n_missing} rows have missing/unparseable target values in {args.target_col!r}.")

    raw["muni_key"] = raw[args.muni_col].apply(canonical_muni_name)

    preds = read_predictions(args.pred_csv)
    join_col = preds["_join_col"].iloc[0]  # "orig_row_index" or "row_index"

    # Merge predictions onto raw rows
    merged = raw.merge(
        preds.drop(columns=["_join_col"]),
        left_on="orig_row_index",
        right_on=join_col,
        how="left",
        validate="one_to_one",
    )

    # Load geometries
    g_all = load_nj_muni_geometries()
    nj_outline = gpd.GeoSeries(g_all.unary_union, crs=g_all.crs)

    # Build plot title
    model_label = (args.model_label.strip() or "model")
    if args.mode == "correctness":
        title = f"NJ Prediction Correctness ({model_label})"
    else:
        title = f"NJ Percent Correct by Municipality ({model_label})"

    # Join to geometries:
    # - For correctness: we want row-level dots, but many rows share the same muni.
    #   We'll collapse to one row per muni by majority vote of correctness, OR keep percent.
    #   Here: correctness dots show each muni once, with correctness = (pct_correct >= 0.5).
    # - For percent_correct: compute pct_correct per muni.
    d = merged[["muni_key", args.muni_col, "y_true_raw", "y_true", "y_pred", "correct"]].copy()
    d = d.dropna(subset=["muni_key"])

    # Only municipalities that actually have predictions
    d_has_pred = d.dropna(subset=["y_pred", "y_true", "correct"]).copy()
    d_has_pred["correct"] = d_has_pred["correct"].astype(int)

    muni_stats = (
        d_has_pred.groupby("muni_key", as_index=False)
        .agg(
            Municipality=(args.muni_col, "first"),
            n=("correct", "size"),
            pct_correct=("correct", "mean"),
        )
    )
    muni_stats["correct_majority"] = (muni_stats["pct_correct"] >= 0.5).astype(int)

    g_join = g_all.merge(muni_stats, on="muni_key", how="left")

    # Prepare dot GeoDataFrame
    g_pts = g_join.copy()
    g_pts["geometry"] = g_pts.geometry.representative_point()
    # For correctness dots, use majority correctness
    g_pts["correct"] = g_pts["correct_majority"]

    # Plot
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 11))
    nj_outline.boundary.plot(ax=ax, linewidth=1)

    if args.mode == "percent_correct":
        if args.polygon:
            plot_percent_correct_polygons(fig, ax, g_join, title=title)

        if args.dot:
            # Optional overlay dots colored by pct_correct continuously
            g_pts2 = g_pts.dropna(subset=["pct_correct"]).copy()
            if len(g_pts2) > 0:
                g_pts2.plot(
                    ax=ax,
                    column="pct_correct",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    markersize=22,
                    legend=False,
                )

        # Add small annotation with counts
        n_muni = int(muni_stats.shape[0])
        ax.text(
            0.02,
            0.02,
            f"Municipalities with predictions: {n_muni}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    else:
        # correctness
        if args.polygon:
            # Fill polygons by majority correctness if you want polygon view
            tmp = g_join.copy()
            has = tmp["correct_majority"].notna()
            if (~has).any():
                tmp.loc[~has].plot(ax=ax, color="lightgray", linewidth=0.2, edgecolor="white")
            tmp.loc[has].plot(
                ax=ax,
                column="correct_majority",
                cmap="RdYlGn",  # 0=red-ish, 1=green-ish
                vmin=0,
                vmax=1,
                linewidth=0.2,
                edgecolor="white",
                legend=True,
                legend_kwds={"ticks": [0, 1], "label": "Majority correctness"},
            )
            ax.set_title(title)
            ax.set_axis_off()

        if args.dot:
            # Dots colored correct/incorrect (majority)
            g_pts3 = g_pts.dropna(subset=["correct"]).copy()
            plot_correctness_dots(ax, g_pts3, title=title)

        n_muni = int(muni_stats.shape[0])
        ax.text(
            0.02,
            0.02,
            f"Municipalities with predictions: {n_muni}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

    # Diagnostics: unmatched munis between your table and TIGER geometries
    unmatched = set(muni_stats["muni_key"]) - set(g_all["muni_key"])
    if unmatched:
        print(f"\n[WARN] Unmatched municipalities ({len(unmatched)}). Examples:")
        for x in sorted(list(unmatched))[:25]:
            print("  -", x)

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
