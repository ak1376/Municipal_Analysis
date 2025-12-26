import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# -----------------------------
# Helpers: name standardization
# -----------------------------
_SUFFIXES = [
    "Township", "Borough", "City", "Town", "Village"
]

plot_dir = Path(__file__).parent.parent / "plots"

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
    s = re.sub(r"[^\w\s\-]", "", s)     # keep letters/numbers/underscore/space/hyphen
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


def normalize_yes_no(x) -> str:
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    if x in {"YES", "Y", "TRUE", "1"}:
        return "YES"
    if x in {"NO", "N", "FALSE", "0"}:
        return "NO"
    return None


# -----------------------------
# Main plotting function
# -----------------------------
def plot_nj_qualification_map(
    df: pd.DataFrame,
    muni_col: str = "Municipality",
    qualified_col: str = "Qualified Municipality",
    figsize=(9, 11),
):
    """
    Plots NJ outline and a dot for each municipality:
      YES -> red
      NO  -> green
    Uses Census TIGER/Line boundaries (COUSUB + PLACE) for matching.
    """

    # Keep only needed columns; de-duplicate municipalities if your table has repeats
    d = df[[muni_col, qualified_col]].copy()
    d[qualified_col] = d[qualified_col].apply(normalize_yes_no)
    d["muni_key"] = d[muni_col].apply(canonical_muni_name)

    # If duplicates exist, keep the first non-null qualification
    d = (
        d.sort_values(by=[muni_col])
         .dropna(subset=["muni_key"])
         .groupby("muni_key", as_index=False)
         .agg({
             muni_col: "first",
             qualified_col: lambda s: next((v for v in s if pd.notna(v)), None),
         })
    )

    # --- Load NJ municipality geometries from TIGER/Line (internet required) ---
    # County subdivisions (includes most townships/boroughs/etc.)
    cousub_url = "https://www2.census.gov/geo/tiger/TIGER2023/COUSUB/tl_2023_34_cousub.zip"
    g_cousub = gpd.read_file(cousub_url)[["NAME", "geometry"]].copy()
    g_cousub["muni_key"] = g_cousub["NAME"].apply(canonical_muni_name)

    # Incorporated places (helps catch a few edge cases)
    place_url = "https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_34_place.zip"
    g_place = gpd.read_file(place_url)[["NAME", "geometry"]].copy()
    g_place["muni_key"] = g_place["NAME"].apply(canonical_muni_name)

    # Combine and drop duplicate keys (prefer COUSUB first, then PLACE)
    g_all = pd.concat([g_cousub, g_place], ignore_index=True)
    g_all = g_all.drop_duplicates(subset=["muni_key"], keep="first")
    g_all = gpd.GeoDataFrame(g_all, geometry="geometry", crs=g_cousub.crs)

    # Join your qualification data onto the geometries
    g_join = g_all.merge(d, on="muni_key", how="inner")

    # Representative points (safer than centroid for weird shapes)
    g_pts = g_join.copy()
    g_pts["geometry"] = g_pts.geometry.representative_point()

    # Make a NJ outline from union of all geometries
    nj_outline = gpd.GeoSeries(g_all.unary_union, crs=g_all.crs)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    nj_outline.boundary.plot(ax=ax, linewidth=1)

    # split by qualification
    yes = g_pts[g_pts[qualified_col] == "YES"]
    no  = g_pts[g_pts[qualified_col] == "NO"]

    # YES -> red, NO -> green (as you requested)
    if len(yes) > 0:
        yes.plot(ax=ax, color="red", markersize=20, label="Qualified (YES)")
    if len(no) > 0:
        no.plot(ax=ax, color="green", markersize=20, label="Unqualified (NO)")

    ax.set_title("New Jersey Municipalities: Qualified vs Unqualified")
    ax.set_axis_off()
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/NJ_map.png')
    # Helpful diagnostics: who didn't match?
    unmatched = set(d["muni_key"]) - set(g_all["muni_key"])
    if unmatched:
        print(f"\nUnmatched municipalities ({len(unmatched)}). Examples:")
        for x in sorted(list(unmatched))[:25]:
            print("  -", x)
