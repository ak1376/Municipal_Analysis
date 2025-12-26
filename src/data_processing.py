from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def read_tsm_excel(path: Path, header_row: int = 2) -> pd.DataFrame:
    df = pd.read_excel(path, header=header_row)
    df.columns = df.columns.map(lambda c: str(c).strip())
    return df


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert messy numeric strings to floats:
      - strips whitespace
      - treats '-', 'N/A', '' as NaN
      - removes commas and percent signs
    """
    x = s.astype(str).str.strip()
    x = x.replace({"-": np.nan, "N/A": np.nan, "n/a": np.nan, "": np.nan})
    x = x.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(x, errors="coerce")


def coerce_object_columns_to_numeric(
    X: pd.DataFrame,
    *,
    min_non_nan_frac: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For object columns, attempt numeric conversion.
    Keep the converted column only if it doesn't destroy the data.
    """
    X = X.copy()
    obj_cols = X.select_dtypes(include=["object"]).columns
    n = len(X)

    for c in obj_cols:
        converted = clean_numeric_series(X[c])
        ok = converted.notna().sum() >= max(1, int(min_non_nan_frac * n))
        if ok:
            X[c] = converted
        else:
            if verbose:
                kept = converted.notna().sum()
                print(f"[warn] Leaving '{c}' as non-numeric (only {kept}/{n} non-NaN after conversion)")
    return X


def build_xy(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    dropna: str = "any",          # "any" or "all" for rowwise NaN in features
    drop_constant: bool = True,
    coerce_objects: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns aligned (X, y) with cleaning applied in one pass.
    """
    ensure_columns(df, feature_cols + [target_col])

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Clean objects -> numeric (optional)
    if coerce_objects:
        X = coerce_object_columns_to_numeric(X, verbose=verbose)

    # Combine, drop rows ONCE so X/y stay aligned
    xy = pd.concat([X, y.rename(target_col)], axis=1)

    if dropna == "any":
        xy = xy.dropna(axis=0, how="any")
    elif dropna == "all":
        xy = xy.dropna(axis=0, how="all")
    else:
        raise ValueError("dropna must be 'any' or 'all'")

    X_clean = xy[feature_cols]
    y_clean = xy[target_col].astype("category")

    # Drop constant columns ONCE at end
    if drop_constant:
        nunique = X_clean.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            if verbose:
                print("[info] Dropping constant columns:", constant_cols)
            X_clean = X_clean.drop(columns=constant_cols)

    if verbose:
        print("[info] X shape:", X_clean.shape, "| y shape:", y_clean.shape)

    return X_clean, y_clean
