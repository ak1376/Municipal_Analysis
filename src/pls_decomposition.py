#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression


def _x_variance_explained_by_scores(X_scaled: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Approximate per-component X-variance explained by PLS scores.

    We compute cumulative variance in X explained by projecting X onto the
    subspace spanned by the first k score vectors T[:, :k], and take differences
    to get per-component contributions.

    Returns
    -------
    per_comp : array shape (K,)
        Fraction of total X variance explained by each component.
    """
    # Total variance in X (sum of variances across columns)
    total_var = float(np.var(X_scaled, axis=0, ddof=0).sum())
    if total_var <= 0:
        return np.zeros(T.shape[1], dtype=float)

    K = T.shape[1]
    cum = np.zeros(K, dtype=float)

    for k in range(1, K + 1):
        Tk = T[:, :k]  # (n, k)
        # Orthonormal basis for span(Tk)
        Q, _ = np.linalg.qr(Tk)
        # Projection of X onto span(Q): X_hat = Q Q^T X
        X_hat = Q @ (Q.T @ X_scaled)
        explained = float(np.var(X_hat, axis=0, ddof=0).sum())
        cum[k - 1] = explained / total_var

    per = np.empty(K, dtype=float)
    per[0] = cum[0]
    per[1:] = np.diff(cum)
    per = np.clip(per, 0.0, 1.0)
    return per


def run_pls_auto(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    *,
    max_components: Optional[int] = None,
    variance_threshold: float = 0.90,
    scale: bool = True,
    scree: bool = True,
    plot_scree: bool = True,
) -> Tuple[pd.DataFrame, PLSRegression, Optional[StandardScaler], pd.DataFrame, Optional[plt.Figure]]:
    """
    Run PLS decomposition (PLSRegression) with automatic component selection based on
    approximate X-variance explained by PLS scores, and named columns PLS1, PLS2, ...

    Notes
    -----
    - PLS is supervised: it uses y to find components.
    - Unlike PCA, PLS does not have a canonical explained_variance_ratio_.
      Here we produce a "scree-like" table using an approximation:
      fraction of X variance explained by projection onto the score subspace.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (rows = samples, columns = features)
    y : Series or array
        Target (binary or continuous). Will be coerced to numeric.
    max_components : int or None
        Upper bound on number of components. Default: min(n_samples-1, n_features)
    variance_threshold : float
        Cumulative X-variance threshold used to pick k (e.g. 0.90)
    scale : bool
        Whether to standardize features before PLS
    scree : bool
        Whether to compute scree-like table
    plot_scree : bool
        Whether to plot scree-like figure

    Returns
    -------
    X_pls_df : DataFrame
        Score space with columns PLS1..PLSk
    pls : PLSRegression
        Fitted PLS model with n_components = k
    scaler : StandardScaler or None
        Fitted scaler (if scale=True)
    scree_df : DataFrame
        Scree-like table (Component, Explained_Variance_Ratio, Cumulative_Explained_Variance)
    fig : matplotlib Figure or None
        Scree-like plot figure if requested
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    y_arr = np.asarray(y).reshape(-1)
    if len(y_arr) != len(X):
        raise ValueError(f"Row mismatch: X has {len(X)} rows but y has {len(y_arr)} entries.")

    # Coerce y to numeric (handles YES/NO if you pass already-mapped y, otherwise map beforehand)
    y_num = pd.to_numeric(pd.Series(y_arr), errors="coerce").to_numpy()
    if np.isnan(y_num).any():
        raise ValueError("y contains non-numeric values after coercion. Map/clean y first.")

    # ---- scale X ----
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X.to_numpy()

    n, p = X_scaled.shape
    if max_components is None:
        max_components = min(n - 1, p)
    max_components = int(max_components)
    if max_components < 1:
        raise ValueError("max_components must be >= 1")
    max_components = min(max_components, n - 1, p)

    # ---- fit max PLS to determine k ----
    pls_full = PLSRegression(n_components=max_components, scale=False)
    pls_full.fit(X_scaled, y_num)

    T_full = pls_full.x_scores_  # (n, max_components)
    per_comp = _x_variance_explained_by_scores(X_scaled, T_full)
    cum_var = np.cumsum(per_comp)

    # Choose k
    k = int(np.searchsorted(cum_var, variance_threshold) + 1)
    k = max(1, min(k, max_components))

    # ---- fit final PLS ----
    pls = PLSRegression(n_components=k, scale=False)
    pls.fit(X_scaled, y_num)
    T = pls.x_scores_  # (n, k)

    pls_cols = [f"PLS{i}" for i in range(1, k + 1)]
    X_pls_df = pd.DataFrame(T, index=X.index, columns=pls_cols)

    # ---- scree-like table ----
    scree_df = pd.DataFrame(
        {
            "Component": np.arange(1, max_components + 1),
            "Explained_Variance_Ratio": per_comp,
            "Cumulative_Explained_Variance": np.cumsum(per_comp),
        }
    )

    fig = None
    if scree and plot_scree:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            scree_df["Component"],
            scree_df["Explained_Variance_Ratio"],
            marker="o",
            label="Approx X variance",
        )
        ax.plot(
            scree_df["Component"],
            scree_df["Cumulative_Explained_Variance"],
            linestyle="--",
            color="black",
            label="Cumulative",
        )
        ax.axhline(variance_threshold, color="red", linestyle=":", label="Threshold")
        ax.set_xlabel("PLS component")
        ax.set_ylabel("Approx. X variance explained")
        ax.set_title(f"PLS Scree-like (k={k} for {variance_threshold:.0%})")
        ax.legend()
        plt.tight_layout()

    return X_pls_df, pls, scaler, scree_df, fig


def pair_plot(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    hue: Optional[Union[str, pd.Series, np.ndarray, Sequence]] = None,
    alpha: float = 0.7,
    s: float = 18.0,
    figsize_per_cell: float = 2.2,
):
    """
    Same matplotlib pair-plot you used for PCA (scatter off-diagonal, hist on diagonal).
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns = list(columns)

    hue_values = None
    hue_name = None
    if hue is None:
        hue_values = None
    elif isinstance(hue, str):
        hue_name = hue
        if hue_name not in df.columns:
            raise KeyError(f"hue='{hue_name}' not found in df.columns")
        hue_values = df[hue_name].to_numpy()
    else:
        hue_name = "hue"
        hue_values = np.asarray(hue)
        if len(hue_values) != len(df):
            raise ValueError(f"hue has length {len(hue_values)} but df has {len(df)} rows")

    if hue_values is None:
        groups = {"": np.ones(len(df), dtype=bool)}
    else:
        uniq = pd.unique(hue_values)
        groups = {str(u): (hue_values == u) for u in uniq}

    k = len(columns)
    fig, axes = plt.subplots(
        k, k, figsize=(figsize_per_cell * k, figsize_per_cell * k), squeeze=False
    )

    for i, ycol in enumerate(columns):
        for j, xcol in enumerate(columns):
            ax = axes[i, j]

            if i == j:
                ax.hist(df[xcol].dropna().to_numpy())
            else:
                for label, mask in groups.items():
                    ax.scatter(
                        df.loc[mask, xcol],
                        df.loc[mask, ycol],
                        s=s,
                        alpha=alpha,
                        label=label if (i == 0 and j == k - 1) else None,
                    )

            if i == k - 1:
                ax.set_xlabel(xcol)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(ycol)
            else:
                ax.set_yticklabels([])

    if hue_values is not None:
        axes[0, k - 1].legend(title=hue_name, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    return fig
