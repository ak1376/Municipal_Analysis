from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Iterable

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca_auto(
    X: pd.DataFrame,
    *,
    variance_threshold: float = 0.90,
    scale: bool = True,
    scree: bool = True,
    plot_scree: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, PCA, Optional[StandardScaler], pd.DataFrame]:
    """
    Run PCA with automatic component selection and named PC columns.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (rows = samples, columns = features)
    variance_threshold : float
        Cumulative explained variance to retain (e.g. 0.90)
    scale : bool
        Whether to standardize features before PCA
    scree : bool
        Whether to compute scree table
    plot_scree : bool
        Whether to plot scree
    random_state : int or None
        Optional random seed

    Returns
    -------
    X_pca_df : DataFrame
        PCA-transformed data with columns PC1, PC2, ...
    pca : PCA
        Fitted PCA object
    scaler : StandardScaler or None
        Fitted scaler (if scale=True)
    scree_df : DataFrame
        Scree table
    """

    # ---- scale ----
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X.values

    # ---- full PCA to determine k ----
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)

    # ---- fit final PCA ----
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    X_pca_df = pd.DataFrame(X_pca, index=X.index, columns=pc_cols)

    # ---- scree table ----
    scree_df = pd.DataFrame({
        "Component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "Explained_Variance_Ratio": pca_full.explained_variance_ratio_,
        "Cumulative_Explained_Variance": cum_var,
    })

    fig = None
    if scree and plot_scree:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            scree_df["Component"],
            scree_df["Explained_Variance_Ratio"],
            marker="o",
            label="Explained variance",
        )
        ax.plot(
            scree_df["Component"],
            scree_df["Cumulative_Explained_Variance"],
            linestyle="--",
            color="black",
            label="Cumulative variance",
        )
        ax.axhline(variance_threshold, color="red", linestyle=":", label="Threshold")
        ax.set_xlabel("PCA component")
        ax.set_ylabel("Variance explained")
        ax.set_title(f"PCA Scree (k={n_components} for {variance_threshold:.0%})")
        ax.legend()
        plt.tight_layout()
        # Do not call plt.show() to avoid blocking

    return X_pca_df, pca, scaler, scree_df, fig

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
    Simple matplotlib pair-plot (scatter off-diagonal, hist on diagonal).

    Parameters
    ----------
    df : DataFrame
        DataFrame containing columns to plot (e.g., PC1..PCk)
    columns : list[str] or None
        Columns to include in the pair plot. If None, use all numeric columns.
    hue : str OR array-like OR Series OR None
        - If str: interpreted as a column name in df.
        - If array-like/Series: used directly as per-row group labels.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns = list(columns)

    # Resolve hue values (either a column name or an explicit vector)
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
            raise ValueError(
                f"hue has length {len(hue_values)} but df has {len(df)} rows"
            )

    # Prepare groups
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
                # diagonal: histogram
                ax.hist(df[xcol].dropna().to_numpy())
            else:
                # off-diagonal: scatter by group
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

    # Add legend once (top-right cell) if hue is used
    if hue_values is not None:
        axes[0, k - 1].legend(title=hue_name, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    return fig