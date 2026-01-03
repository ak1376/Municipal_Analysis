# src/decision_tree_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, balanced_accuracy_score


@dataclass(frozen=True)
class TreeParams:
    max_depth: Optional[int]
    min_samples_leaf: int
    class_weight: Optional[str]  # None or "balanced"
    random_state: int


def build_decision_tree(params: TreeParams) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=params.max_depth,
        min_samples_leaf=int(params.min_samples_leaf),
        class_weight=params.class_weight,
        random_state=int(params.random_state),
    )


def probs_to_preds(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probs).ravel() >= float(threshold)).astype(int)


def save_feature_importances(clf: DecisionTreeClassifier, feature_names: List[str], out_path: Path) -> None:
    imp = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(clf.feature_importances_).ravel()}
    ).sort_values("importance", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_path, index=False)


def plot_and_save_tree_readable(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    out_path: Path,
    *,
    max_depth_plot: int = 3,
    class_names: Tuple[str, str] = ("NO", "YES"),
    fontsize: int = 12,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = int(clf.tree_.node_count)
    w = int(min(60, max(18, 8 + 2 * max_depth_plot)))
    h = int(min(30, max(10, 6 + 2 * max_depth_plot)))

    fig, ax = plt.subplots(figsize=(w, h))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=list(class_names),
        filled=True,
        rounded=True,
        max_depth=int(max_depth_plot),
        proportion=False,
        impurity=True,
        label="all",
        precision=3,
        fontsize=int(fontsize),
        ax=ax,
    )
    ax.set_title(f"Decision Tree (shown up to depth={max_depth_plot})  |  nodes={n_nodes}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_tree_rules_text(clf: DecisionTreeClassifier, feature_names: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rules = export_text(clf, feature_names=feature_names, decimals=3)
    out_path.write_text(rules + "\n")
