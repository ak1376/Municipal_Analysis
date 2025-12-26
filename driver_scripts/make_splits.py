#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------
# Repo import setup
# ----------------------------
def _add_repo_root_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]  # .../modeling
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _add_repo_root_to_syspath()


# ----------------------------
# YAML loading
# ----------------------------
def read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: PyYAML.\n"
            "Install it in your environment:\n"
            "  pip install pyyaml\n"
            "or:\n"
            "  conda install -c conda-forge pyyaml\n"
        ) from e

    obj = yaml.safe_load(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"YAML must parse to a dict: {path}")
    return obj


def yaml_dump(obj: Any) -> str:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: PyYAML.\n"
            "Install it in your environment:\n"
            "  pip install pyyaml\n"
            "or:\n"
            "  conda install -c conda-forge pyyaml\n"
        ) from e
    return yaml.safe_dump(obj, sort_keys=False)


# ----------------------------
# Small utils
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _indices_to_txt(indices: np.ndarray) -> str:
    return " ".join(map(str, indices.astype(int).tolist()))


# ----------------------------
# Coercions / fingerprinting
# ----------------------------
def _as_int_y(y: pd.Series) -> np.ndarray:
    """
    Accept category/object/bool/0-1; force 0/1 int numpy array.
    """
    yv = y.astype(str) if y.dtype.name == "category" else y

    if yv.dtype == object:
        s = yv.astype(str).str.strip().str.upper()
        mapping = {"YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
        yy = s.map(mapping)
        if yy.isna().any():
            bad = yv[yy.isna()].head(10).tolist()
            raise ValueError(f"Target has non-binary/uncoercible values (examples): {bad}")
        yy = yy.astype(int)
    else:
        yy = pd.to_numeric(yv, errors="coerce")
        if yy.isna().any():
            bad = yv[yy.isna()].head(10).tolist()
            raise ValueError(f"Target has NaNs after numeric coercion (examples): {bad}")
        yy = yy.astype(int)

    uniq = sorted(pd.unique(yy).tolist())
    if uniq not in ([0, 1], [0], [1]):
        raise ValueError(f"Target must be binary 0/1; got {uniq}")
    return yy.to_numpy()


def coerce_X_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all columns to numeric (hard error if coercion introduces NaNs).
    """
    Xn = X.copy()
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    if Xn.isna().any().any():
        bad_cols = Xn.columns[Xn.isna().any()].tolist()
        raise ValueError(
            "X contains NaNs after numeric coercion. "
            f"Problem columns (examples): {bad_cols[:20]}"
        )
    return Xn


def fingerprint_xy(X: pd.DataFrame, y_int: np.ndarray) -> Dict[str, Any]:
    cols = list(X.columns)
    shape = (int(X.shape[0]), int(X.shape[1]))

    X_arr = X.to_numpy()
    if np.issubdtype(X_arr.dtype, np.floating):
        X_bytes = np.ascontiguousarray(np.round(X_arr, 12)).tobytes()
    else:
        X_bytes = np.ascontiguousarray(X_arr).tobytes()

    y_bytes = np.ascontiguousarray(y_int.astype(int)).tobytes()

    h = hashlib.sha256()
    h.update(str(cols).encode("utf-8"))
    h.update(str(shape).encode("utf-8"))
    h.update(X_bytes)
    h.update(y_bytes)

    return {
        "n": shape[0],
        "p": shape[1],
        "columns": cols,
        "sha256": h.hexdigest(),
    }


# ----------------------------
# Dataset loading
# ----------------------------
def load_xy_from_config(cfg: Dict[str, Any], *, repo_root: Path) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = cfg.get("dataset", {})
    mode = dataset.get("mode", "processed_csv")

    if mode == "processed_csv":
        proc = dataset.get("processed", {})
        x_csv = (repo_root / proc["x_csv"]).resolve()
        y_csv = (repo_root / proc["y_csv"]).resolve()
        target_col = proc.get("target_col", "Qualified Municipality")

        X = pd.read_csv(x_csv)
        y_df = pd.read_csv(y_csv)
        if target_col not in y_df.columns:
            raise KeyError(f"target_col {target_col!r} not found in {y_csv}")
        y = y_df[target_col]

        if len(X) != len(y):
            raise ValueError(f"Row mismatch: X={len(X)} y={len(y)}. Ensure they were saved in same order.")
        return X.reset_index(drop=True), y.reset_index(drop=True)

    raise ValueError(f"Unknown dataset.mode: {mode}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create and save deterministic train/test splits from a YAML config.\n"
            "Writes indices and manifests so ALL model drivers can do 1-to-1 comparisons.\n"
            "NO cross-validation folds are produced."
        )
    )
    ap.add_argument("--config", type=Path, required=True, help="Path to YAML split config.")
    args = ap.parse_args()

    cfg = read_yaml(args.config)

    # --- load X/y early (for fingerprint + naming)
    X, y = load_xy_from_config(cfg, repo_root=REPO_ROOT)
    Xn = coerce_X_to_numeric(X)
    y_int = _as_int_y(y)
    fp = fingerprint_xy(Xn, y_int)

    # --- output config
    out_cfg = cfg.get("output", {})
    root_dir = Path(out_cfg.get("root_dir", "data/splits"))

    name_prefix = out_cfg.get("name_prefix", None)
    name_fixed = out_cfg.get("name", None)

    if name_prefix is not None and str(name_prefix).strip():
        name = f"{str(name_prefix).strip()}__{fp['sha256'][:12]}"
    elif name_fixed is not None and str(name_fixed).strip():
        name = str(name_fixed).strip()
    else:
        name = f"split__{fp['sha256'][:12]}"

    out_dir = (REPO_ROOT / root_dir / name).resolve()
    ensure_dir(out_dir)

    # --- splitting params
    split_cfg = cfg.get("split", {})
    strategy = split_cfg.get("strategy", "train_test")  # "train_test" or "all_train"
    test_size = float(split_cfg.get("test_size", 0.25))
    random_state = int(split_cfg.get("random_state", 0))
    stratify = bool(split_cfg.get("stratify", True))

    n = int(len(y_int))
    all_idx = np.arange(n, dtype=int)

    if strategy == "all_train":
        train_idx = all_idx
        test_idx = np.array([], dtype=int)
    elif strategy == "train_test":
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=y_int if stratify else None,
        )
    else:
        raise ValueError(f"Unknown split.strategy: {strategy} (use 'train_test' or 'all_train')")

    # --- Save artifacts
    (out_dir / "split_config.yaml").write_text(yaml_dump(cfg))
    write_json(out_dir / "dataset_fingerprint.json", fp)

    np.save(out_dir / "indices_train.npy", train_idx)
    np.save(out_dir / "indices_test.npy", test_idx)
    (out_dir / "indices_train.txt").write_text(_indices_to_txt(train_idx) + "\n")
    (out_dir / "indices_test.txt").write_text(_indices_to_txt(test_idx) + "\n")

    manifest = {
        "out_dir": str(out_dir),
        "strategy": strategy,
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "cv_enabled": False,
        "split_random_state": random_state,
        "test_size": test_size,
        "stratify": bool(stratify),
        "dataset_fingerprint_sha256": fp["sha256"],
        "dataset_fingerprint_short": fp["sha256"][:12],
    }
    write_json(out_dir / "manifest.json", manifest)

    (out_dir / "manifest.txt").write_text(
        "\n".join(
            [
                f"out_dir: {out_dir}",
                f"strategy: {strategy}",
                f"n_total: {n}",
                f"n_train: {len(train_idx)}",
                f"n_test: {len(test_idx)}",
                f"cv_enabled: False",
                f"dataset_fingerprint_sha256: {fp['sha256']}",
            ]
        )
        + "\n"
    )

    print(f"[OK] Wrote split artifacts to: {out_dir}")
    print(f"[OK] n_train={len(train_idx)} n_test={len(test_idx)} fp={fp['sha256'][:12]}...")


if __name__ == "__main__":
    main()
