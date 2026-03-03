"""I/O utilities for saving/loading embeddings, metadata, and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def save_embeddings(
    embeddings: np.ndarray,
    ids: list[str],
    save_dir: str | Path,
    prefix: str = "embeddings",
) -> dict[str, Path]:
    """Save embedding vectors and their IDs to disk.

    Args:
        embeddings: Array of shape (N, D).
        ids: List of sample IDs corresponding to each embedding.
        save_dir: Directory to save files.
        prefix: Filename prefix.

    Returns:
        Dictionary mapping 'vectors' and 'ids' to their file paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_path = save_dir / f"{prefix}_vectors.npy"
    ids_path = save_dir / f"{prefix}_ids.json"

    np.save(str(vec_path), embeddings)
    with open(ids_path, "w") as f:
        json.dump(ids, f)

    return {"vectors": vec_path, "ids": ids_path}


def load_embeddings(
    save_dir: str | Path,
    prefix: str = "embeddings",
) -> tuple[np.ndarray, list[str]]:
    """Load saved embeddings and IDs.

    Returns:
        Tuple of (embeddings_array, ids_list).
    """
    save_dir = Path(save_dir)
    vec_path = save_dir / f"{prefix}_vectors.npy"
    ids_path = save_dir / f"{prefix}_ids.json"

    embeddings = np.load(str(vec_path))
    with open(ids_path, "r") as f:
        ids = json.load(f)

    return embeddings, ids


def save_metrics(metrics: Dict[str, Any], save_path: str | Path) -> None:
    """Save metrics dictionary to JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(load_path: str | Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(load_path, "r") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: str | Path, format: str = "parquet") -> None:
    """Save a DataFrame in specified format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "parquet":
        df.to_parquet(str(path), index=False)
    elif format == "csv":
        df.to_csv(str(path), index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from parquet or CSV."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(str(path))
    elif path.suffix == ".csv":
        return pd.read_csv(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
