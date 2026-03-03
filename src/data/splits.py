"""Train/val/test split logic with leakage prevention."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def create_splits(
    manifest: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_column: str = "image_id",
    seed: int = 42,
    stratify_column: Optional[str] = "language",
) -> pd.DataFrame:
    """Create train/val/test splits with leakage prevention.

    Splits on unique values of `split_column` (typically image_id) to
    prevent the same image from appearing in multiple splits.

    Args:
        manifest: DataFrame with at least [split_column, language] columns.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        split_column: Column to split on (prevents leakage).
        seed: Random seed.
        stratify_column: Optional column for stratified splitting.

    Returns:
        DataFrame with an added 'split' column.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    rng = np.random.RandomState(seed)

    # Get unique split entities (e.g., unique image_ids)
    unique_entities = manifest[split_column].unique()
    rng.shuffle(unique_entities)

    n = len(unique_entities)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_entities = set(unique_entities[:n_train])
    val_entities = set(unique_entities[n_train : n_train + n_val])
    test_entities = set(unique_entities[n_train + n_val :])

    # Assign splits
    def assign_split(entity_id):
        if entity_id in train_entities:
            return "train"
        elif entity_id in val_entities:
            return "val"
        else:
            return "test"

    manifest = manifest.copy()
    manifest["split"] = manifest[split_column].apply(assign_split)

    return manifest


def create_cross_lingual_eval_split(
    manifest: pd.DataFrame,
    languages: list[str],
    samples_per_language: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a balanced cross-lingual evaluation split.

    Ensures equal representation of selected languages in the eval set.

    Args:
        manifest: Full manifest DataFrame.
        languages: List of language codes to include.
        samples_per_language: Number of samples per language.
        seed: Random seed.

    Returns:
        Filtered DataFrame with balanced language representation.
    """
    rng = np.random.RandomState(seed)
    subsets = []

    for lang in languages:
        lang_df = manifest[manifest["language"] == lang]
        if len(lang_df) >= samples_per_language:
            sampled = lang_df.sample(n=samples_per_language, random_state=rng)
        else:
            sampled = lang_df  # Use all available if fewer than requested
        subsets.append(sampled)

    return pd.concat(subsets, ignore_index=True)


def verify_no_leakage(
    manifest: pd.DataFrame,
    split_column: str = "image_id",
) -> bool:
    """Verify that no entity appears in multiple splits.

    Returns:
        True if no leakage detected.
    """
    train_ids = set(manifest[manifest["split"] == "train"][split_column])
    val_ids = set(manifest[manifest["split"] == "val"][split_column])
    test_ids = set(manifest[manifest["split"] == "test"][split_column])

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    if train_val or train_test or val_test:
        raise ValueError(
            f"Data leakage detected! "
            f"train∩val: {len(train_val)}, "
            f"train∩test: {len(train_test)}, "
            f"val∩test: {len(val_test)}"
        )
    return True
