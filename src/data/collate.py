"""Batch collation for multimodal image-text pairs."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of multimodal samples into padded tensors.

    Args:
        batch: List of sample dicts from WITDataset.__getitem__.

    Returns:
        Collated batch dictionary with stacked tensors and metadata lists.
    """
    images = torch.stack([sample["image"] for sample in batch])
    input_ids = torch.stack([sample["input_ids"] for sample in batch])
    attention_mask = torch.stack([sample["attention_mask"] for sample in batch])

    sample_ids = [sample["sample_id"] for sample in batch]
    languages = [sample["language"] for sample in batch]
    texts_raw = [sample["text_raw"] for sample in batch]

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sample_ids": sample_ids,
        "languages": languages,
        "texts_raw": texts_raw,
    }
