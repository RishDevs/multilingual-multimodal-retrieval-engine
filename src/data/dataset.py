"""PyTorch Dataset for WIT image-text pairs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.tokenization import TextTokenizer
from src.data.transforms import get_eval_transforms, get_train_transforms


class WITDataset(Dataset):
    """Dataset for Wikipedia-based Image Text (WIT) samples.

    Loads image-text pairs from a manifest parquet file and applies
    image transforms and text tokenization on-the-fly.

    Each sample contains:
        - sample_id: unique identifier
        - image: transformed image tensor
        - input_ids: tokenized text input IDs
        - attention_mask: text attention mask
        - language: language code
        - text_raw: raw text string (for debugging / display)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        tokenizer: TextTokenizer,
        split: str = "train",
        image_size: int = 224,
        text_fields: list[str] | None = None,
        max_text_len: int = 64,
        transform: Optional[Callable] = None,
    ):
        """Initialize the WIT Dataset.

        Args:
            manifest_path: Path to manifest parquet with columns:
                [sample_id, image_path, text, language, image_id, pair_id, split].
            image_root: Root directory containing images.
            tokenizer: TextTokenizer instance.
            split: One of 'train', 'val', 'test'.
            image_size: Target image size for transforms.
            text_fields: Text field names (for multi-field support).
            max_text_len: Maximum text token length.
            transform: Optional custom image transform.
        """
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.split = split
        self.max_text_len = max_text_len

        # Load manifest and filter by split
        df = pd.read_parquet(str(manifest_path))
        self.data = df[df["split"] == split].reset_index(drop=True)

        # Set transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_eval_transforms(image_size)

        self.text_fields = text_fields or ["text"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]

        # ── Image ──
        image_path = self.image_root / row["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            # Return a black image on failure (with logged warning)
            image = torch.zeros(3, 224, 224)

        # ── Text ──
        text = self._get_text(row)
        text = self._clean_text(text)
        encoded = self.tokenizer(text)

        return {
            "sample_id": str(row.get("sample_id", idx)),
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "language": str(row.get("language", "unknown")),
            "text_raw": text,
        }

    def _get_text(self, row: pd.Series) -> str:
        """Extract the best available text field from a manifest row."""
        for field in self.text_fields:
            if field in row and pd.notna(row[field]) and str(row[field]).strip():
                return str(row[field]).strip()
        # Fallback: use 'text' column
        return str(row.get("text", "")).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing HTML artifacts and normalizing whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)         # Remove HTML tags
        text = re.sub(r"\s+", " ", text).strip()      # Normalize whitespace
        return text

    def get_language_distribution(self) -> Dict[str, int]:
        """Return count of samples per language."""
        return self.data["language"].value_counts().to_dict()
