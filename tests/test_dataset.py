"""Tests for WIT dataset loader."""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestWITDataset:
    """Test the WIT dataset class."""

    @pytest.fixture
    def sample_manifest(self, tmp_path):
        """Create a sample manifest for testing."""
        data = {
            "sample_id": [f"s{i}" for i in range(20)],
            "image_path": [f"img_{i}.jpg" for i in range(20)],
            "text": [f"This is sample text number {i}" for i in range(20)],
            "language": ["en"] * 10 + ["es"] * 5 + ["fr"] * 5,
            "image_id": [f"img_{i}" for i in range(20)],
            "pair_id": [f"p{i}" for i in range(20)],
            "split": ["train"] * 14 + ["val"] * 3 + ["test"] * 3,
        }
        df = pd.DataFrame(data)
        manifest_path = tmp_path / "manifest.parquet"
        df.to_parquet(str(manifest_path))
        return manifest_path

    @pytest.fixture
    def image_dir(self, tmp_path):
        """Create sample images for testing."""
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(20):
            img = Image.new("RGB", (256, 256), color=(i * 10 % 256, 100, 150))
            img.save(str(img_dir / f"img_{i}.jpg"))
        return img_dir

    def test_dataset_length(self, sample_manifest, image_dir):
        """Test dataset returns correct length per split."""
        from src.data.tokenization import TextTokenizer
        from src.data.dataset import WITDataset

        tokenizer = TextTokenizer.__new__(TextTokenizer)
        tokenizer.tokenizer = MagicMock()
        tokenizer.max_length = 64
        tokenizer.padding = "max_length"
        tokenizer.truncation = True

        # Mock tokenizer call to return proper tensors
        mock_output = {
            "input_ids": torch.ones(1, 64, dtype=torch.long),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }
        tokenizer.__call__ = MagicMock(return_value=mock_output)

        dataset = WITDataset(
            manifest_path=sample_manifest,
            image_root=image_dir,
            tokenizer=tokenizer,
            split="train",
        )
        assert len(dataset) == 14

    def test_dataset_item_structure(self, sample_manifest, image_dir):
        """Test that __getitem__ returns expected keys."""
        from src.data.tokenization import TextTokenizer
        from src.data.dataset import WITDataset

        tokenizer = TextTokenizer.__new__(TextTokenizer)
        tokenizer.tokenizer = MagicMock()
        tokenizer.max_length = 64
        tokenizer.padding = "max_length"
        tokenizer.truncation = True

        mock_output = {
            "input_ids": torch.ones(1, 64, dtype=torch.long),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }
        tokenizer.__call__ = MagicMock(return_value=mock_output)

        dataset = WITDataset(
            manifest_path=sample_manifest,
            image_root=image_dir,
            tokenizer=tokenizer,
            split="train",
        )

        item = dataset[0]
        assert "image" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "sample_id" in item
        assert "language" in item

    def test_language_distribution(self, sample_manifest, image_dir):
        """Test language distribution reporting."""
        from src.data.tokenization import TextTokenizer
        from src.data.dataset import WITDataset

        tokenizer = TextTokenizer.__new__(TextTokenizer)
        tokenizer.tokenizer = MagicMock()
        tokenizer.max_length = 64
        tokenizer.padding = "max_length"
        tokenizer.truncation = True

        mock_output = {
            "input_ids": torch.ones(1, 64, dtype=torch.long),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }
        tokenizer.__call__ = MagicMock(return_value=mock_output)

        dataset = WITDataset(
            manifest_path=sample_manifest,
            image_root=image_dir,
            tokenizer=tokenizer,
            split="train",
        )

        lang_dist = dataset.get_language_distribution()
        assert isinstance(lang_dist, dict)
        assert sum(lang_dist.values()) == 14
