"""Text tokenization wrapper for multilingual models."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


class TextTokenizer:
    """Wrapper around HuggingFace tokenizer for consistent text processing.

    Handles tokenization for multilingual models (XLM-R, MiniLM, etc.)
    with configurable max length and padding strategy.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 64,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """Initialize the tokenizer.

        Args:
            model_name: HuggingFace model name/path.
            max_length: Maximum token sequence length.
            padding: Padding strategy ('max_length', 'longest', or False).
            truncation: Whether to truncate sequences exceeding max_length.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.model_name = model_name

    def __call__(
        self,
        texts: str | List[str],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text(s) and return model-ready tensors.

        Args:
            texts: Single text string or list of strings.
            return_tensors: Output tensor format ('pt' for PyTorch).

        Returns:
            Dictionary with 'input_ids', 'attention_mask', etc.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode token IDs back to text strings."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        """Return pad token ID."""
        return self.tokenizer.pad_token_id
