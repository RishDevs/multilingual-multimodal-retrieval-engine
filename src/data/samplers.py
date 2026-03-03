"""Balanced and language-aware samplers for multilingual training."""

from __future__ import annotations

from collections import Counter
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class BalancedLanguageSampler(Sampler[int]):
    """Sampler that balances language distribution across batches.

    Upsamples under-represented languages and downsamples over-represented
    ones to maintain roughly equal language exposure during training.

    Args:
        languages: List of language codes for each sample.
        num_samples: Total samples to draw per epoch (default: len(languages)).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        languages: List[str],
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.languages = np.array(languages)
        self.num_samples = num_samples or len(languages)
        self.seed = seed
        self.epoch = 0

        # Build per-language index maps
        self.lang_to_indices: dict[str, np.ndarray] = {}
        for lang in np.unique(self.languages):
            self.lang_to_indices[lang] = np.where(self.languages == lang)[0]

        self.num_languages = len(self.lang_to_indices)
        self.samples_per_lang = self.num_samples // self.num_languages

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = []

        for lang, lang_indices in self.lang_to_indices.items():
            if len(lang_indices) >= self.samples_per_lang:
                chosen = rng.choice(lang_indices, size=self.samples_per_lang, replace=False)
            else:
                # Upsample small languages
                chosen = rng.choice(lang_indices, size=self.samples_per_lang, replace=True)
            indices.extend(chosen.tolist())

        # Fill remaining slots randomly
        remaining = self.num_samples - len(indices)
        if remaining > 0:
            all_indices = np.arange(len(self.languages))
            extra = rng.choice(all_indices, size=remaining, replace=False)
            indices.extend(extra.tolist())

        rng.shuffle(indices)
        return iter(indices[:self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling."""
        self.epoch = epoch


class SequentialSubsetSampler(Sampler[int]):
    """Sequentially samples from a subset of indices (for evaluation)."""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
