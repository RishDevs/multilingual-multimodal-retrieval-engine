"""Hard negative mining utilities for contrastive learning."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class HardNegativeMiner:
    """In-batch hard negative mining for contrastive learning.

    Identifies the most confusing negative samples within each batch
    to focus training on the decision boundary.

    Strategies:
        - in_batch_topk_semantic: Select top-k most similar non-matching pairs
        - in_batch_random: Randomly select k negatives per anchor
        - language_aware: Prefer negatives from the same language (harder)

    Args:
        strategy: Mining strategy name.
        topk: Number of hard negatives to select per anchor.
        start_epoch: Epoch to begin hard negative mining (warmup period).
    """

    def __init__(
        self,
        strategy: str = "in_batch_topk_semantic",
        topk: int = 5,
        start_epoch: int = 3,
    ):
        self.strategy = strategy
        self.topk = topk
        self.start_epoch = start_epoch

    def mine(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        current_epoch: int,
        languages: Optional[list[str]] = None,
    ) -> Optional[torch.Tensor]:
        """Mine hard negatives and return a binary mask.

        Args:
            image_embeddings: (B, D) image embeddings.
            text_embeddings: (B, D) text embeddings.
            current_epoch: Current training epoch.
            languages: Optional list of language codes per sample.

        Returns:
            Boolean mask of shape (B, B) marking hard negative positions,
            or None if mining is not active yet.
        """
        if current_epoch < self.start_epoch:
            return None

        if self.strategy == "in_batch_topk_semantic":
            return self._mine_topk_semantic(image_embeddings, text_embeddings)
        elif self.strategy == "in_batch_random":
            return self._mine_random(image_embeddings.shape[0], image_embeddings.device)
        elif self.strategy == "language_aware":
            return self._mine_language_aware(
                image_embeddings, text_embeddings, languages
            )
        else:
            raise ValueError(f"Unknown mining strategy: {self.strategy}")

    def _mine_topk_semantic(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Select top-k most similar non-matching pairs as hard negatives.

        For each image, finds the k most similar texts that are NOT the
        correct match — these are the hardest negatives to distinguish from.

        Returns:
            Boolean mask (B, B) with True at hard negative positions.
        """
        batch_size = image_embeddings.shape[0]

        # Compute similarity matrix
        with torch.no_grad():
            sim_matrix = image_embeddings @ text_embeddings.T

            # Mask out positive pairs (diagonal)
            mask = torch.eye(batch_size, device=sim_matrix.device).bool()
            sim_matrix[mask] = float("-inf")

            # Find top-k hardest negatives for each image
            _, topk_indices = sim_matrix.topk(
                min(self.topk, batch_size - 1), dim=1
            )

            # Create hard negative mask
            hard_neg_mask = torch.zeros(
                batch_size, batch_size,
                dtype=torch.bool, device=sim_matrix.device,
            )
            for i in range(batch_size):
                hard_neg_mask[i, topk_indices[i]] = True

        return hard_neg_mask

    def _mine_random(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Randomly select k negatives per anchor."""
        hard_neg_mask = torch.zeros(
            batch_size, batch_size,
            dtype=torch.bool, device=device,
        )
        for i in range(batch_size):
            neg_indices = list(range(batch_size))
            neg_indices.remove(i)
            import random
            selected = random.sample(neg_indices, min(self.topk, len(neg_indices)))
            hard_neg_mask[i, selected] = True

        return hard_neg_mask

    def _mine_language_aware(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        languages: Optional[list[str]],
    ) -> torch.Tensor:
        """Prefer hard negatives from the same language.

        Same-language negatives are typically harder because they share
        linguistic features with the positive text.
        """
        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device

        if languages is None:
            return self._mine_topk_semantic(image_embeddings, text_embeddings)

        with torch.no_grad():
            sim_matrix = image_embeddings @ text_embeddings.T
            mask = torch.eye(batch_size, device=device).bool()
            sim_matrix[mask] = float("-inf")

            # Build same-language preference mask
            lang_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and languages[i] == languages[j]:
                        lang_mask[i, j] = True

            # Boost same-language similarities for mining
            boosted_sim = sim_matrix.clone()
            boosted_sim[lang_mask] += 1.0  # Prefer same-language negatives

            _, topk_indices = boosted_sim.topk(
                min(self.topk, batch_size - 1), dim=1
            )

            hard_neg_mask = torch.zeros(
                batch_size, batch_size,
                dtype=torch.bool, device=device,
            )
            for i in range(batch_size):
                hard_neg_mask[i, topk_indices[i]] = True

        return hard_neg_mask


def build_hard_negative_miner(cfg: dict) -> Optional[HardNegativeMiner]:
    """Build a hard negative miner from config.

    Args:
        cfg: Hard negative mining config dict.

    Returns:
        HardNegativeMiner if enabled, else None.
    """
    if not cfg.get("enabled", False):
        return None

    return HardNegativeMiner(
        strategy=cfg.get("strategy", "in_batch_topk_semantic"),
        topk=cfg.get("topk", 5),
        start_epoch=cfg.get("start_epoch", 3),
    )
