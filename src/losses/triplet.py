"""Triplet loss for baseline dual encoder training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet margin loss for baseline contrastive training.

    For each anchor, selects the hardest positive and hardest negative
    within the batch (online hard mining variant).

    Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))

    Args:
        margin: Minimum desired gap between positive and negative distances.
        distance: Distance function ('cosine' or 'euclidean').
    """

    def __init__(
        self,
        margin: float = 0.2,
        distance: str = "cosine",
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute triplet loss with in-batch hard mining.

        Each image is anchored with its paired text as positive.
        The hardest negative is the most similar non-matching text.

        Args:
            image_embeddings: (B, D) L2-normalized image embeddings.
            text_embeddings: (B, D) L2-normalized text embeddings.

        Returns:
            Dictionary with 'loss', 'avg_positive_dist', 'avg_negative_dist'.
        """
        batch_size = image_embeddings.shape[0]

        if self.distance == "cosine":
            # Cosine distance = 1 - cosine_similarity
            sim_matrix = image_embeddings @ text_embeddings.T
            dist_matrix = 1.0 - sim_matrix
        else:
            # Euclidean distance
            dist_matrix = torch.cdist(image_embeddings, text_embeddings, p=2)

        # Positive distances (diagonal elements)
        positive_dist = dist_matrix.diag()

        # Hard negative mining: for each image, find most similar non-matching text
        mask = torch.eye(batch_size, device=dist_matrix.device).bool()
        dist_matrix_neg = dist_matrix.clone()
        dist_matrix_neg[mask] = float("inf")  # Exclude positives
        hardest_negative_dist, _ = dist_matrix_neg.min(dim=1)

        # Triplet loss
        losses = F.relu(positive_dist - hardest_negative_dist + self.margin)
        loss = losses.mean()

        return {
            "loss": loss,
            "avg_positive_dist": positive_dist.mean(),
            "avg_negative_dist": hardest_negative_dist.mean(),
        }
