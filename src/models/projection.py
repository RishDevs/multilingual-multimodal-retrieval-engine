"""MLP projection heads for mapping encoder outputs to shared embedding space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head that maps encoder features to shared embedding space.

    Architecture: Linear → GELU → Dropout → Linear → (optional L2 normalize)

    This projects modality-specific features into a common space where
    cosine similarity can be used for cross-modal matching.

    Args:
        input_dim: Dimension of the encoder output.
        embedding_dim: Dimension of the shared embedding space.
        hidden_dim: Dimension of the hidden layer.
        dropout: Dropout probability.
        l2_normalize: Whether to L2-normalize the output embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.l2_normalize = l2_normalize

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights with Xavier uniform."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to shared embedding space.

        Args:
            features: Encoder output of shape (B, input_dim).

        Returns:
            Projected embeddings of shape (B, embedding_dim).
        """
        projected = self.projection(features)

        if self.l2_normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected


def build_projection_head(input_dim: int, cfg: dict) -> ProjectionHead:
    """Factory function to build a projection head from config.

    Args:
        input_dim: Dimension of the encoder output.
        cfg: Dictionary with keys: embedding_dim, hidden_dim, dropout, l2_normalize.

    Returns:
        Configured ProjectionHead.
    """
    return ProjectionHead(
        input_dim=input_dim,
        embedding_dim=cfg.get("embedding_dim", 512),
        hidden_dim=cfg.get("hidden_dim", 1024),
        dropout=cfg.get("dropout", 0.1),
        l2_normalize=cfg.get("l2_normalize", True),
    )
