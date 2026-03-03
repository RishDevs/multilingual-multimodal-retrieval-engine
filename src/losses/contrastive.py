"""Symmetric CLIP-style contrastive loss (InfoNCE)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPContrastiveLoss(nn.Module):
    """Symmetric CLIP-style InfoNCE contrastive loss.

    For a batch of N image-text pairs, treats diagonal elements as positives
    and all off-diagonal elements as in-batch negatives.

    Loss = 0.5 * (CE(logits_per_image, targets) + CE(logits_per_text, targets))

    This is the standard CLIP training objective that maximizes agreement
    between matching image-text pairs while pushing non-matching pairs apart.

    Args:
        label_smoothing: Label smoothing for cross-entropy (0.0 = no smoothing).
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute symmetric contrastive loss.

        Args:
            logits_per_image: Similarity matrix (B, B), image→text direction.
            logits_per_text: Similarity matrix (B, B), text→image direction.

        Returns:
            Dictionary with 'loss', 'loss_i2t', 'loss_t2i'.
        """
        batch_size = logits_per_image.shape[0]
        targets = torch.arange(batch_size, device=logits_per_image.device)

        # Image-to-Text loss: each image should match its paired text
        loss_i2t = F.cross_entropy(
            logits_per_image, targets, label_smoothing=self.label_smoothing
        )

        # Text-to-Image loss: each text should match its paired image
        loss_t2i = F.cross_entropy(
            logits_per_text, targets, label_smoothing=self.label_smoothing
        )

        # Symmetric loss
        total_loss = (loss_i2t + loss_t2i) / 2.0

        return {
            "loss": total_loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
        }


class ContrastiveLossWithHardNegatives(nn.Module):
    """CLIP contrastive loss with hard negative mining support.

    Extends the standard CLIP loss by allowing hard negative logit
    augmentation. Hard negatives (most confusing non-matching samples)
    are up-weighted in the similarity matrix.

    Args:
        label_smoothing: Label smoothing for cross-entropy.
        hard_negative_weight: Weight multiplier for hard negative logits.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
        hard_negative_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss with optional hard negative emphasis.

        Args:
            logits_per_image: Similarity matrix (B, B).
            logits_per_text: Similarity matrix (B, B).
            hard_negative_mask: Boolean mask (B, B) where True indicates
                a hard negative pair to up-weight.

        Returns:
            Dictionary with 'loss', 'loss_i2t', 'loss_t2i'.
        """
        batch_size = logits_per_image.shape[0]
        targets = torch.arange(batch_size, device=logits_per_image.device)

        # Apply hard negative weighting if mask provided
        if hard_negative_mask is not None:
            # Up-weight hard negatives in the logit matrix
            hn_boost = hard_negative_mask.float() * self.hard_negative_weight
            logits_i2t = logits_per_image + hn_boost
            logits_t2i = logits_per_text + hn_boost.T
        else:
            logits_i2t = logits_per_image
            logits_t2i = logits_per_text

        loss_i2t = F.cross_entropy(
            logits_i2t, targets, label_smoothing=self.label_smoothing
        )
        loss_t2i = F.cross_entropy(
            logits_t2i, targets, label_smoothing=self.label_smoothing
        )

        total_loss = (loss_i2t + loss_t2i) / 2.0

        return {
            "loss": total_loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
        }
