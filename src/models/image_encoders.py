"""Image encoder wrappers (ResNet, ViT) using timm."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Configurable image encoder backed by timm.

    Supports ResNet, ViT, and any timm model. Extracts feature vectors
    from images for the dual encoder pipeline.

    Args:
        model_name: timm model identifier (e.g., 'resnet50', 'vit_base_patch16_224').
        pretrained: Whether to load pretrained weights.
        output_dim: Expected output feature dimension (for verification).
        trainable_layers: 'all', 'none', or number of last layers to unfreeze.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        output_dim: Optional[int] = None,
        trainable_layers: str = "all",
    ):
        super().__init__()
        self.model_name = model_name

        # Create model with no classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head → feature extractor
        )

        # Determine actual output dimension
        self.output_dim = self.backbone.num_features

        # Handle layer freezing
        self._configure_trainable_layers(trainable_layers)

    def _configure_trainable_layers(self, trainable_layers: str) -> None:
        """Configure which layers are trainable."""
        if trainable_layers == "none":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif trainable_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif trainable_layers.isdigit():
            # Freeze all, then unfreeze last N layers
            n = int(trainable_layers)
            for param in self.backbone.parameters():
                param.requires_grad = False
            params = list(self.backbone.parameters())
            for param in params[-n:]:
                param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            Feature tensor of shape (B, output_dim).
        """
        return self.backbone(images)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def build_image_encoder(cfg: dict) -> ImageEncoder:
    """Factory function to build an image encoder from config.

    Args:
        cfg: Dictionary with keys: name, pretrained, trainable_layers, output_dim.

    Returns:
        Configured ImageEncoder.
    """
    return ImageEncoder(
        model_name=cfg.get("name", "vit_base_patch16_224"),
        pretrained=cfg.get("pretrained", True),
        output_dim=cfg.get("output_dim", None),
        trainable_layers=str(cfg.get("trainable_layers", "all")),
    )
