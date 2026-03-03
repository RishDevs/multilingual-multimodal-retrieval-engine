"""CLIP-style dual encoder model combining image and text encoders."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.image_encoders import ImageEncoder, build_image_encoder
from src.models.projection import ProjectionHead, build_projection_head
from src.models.text_encoders import TextEncoder, build_text_encoder


class DualEncoder(nn.Module):
    """CLIP-style dual encoder for multimodal image-text matching.

    Encodes images and texts into a shared embedding space using
    separate encoders + projection heads. Computes cross-modal
    similarity via dot product of L2-normalized embeddings.

    Architecture:
        Image: backbone → projection → L2 normalize
        Text:  backbone → projection → L2 normalize
        Similarity: image_emb @ text_emb.T / temperature

    Args:
        image_encoder: Image feature extractor.
        text_encoder: Text feature extractor.
        image_projection: Image-to-embedding projection head.
        text_projection: Text-to-embedding projection head.
        temperature_init: Initial temperature for similarity scaling.
        learnable_temperature: Whether temperature is a learnable parameter.
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        image_projection: ProjectionHead,
        text_projection: ProjectionHead,
        temperature_init: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.text_projection = text_projection

        # Learnable temperature parameter (log-scale for numerical stability)
        log_temp = torch.log(torch.tensor(temperature_init))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value."""
        return torch.exp(self.log_temperature)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to shared embedding space.

        Args:
            images: Image tensor of shape (B, 3, H, W).

        Returns:
            L2-normalized image embeddings of shape (B, embedding_dim).
        """
        features = self.image_encoder(images)
        return self.image_projection(features)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to shared embedding space.

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            L2-normalized text embeddings of shape (B, embedding_dim).
        """
        features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(features)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing image and text embeddings + similarity matrix.

        For a batch of N paired image-text samples:
        1. Encode all images → I ∈ R^(N×d)
        2. Encode all texts  → T ∈ R^(N×d)
        3. Compute similarity S = I·Tᵀ / temperature

        Args:
            images: Image tensor (B, 3, H, W).
            input_ids: Token IDs (B, L).
            attention_mask: Attention mask (B, L).

        Returns:
            Dictionary with:
                - image_embeddings: (B, d)
                - text_embeddings: (B, d)
                - logits_per_image: (B, B) similarity scores (image → text)
                - logits_per_text: (B, B) similarity scores (text → image)
                - temperature: current temperature scalar
        """
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)

        # Compute scaled similarity matrix
        logits_per_image = (image_embeddings @ text_embeddings.T) / self.temperature
        logits_per_text = logits_per_image.T

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "temperature": self.temperature,
        }

    def get_param_groups(
        self,
        lr_image: float = 1e-5,
        lr_text: float = 1e-5,
        lr_projection: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """Get parameter groups with separate learning rates.

        This enables differential learning rates for encoders vs projection heads,
        which is critical for fine-tuning pretrained models.

        Returns:
            List of parameter group dictionaries for the optimizer.
        """
        param_groups = [
            {
                "params": self.image_encoder.parameters(),
                "lr": lr_image,
                "weight_decay": weight_decay,
                "name": "image_encoder",
            },
            {
                "params": self.text_encoder.parameters(),
                "lr": lr_text,
                "weight_decay": weight_decay,
                "name": "text_encoder",
            },
            {
                "params": list(self.image_projection.parameters())
                + list(self.text_projection.parameters()),
                "lr": lr_projection,
                "weight_decay": weight_decay,
                "name": "projection_heads",
            },
        ]

        # Add temperature if learnable
        if isinstance(self.log_temperature, nn.Parameter):
            param_groups.append({
                "params": [self.log_temperature],
                "lr": lr_projection,
                "weight_decay": 0.0,
                "name": "temperature",
            })

        return param_groups

    def get_model_info(self) -> Dict[str, Any]:
        """Return model architecture summary."""
        return {
            "image_encoder": self.image_encoder.model_name,
            "image_encoder_params": self.image_encoder.get_total_params(),
            "image_encoder_trainable": self.image_encoder.get_trainable_params(),
            "text_encoder": self.text_encoder.model_name,
            "text_encoder_params": self.text_encoder.get_total_params(),
            "text_encoder_trainable": self.text_encoder.get_trainable_params(),
            "temperature": self.temperature.item(),
            "learnable_temperature": isinstance(self.log_temperature, nn.Parameter),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def build_dual_encoder(cfg: dict) -> DualEncoder:
    """Factory function to build a DualEncoder from config.

    Args:
        cfg: Model config dict with keys: image_encoder, text_encoder,
             projection_head, temperature.

    Returns:
        Configured DualEncoder model.
    """
    # Build encoders
    image_encoder = build_image_encoder(cfg["image_encoder"])
    text_encoder = build_text_encoder(cfg["text_encoder"])

    # Build projection heads
    image_projection = build_projection_head(
        input_dim=image_encoder.output_dim,
        cfg=cfg["projection_head"],
    )
    text_projection = build_projection_head(
        input_dim=text_encoder.output_dim,
        cfg=cfg["projection_head"],
    )

    # Build dual encoder
    temp_cfg = cfg.get("temperature", {})
    model = DualEncoder(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_projection=image_projection,
        text_projection=text_projection,
        temperature_init=temp_cfg.get("init_value", 0.07),
        learnable_temperature=temp_cfg.get("learnable", True),
    )

    return model
