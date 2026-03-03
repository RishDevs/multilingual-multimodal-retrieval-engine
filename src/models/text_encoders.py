"""Text encoder wrappers (XLM-R, MiniLM) using HuggingFace transformers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Configurable multilingual text encoder backed by HuggingFace transformers.

    Supports XLM-RoBERTa, multilingual MiniLM, and any HuggingFace model.
    Extracts text feature vectors using CLS pooling or mean pooling.

    Args:
        model_name: HuggingFace model identifier.
        pretrained: Whether to load pretrained weights.
        pooling: Pooling strategy ('cls' or 'mean').
        output_dim: Expected output dimension (for verification).
        trainable_layers: 'all', 'none', or number of last layers to unfreeze.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        pretrained: bool = True,
        pooling: str = "cls",
        output_dim: Optional[int] = None,
        trainable_layers: str = "all",
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        self.output_dim = self.backbone.config.hidden_size

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
            n = int(trainable_layers)
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last N encoder layers
            if hasattr(self.backbone, "encoder"):
                layers = list(self.backbone.encoder.layer)
                for layer in layers[-n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract text features.

        Args:
            input_ids: Token IDs, shape (B, L).
            attention_mask: Attention mask, shape (B, L).

        Returns:
            Feature tensor of shape (B, output_dim).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def build_text_encoder(cfg: dict) -> TextEncoder:
    """Factory function to build a text encoder from config.

    Args:
        cfg: Dictionary with keys: name, pretrained, pooling, output_dim, trainable_layers.

    Returns:
        Configured TextEncoder.
    """
    return TextEncoder(
        model_name=cfg.get("name", "xlm-roberta-base"),
        pretrained=cfg.get("pretrained", True),
        pooling=cfg.get("pooling", "cls"),
        output_dim=cfg.get("output_dim", None),
        trainable_layers=str(cfg.get("trainable_layers", "all")),
    )
