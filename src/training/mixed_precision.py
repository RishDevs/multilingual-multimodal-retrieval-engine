"""Mixed precision training utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast


class MixedPrecisionManager:
    """Manages mixed precision (fp16/bf16) training.

    Handles autocast contexts and gradient scaling for stable
    mixed precision training on CUDA GPUs.

    Args:
        enabled: Whether to enable mixed precision.
        dtype: Precision type ('fp16' or 'bf16').
        device_type: Device type ('cuda', 'cpu').
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: str = "fp16",
        device_type: str = "cuda",
    ):
        self.enabled = enabled and device_type == "cuda"
        self.device_type = device_type

        if dtype == "fp16":
            self.torch_dtype = torch.float16
        elif dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
            self.enabled = False

        self.scaler = GradScaler(enabled=self.enabled) if self.enabled else None

    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision forward pass."""
        if self.enabled:
            with autocast(dtype=self.torch_dtype):
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with gradient unscaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
