"""Learning rate scheduler factory."""

from __future__ import annotations

import math
from typing import Optional

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    StepLR,
    _LRScheduler,
)


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    num_training_steps: Optional[int] = None,
    num_warmup_steps: int = 0,
    **kwargs,
) -> _LRScheduler:
    """Build a learning rate scheduler with optional warmup.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_name: One of 'cosine', 'linear', 'step', 'cosine_warmup'.
        num_training_steps: Total training steps (required for cosine/linear).
        num_warmup_steps: Number of warmup steps.
        **kwargs: Additional scheduler-specific parameters.

    Returns:
        Configured LR scheduler.
    """
    if scheduler_name == "cosine" and num_warmup_steps > 0:
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps or 10000,
        )
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps or 10000,
            eta_min=1e-7,
        )
    elif scheduler_name == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=num_training_steps or 10000,
        )
    elif scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 5),
            gamma=kwargs.get("gamma", 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """Create cosine LR schedule with linear warmup.

    Standard schedule used in CLIP and BERT-style training:
    - Linear warmup from 0 to base LR
    - Cosine decay from base LR to 0

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Steps for linear warmup.
        num_training_steps: Total training steps.
        num_cycles: Number of cosine annealing cycles.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)
