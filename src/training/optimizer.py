"""Optimizer factory with support for differential learning rates."""

from __future__ import annotations

from typing import List

import torch.optim as optim


def build_optimizer(
    param_groups: List[dict],
    optimizer_name: str = "adamw",
    default_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> optim.Optimizer:
    """Build optimizer with differential learning rate support.

    Args:
        param_groups: List of parameter group dicts from model.get_param_groups().
        optimizer_name: One of 'adamw', 'adam', 'sgd'.
        default_lr: Default learning rate (used if group doesn't specify lr).
        weight_decay: Default weight decay.

    Returns:
        Configured optimizer.
    """
    # Ensure each group has lr and weight_decay
    for group in param_groups:
        if "lr" not in group:
            group["lr"] = default_lr
        if "weight_decay" not in group:
            group["weight_decay"] = weight_decay

    if optimizer_name.lower() == "adamw":
        return optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
