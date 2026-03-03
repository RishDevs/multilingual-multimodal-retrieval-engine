"""Device selection and management utilities."""

from __future__ import annotations

import torch


def get_device(accelerator: str = "auto") -> torch.device:
    """Select the best available device.

    Args:
        accelerator: One of 'auto', 'cuda', 'mps', 'cpu'.

    Returns:
        torch.device for computation.
    """
    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    elif accelerator == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """Get information about the computing device.

    Returns:
        Dictionary with device metadata.
    """
    info = {
        "device_type": device.type,
        "device_index": device.index,
    }
    if device.type == "cuda":
        info.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_memory_total_gb": torch.cuda.get_device_properties(device).total_mem / 1e9,
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "cuda_version": torch.version.cuda,
        })
    return info
