"""System performance metrics: latency, QPS, memory."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import numpy as np


def measure_embedding_generation_time(
    generate_fn,
    dataloader,
    num_runs: int = 3,
) -> Dict[str, float]:
    """Measure embedding generation throughput.

    Args:
        generate_fn: Callable that generates embeddings from dataloader.
        dataloader: DataLoader to process.
        num_runs: Number of runs to average.

    Returns:
        Dictionary with timing statistics.
    """
    times = []
    total_samples = 0

    for _ in range(num_runs):
        start = time.time()
        result = generate_fn(dataloader)
        elapsed = time.time() - start
        times.append(elapsed)

        # Count samples from result
        if isinstance(result, tuple):
            total_samples = len(result[0])
        elif isinstance(result, dict):
            first_key = list(result.keys())[0]
            if isinstance(result[first_key], tuple):
                total_samples = len(result[first_key][0])

    mean_time = np.mean(times)
    return {
        "mean_time_s": float(mean_time),
        "std_time_s": float(np.std(times)),
        "samples_per_second": float(total_samples / mean_time) if mean_time > 0 else 0.0,
        "total_samples": total_samples,
        "num_runs": num_runs,
    }


def measure_index_build_time(
    build_fn,
    embeddings: np.ndarray,
    num_runs: int = 3,
) -> Dict[str, float]:
    """Measure FAISS index build time.

    Args:
        build_fn: Callable that builds the index from embeddings.
        embeddings: Embedding vectors.
        num_runs: Number of runs to average.

    Returns:
        Dictionary with timing statistics.
    """
    times = []

    for _ in range(num_runs):
        start = time.time()
        build_fn(embeddings)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "mean_build_time_s": float(np.mean(times)),
        "std_build_time_s": float(np.std(times)),
        "num_vectors": len(embeddings),
        "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else 0,
        "num_runs": num_runs,
    }


def get_memory_usage() -> Dict[str, float]:
    """Get current process memory usage.

    Returns:
        Dictionary with memory stats in GB.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_gb": mem_info.rss / 1e9,
            "vms_gb": mem_info.vms / 1e9,
        }
    except ImportError:
        return {"rss_gb": -1, "vms_gb": -1}


def estimate_index_memory(
    num_vectors: int,
    embedding_dim: int,
    dtype_bytes: int = 4,
) -> Dict[str, float]:
    """Estimate memory footprint of a FAISS index.

    Args:
        num_vectors: Number of indexed vectors.
        embedding_dim: Dimension of each vector.
        dtype_bytes: Bytes per element (4 for float32).

    Returns:
        Dictionary with memory estimates.
    """
    raw_bytes = num_vectors * embedding_dim * dtype_bytes
    return {
        "raw_vectors_gb": raw_bytes / 1e9,
        "raw_vectors_mb": raw_bytes / 1e6,
        "num_vectors": num_vectors,
        "embedding_dim": embedding_dim,
    }
