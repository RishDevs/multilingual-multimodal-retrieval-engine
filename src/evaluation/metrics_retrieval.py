"""Core retrieval metrics: R@K, mAP, MRR, NDCG@K."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def recall_at_k(
    similarity_matrix: np.ndarray,
    ks: List[int] = [1, 5, 10],
    direction: str = "text_to_image",
) -> Dict[str, float]:
    """Compute Recall@K for retrieval.

    For paired data where sample i's match is at position i:
    - Text→Image: rows = text queries, cols = image candidates
    - Image→Text: rows = image queries, cols = text candidates

    Args:
        similarity_matrix: (N, N) similarity scores.
        ks: List of K values.
        direction: 'text_to_image' or 'image_to_text'.

    Returns:
        Dictionary mapping 'r@k' to recall value.
    """
    if direction == "image_to_text":
        sim = similarity_matrix
    else:
        sim = similarity_matrix.T

    n = sim.shape[0]
    targets = np.arange(n)
    results = {}

    for k in ks:
        topk_indices = np.argsort(-sim, axis=1)[:, :k]
        correct = np.any(topk_indices == targets[:, np.newaxis], axis=1)
        results[f"r@{k}"] = float(np.mean(correct))

    return results


def mean_average_precision(
    similarity_matrix: np.ndarray,
    direction: str = "text_to_image",
) -> float:
    """Compute Mean Average Precision (mAP).

    For each query, computes Average Precision over the sorted result list.
    Assumes one relevant item per query (standard for paired retrieval).

    Args:
        similarity_matrix: (N, N) similarity scores.
        direction: 'text_to_image' or 'image_to_text'.

    Returns:
        mAP score.
    """
    if direction == "image_to_text":
        sim = similarity_matrix
    else:
        sim = similarity_matrix.T

    n = sim.shape[0]
    targets = np.arange(n)

    aps = []
    for i in range(n):
        sorted_indices = np.argsort(-sim[i])
        rank = np.where(sorted_indices == targets[i])[0][0] + 1  # 1-indexed rank
        ap = 1.0 / rank
        aps.append(ap)

    return float(np.mean(aps))


def mean_reciprocal_rank(
    similarity_matrix: np.ndarray,
    direction: str = "text_to_image",
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    MRR = mean(1/rank_of_correct_item).

    Args:
        similarity_matrix: (N, N) similarity scores.
        direction: 'text_to_image' or 'image_to_text'.

    Returns:
        MRR score.
    """
    # For single-relevant retrieval, MRR == mAP
    return mean_average_precision(similarity_matrix, direction)


def ndcg_at_k(
    similarity_matrix: np.ndarray,
    k: int = 10,
    direction: str = "text_to_image",
) -> float:
    """Compute normalized Discounted Cumulative Gain at K.

    For single-relevant-item retrieval, NDCG@K simplifies to:
    1/log2(rank+1) if the relevant item is in top-K, else 0.

    Args:
        similarity_matrix: (N, N) similarity scores.
        k: Cutoff position.
        direction: 'text_to_image' or 'image_to_text'.

    Returns:
        NDCG@K score.
    """
    if direction == "image_to_text":
        sim = similarity_matrix
    else:
        sim = similarity_matrix.T

    n = sim.shape[0]
    targets = np.arange(n)

    dcgs = []
    for i in range(n):
        sorted_indices = np.argsort(-sim[i])[:k]
        if targets[i] in sorted_indices:
            rank = np.where(sorted_indices == targets[i])[0][0] + 1
            dcgs.append(1.0 / np.log2(rank + 1))
        else:
            dcgs.append(0.0)

    # Ideal DCG is 1/log2(2) = 1.0 for single relevant item
    ideal_dcg = 1.0

    return float(np.mean(dcgs) / ideal_dcg)


def compute_all_retrieval_metrics(
    similarity_matrix: np.ndarray,
    ks: List[int] = [1, 5, 10],
    directions: List[str] = ["text_to_image", "image_to_text"],
) -> Dict[str, Dict[str, float]]:
    """Compute all retrieval metrics.

    Args:
        similarity_matrix: (N, N) similarity scores where sim[i][j]
            is image_i · text_j similarity.
        ks: List of K values for R@K and NDCG@K.
        directions: Retrieval directions to evaluate.

    Returns:
        Nested dictionary: {direction: {metric_name: value}}.
    """
    results = {}

    for direction in directions:
        dir_results = {}

        # Recall@K
        r_at_k = recall_at_k(similarity_matrix, ks, direction)
        dir_results.update(r_at_k)

        # mAP
        dir_results["map"] = mean_average_precision(similarity_matrix, direction)

        # MRR
        dir_results["mrr"] = mean_reciprocal_rank(similarity_matrix, direction)

        # NDCG@K
        for k in ks:
            dir_results[f"ndcg@{k}"] = ndcg_at_k(similarity_matrix, k, direction)

        results[direction] = dir_results

    return results
