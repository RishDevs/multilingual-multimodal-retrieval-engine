"""Post-processing utilities for retrieval results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_results(
    results: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Format retrieval results for display or output.

    Args:
        results: List of result dicts from ANNSearcher.
        include_fields: Optional list of fields to include.

    Returns:
        Filtered and formatted result list.
    """
    if include_fields is None:
        return results

    formatted = []
    for result in results:
        filtered = {k: v for k, v in result.items() if k in include_fields}
        formatted.append(filtered)

    return formatted


def deduplicate_results(
    results: List[Dict[str, Any]],
    id_field: str = "id",
) -> List[Dict[str, Any]]:
    """Remove duplicate results based on ID field.

    Args:
        results: List of result dicts.
        id_field: Field name to use for deduplication.

    Returns:
        Deduplicated results preserving original order.
    """
    seen = set()
    unique = []
    for result in results:
        rid = result.get(id_field)
        if rid not in seen:
            seen.add(rid)
            unique.append(result)
    return unique


def merge_reranked_results(
    initial_results: List[Dict[str, Any]],
    reranked_indices: List[tuple[int, float]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Merge reranking scores into initial retrieval results.

    Args:
        initial_results: Original retrieval results.
        reranked_indices: List of (original_index, rerank_score) tuples.
        top_k: Number of final results to return.

    Returns:
        Reranked results with updated scores and ranks.
    """
    reranked = []
    for rank, (orig_idx, rerank_score) in enumerate(reranked_indices[:top_k], start=1):
        if orig_idx < len(initial_results):
            result = initial_results[orig_idx].copy()
            result["original_score"] = result.get("score", 0.0)
            result["rerank_score"] = rerank_score
            result["score"] = rerank_score
            result["rank"] = rank
            reranked.append(result)

    return reranked
