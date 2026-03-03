"""ANN search interface over FAISS indexes."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


class ANNSearcher:
    """Search interface for FAISS-indexed embeddings.

    Provides top-k nearest neighbor search with latency tracking.

    Args:
        index: Built FAISS index.
        ids: List of sample IDs corresponding to indexed vectors.
        metadata: Optional metadata dict for each sample.
    """

    def __init__(
        self,
        index: faiss.Index,
        ids: List[str],
        metadata: Optional[Dict[str, List[Any]]] = None,
    ):
        self.index = index
        self.ids = ids
        self.metadata = metadata or {}

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search for nearest neighbors.

        Args:
            query_embeddings: Query vectors of shape (Q, D).
            top_k: Number of nearest neighbors to return.

        Returns:
            Tuple of (scores [Q, K], indices [Q, K], latency_ms).
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        query_embeddings = query_embeddings.astype(np.float32)

        # Normalize queries if index expects normalized vectors
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        query_embeddings = query_embeddings / norms

        start_time = time.perf_counter()
        scores, indices = self.index.search(query_embeddings, top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return scores, indices, latency_ms

    def search_with_metadata(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """Search and return results with metadata.

        Args:
            query_embeddings: Query vectors of shape (Q, D).
            top_k: Number of results per query.

        Returns:
            List of lists of result dicts, each containing:
                - id: sample ID
                - score: similarity score
                - rank: 1-indexed rank
                - metadata fields (if available)
        """
        scores, indices, latency_ms = self.search(query_embeddings, top_k)
        results = []

        for q_idx in range(len(query_embeddings)):
            query_results = []
            for rank, (idx, score) in enumerate(
                zip(indices[q_idx], scores[q_idx]), start=1
            ):
                if idx == -1:
                    continue  # No more results

                result = {
                    "id": self.ids[idx] if idx < len(self.ids) else str(idx),
                    "score": float(score),
                    "rank": rank,
                }

                # Add metadata fields
                for key, values in self.metadata.items():
                    if idx < len(values):
                        result[key] = values[idx]

                query_results.append(result)

            results.append(query_results)

        return results

    def benchmark_latency(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        num_warmup: int = 100,
        num_queries: int = 1000,
        percentiles: List[int] = [50, 95, 99],
    ) -> Dict[str, float]:
        """Benchmark search latency.

        Args:
            query_embeddings: Pool of query vectors to sample from.
            top_k: Number of results per query.
            num_warmup: Warmup queries to exclude.
            num_queries: Number of queries to benchmark.
            percentiles: Latency percentiles to report.

        Returns:
            Dictionary with latency statistics.
        """
        # Warmup
        for _ in range(min(num_warmup, len(query_embeddings))):
            idx = np.random.randint(0, len(query_embeddings))
            self.search(query_embeddings[idx:idx+1], top_k)

        # Benchmark
        latencies = []
        for _ in range(num_queries):
            idx = np.random.randint(0, len(query_embeddings))
            _, _, latency_ms = self.search(query_embeddings[idx:idx+1], top_k)
            latencies.append(latency_ms)

        latencies = np.array(latencies)

        stats = {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "qps": float(1000.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0.0,
            "num_queries": num_queries,
            "top_k": top_k,
            "index_size": self.index.ntotal,
        }

        for p in percentiles:
            stats[f"p{p}_latency_ms"] = float(np.percentile(latencies, p))

        return stats
