"""Optional cross-encoder reranker for top-k refinement."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker(nn.Module):
    """Cross-encoder reranker for refining top-k retrieval results.

    Unlike the dual encoder which encodes queries and candidates independently,
    the cross-encoder jointly encodes query-candidate pairs, enabling richer
    interaction modeling at the cost of higher latency.

    Typically used to rerank the top-k (e.g., top-50) candidates from the
    fast dual encoder retrieval stage.

    Args:
        model_name: HuggingFace cross-encoder model name.
        max_length: Maximum sequence length for paired input.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 128,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def score_pairs(
        self,
        queries: List[str],
        candidates: List[str],
    ) -> torch.Tensor:
        """Score query-candidate pairs for relevance.

        Args:
            queries: List of query texts.
            candidates: List of candidate texts (same length as queries).

        Returns:
            Relevance scores tensor of shape (N,).
        """
        assert len(queries) == len(candidates), "Queries and candidates must be same length."

        encoded = self.tokenizer(
            queries,
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            scores = outputs.logits.squeeze(-1)

        return scores

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Rerank candidates for a single query.

        Args:
            query: Query text.
            candidates: List of candidate texts to rerank.
            top_k: Number of top results to return.

        Returns:
            List of (original_index, score) tuples, sorted by relevance.
        """
        queries = [query] * len(candidates)
        scores = self.score_pairs(queries, candidates)

        # Sort by score descending
        sorted_indices = scores.argsort(descending=True)
        results = [
            (idx.item(), scores[idx].item())
            for idx in sorted_indices[:top_k]
        ]

        return results
