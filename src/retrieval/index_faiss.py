"""FAISS index builders for ANN retrieval."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import faiss
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("faiss_index")


class FAISSIndexBuilder:
    """Builds and manages FAISS indexes for approximate nearest neighbor search.

    Supports:
        - IndexFlatIP: Exact inner product search (baseline)
        - IVF_PQ: Inverted file with product quantization (scalable)
        - HNSW: Hierarchical navigable small world graph (fast)

    Args:
        index_type: Type of FAISS index.
        embedding_dim: Dimension of embedding vectors.
        normalize: Whether to L2-normalize vectors before indexing.
    """

    def __init__(
        self,
        index_type: str = "IndexFlatIP",
        embedding_dim: int = 512,
        normalize: bool = True,
        **kwargs,
    ):
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.kwargs = kwargs
        self.index: Optional[faiss.Index] = None
        self.build_time: float = 0.0

    def build(
        self,
        embeddings: np.ndarray,
        train_size: Optional[int] = None,
    ) -> faiss.Index:
        """Build a FAISS index from embedding vectors.

        Args:
            embeddings: Array of shape (N, D).
            train_size: Number of vectors to use for index training
                (for IVF/PQ indexes; if None, uses all).

        Returns:
            Built FAISS index.
        """
        if self.normalize:
            embeddings = self._normalize(embeddings)

        start_time = time.time()

        if self.index_type == "IndexFlatIP":
            self.index = self._build_flat_ip(embeddings)
        elif self.index_type == "IVF_PQ":
            self.index = self._build_ivf_pq(embeddings, train_size)
        elif self.index_type == "HNSW":
            self.index = self._build_hnsw(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.build_time = time.time() - start_time
        logger.info(
            f"Built {self.index_type} index: "
            f"{self.index.ntotal} vectors, "
            f"dim={self.embedding_dim}, "
            f"build_time={self.build_time:.2f}s"
        )

        return self.index

    def _build_flat_ip(self, embeddings: np.ndarray) -> faiss.Index:
        """Build exact inner product search index."""
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings.astype(np.float32))
        return index

    def _build_ivf_pq(
        self,
        embeddings: np.ndarray,
        train_size: Optional[int] = None,
    ) -> faiss.Index:
        """Build IVF-PQ approximate index."""
        nlist = self.kwargs.get("nlist", 256)
        m = self.kwargs.get("m", 32)
        nbits = self.kwargs.get("nbits", 8)
        nprobe = self.kwargs.get("nprobe", 16)

        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFPQ(
            quantizer, self.embedding_dim, nlist, m, nbits
        )

        # Train the index
        train_data = embeddings.astype(np.float32)
        if train_size and train_size < len(train_data):
            indices = np.random.choice(len(train_data), train_size, replace=False)
            train_data_subset = train_data[indices]
        else:
            train_data_subset = train_data

        index.train(train_data_subset)
        index.add(train_data)
        index.nprobe = nprobe

        return index

    def _build_hnsw(self, embeddings: np.ndarray) -> faiss.Index:
        """Build HNSW graph-based index."""
        M = self.kwargs.get("M", 32)
        ef_construction = self.kwargs.get("ef_construction", 200)
        ef_search = self.kwargs.get("ef_search", 128)

        index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search

        index.add(embeddings.astype(np.float32))

        return index

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embedding vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        return embeddings / norms

    def save(self, path: str | Path) -> None:
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save. Build first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info(f"Saved index to {path}")

    def load(self, path: str | Path) -> faiss.Index:
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(str(path))
        logger.info(f"Loaded index from {path}: {self.index.ntotal} vectors")
        return self.index

    def get_index_info(self) -> Dict[str, Any]:
        """Return index metadata."""
        info = {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "normalize": self.normalize,
            "build_time_s": self.build_time,
        }
        if self.index is not None:
            info["num_vectors"] = self.index.ntotal
        return info


def build_faiss_index(cfg: dict, embeddings: np.ndarray) -> FAISSIndexBuilder:
    """Factory function to build a FAISS index from config.

    Args:
        cfg: FAISS config dict.
        embeddings: Embedding vectors to index.

    Returns:
        Built FAISSIndexBuilder.
    """
    builder = FAISSIndexBuilder(
        index_type=cfg.get("index_type", "IndexFlatIP"),
        embedding_dim=cfg.get("embedding_dim", 512),
        normalize=cfg.get("normalize", True),
        **{k: v for k, v in cfg.items()
           if k not in ("index_type", "embedding_dim", "normalize",
                        "index_path", "metadata_path")},
    )
    builder.build(embeddings, train_size=cfg.get("train_size"))
    return builder
