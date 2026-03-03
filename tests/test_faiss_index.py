"""Tests for FAISS index building and search."""

import pytest
import numpy as np

from src.retrieval.index_faiss import FAISSIndexBuilder
from src.retrieval.search import ANNSearcher


class TestFAISSIndex:
    """Test FAISS index builders."""

    @pytest.fixture
    def sample_embeddings(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(1000, 128).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    @pytest.fixture
    def sample_ids(self):
        return [f"sample_{i}" for i in range(1000)]

    def test_flat_ip_build(self, sample_embeddings):
        builder = FAISSIndexBuilder(
            index_type="IndexFlatIP",
            embedding_dim=128,
            normalize=False,
        )
        index = builder.build(sample_embeddings)
        assert index.ntotal == 1000

    def test_hnsw_build(self, sample_embeddings):
        builder = FAISSIndexBuilder(
            index_type="HNSW",
            embedding_dim=128,
            normalize=False,
            M=16, ef_construction=100, ef_search=64,
        )
        index = builder.build(sample_embeddings)
        assert index.ntotal == 1000

    def test_save_load(self, sample_embeddings, tmp_path):
        builder = FAISSIndexBuilder(
            index_type="IndexFlatIP",
            embedding_dim=128,
            normalize=False,
        )
        builder.build(sample_embeddings)

        path = tmp_path / "test.index"
        builder.save(path)

        # Load into new builder
        builder2 = FAISSIndexBuilder(embedding_dim=128)
        index = builder2.load(path)
        assert index.ntotal == 1000

    def test_index_info(self, sample_embeddings):
        builder = FAISSIndexBuilder(
            index_type="IndexFlatIP",
            embedding_dim=128,
            normalize=False,
        )
        builder.build(sample_embeddings)
        info = builder.get_index_info()

        assert info["index_type"] == "IndexFlatIP"
        assert info["num_vectors"] == 1000
        assert info["build_time_s"] > 0


class TestANNSearcher:
    """Test ANN search interface."""

    @pytest.fixture
    def searcher(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(500, 64).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        builder = FAISSIndexBuilder(
            index_type="IndexFlatIP",
            embedding_dim=64,
            normalize=False,
        )
        index = builder.build(embeddings)
        ids = [f"id_{i}" for i in range(500)]

        return ANNSearcher(index=index, ids=ids), embeddings

    def test_search_returns_correct_shape(self, searcher):
        searcher_obj, embeddings = searcher
        query = embeddings[0:1]
        scores, indices, latency = searcher_obj.search(query, top_k=10)

        assert scores.shape == (1, 10)
        assert indices.shape == (1, 10)
        assert latency > 0

    def test_self_search(self, searcher):
        """A vector should be its own nearest neighbor."""
        searcher_obj, embeddings = searcher
        query = embeddings[42:43]
        scores, indices, _ = searcher_obj.search(query, top_k=1)

        assert indices[0, 0] == 42

    def test_search_with_metadata(self, searcher):
        searcher_obj, embeddings = searcher
        query = embeddings[0:1]
        results = searcher_obj.search_with_metadata(query, top_k=5)

        assert len(results) == 1
        assert len(results[0]) == 5
        assert "id" in results[0][0]
        assert "score" in results[0][0]
        assert "rank" in results[0][0]

    def test_benchmark_latency(self, searcher):
        searcher_obj, embeddings = searcher
        stats = searcher_obj.benchmark_latency(
            embeddings,
            top_k=10,
            num_warmup=10,
            num_queries=50,
        )

        assert "mean_latency_ms" in stats
        assert "p95_latency_ms" in stats
        assert "qps" in stats
        assert stats["qps"] > 0
