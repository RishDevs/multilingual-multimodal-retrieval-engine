"""Tests for retrieval metrics."""

import pytest
import numpy as np

from src.evaluation.metrics_retrieval import (
    compute_all_retrieval_metrics,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


class TestRecallAtK:
    """Test Recall@K computation."""

    def test_perfect_retrieval(self):
        """Perfect diagonal similarity should give R@1 = 1.0."""
        n = 100
        sim = np.eye(n) * 10.0 + np.random.randn(n, n) * 0.01
        result = recall_at_k(sim, ks=[1, 5, 10], direction="text_to_image")

        assert result["r@1"] == pytest.approx(1.0, abs=0.01)
        assert result["r@5"] == pytest.approx(1.0, abs=0.01)
        assert result["r@10"] == pytest.approx(1.0, abs=0.01)

    def test_random_retrieval(self):
        """Random similarity should give R@K ≈ K/N."""
        n = 100
        sim = np.random.randn(n, n)
        result = recall_at_k(sim, ks=[1, 10], direction="text_to_image")

        # R@1 should be close to 1/100 = 0.01
        assert result["r@1"] < 0.2
        # R@10 should be close to 10/100 = 0.10
        assert result["r@10"] < 0.5

    def test_both_directions(self):
        """Both directions should work."""
        sim = np.eye(10) * 10.0
        t2i = recall_at_k(sim, ks=[1], direction="text_to_image")
        i2t = recall_at_k(sim, ks=[1], direction="image_to_text")

        assert t2i["r@1"] == pytest.approx(1.0, abs=0.01)
        assert i2t["r@1"] == pytest.approx(1.0, abs=0.01)


class TestMAP:
    """Test Mean Average Precision."""

    def test_perfect_map(self):
        sim = np.eye(50) * 10.0 + np.random.randn(50, 50) * 0.01
        m = mean_average_precision(sim, direction="text_to_image")
        assert m == pytest.approx(1.0, abs=0.01)

    def test_map_range(self):
        sim = np.random.randn(50, 50)
        m = mean_average_precision(sim, direction="text_to_image")
        assert 0.0 <= m <= 1.0


class TestMRR:
    """Test Mean Reciprocal Rank."""

    def test_perfect_mrr(self):
        sim = np.eye(50) * 10.0 + np.random.randn(50, 50) * 0.01
        m = mean_reciprocal_rank(sim, direction="text_to_image")
        assert m == pytest.approx(1.0, abs=0.01)


class TestNDCG:
    """Test NDCG@K."""

    def test_perfect_ndcg(self):
        sim = np.eye(50) * 10.0 + np.random.randn(50, 50) * 0.01
        n = ndcg_at_k(sim, k=10, direction="text_to_image")
        assert n == pytest.approx(1.0, abs=0.01)

    def test_ndcg_range(self):
        sim = np.random.randn(50, 50)
        n = ndcg_at_k(sim, k=10, direction="text_to_image")
        assert 0.0 <= n <= 1.0


class TestComputeAll:
    """Test combined metric computation."""

    def test_compute_all(self):
        sim = np.random.randn(50, 50)
        results = compute_all_retrieval_metrics(sim, ks=[1, 5, 10])

        assert "text_to_image" in results
        assert "image_to_text" in results
        assert "r@1" in results["text_to_image"]
        assert "map" in results["text_to_image"]
        assert "mrr" in results["text_to_image"]
