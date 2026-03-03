"""End-to-end smoke test for the full pipeline."""

import pytest
import torch
import numpy as np


class TestEndToEndSmoke:
    """Smoke tests to verify the pipeline works end to end."""

    def test_model_builds_from_config(self):
        """Test building a dual encoder from config dict."""
        from src.models.dual_encoder import build_dual_encoder

        cfg = {
            "image_encoder": {
                "name": "resnet18",
                "pretrained": False,
                "trainable_layers": "all",
            },
            "text_encoder": {
                "name": "xlm-roberta-base",
                "pretrained": False,
                "pooling": "cls",
            },
            "projection_head": {
                "embedding_dim": 256,
                "hidden_dim": 512,
                "dropout": 0.1,
                "l2_normalize": True,
            },
            "temperature": {
                "init_value": 0.07,
                "learnable": True,
            },
        }

        model = build_dual_encoder(cfg)
        assert model is not None
        assert model.get_model_info()["trainable_params"] > 0

    def test_loss_backward_pass(self):
        """Test that loss computation produces gradients."""
        from src.losses.contrastive import CLIPContrastiveLoss

        loss_fn = CLIPContrastiveLoss()

        logits = torch.randn(4, 4, requires_grad=True)
        result = loss_fn(logits, logits.T)
        result["loss"].backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_hard_negative_miner(self):
        """Test hard negative mining produces valid masks."""
        from src.losses.hard_negative import HardNegativeMiner

        miner = HardNegativeMiner(strategy="in_batch_topk_semantic", topk=3, start_epoch=0)
        img_emb = torch.randn(8, 256)
        txt_emb = torch.randn(8, 256)

        mask = miner.mine(img_emb, txt_emb, current_epoch=0)

        assert mask is not None
        assert mask.shape == (8, 8)
        assert mask.dtype == torch.bool
        # Each row should have at most topk=3 True values
        assert mask.sum(dim=1).max().item() <= 3

    def test_retrieval_metrics(self):
        """Test retrieval metrics on simple case."""
        from src.evaluation.metrics_retrieval import compute_all_retrieval_metrics

        # Near-perfect similarity (identity + noise)
        sim = np.eye(20) * 10.0 + np.random.randn(20, 20) * 0.1
        results = compute_all_retrieval_metrics(sim, ks=[1, 5, 10])

        assert results["text_to_image"]["r@1"] > 0.9
        assert results["image_to_text"]["r@1"] > 0.9

    def test_faiss_roundtrip(self):
        """Test FAISS index build + search roundtrip."""
        from src.retrieval.index_faiss import FAISSIndexBuilder
        from src.retrieval.search import ANNSearcher

        rng = np.random.RandomState(42)
        embeddings = rng.randn(100, 64).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        builder = FAISSIndexBuilder(
            index_type="IndexFlatIP",
            embedding_dim=64,
            normalize=False,
        )
        index = builder.build(embeddings)

        ids = [str(i) for i in range(100)]
        searcher = ANNSearcher(index=index, ids=ids)

        # Search for a known vector
        _, indices, _ = searcher.search(embeddings[0:1], top_k=1)
        assert indices[0, 0] == 0

    def test_config_load_merge(self):
        """Test config loading and merging."""
        from src.utils.config import load_config
        import tempfile, yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            yaml.dump({"project": {"name": "test", "seed": 42}}, f1)
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            yaml.dump({"train": {"epochs": 10}}, f2)
            path2 = f2.name

        cfg = load_config([path1, path2])
        assert cfg["project"]["name"] == "test"
        assert cfg["train"]["epochs"] == 10

    def test_data_splits_no_leakage(self):
        """Test that split creation prevents leakage."""
        import pandas as pd
        from src.data.splits import create_splits, verify_no_leakage

        df = pd.DataFrame({
            "sample_id": range(100),
            "image_id": [f"img_{i // 2}" for i in range(100)],
            "language": ["en"] * 50 + ["fr"] * 50,
        })

        df = create_splits(df, split_column="image_id", seed=42)
        assert verify_no_leakage(df, split_column="image_id")

    def test_early_stopping(self):
        """Test early stopping callback."""
        from src.training.callbacks import EarlyStopping

        es = EarlyStopping(patience=3, metric="val_loss", mode="min")

        # Improving
        assert not es({"val_loss": 1.0})
        assert not es({"val_loss": 0.9})
        assert not es({"val_loss": 0.8})

        # Not improving
        assert not es({"val_loss": 0.85})
        assert not es({"val_loss": 0.82})
        assert es({"val_loss": 0.81})  # Should trigger after patience=3
