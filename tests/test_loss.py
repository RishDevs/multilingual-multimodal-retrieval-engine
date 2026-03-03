"""Tests for loss functions."""

import pytest
import torch


class TestCLIPContrastiveLoss:
    """Test CLIP-style contrastive loss."""

    def test_basic_forward(self):
        from src.losses.contrastive import CLIPContrastiveLoss

        loss_fn = CLIPContrastiveLoss()
        logits_i2t = torch.randn(8, 8)
        logits_t2i = logits_i2t.T

        result = loss_fn(logits_i2t, logits_t2i)

        assert "loss" in result
        assert "loss_i2t" in result
        assert "loss_t2i" in result
        assert result["loss"].item() > 0

    def test_perfect_alignment(self):
        from src.losses.contrastive import CLIPContrastiveLoss

        loss_fn = CLIPContrastiveLoss()
        # Create logits where diagonal is much higher
        logits = torch.eye(8) * 10.0 - 5.0
        result = loss_fn(logits, logits.T)

        # Loss should be low for perfect alignment
        assert result["loss"].item() < 1.0

    def test_with_label_smoothing(self):
        from src.losses.contrastive import CLIPContrastiveLoss

        loss_smooth = CLIPContrastiveLoss(label_smoothing=0.1)
        loss_no_smooth = CLIPContrastiveLoss(label_smoothing=0.0)

        logits = torch.randn(8, 8)
        r1 = loss_smooth(logits, logits.T)
        r2 = loss_no_smooth(logits, logits.T)

        # Both should produce valid losses
        assert r1["loss"].item() > 0
        assert r2["loss"].item() > 0

    def test_gradient_flow(self):
        from src.losses.contrastive import CLIPContrastiveLoss

        loss_fn = CLIPContrastiveLoss()
        logits = torch.randn(4, 4, requires_grad=True)
        result = loss_fn(logits, logits.T)
        result["loss"].backward()

        assert logits.grad is not None


class TestTripletLoss:
    """Test triplet loss."""

    def test_basic_forward(self):
        from src.losses.triplet import TripletLoss

        loss_fn = TripletLoss(margin=0.2)
        image_emb = torch.randn(8, 512)
        text_emb = torch.randn(8, 512)

        # Normalize
        image_emb = torch.nn.functional.normalize(image_emb, dim=1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=1)

        result = loss_fn(image_emb, text_emb)

        assert "loss" in result
        assert result["loss"].item() >= 0


class TestContrastiveLossWithHardNegatives:
    """Test contrastive loss with hard negative support."""

    def test_without_mask(self):
        from src.losses.contrastive import ContrastiveLossWithHardNegatives

        loss_fn = ContrastiveLossWithHardNegatives()
        logits = torch.randn(8, 8)
        result = loss_fn(logits, logits.T)

        assert result["loss"].item() > 0

    def test_with_mask(self):
        from src.losses.contrastive import ContrastiveLossWithHardNegatives

        loss_fn = ContrastiveLossWithHardNegatives()
        logits = torch.randn(8, 8)
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[0, 1] = True  # Mark one hard negative

        result = loss_fn(logits, logits.T, hard_negative_mask=mask)
        assert result["loss"].item() > 0
