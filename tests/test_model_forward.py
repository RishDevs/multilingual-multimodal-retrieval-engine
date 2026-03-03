"""Tests for model forward pass."""

import pytest
import torch


class TestImageEncoder:
    """Test image encoder outputs."""

    def test_resnet_forward(self):
        from src.models.image_encoders import ImageEncoder

        encoder = ImageEncoder(model_name="resnet18", pretrained=False)
        images = torch.randn(2, 3, 224, 224)
        output = encoder(images)

        assert output.shape[0] == 2
        assert output.ndim == 2
        assert output.shape[1] == encoder.output_dim

    def test_vit_forward(self):
        from src.models.image_encoders import ImageEncoder

        encoder = ImageEncoder(model_name="vit_tiny_patch16_224", pretrained=False)
        images = torch.randn(2, 3, 224, 224)
        output = encoder(images)

        assert output.shape[0] == 2
        assert output.ndim == 2

    def test_layer_freezing(self):
        from src.models.image_encoders import ImageEncoder

        encoder = ImageEncoder(model_name="resnet18", pretrained=False, trainable_layers="none")
        assert encoder.get_trainable_params() == 0

        encoder_all = ImageEncoder(model_name="resnet18", pretrained=False, trainable_layers="all")
        assert encoder_all.get_trainable_params() > 0


class TestProjectionHead:
    """Test projection head."""

    def test_projection_shape(self):
        from src.models.projection import ProjectionHead

        proj = ProjectionHead(input_dim=768, embedding_dim=512, hidden_dim=1024)
        features = torch.randn(4, 768)
        output = proj(features)

        assert output.shape == (4, 512)

    def test_l2_normalization(self):
        from src.models.projection import ProjectionHead

        proj = ProjectionHead(input_dim=768, embedding_dim=512, l2_normalize=True)
        features = torch.randn(4, 768)
        output = proj(features)

        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestDualEncoder:
    """Test dual encoder model."""

    @pytest.fixture
    def dual_encoder(self):
        from src.models.image_encoders import ImageEncoder
        from src.models.text_encoders import TextEncoder
        from src.models.projection import ProjectionHead
        from src.models.dual_encoder import DualEncoder

        image_enc = ImageEncoder("resnet18", pretrained=False)
        # Use a tiny config for text encoder to avoid large downloads
        text_enc = TextEncoder.__new__(TextEncoder)
        text_enc.model_name = "test"
        text_enc.pooling = "cls"
        text_enc.output_dim = 768

        # Mock text encoder
        text_enc.backbone = torch.nn.Linear(64, 768)  # Simplified
        text_enc.forward = lambda input_ids, attention_mask: torch.randn(
            input_ids.shape[0], 768
        )
        text_enc.get_trainable_params = lambda: 0
        text_enc.get_total_params = lambda: 0

        img_proj = ProjectionHead(image_enc.output_dim, 512, 1024)
        txt_proj = ProjectionHead(768, 512, 1024)

        model = DualEncoder(
            image_encoder=image_enc,
            text_encoder=text_enc,
            image_projection=img_proj,
            text_projection=txt_proj,
        )
        return model

    def test_forward_pass(self, dual_encoder):
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 64))
        attention_mask = torch.ones(4, 64, dtype=torch.long)

        outputs = dual_encoder(images, input_ids, attention_mask)

        assert "image_embeddings" in outputs
        assert "text_embeddings" in outputs
        assert "logits_per_image" in outputs
        assert "logits_per_text" in outputs
        assert "temperature" in outputs

        assert outputs["image_embeddings"].shape == (4, 512)
        assert outputs["text_embeddings"].shape == (4, 512)
        assert outputs["logits_per_image"].shape == (4, 4)

    def test_encode_image(self, dual_encoder):
        images = torch.randn(2, 3, 224, 224)
        emb = dual_encoder.encode_image(images)
        assert emb.shape == (2, 512)

        # Check L2 normalization
        norms = torch.norm(emb, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_param_groups(self, dual_encoder):
        groups = dual_encoder.get_param_groups()
        assert len(groups) >= 3  # image, text, projection (+temperature)

    def test_model_info(self, dual_encoder):
        info = dual_encoder.get_model_info()
        assert "image_encoder" in info
        assert "total_params" in info
        assert "trainable_params" in info
        assert "temperature" in info
