"""Batch embedding generation from trained dual encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dual_encoder import DualEncoder


class EmbeddingGenerator:
    """Generate embeddings from a trained dual encoder model.

    Efficiently encodes a dataset into image and/or text embeddings
    for downstream indexing and retrieval.

    Args:
        model: Trained DualEncoder.
        device: Torch device for inference.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model: DualEncoder,
        device: torch.device,
        batch_size: int = 256,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def generate_image_embeddings(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate image embeddings for all samples in dataloader.

        Args:
            dataloader: DataLoader yielding batches with 'images' and 'sample_ids'.

        Returns:
            Tuple of (embeddings_array [N, D], sample_ids_list).
        """
        all_embeddings = []
        all_ids = []

        for batch in tqdm(dataloader, desc="Encoding images"):
            images = batch["images"].to(self.device)
            embeddings = self.model.encode_image(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(batch["sample_ids"])

        return np.concatenate(all_embeddings, axis=0), all_ids

    @torch.no_grad()
    def generate_text_embeddings(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate text embeddings for all samples in dataloader.

        Args:
            dataloader: DataLoader yielding batches with 'input_ids',
                'attention_mask', and 'sample_ids'.

        Returns:
            Tuple of (embeddings_array [N, D], sample_ids_list).
        """
        all_embeddings = []
        all_ids = []

        for batch in tqdm(dataloader, desc="Encoding texts"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            embeddings = self.model.encode_text(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(batch["sample_ids"])

        return np.concatenate(all_embeddings, axis=0), all_ids

    @torch.no_grad()
    def generate_all_embeddings(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """Generate both image and text embeddings.

        Returns:
            Dictionary with 'image' and 'text' keys, each containing
            (embeddings_array, sample_ids_list).
        """
        all_image_emb = []
        all_text_emb = []
        all_ids = []
        all_languages = []

        for batch in tqdm(dataloader, desc="Encoding all"):
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            image_emb = self.model.encode_image(images)
            text_emb = self.model.encode_text(input_ids, attention_mask)

            all_image_emb.append(image_emb.cpu().numpy())
            all_text_emb.append(text_emb.cpu().numpy())
            all_ids.extend(batch["sample_ids"])
            all_languages.extend(batch.get("languages", ["unknown"] * len(batch["sample_ids"])))

        return {
            "image": (np.concatenate(all_image_emb, axis=0), all_ids),
            "text": (np.concatenate(all_text_emb, axis=0), all_ids),
            "languages": all_languages,
        }
