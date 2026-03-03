#!/usr/bin/env python3
"""
Generate and save embeddings from a trained model.

Usage:
    python scripts/generate_embeddings.py \
        --checkpoint outputs/best.ckpt \
        --config configs/model/vit_xlmr_clipstyle.yaml \
        --config configs/data/wit_subset.yaml \
        --output-dir data/artifacts/
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.collate import multimodal_collate_fn
from src.data.dataset import WITDataset
from src.data.tokenization import TextTokenizer
from src.models.dual_encoder import build_dual_encoder
from src.retrieval.embed import EmbeddingGenerator
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import save_embeddings
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="data/artifacts/")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    logger = setup_logger("generate_embeddings")

    device = get_device(cfg.get("device", {}).get("accelerator", "auto"))

    # Load model
    model_cfg = cfg.get("model", {})
    model = build_dual_encoder(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Dataset
    data_cfg = cfg.get("data", {})
    tokenizer = TextTokenizer(
        model_name=model_cfg.get("text_encoder", {}).get("name", "xlm-roberta-base"),
        max_length=data_cfg.get("max_text_len", 64),
    )

    dataset = WITDataset(
        manifest_path=data_cfg.get("subset_manifest"),
        image_root=data_cfg.get("image_root"),
        tokenizer=tokenizer,
        split=args.split,
        image_size=data_cfg.get("image_size", 224),
    )

    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("eval_batch_size", 256),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=multimodal_collate_fn,
    )

    # Generate
    generator = EmbeddingGenerator(model, device)
    result = generator.generate_all_embeddings(loader)

    # Save
    output_dir = Path(args.output_dir)
    save_embeddings(result["image"][0], result["image"][1], output_dir, prefix=f"{args.split}_image")
    save_embeddings(result["text"][0], result["text"][1], output_dir, prefix=f"{args.split}_text")

    logger.info(f"Saved embeddings to {output_dir}")
    logger.info(f"Image embeddings: {result['image'][0].shape}")
    logger.info(f"Text embeddings: {result['text'][0].shape}")


if __name__ == "__main__":
    main()
