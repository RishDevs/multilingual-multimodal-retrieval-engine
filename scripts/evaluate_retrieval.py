#!/usr/bin/env python3
"""
Evaluate retrieval performance of a trained model.

Computes R@K, mAP, MRR, NDCG@K for text→image and image→text,
with optional multilingual per-language breakdown.

Usage:
    python scripts/evaluate_retrieval.py \
        --config configs/eval/retrieval_eval.yaml \
        --model-config configs/model/vit_xlmr_clipstyle.yaml \
        --data-config configs/data/wit_subset.yaml \
        --checkpoint outputs/best.ckpt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.collate import multimodal_collate_fn
from src.data.dataset import WITDataset
from src.data.tokenization import TextTokenizer
from src.evaluation.metrics_retrieval import compute_all_retrieval_metrics
from src.evaluation.multilingual_eval import (
    compute_multilingual_summary,
    format_multilingual_table,
    per_language_retrieval_metrics,
)
from src.models.dual_encoder import build_dual_encoder
from src.retrieval.embed import EmbeddingGenerator
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import save_metrics
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--config", nargs="+", required=True, help="Config file(s)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    logger = setup_logger("evaluate")

    # ── Device ──
    device = get_device(cfg.get("device", {}).get("accelerator", "auto"))
    logger.info(f"Device: {device}")

    # ── Load model ──
    model = build_dual_encoder(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Tokenizer + Dataset ──
    text_encoder_name = model_cfg.get("text_encoder", {}).get("name", "xlm-roberta-base")
    tokenizer = TextTokenizer(model_name=text_encoder_name, max_length=data_cfg.get("max_text_len", 64))

    test_dataset = WITDataset(
        manifest_path=data_cfg.get("subset_manifest", "data/interim/wit_subset_manifest.parquet"),
        image_root=data_cfg.get("image_root", "data/raw/images/"),
        tokenizer=tokenizer,
        split="test",
        image_size=data_cfg.get("image_size", 224),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get("eval_batch_size", 256),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=multimodal_collate_fn,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # ── Generate embeddings ──
    generator = EmbeddingGenerator(model, device)
    result = generator.generate_all_embeddings(test_loader)

    image_embeddings, image_ids = result["image"]
    text_embeddings, text_ids = result["text"]
    languages = result["languages"]

    logger.info(f"Generated embeddings: images={image_embeddings.shape}, texts={text_embeddings.shape}")

    # ── Compute similarity matrix ──
    sim_matrix = image_embeddings @ text_embeddings.T

    # ── Retrieval metrics ──
    ks = eval_cfg.get("ks", [1, 5, 10])
    directions = eval_cfg.get("directions", ["text_to_image", "image_to_text"])

    metrics = compute_all_retrieval_metrics(sim_matrix, ks=ks, directions=directions)
    logger.info("Retrieval metrics:")
    for direction, dir_metrics in metrics.items():
        logger.info(f"  {direction}: {dir_metrics}")

    # ── Multilingual breakdown ──
    multilingual_metrics = {}
    if eval_cfg.get("multilingual_breakdown", False):
        per_lang = per_language_retrieval_metrics(sim_matrix, languages, ks=ks, directions=directions)
        table = format_multilingual_table(per_lang)
        logger.info(f"\nMultilingual Results:\n{table}")

        summary = compute_multilingual_summary(per_lang, metric_key="t2i_r@10")
        logger.info(f"Multilingual summary: {summary}")

        multilingual_metrics = {
            "per_language": per_lang,
            "summary": summary,
        }

    # ── Save results ──
    output_file = eval_cfg.get("output_file", "reports/tables/retrieval_metrics.json")
    all_metrics = {
        **metrics,
        "multilingual": multilingual_metrics,
        "checkpoint": args.checkpoint,
        "num_test_samples": len(test_dataset),
    }
    save_metrics(all_metrics, output_file)
    logger.info(f"Saved metrics to {output_file}")

    # ── Pretty print ──
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    for direction in directions:
        d_key = "T→I" if direction == "text_to_image" else "I→T"
        dir_m = metrics[direction]
        print(f"\n{d_key}:")
        for k in ks:
            print(f"  R@{k}: {dir_m.get(f'r@{k}', 0)*100:.1f}%")
        print(f"  mAP: {dir_m.get('map', 0):.4f}")
        print(f"  MRR: {dir_m.get('mrr', 0):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
