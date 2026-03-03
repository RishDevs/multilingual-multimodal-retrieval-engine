#!/usr/bin/env python3
"""
Train the dual encoder retrieval model.

Supports stacked configs and CLI overrides.

Usage:
    python scripts/train.py \
        --config configs/base.yaml \
        --config configs/data/wit_subset.yaml \
        --config configs/model/vit_xlmr_clipstyle.yaml \
        --config configs/train/train_flagship.yaml \
        train.epochs=15 \
        project.experiment_name=vit_xlmr_hn_on_run03
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.collate import multimodal_collate_fn
from src.data.dataset import WITDataset
from src.data.samplers import BalancedLanguageSampler
from src.data.tokenization import TextTokenizer
from src.models.dual_encoder import build_dual_encoder
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.device import get_device, get_device_info
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    # ── Parse CLI ──
    parser = argparse.ArgumentParser(description="Train dual encoder")
    parser.add_argument("--config", nargs="+", required=True, help="Config file(s)")
    args, overrides = parser.parse_known_args()

    # Load and merge configs
    cfg = load_config(args.config, overrides if overrides else None)

    # ── Setup ──
    project_cfg = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    experiment_name = project_cfg.get("experiment_name", "experiment")
    output_dir = Path(project_cfg.get("output_dir", "outputs")) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "train",
        log_file=str(output_dir / "train.log"),
    )

    # Save merged config
    save_config(cfg, output_dir / "config_merged.yaml")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")

    # ── Device ──
    device_cfg = cfg.get("device", {})
    device = get_device(device_cfg.get("accelerator", "auto"))
    logger.info(f"Device: {device} | Info: {get_device_info(device)}")

    # ── Tokenizer ──
    model_cfg = cfg.get("model", {})
    text_encoder_name = model_cfg.get("text_encoder", {}).get("name", "xlm-roberta-base")
    max_text_len = data_cfg.get("max_text_len", 64)

    tokenizer = TextTokenizer(
        model_name=text_encoder_name,
        max_length=max_text_len,
    )
    logger.info(f"Tokenizer: {text_encoder_name} (max_len={max_text_len})")

    # ── Datasets ──
    manifest_path = data_cfg.get("subset_manifest", "data/interim/wit_subset_manifest.parquet")
    image_root = data_cfg.get("image_root", "data/raw/images/")
    image_size = data_cfg.get("image_size", 224)
    text_fields = data_cfg.get("text_fields", ["text"])

    train_dataset = WITDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        tokenizer=tokenizer,
        split="train",
        image_size=image_size,
        text_fields=text_fields,
    )

    val_dataset = WITDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        tokenizer=tokenizer,
        split="val",
        image_size=image_size,
        text_fields=text_fields,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Train language distribution: {train_dataset.get_language_distribution()}")

    # ── Sampler ──
    train_sampler = None
    shuffle = True
    if data_cfg.get("balanced_language_sampling", False):
        train_sampler = BalancedLanguageSampler(
            languages=train_dataset.data["language"].tolist(),
            seed=seed,
        )
        shuffle = False
        logger.info("Using balanced language sampling")

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("train_batch_size", 128),
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("eval_batch_size", 256),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=multimodal_collate_fn,
    )

    # ── Model ──
    model = build_dual_encoder(model_cfg)
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    # ── Train ──
    results = trainer.train()

    logger.info("Training complete!")
    logger.info(f"Best epoch: {results['best_epoch']}")
    logger.info(f"Output: {results['output_dir']}")


if __name__ == "__main__":
    main()
