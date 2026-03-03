#!/usr/bin/env python3
"""
Build train/val/test splits from a WIT subset manifest.

Splits on image_id to prevent data leakage. Optionally creates
a balanced cross-lingual evaluation split.

Usage:
    python scripts/build_splits.py --config configs/data/wit_subset.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data.splits import create_cross_lingual_eval_split, create_splits, verify_no_leakage
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test splits")
    parser.add_argument("--config", nargs="+", required=True, help="Config file(s)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    logger = setup_logger("build_splits")

    # ── Load manifest ──
    manifest_path = data_cfg.get("subset_manifest", "data/interim/wit_subset_manifest.parquet")
    logger.info(f"Loading manifest from {manifest_path}")
    df = pd.read_parquet(manifest_path)
    logger.info(f"Loaded {len(df)} samples")

    # ── Create splits ──
    logger.info("Creating train/val/test splits (splitting on image_id)...")
    df = create_splits(
        df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        split_column="image_id",
        seed=seed,
    )

    # ── Verify no leakage ──
    verify_no_leakage(df, split_column="image_id")
    logger.info("✓ No data leakage detected")

    # ── Save splits ──
    output_path = Path(manifest_path)
    df.to_parquet(str(output_path), index=False)
    logger.info(f"Saved splits to {output_path}")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        print(f"\n{split.upper()}: {len(split_df)} samples")
        print(f"  Languages: {dict(split_df['language'].value_counts())}")
        print(f"  Unique images: {split_df['image_id'].nunique()}")

    # ── Create cross-lingual eval split (optional) ──
    if data_cfg.get("cross_lingual_eval", False):
        languages = data_cfg.get("languages", ["en", "hi", "es", "fr", "de", "pt"])
        samples_per_lang = data_cfg.get("samples_per_language", 1000)

        test_df = df[df["split"] == "test"]
        cross_lingual = create_cross_lingual_eval_split(
            test_df, languages, samples_per_lang, seed=seed,
        )

        cross_path = Path("data/interim/wit_cross_lingual_eval.parquet")
        cross_path.parent.mkdir(parents=True, exist_ok=True)
        cross_lingual.to_parquet(str(cross_path), index=False)
        logger.info(f"Saved cross-lingual eval split to {cross_path} ({len(cross_lingual)} samples)")

    print("=" * 60)


if __name__ == "__main__":
    main()
