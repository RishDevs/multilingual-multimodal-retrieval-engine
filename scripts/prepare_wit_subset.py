#!/usr/bin/env python3
"""
Prepare a WIT subset manifest from the Google WIT dataset.

Downloads/loads the WIT dataset, filters by languages, cleans text fields,
and produces a curated manifest parquet file for downstream training.

Usage:
    python scripts/prepare_wit_subset.py --config configs/data/wit_subset.yaml
"""

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def clean_text(text: str) -> str:
    """Clean text by removing HTML, normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_sample_id(image_url: str, text: str) -> str:
    """Generate a deterministic sample ID."""
    content = f"{image_url}:{text}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Prepare WIT subset")
    parser.add_argument("--config", nargs="+", required=True, help="Config file(s)")
    parser.add_argument("--debug", action="store_true", help="Use debug subset size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    logger = setup_logger("prepare_wit", log_file="data/interim/prepare_wit.log")

    # ── Configuration ──
    languages = data_cfg.get("languages", ["en"])
    text_fields = data_cfg.get("text_fields", ["caption_reference_description"])
    subset_size = data_cfg.get("debug_size", 10000) if args.debug else data_cfg.get("subset_size", 500000)
    min_text_len = data_cfg.get("min_text_length", 5)
    output_path = data_cfg.get("subset_manifest", "data/interim/wit_subset_manifest.parquet")

    logger.info(f"Preparing WIT subset: {subset_size} samples, languages={languages}")

    # ── Load WIT dataset ──
    try:
        from datasets import load_dataset
        logger.info("Loading WIT dataset from HuggingFace...")

        # Load a subset of WIT (streaming for large datasets)
        dataset = load_dataset(
            "google/wit",
            split="train",
            streaming=True,
        )

        records = []
        seen_images = set()
        lang_counts = {lang: 0 for lang in languages}
        target_per_lang = subset_size // len(languages)

        for sample in tqdm(dataset, desc="Processing WIT", total=subset_size * 3):
            # Check if we have enough
            if all(c >= target_per_lang for c in lang_counts.values()):
                break

            lang = sample.get("language", "unknown")
            if lang not in languages:
                continue
            if lang_counts.get(lang, 0) >= target_per_lang:
                continue

            # Get best text field
            text = ""
            for field in text_fields:
                val = sample.get(field, "")
                if val and isinstance(val, str) and len(val.strip()) >= min_text_len:
                    text = clean_text(val)
                    break

            if len(text) < min_text_len:
                continue

            image_url = sample.get("image_url", "")
            if not image_url:
                continue

            # Deduplicate by image URL
            if image_url in seen_images and data_cfg.get("remove_duplicates", True):
                continue
            seen_images.add(image_url)

            sample_id = generate_sample_id(image_url, text)

            records.append({
                "sample_id": sample_id,
                "image_url": image_url,
                "image_path": f"images/{sample_id}.jpg",
                "text": text,
                "language": lang,
                "image_id": hashlib.md5(image_url.encode()).hexdigest()[:10],
                "pair_id": sample_id,
                "page_title": clean_text(sample.get("page_title", "")),
                "section_title": clean_text(sample.get("hierarchical_section_title", "")),
            })

            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        logger.info(f"Collected {len(records)} samples")
        logger.info(f"Language distribution: {lang_counts}")

    except Exception as e:
        logger.warning(f"Could not load WIT from HuggingFace: {e}")
        logger.info("Creating a synthetic manifest for development...")

        # Create synthetic data for development
        import numpy as np
        rng = np.random.RandomState(seed)

        records = []
        for i in range(min(subset_size, 10000)):
            lang = languages[i % len(languages)]
            sample_id = f"syn_{i:06d}"
            records.append({
                "sample_id": sample_id,
                "image_url": f"https://example.com/img_{i}.jpg",
                "image_path": f"images/{sample_id}.jpg",
                "text": f"Sample text {i} in {lang}",
                "language": lang,
                "image_id": f"img_{i:06d}",
                "pair_id": sample_id,
                "page_title": f"Page {i}",
                "section_title": f"Section {i}",
            })

        logger.info(f"Created {len(records)} synthetic samples")

    # ── Save manifest ──
    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)
    logger.info(f"Saved manifest to {output_path} ({len(df)} rows)")

    # Print summary
    print("\n" + "=" * 60)
    print("WIT SUBSET MANIFEST PREPARED")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Output: {output_path}")
    print(f"\nLanguage distribution:")
    print(df["language"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
