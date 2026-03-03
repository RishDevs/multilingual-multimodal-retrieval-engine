#!/usr/bin/env python3
"""
Run ablation studies to compare model variants.

Usage:
    python scripts/run_ablations.py --config configs/base.yaml
"""

import argparse
from pathlib import Path

from src.evaluation.ablations import AblationTracker
from src.utils.config import load_config
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--results-dir", default="reports/tables/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("ablations")

    tracker = AblationTracker()

    # ── Define ablation variants ──
    # In practice, each variant would train/evaluate a model.
    # Here we define the framework for registering results.

    ablation_configs = [
        {
            "name": "ResNet + MiniLM (baseline)",
            "config_diff": {
                "image_encoder": "resnet50",
                "text_encoder": "MiniLM",
                "hard_negatives": False,
                "embedding_dim": 256,
            },
        },
        {
            "name": "ViT + XLM-R (no hard neg)",
            "config_diff": {
                "image_encoder": "vit_base_patch16_224",
                "text_encoder": "xlm-roberta-base",
                "hard_negatives": False,
                "embedding_dim": 512,
            },
        },
        {
            "name": "ViT + XLM-R + hard neg",
            "config_diff": {
                "image_encoder": "vit_base_patch16_224",
                "text_encoder": "xlm-roberta-base",
                "hard_negatives": True,
                "embedding_dim": 512,
            },
        },
        {
            "name": "ViT + XLM-R + hard neg + rerank",
            "config_diff": {
                "image_encoder": "vit_base_patch16_224",
                "text_encoder": "xlm-roberta-base",
                "hard_negatives": True,
                "embedding_dim": 512,
                "reranker": True,
            },
        },
    ]

    logger.info(f"Defined {len(ablation_configs)} ablation variants")
    logger.info(
        "To run ablations, train each variant and register results. "
        "Example:\n"
        "  tracker.add_result(\n"
        "    variant_name='ViT + XLM-R + hard neg',\n"
        "    config_diff={...},\n"
        "    metrics={'t2i_r@10': 0.816, 'i2t_r@10': 0.840, 'map': 0.612},\n"
        "    system_metrics={'p95_latency_ms': 31.0},\n"
        "  )"
    )

    # Example: registering placeholder results
    for ablation in ablation_configs:
        logger.info(f"Registered placeholder for: {ablation['name']}")
        tracker.add_result(
            variant_name=ablation["name"],
            config_diff=ablation["config_diff"],
            metrics={},  # Fill with actual metrics after training
            system_metrics={},
        )

    # Save
    results_path = Path(args.results_dir) / "ablation_results.json"
    tracker.save(results_path)
    logger.info(f"Saved ablation framework to {results_path}")

    print("\n" + "=" * 70)
    print("ABLATION FRAMEWORK")
    print("=" * 70)
    print(tracker.get_comparison_table())
    print("=" * 70)


if __name__ == "__main__":
    main()
