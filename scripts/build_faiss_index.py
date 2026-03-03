#!/usr/bin/env python3
"""
Build a FAISS index from pre-generated embeddings.

Usage:
    python scripts/build_faiss_index.py --config configs/faiss/flatip.yaml
"""

import argparse
from pathlib import Path

import numpy as np

from src.retrieval.index_faiss import FAISSIndexBuilder
from src.utils.config import load_config
from src.utils.io import load_embeddings
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--embeddings-dir", default="data/artifacts/")
    parser.add_argument("--prefix", default="test_image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    faiss_cfg = cfg.get("faiss", {})
    logger = setup_logger("build_faiss")

    # Load embeddings
    embeddings, ids = load_embeddings(args.embeddings_dir, prefix=args.prefix)
    logger.info(f"Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # Build index
    builder = FAISSIndexBuilder(
        index_type=faiss_cfg.get("index_type", "IndexFlatIP"),
        embedding_dim=faiss_cfg.get("embedding_dim", embeddings.shape[1]),
        normalize=faiss_cfg.get("normalize", True),
        **{k: v for k, v in faiss_cfg.items()
           if k not in ("index_type", "embedding_dim", "normalize", "index_path", "metadata_path")},
    )

    builder.build(embeddings, train_size=faiss_cfg.get("train_size"))

    # Save
    index_path = faiss_cfg.get("index_path", "data/artifacts/faiss.index")
    builder.save(index_path)

    # Print info
    info = builder.get_index_info()
    print("\n" + "=" * 50)
    print("FAISS INDEX BUILT")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
