#!/usr/bin/env python3
"""
Benchmark retrieval latency across different FAISS index types and corpus sizes.

Usage:
    python scripts/benchmark_latency.py --config configs/eval/latency_eval.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.retrieval.index_faiss import FAISSIndexBuilder
from src.retrieval.search import ANNSearcher
from src.utils.config import load_config
from src.utils.io import save_metrics
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Benchmark latency")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--embedding-dim", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", {}).get("latency", {})
    logger = setup_logger("benchmark")

    num_warmup = eval_cfg.get("num_warmup_queries", 100)
    num_benchmark = eval_cfg.get("num_benchmark_queries", 1000)
    batch_sizes = eval_cfg.get("batch_sizes", [1, 16, 64])
    corpus_sizes = eval_cfg.get("corpus_sizes", [10000, 100000])
    top_k = eval_cfg.get("top_k", 10)
    percentiles = eval_cfg.get("percentiles", [50, 95, 99])
    embedding_dim = args.embedding_dim

    all_results = {}

    for corpus_size in corpus_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Benchmarking corpus size: {corpus_size}")

        # Generate random embeddings for benchmarking
        rng = np.random.RandomState(42)
        corpus_embeddings = rng.randn(corpus_size, embedding_dim).astype(np.float32)
        corpus_embeddings /= np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        query_embeddings = rng.randn(max(num_warmup + num_benchmark, 2000), embedding_dim).astype(np.float32)
        query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        ids = [str(i) for i in range(corpus_size)]

        # Test each index type
        for index_type in ["IndexFlatIP", "HNSW"]:
            logger.info(f"  Index type: {index_type}")

            builder = FAISSIndexBuilder(
                index_type=index_type,
                embedding_dim=embedding_dim,
                normalize=False,  # Already normalized
                M=32, ef_construction=200, ef_search=128,
            )
            builder.build(corpus_embeddings)

            searcher = ANNSearcher(index=builder.index, ids=ids)

            stats = searcher.benchmark_latency(
                query_embeddings,
                top_k=top_k,
                num_warmup=num_warmup,
                num_queries=num_benchmark,
                percentiles=percentiles,
            )

            stats["index_type"] = index_type
            stats["corpus_size"] = corpus_size
            stats["build_time_s"] = builder.build_time

            key = f"{index_type}_corpus{corpus_size}"
            all_results[key] = stats

            logger.info(f"    P50: {stats['p50_latency_ms']:.2f}ms")
            logger.info(f"    P95: {stats['p95_latency_ms']:.2f}ms")
            logger.info(f"    P99: {stats['p99_latency_ms']:.2f}ms")
            logger.info(f"    QPS: {stats['qps']:.1f}")
            logger.info(f"    Build: {stats['build_time_s']:.2f}s")

    # Save results
    report_file = eval_cfg.get("report_file", "reports/tables/latency_report.json")
    save_metrics(all_results, report_file)
    logger.info(f"\nSaved latency report to {report_file}")

    # Pretty print
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    header = f"{'Config':>30s} | {'P50 (ms)':>10s} | {'P95 (ms)':>10s} | {'P99 (ms)':>10s} | {'QPS':>10s} | {'Build (s)':>10s}"
    print(header)
    print("-" * len(header))
    for key, stats in all_results.items():
        print(
            f"{key:>30s} | "
            f"{stats['p50_latency_ms']:>10.2f} | "
            f"{stats['p95_latency_ms']:>10.2f} | "
            f"{stats['p99_latency_ms']:>10.2f} | "
            f"{stats['qps']:>10.1f} | "
            f"{stats['build_time_s']:>10.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
