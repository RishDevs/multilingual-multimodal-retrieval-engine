"""Ablation study framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class AblationTracker:
    """Track and compare ablation study results.

    Stores results from multiple experiment variants for side-by-side
    comparison, enabling systematic analysis of architectural choices.

    Example variants:
        - ResNet vs ViT
        - MiniLM vs XLM-R
        - Hard negatives ON vs OFF
        - Embedding dim 256 vs 512 vs 768
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        variant_name: str,
        config_diff: Dict[str, Any],
        metrics: Dict[str, float],
        system_metrics: Dict[str, float] | None = None,
    ) -> None:
        """Add an ablation variant result.

        Args:
            variant_name: Human-readable variant label.
            config_diff: What differs from the default config.
            metrics: Retrieval metrics.
            system_metrics: Optional system performance metrics.
        """
        self.results.append({
            "variant": variant_name,
            "config_diff": config_diff,
            "metrics": metrics,
            "system_metrics": system_metrics or {},
        })

    def get_comparison_table(
        self,
        metrics_to_compare: List[str] = [
            "t2i_r@10", "i2t_r@10", "map", "p95_latency_ms"
        ],
    ) -> str:
        """Generate a comparison table of all ablation variants.

        Returns:
            Formatted comparison table string.
        """
        if not self.results:
            return "No ablation results recorded."

        headers = ["Variant"] + metrics_to_compare
        header_line = " | ".join(f"{h:>25s}" for h in headers)
        separator = "-" * len(header_line)

        lines = [header_line, separator]

        for result in self.results:
            all_metrics = {**result["metrics"], **result["system_metrics"]}
            values = [result["variant"]]
            for metric in metrics_to_compare:
                val = all_metrics.get(metric, "—")
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))

            lines.append(" | ".join(f"{v:>25s}" for v in values))

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save ablation results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def load(self, path: str | Path) -> None:
        """Load ablation results from JSON."""
        with open(path, "r") as f:
            self.results = json.load(f)

    def get_best_variant(self, metric: str, mode: str = "max") -> Dict[str, Any]:
        """Find the best variant by a specific metric."""
        if not self.results:
            return {}

        def get_metric_value(result):
            all_metrics = {**result["metrics"], **result["system_metrics"]}
            return all_metrics.get(metric, float("-inf") if mode == "max" else float("inf"))

        if mode == "max":
            return max(self.results, key=get_metric_value)
        else:
            return min(self.results, key=get_metric_value)
