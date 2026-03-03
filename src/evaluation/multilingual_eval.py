"""Multilingual evaluation with per-language breakdown."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.evaluation.metrics_retrieval import recall_at_k, mean_average_precision


def per_language_retrieval_metrics(
    similarity_matrix: np.ndarray,
    languages: List[str],
    ks: List[int] = [1, 5, 10],
    directions: List[str] = ["text_to_image", "image_to_text"],
) -> Dict[str, Dict[str, float]]:
    """Compute retrieval metrics broken down by language.

    Args:
        similarity_matrix: (N, N) similarity scores.
        languages: List of language codes per sample.
        ks: K values for R@K.
        directions: Retrieval directions.

    Returns:
        Nested dict: {language: {direction: {metric: value}}}.
    """
    languages_arr = np.array(languages)
    unique_languages = sorted(set(languages))
    results = {}

    for lang in unique_languages:
        lang_mask = languages_arr == lang
        lang_indices = np.where(lang_mask)[0]

        if len(lang_indices) < 2:
            continue

        # Extract sub-matrix for this language
        sub_sim = similarity_matrix[np.ix_(lang_indices, lang_indices)]

        lang_results = {}
        for direction in directions:
            dir_key = "t2i" if direction == "text_to_image" else "i2t"
            r_metrics = recall_at_k(sub_sim, ks, direction)
            for k_key, v in r_metrics.items():
                lang_results[f"{dir_key}_{k_key}"] = v

            lang_results[f"{dir_key}_map"] = mean_average_precision(sub_sim, direction)

        lang_results["num_samples"] = int(len(lang_indices))
        results[lang] = lang_results

    return results


def compute_multilingual_summary(
    per_lang_metrics: Dict[str, Dict[str, float]],
    metric_key: str = "t2i_r@10",
) -> Dict[str, float]:
    """Compute multilingual summary statistics.

    Args:
        per_lang_metrics: Output from per_language_retrieval_metrics.
        metric_key: Metric to summarize across languages.

    Returns:
        Dictionary with macro/micro averages and performance gap.
    """
    values = []
    en_value = None

    for lang, metrics in per_lang_metrics.items():
        if metric_key in metrics:
            values.append(metrics[metric_key])
            if lang == "en":
                en_value = metrics[metric_key]

    if not values:
        return {}

    summary = {
        "macro_avg": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "num_languages": len(values),
    }

    if en_value is not None:
        non_en = [v for v in values if v != en_value] or values
        summary["en_value"] = en_value
        summary["non_en_avg"] = float(np.mean(non_en))
        summary["en_vs_non_en_gap"] = float(en_value - np.mean(non_en))

    return summary


def format_multilingual_table(
    per_lang_metrics: Dict[str, Dict[str, float]],
    directions: List[str] = ["text_to_image", "image_to_text"],
    ks: List[int] = [10],
) -> str:
    """Format per-language metrics as a pretty table string.

    Returns:
        Formatted table string for logging/reporting.
    """
    lines = []
    header_parts = ["Language", "N"]

    for direction in directions:
        d_key = "t2i" if direction == "text_to_image" else "i2t"
        for k in ks:
            header_parts.append(f"{d_key.upper()} R@{k}")
        header_parts.append(f"{d_key.upper()} mAP")

    header = " | ".join(f"{h:>12s}" for h in header_parts)
    lines.append(header)
    lines.append("-" * len(header))

    for lang in sorted(per_lang_metrics.keys()):
        metrics = per_lang_metrics[lang]
        row_parts = [lang, str(metrics.get("num_samples", 0))]

        for direction in directions:
            d_key = "t2i" if direction == "text_to_image" else "i2t"
            for k in ks:
                val = metrics.get(f"{d_key}_r@{k}", 0.0)
                row_parts.append(f"{val*100:.1f}")
            val = metrics.get(f"{d_key}_map", 0.0)
            row_parts.append(f"{val:.4f}")

        row = " | ".join(f"{p:>12s}" for p in row_parts)
        lines.append(row)

    return "\n".join(lines)
