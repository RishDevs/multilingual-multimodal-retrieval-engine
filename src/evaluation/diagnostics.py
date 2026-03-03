"""Embedding diagnostics and error analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embedding_space(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    languages: Optional[List[str]] = None,
    method: str = "pca",
    n_samples: int = 500,
    save_path: Optional[str] = None,
    title: str = "Embedding Space Visualization",
) -> None:
    """Visualize the joint embedding space.

    Args:
        image_embeddings: (N, D) image embeddings.
        text_embeddings: (N, D) text embeddings.
        languages: Optional language codes for coloring.
        method: Dimensionality reduction method ('pca' or 'tsne').
        n_samples: Number of samples to visualize.
        save_path: Path to save the plot.
        title: Plot title.
    """
    # Subsample if needed
    n = min(n_samples, len(image_embeddings), len(text_embeddings))
    idx = np.random.choice(len(image_embeddings), n, replace=False)

    img_emb = image_embeddings[idx]
    txt_emb = text_embeddings[idx]

    # Combine for joint projection
    combined = np.concatenate([img_emb, txt_emb], axis=0)

    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    reduced = reducer.fit_transform(combined)

    img_2d = reduced[:n]
    txt_2d = reduced[n:]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if languages:
        langs = np.array(languages)[idx]
        unique_langs = sorted(set(langs))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_langs)))
        lang_color_map = dict(zip(unique_langs, colors))

        for lang in unique_langs:
            mask = langs == lang
            ax.scatter(
                img_2d[mask, 0], img_2d[mask, 1],
                c=[lang_color_map[lang]], marker="o", alpha=0.5,
                label=f"img-{lang}", s=20,
            )
            ax.scatter(
                txt_2d[mask, 0], txt_2d[mask, 1],
                c=[lang_color_map[lang]], marker="x", alpha=0.5,
                label=f"txt-{lang}", s=20,
            )
    else:
        ax.scatter(img_2d[:, 0], img_2d[:, 1], c="blue", marker="o", alpha=0.5, label="Images", s=20)
        ax.scatter(txt_2d[:, 0], txt_2d[:, 1], c="red", marker="x", alpha=0.5, label="Texts", s=20)

    # Draw lines between matched pairs
    for i in range(min(50, n)):
        ax.plot(
            [img_2d[i, 0], txt_2d[i, 0]],
            [img_2d[i, 1], txt_2d[i, 1]],
            "k-", alpha=0.1, linewidth=0.5,
        )

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_failures(
    similarity_matrix: np.ndarray,
    sample_ids: List[str],
    texts: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    top_k: int = 5,
    num_failures: int = 10,
) -> List[Dict[str, Any]]:
    """Analyze retrieval failures for error analysis.

    Identifies cases where the correct match is not in the top-k
    and provides context for understanding the failure.

    Args:
        similarity_matrix: (N, N) similarity scores.
        sample_ids: List of sample IDs.
        texts: Optional list of text strings.
        languages: Optional list of language codes.
        top_k: K for determining failure.
        num_failures: Number of failures to return.

    Returns:
        List of failure analysis dicts.
    """
    n = similarity_matrix.shape[0]
    targets = np.arange(n)

    failures = []

    for i in range(n):
        sorted_indices = np.argsort(-similarity_matrix[i])
        rank = np.where(sorted_indices == targets[i])[0][0] + 1

        if rank > top_k:
            failure = {
                "query_id": sample_ids[i] if i < len(sample_ids) else str(i),
                "correct_rank": int(rank),
                "correct_score": float(similarity_matrix[i, targets[i]]),
                "top_k_retrieved": [],
            }

            if texts and i < len(texts):
                failure["query_text"] = texts[i]
            if languages and i < len(languages):
                failure["language"] = languages[i]

            # Top-k retrieved items
            for j, idx in enumerate(sorted_indices[:top_k]):
                retrieved = {
                    "rank": j + 1,
                    "id": sample_ids[idx] if idx < len(sample_ids) else str(idx),
                    "score": float(similarity_matrix[i, idx]),
                }
                if texts and idx < len(texts):
                    retrieved["text"] = texts[idx]
                failure["top_k_retrieved"].append(retrieved)

            failures.append(failure)

    # Sort by worst rank and return top failures
    failures.sort(key=lambda x: x["correct_rank"], reverse=True)
    return failures[:num_failures]


def compute_embedding_statistics(
    embeddings: np.ndarray,
) -> Dict[str, float]:
    """Compute statistics about embedding vectors.

    Args:
        embeddings: (N, D) embedding vectors.

    Returns:
        Dictionary with embedding statistics.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim = embeddings @ embeddings.T
    np.fill_diagonal(cos_sim, 0)

    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "mean_pairwise_similarity": float(np.mean(cos_sim)),
        "mean_abs_pairwise_similarity": float(np.mean(np.abs(cos_sim))),
        "embedding_dim": embeddings.shape[1],
        "num_embeddings": embeddings.shape[0],
    }
