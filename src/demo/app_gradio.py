"""Gradio demo for interactive multimodal retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.data.tokenization import TextTokenizer
from src.data.transforms import get_eval_transforms
from src.models.dual_encoder import DualEncoder, build_dual_encoder
from src.retrieval.search import ANNSearcher
from src.retrieval.index_faiss import FAISSIndexBuilder
from src.utils.config import load_config
from src.utils.device import get_device


def create_gradio_demo(
    checkpoint_path: str,
    config_path: str,
    faiss_index_path: str,
    metadata_path: str,
    image_root: str = "data/raw/images/",
) -> gr.Blocks:
    """Create a Gradio demo for interactive retrieval.

    Supports:
        - Text → Image retrieval
        - Image → Text retrieval
        - Language selection
        - Top-k slider
        - Latency display

    Args:
        checkpoint_path: Path to trained model checkpoint.
        config_path: Path to model config YAML.
        faiss_index_path: Path to built FAISS index.
        metadata_path: Path to corpus metadata JSON.
        image_root: Root directory of images.

    Returns:
        Gradio Blocks interface.
    """
    # ── Load model ──
    device = get_device("auto")
    cfg = load_config([config_path])
    model = build_dual_encoder(cfg["model"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Load index ──
    index_builder = FAISSIndexBuilder()
    index = index_builder.load(faiss_index_path)

    # ── Load metadata ──
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    ids = metadata.get("ids", [])
    texts = metadata.get("texts", [])
    languages_list = metadata.get("languages", [])
    image_paths = metadata.get("image_paths", [])

    searcher = ANNSearcher(
        index=index,
        ids=ids,
        metadata={"text": texts, "language": languages_list, "image_path": image_paths},
    )

    # ── Tokenizer + transforms ──
    tokenizer = TextTokenizer(
        model_name=cfg["model"]["text_encoder"]["name"],
        max_length=cfg.get("data", {}).get("max_text_len", 64),
    )
    transform = get_eval_transforms(cfg.get("data", {}).get("image_size", 224))

    # ── Search functions ──
    def text_to_image_search(query: str, top_k: int):
        """Search for images matching a text query."""
        encoded = tokenizer(query)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            query_emb = model.encode_text(input_ids, attention_mask)
            query_np = query_emb.cpu().numpy()

        results = searcher.search_with_metadata(query_np, top_k=top_k)

        output_images = []
        output_info = []
        for result in results[0]:
            img_path = Path(image_root) / result.get("image_path", "")
            if img_path.exists():
                output_images.append(str(img_path))
            output_info.append(
                f"Rank {result['rank']} | Score: {result['score']:.4f} | "
                f"Lang: {result.get('language', '?')}"
            )

        return output_images, "\n".join(output_info)

    def image_to_text_search(image: Image.Image, top_k: int):
        """Search for texts matching an image query."""
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            query_emb = model.encode_image(image_tensor)
            query_np = query_emb.cpu().numpy()

        results = searcher.search_with_metadata(query_np, top_k=top_k)

        output_texts = []
        for result in results[0]:
            output_texts.append(
                f"**Rank {result['rank']}** (Score: {result['score']:.4f})\n"
                f"Language: {result.get('language', '?')}\n"
                f"Text: {result.get('text', 'N/A')}\n"
            )

        return "\n---\n".join(output_texts)

    # ── Build Gradio UI ──
    with gr.Blocks(
        title="Multilingual Multimodal Retrieval Engine",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🌍 Multilingual Multimodal Retrieval Engine
            ### Cross-lingual image–text retrieval powered by CLIP-style dual encoder + FAISS

            Search for images using text queries (in any supported language)
            or find relevant captions for uploaded images.
            """
        )

        with gr.Tab("🔍 Text → Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Enter your search query",
                        placeholder="e.g., 'a sunset over the mountains' or 'पहाड़ों पर सूर्यास्त'",
                        lines=2,
                    )
                    t2i_topk = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Number of results (top-k)",
                    )
                    t2i_btn = gr.Button("🔍 Search", variant="primary")

                with gr.Column(scale=2):
                    t2i_gallery = gr.Gallery(label="Retrieved Images")
                    t2i_info = gr.Textbox(label="Result Details", lines=5)

            t2i_btn.click(
                text_to_image_search,
                inputs=[text_input, t2i_topk],
                outputs=[t2i_gallery, t2i_info],
            )

        with gr.Tab("🖼️ Image → Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload an image")
                    i2t_topk = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Number of results (top-k)",
                    )
                    i2t_btn = gr.Button("🔍 Search", variant="primary")

                with gr.Column(scale=2):
                    i2t_output = gr.Markdown(label="Retrieved Texts")

            i2t_btn.click(
                image_to_text_search,
                inputs=[image_input, i2t_topk],
                outputs=[i2t_output],
            )

        gr.Markdown(
            """
            ---
            **Model**: ViT-B/16 + XLM-R | **Index**: FAISS
            | **Languages**: en, hi, es, fr, de, pt
            """
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch retrieval demo")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--index", required=True, help="FAISS index path")
    parser.add_argument("--metadata", required=True, help="Corpus metadata JSON path")
    parser.add_argument("--image-root", default="data/raw/images/")
    parser.add_argument("--port", type=int, default=7860)

    args = parser.parse_args()

    demo = create_gradio_demo(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        faiss_index_path=args.index,
        metadata_path=args.metadata,
        image_root=args.image_root,
    )
    demo.launch(server_port=args.port, share=False)
