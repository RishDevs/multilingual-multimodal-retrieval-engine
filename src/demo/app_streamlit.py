"""Streamlit demo for interactive multimodal retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data.tokenization import TextTokenizer
from src.data.transforms import get_eval_transforms
from src.models.dual_encoder import build_dual_encoder
from src.retrieval.index_faiss import FAISSIndexBuilder
from src.retrieval.search import ANNSearcher
from src.utils.config import load_config
from src.utils.device import get_device


@st.cache_resource
def load_model_and_index(checkpoint_path, config_path, index_path, metadata_path):
    """Load and cache model + index for Streamlit."""
    device = get_device("auto")
    cfg = load_config([config_path])

    model = build_dual_encoder(cfg["model"])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    builder = FAISSIndexBuilder()
    index = builder.load(index_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    tokenizer = TextTokenizer(
        model_name=cfg["model"]["text_encoder"]["name"],
        max_length=cfg.get("data", {}).get("max_text_len", 64),
    )
    transform = get_eval_transforms(cfg.get("data", {}).get("image_size", 224))

    return model, index, metadata, tokenizer, transform, device


def main():
    st.set_page_config(
        page_title="Multilingual Multimodal Retrieval",
        page_icon="🌍",
        layout="wide",
    )

    st.title("🌍 Multilingual Multimodal Retrieval Engine")
    st.markdown(
        "Cross-lingual image–text retrieval powered by "
        "**CLIP-style dual encoder** + **FAISS ANN search**"
    )

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    mode = st.sidebar.radio("Search Mode", ["Text → Image", "Image → Text"])
    top_k = st.sidebar.slider("Top-K Results", 1, 20, 5)

    st.sidebar.header("📊 Model Info")
    st.sidebar.markdown("""
    - **Image Encoder**: ViT-B/16
    - **Text Encoder**: XLM-R Base
    - **Embedding Dim**: 512
    - **Languages**: en, hi, es, fr, de, pt
    """)

    # Note: In production, these would come from CLI args or env vars
    st.info("Configure checkpoint/index paths to enable search functionality.")

    if mode == "Text → Image":
        st.header("🔍 Text → Image Search")
        query = st.text_input(
            "Enter your query (any supported language)",
            placeholder="a sunset over mountains | पहाड़ों पर सूर्यास्त",
        )

        if st.button("Search", type="primary"):
            if query.strip():
                st.info(f"Searching for: '{query}' (top-{top_k})")
                # Placeholder for actual search
                st.markdown("*Connect model checkpoint and FAISS index to see results.*")

    else:
        st.header("🖼️ Image → Text Search")
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", width=300)

            if st.button("Search", type="primary"):
                st.info(f"Searching for matching texts (top-{top_k})")
                st.markdown("*Connect model checkpoint and FAISS index to see results.*")


if __name__ == "__main__":
    main()
