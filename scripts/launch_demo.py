#!/usr/bin/env python3
"""
Launch the interactive retrieval demo.

Usage:
    python scripts/launch_demo.py \
        --checkpoint outputs/best.ckpt \
        --config configs/model/vit_xlmr_clipstyle.yaml \
        --index data/artifacts/faiss_flatip.index \
        --metadata data/artifacts/corpus_metadata.json \
        --mode gradio
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch demo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", default="data/raw/images/")
    parser.add_argument("--mode", choices=["gradio", "streamlit"], default="gradio")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.mode == "gradio":
        from src.demo.app_gradio import create_gradio_demo
        demo = create_gradio_demo(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            faiss_index_path=args.index,
            metadata_path=args.metadata,
            image_root=args.image_root,
        )
        demo.launch(server_port=args.port, share=False)

    elif args.mode == "streamlit":
        import subprocess
        subprocess.run([
            "streamlit", "run", "src/demo/app_streamlit.py",
            "--server.port", str(args.port),
        ])


if __name__ == "__main__":
    main()
