# Final Report: Multilingual Multimodal Retrieval Engine

## 1. Overview

This report documents the development, training, and evaluation of a multilingual
multimodal retrieval system built on the Google WIT dataset.

## 2. Dataset

- **Source**: Google WIT (Wikipedia-based Image Text)
- **Subset Size**: TBD
- **Languages**: en, hi, es, fr, de, pt
- **Split**: 80/10/10 (train/val/test, split on image_id)

## 3. Model Architecture

- **Image Encoder**: ViT-B/16 (pretrained on ImageNet)
- **Text Encoder**: XLM-RoBERTa Base
- **Projection**: MLP (hidden=1024 → embedding=512)
- **Training Objective**: Symmetric CLIP InfoNCE loss
- **Hard Negatives**: In-batch top-k semantic mining (after epoch 3)

## 4. Results

### 4.1 Retrieval Metrics
*(To be filled after training)*

| Direction | R@1 | R@5 | R@10 | mAP | MRR |
|-----------|-----|-----|------|-----|-----|
| T → I     |     |     |      |     |     |
| I → T     |     |     |      |     |     |

### 4.2 Multilingual Breakdown
*(To be filled after evaluation)*

| Language | T→I R@10 | I→T R@10 | mAP (T→I) |
|----------|----------|----------|-----------|
| en       |          |          |           |
| hi       |          |          |           |
| es       |          |          |           |

### 4.3 Ablation Studies
*(To be filled after ablation experiments)*

### 4.4 System Performance
*(To be filled after benchmarking)*

## 5. Error Analysis

*(To be filled with failure case examples)*

## 6. Conclusions

*(To be written after experiments)*
