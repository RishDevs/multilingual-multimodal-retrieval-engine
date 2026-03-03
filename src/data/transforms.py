"""Image transforms for training and evaluation."""

from __future__ import annotations

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Get training image transforms with data augmentation.

    Includes random resized crop and horizontal flip for training.

    Args:
        image_size: Target image size (square).

    Returns:
        Composed torchvision transforms.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],   # CLIP/ImageNet stats
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Get evaluation image transforms (deterministic, no augmentation).

    Args:
        image_size: Target image size (square).

    Returns:
        Composed torchvision transforms.
    """
    return transforms.Compose([
        transforms.Resize(
            int(image_size * 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
