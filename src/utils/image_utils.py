"""Image decoding and validation utilities."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import tensorflow as tf


def safe_decode_image(
    path: tf.Tensor,
    image_size: tuple[int, int] | None = None,
    training: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Safely decodes an image, returning a boolean indicating success.

    Args:
        path: Path to the image file.
        image_size: Optional target size for the image.
        training: Whether we are in training mode (for data augmentation).

    Returns:
        A tuple of (image, success_flag).
        If decoding fails, image is a zero-filled tensor and success_flag is False.
    """
    try:
        # Convert path to string if it's a tensor
        if isinstance(path, tf.Tensor):
            path_str = path.numpy().decode("utf-8")
        else:
            path_str = str(path)

        with open(path_str, "rb") as f:
            img_bytes = f.read()
        
        # Use decode_image which is more robust for various formats
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        # Ensure 3D [H, W, 3]
        if len(img.shape) != 3:
            raise ValueError(f"Invalid image shape: {img.shape}")

        if image_size:
            if training:
                # Resize to slightly larger then crop for augmentation
                img = tf.image.resize(img, [256, 256])
                img = tf.image.random_crop(img, [image_size[0], image_size[1], 3])
                img = tf.image.random_flip_left_right(img)
            else:
                img = tf.image.resize(img, image_size)
        
        # Final safety check on shape
        if image_size and (img.shape[0] != image_size[0] or img.shape[1] != image_size[1]):
             img = tf.image.resize(img, image_size)

        return img, tf.constant(True)
    except Exception:
        # Fallback for any error
        # Force allocation on CPU to avoid "Expected size[0] in [0, ...], but got 224" GPU errors inside py_function
        with tf.device('/CPU:0'):
            if image_size:
                return tf.zeros((*image_size, 3), dtype=tf.float32), tf.constant(False)
            return tf.zeros((1, 1, 3), dtype=tf.float32), tf.constant(False)


def is_image_valid(path: str) -> bool:
    """Checks if an image file is valid and can be decoded."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
        if width < 2 or height < 2:
            return False
        return True
    except Exception:
        return False


def collect_invalid_image_paths(paths: list[str]) -> list[str]:
    """Returns image paths that are missing or cannot be decoded as RGB images."""
    max_workers = min(32, max(4, (os.cpu_count() or 4) * 2))

    def _is_invalid(path: str) -> bool:
        return not Path(path).exists() or not is_image_valid(path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        invalid_flags = list(executor.map(_is_invalid, paths))
    return [path for path, is_invalid in zip(paths, invalid_flags) if is_invalid]
