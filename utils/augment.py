"""
utils/augment.py
----------------
Data augmentation helpers used during training.
Provides both Keras-layer-based augmentation (for tf.data pipelines)
and standalone NumPy-based transforms for offline inspection.
"""

import numpy as np
import tensorflow as tf
import cv2
import random


# ── Keras Sequential Augmentation Layer ──────────────────────────────────────

def get_augmentation_layer():
    """
    Return a tf.keras Sequential augmentation model that can be embedded
    directly inside the Keras model graph (runs on GPU during training).

    Usage in model:
        x = get_augmentation_layer()(inputs, training=True)
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),           # ±15 % of full rotation
        tf.keras.layers.RandomZoom(0.20),               # ±20 % zoom
        tf.keras.layers.RandomTranslation(0.10, 0.10),  # ±10 % translation
        tf.keras.layers.RandomBrightness(0.20),         # ±20 % brightness
        tf.keras.layers.RandomContrast(0.15),           # ±15 % contrast
    ], name="augmentation")


# ── Standalone NumPy Augmentation (for EDA / visualization) ──────────────────

def random_flip(img: np.ndarray) -> np.ndarray:
    """Randomly flip image horizontally."""
    if random.random() > 0.5:
        return np.fliplr(img)
    return img


def random_rotation(img: np.ndarray, max_angle: float = 25.0) -> np.ndarray:
    """Rotate image by a random angle within ±max_angle degrees."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_zoom(img: np.ndarray, zoom_range: float = 0.20) -> np.ndarray:
    """Randomly zoom in/out up to zoom_range fraction."""
    factor = random.uniform(1 - zoom_range, 1 + zoom_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (new_w, new_h))
    # Crop or pad back to original size
    if factor > 1:
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return resized[start_y:start_y + h, start_x:start_x + w]
    else:
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        padded = np.zeros_like(img)
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return padded


def random_brightness(img: np.ndarray, delta: float = 0.25) -> np.ndarray:
    """Adjust brightness by a random amount within ±delta."""
    shift = random.uniform(-delta, delta)
    out = img.astype(np.float32) + shift
    return np.clip(out, 0.0, 1.0).astype(img.dtype)


def random_contrast(img: np.ndarray, lower: float = 0.75, upper: float = 1.25) -> np.ndarray:
    """Adjust contrast by scaling pixel values around their mean."""
    factor = random.uniform(lower, upper)
    mean = img.mean()
    out = (img.astype(np.float32) - mean) * factor + mean
    return np.clip(out, 0.0, 1.0).astype(img.dtype)


def apply_augmentation_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Apply the full augmentation pipeline to a single image (NumPy float32 [0,1]).
    Useful for visualizing what augmented samples look like.

    Args:
        img: np.ndarray of shape (H, W, 3) with values in [0, 1].

    Returns:
        Augmented image of the same shape.
    """
    img = random_flip(img)
    img = random_rotation(img)
    img = random_zoom(img)
    img = random_brightness(img)
    img = random_contrast(img)
    return img


def generate_augmented_samples(img: np.ndarray, n: int = 6) -> list:
    """
    Generate n augmented versions of a single image for visualization.

    Args:
        img: Source image (H, W, 3) float32 in [0,1].
        n  : Number of augmented copies to generate.

    Returns:
        List of n augmented images.
    """
    return [apply_augmentation_pipeline(img.copy()) for _ in range(n)]
