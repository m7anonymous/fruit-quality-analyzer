"""
utils/preprocess.py
-------------------
Image preprocessing utilities for the Fruit Quality Analyzer.
Handles loading, resizing, normalization, and dataset splitting.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)          # Input size expected by MobileNetV2 / ResNet50
BATCH_SIZE  = 32
SEED        = 42

# Quality → shelf-life mapping (used by predict.py and app.py)
SHELF_LIFE_MAP = {
    "fresh":    {"days": "7–14 days",  "emoji": "🟢", "status": "Excellent"},
    "good":     {"days": "7–14 days",  "emoji": "🟢", "status": "Excellent"},
    "ripe":     {"days": "3–7 days",   "emoji": "🟡", "status": "Good"},
    "medium":   {"days": "3–5 days",   "emoji": "🟡", "status": "Fair"},
    "overripe": {"days": "1–3 days",   "emoji": "🟠", "status": "Poor"},
    "rotten":   {"days": "Expired",    "emoji": "🔴", "status": "Expired"},
    "bad":      {"days": "Expired",    "emoji": "🔴", "status": "Expired"},
    "damaged":  {"days": "0–1 days",   "emoji": "🟠", "status": "Very Poor"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load a single image from disk, resize to IMG_SIZE and normalize to [0,1].

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        Numpy array of shape (224, 224, 3), dtype float32.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV reads BGR → convert RGB
    img = cv2.resize(img, IMG_SIZE)                      # Resize to 224×224
    img = img.astype(np.float32) / 255.0                 # Normalize to [0, 1]
    return img


def get_class_names(dataset_dir: str) -> list:
    """
    Return sorted list of class names found as sub-directories.

    Args:
        dataset_dir: Root directory containing one sub-folder per class.

    Returns:
        Sorted list of class name strings.
    """
    return sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])


def build_file_lists(dataset_dir: str):
    """
    Walk through dataset_dir and build parallel lists of file paths and labels.

    Expected layout:
        dataset_dir/
            class_a/  img1.jpg  img2.jpg ...
            class_b/  img1.jpg  ...

    Returns:
        (file_paths, labels, class_names)
        - file_paths : list of str
        - labels     : list of int (index into class_names)
        - class_names: list of str
    """
    class_names = get_class_names(dataset_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    file_paths, labels = [], []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if Path(fname).suffix.lower() in valid_exts:
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])

    return file_paths, labels, class_names


def split_dataset(file_paths: list, labels: list, val_size=0.15, test_size=0.15):
    """
    Stratified split into train / validation / test sets.

    Returns:
        Six lists: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split off the test set
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        file_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=SEED
    )
    # From the remaining, split validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_ratio,
        stratify=y_tmp,
        random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_generators(dataset_dir: str):
    """
    Create Keras ImageDataGenerators for train / val / test splits
    using flow_from_directory (simpler for directory-based datasets).

    Returns:
        (train_gen, val_gen, test_gen, class_names)
    """
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        zoom_range=0.25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.75, 1.25],
        fill_mode="nearest",
        validation_split=0.15,
    )

    # Validation / test generator — only rescale, NO augmentation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
    )

    common_kwargs = dict(
        directory=dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )

    train_gen = train_datagen.flow_from_directory(
        subset="training", shuffle=True, **common_kwargs
    )
    val_gen = val_datagen.flow_from_directory(
        subset="validation", shuffle=False, **common_kwargs
    )

    class_names = list(train_gen.class_indices.keys())
    return train_gen, val_gen, class_names


def get_shelf_life(predicted_class: str) -> dict:
    """
    Map a predicted quality class name to a shelf-life estimate.

    Args:
        predicted_class: Raw class name string (case-insensitive).

    Returns:
        Dict with keys: days, emoji, status
    """
    key = predicted_class.lower()
    # Try exact match first, then partial match
    if key in SHELF_LIFE_MAP:
        return SHELF_LIFE_MAP[key]
    for k, v in SHELF_LIFE_MAP.items():
        if k in key or key in k:
            return v
    # Fallback heuristic
    if any(w in key for w in ["fresh", "good", "best"]):
        return SHELF_LIFE_MAP["fresh"]
    if any(w in key for w in ["rot", "bad", "spoil", "waste"]):
        return SHELF_LIFE_MAP["rotten"]
    return {"days": "Unknown", "emoji": "⚪", "status": "Unknown"}
