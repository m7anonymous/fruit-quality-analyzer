"""
predict.py
----------
Standalone inference script for the Fruit Quality Analyzer.

Usage:
    # Single image
    python predict.py --image path/to/apple.jpg

    # Batch of images in a folder
    python predict.py --folder path/to/images/

    # Save results to CSV
    python predict.py --folder path/to/images/ --output results.csv
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocess import load_image, get_shelf_life, IMG_SIZE
from database import save_prediction

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fruit_quality_model.h5"
META_PATH  = MODEL_DIR / "model_meta.json"
LOG_DIR    = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached globally for repeated calls from app.py / api.py)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL       = None
_CLASS_NAMES = None


def load_model_and_meta():
    """Load model and class names from disk (lazy, cached)."""
    global _MODEL, _CLASS_NAMES

    if _MODEL is not None:
        return _MODEL, _CLASS_NAMES

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}.\n"
            "Please run: python train.py --dataset dataset/"
        )

    log.info(f"Loading model from {MODEL_PATH} …")
    _MODEL = tf.keras.models.load_model(str(MODEL_PATH))

    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        _CLASS_NAMES = meta["class_names"]
        log.info(f"Classes: {_CLASS_NAMES}")
    else:
        # Fall back to numeric class names
        n = _MODEL.output_shape[-1]
        _CLASS_NAMES = [str(i) for i in range(n)]
        log.warning("model_meta.json not found; using numeric class names.")

    return _MODEL, _CLASS_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(image_path: str, save_result: bool = True) -> dict:
    """
    Run inference on a single image file.

    Args:
        image_path  : Path to the image to analyse.
        save_result : Whether to persist the result to the database.

    Returns:
        dict with keys:
            predicted_class, confidence, shelf_life, top3,
            class_names, all_probs
    """
    model, class_names = load_model_and_meta()

    # ── Pre-process ───────────────────────────────────────────────────────
    img = load_image(image_path)                    # (224, 224, 3) float32
    img_batch = np.expand_dims(img, axis=0)         # (1, 224, 224, 3)

    # ── Inference ─────────────────────────────────────────────────────────
    probs = model.predict(img_batch, verbose=0)[0]  # shape: (num_classes,)

    # ── Decode results ────────────────────────────────────────────────────
    top_idx        = int(np.argmax(probs))
    predicted_class = class_names[top_idx]
    confidence      = float(probs[top_idx])

    # Top-3 predictions
    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {"class": class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top3_indices
    ]

    # ── Shelf-life ────────────────────────────────────────────────────────
    shelf = get_shelf_life(predicted_class)

    result = {
        "predicted_class": predicted_class,
        "confidence":       round(confidence, 4),
        "shelf_life":       shelf,
        "top3":             top3,
        "class_names":      class_names,
        "all_probs":        {cls: round(float(p), 4) for cls, p in zip(class_names, probs)},
        "image_path":       image_path,
    }

    # ── Persist ───────────────────────────────────────────────────────────
    if save_result:
        try:
            save_prediction(
                filename        = Path(image_path).name,
                predicted_class = predicted_class,
                confidence      = confidence,
                shelf_life_days = shelf["days"],
                shelf_status    = shelf["status"],
                top3            = top3,
            )
        except Exception as e:
            log.warning(f"Could not save to database: {e}")

    return result


def predict_from_array(img_array: np.ndarray, filename: str = "upload") -> dict:
    """
    Run inference on an already-loaded numpy image array.

    Args:
        img_array : (H, W, 3) uint8 or float32 array.
        filename  : Label for DB storage.

    Returns:
        Same dict structure as predict_image().
    """
    import tempfile, uuid
    # Write to temp file so predict_image() can reuse its logic
    tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.jpg"
    if img_array.dtype != np.uint8:
        img_save = (img_array * 255).clip(0, 255).astype(np.uint8)
    else:
        img_save = img_array
    cv2.imwrite(str(tmp), cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
    result = predict_image(str(tmp), save_result=True)
    tmp.unlink(missing_ok=True)
    result["image_path"] = filename
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def visualise_prediction(result: dict, output_path: str = None) -> str:
    """
    Create a matplotlib figure showing the image, prediction, and confidence bar.

    Args:
        result      : Output dict from predict_image().
        output_path : If given, save figure there; else auto-generate path.

    Returns:
        Path to saved figure.
    """
    img   = load_image(result["image_path"])
    shelf = result["shelf_life"]
    top3  = result["top3"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: image
    axes[0].imshow(img)
    axes[0].axis("off")
    status_color = {"Excellent": "#4CAF50", "Good": "#8BC34A", "Fair": "#FF9800",
                    "Poor": "#FF5722", "Very Poor": "#F44336", "Expired": "#B71C1C"}.get(
        shelf["status"], "#9E9E9E"
    )
    axes[0].set_title(
        f"{shelf['emoji']}  {result['predicted_class'].upper()}\n"
        f"Confidence: {result['confidence']*100:.1f}%  |  Shelf life: {shelf['days']}",
        fontsize=13, fontweight="bold", color=status_color,
    )

    # Right: probability bar chart
    classes = [t["class"] for t in top3]
    confs   = [t["confidence"] for t in top3]
    bar_colors = [status_color if i == 0 else "#B0BEC5" for i in range(len(classes))]
    bars = axes[1].barh(classes[::-1], confs[::-1], color=bar_colors[::-1], edgecolor="white")
    axes[1].bar_label(bars, labels=[f"{c*100:.1f}%" for c in confs[::-1]], padding=4, fontsize=10)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_title("Top-3 Predictions", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Confidence")
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("🍎 Fruit Quality Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        output_path = str(LOG_DIR / f"prediction_{Path(result['image_path']).stem}.png")

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_folder(folder: str) -> list:
    """Run inference on every image in `folder`. Returns list of result dicts."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in valid_exts]
    if not paths:
        log.warning(f"No images found in {folder}")
        return []

    results = []
    for p in paths:
        try:
            res = predict_image(str(p))
            results.append(res)
            log.info(
                f"{p.name:40s} → {res['predicted_class']:12s}  "
                f"({res['confidence']*100:.1f}%)  "
                f"Shelf: {res['shelf_life']['days']}"
            )
        except Exception as e:
            log.error(f"Failed on {p.name}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fruit Quality Inference")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image",  help="Single image path")
    g.add_argument("--folder", help="Folder of images for batch inference")
    p.add_argument("--output", help="CSV file path for batch results")
    p.add_argument("--visualise", action="store_true", help="Save visualisation figure")
    return p.parse_args()


def main():
    args = parse_args()

    if args.image:
        result = predict_image(args.image)
        print("\n" + "=" * 50)
        print(f"  Predicted class : {result['predicted_class']}")
        print(f"  Confidence      : {result['confidence']*100:.2f}%")
        print(f"  Shelf life      : {result['shelf_life']['days']}")
        print(f"  Status          : {result['shelf_life']['status']}")
        print("\n  Top-3 predictions:")
        for t in result["top3"]:
            print(f"    {t['class']:15s}  {t['confidence']*100:.1f}%")
        print("=" * 50)

        if args.visualise:
            fig_path = visualise_prediction(result)
            print(f"\nFigure saved → {fig_path}")

    elif args.folder:
        results = predict_folder(args.folder)
        if args.output and results:
            import csv
            fieldnames = ["filename", "predicted_class", "confidence", "shelf_life_days", "shelf_status"]
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "filename":        Path(r["image_path"]).name,
                        "predicted_class": r["predicted_class"],
                        "confidence":      f"{r['confidence']*100:.2f}%",
                        "shelf_life_days": r["shelf_life"]["days"],
                        "shelf_status":    r["shelf_life"]["status"],
                    })
            log.info(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
