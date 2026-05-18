"""
train.py
--------
End-to-end training pipeline for the Fruit Quality Analyzer.

Run:
    python train.py --dataset dataset/ --epochs 20 --model mobilenet

Dataset layout expected:
    dataset/
        fresh/    *.jpg ...
        rotten/   *.jpg ...
        medium/   *.jpg ...
        ...

Trained model is saved to: model/fruit_quality_model.h5
"""

import os
import sys
import argparse
import json
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import classification_report, confusion_matrix

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocess import (
    create_data_generators, get_class_names,
    IMG_SIZE, BATCH_SIZE, SEED
)
from utils.augment import get_augmentation_layer

# ── Reproducibility ───────────────────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "model"
LOG_DIR    = BASE_DIR / "logs"
MODEL_PATH = MODEL_DIR / "fruit_quality_model.h5"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1 ▸ EDA helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(generator, class_names: list, save_dir: Path):
    """Bar chart of per-class image counts."""
    counts = {}
    for cls, idx in generator.class_indices.items():
        counts[cls] = np.sum(generator.classes == idx)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.2), 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(list(counts.keys()), list(counts.values()), color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out = save_dir / "class_distribution.png"
    plt.savefig(out, dpi=120)
    plt.close()
    log.info(f"Saved class distribution → {out}")


def plot_sample_images(generator, class_names: list, save_dir: Path, n_per_class: int = 3):
    """Grid of sample images per class."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 3))
    if n_classes == 1:
        axes = [axes]

    # Collect one batch
    batch_imgs, batch_labels = next(generator)
    generator.reset()

    for row, cls in enumerate(class_names):
        cls_idx = generator.class_indices[cls]
        selected = batch_imgs[batch_labels.argmax(axis=1) == cls_idx][:n_per_class]
        for col in range(n_per_class):
            ax = axes[row][col] if n_per_class > 1 else axes[row]
            if col < len(selected):
                ax.imshow(selected[col])
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=0, labelpad=50, va="center")

    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "sample_images.png"
    plt.savefig(out, dpi=120)
    plt.close()
    log.info(f"Saved sample images → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2 ▸ Model building
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, backbone: str = "mobilenet") -> keras.Model:
    """
    Build a transfer-learning CNN with a custom classification head.

    Architecture:
        Input (224×224×3)
            → Augmentation layers (training only)
            → Pretrained backbone (frozen initially)
            → GlobalAveragePooling2D
            → Dense(256, ReLU) + BatchNorm + Dropout(0.4)
            → Dense(128, ReLU) + BatchNorm + Dropout(0.3)
            → Dense(num_classes, Softmax)

    Args:
        num_classes: Number of output classes.
        backbone   : "mobilenet" or "resnet50".

    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # ── Augmentation (applied only during training) ───────────────────────
    x = get_augmentation_layer()(inputs, training=True)

    # ── Backbone ──────────────────────────────────────────────────────────
    if backbone == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
        log.info("Using ResNet50 backbone")
    else:
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
        log.info("Using MobileNetV2 backbone")

    base.trainable = False          # Freeze pretrained weights initially
    x = base(x, training=False)

    # ── Custom head ───────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"FruitQuality_{backbone}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_and_finetune(model: keras.Model, base_model, num_layers_to_unfreeze: int = 30):
    """
    Unfreeze the top N layers of the backbone for fine-tuning with a lower LR.
    Called after initial training converges.
    """
    base_model.trainable = True
    # Freeze all but the last `num_layers_to_unfreeze` layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    log.info(f"Fine-tuning top {num_layers_to_unfreeze} layers of backbone")


# ─────────────────────────────────────────────────────────────────────────────
# 3 ▸ Training
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(checkpoint_path: Path) -> list:
    """Return standard training callbacks."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(LOG_DIR / "training_log.csv"), append=True),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4 ▸ Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_dir: Path):
    """Accuracy and loss curves side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"],     label="Train Acc",  color="#2196F3", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Acc",    color="#FF5722", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train Loss", color="#4CAF50", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#9C27B0", linewidth=2)
    ax2.set_title("Model Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Saved training curves → {out}")


def evaluate_model(model: keras.Model, val_gen, class_names: list, save_dir: Path):
    """Confusion matrix + classification report."""
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=0)
    y_pred = y_pred_proba.argmax(axis=1)
    y_true = val_gen.classes

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    out = save_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=120)
    plt.close()
    log.info(f"Saved confusion matrix → {out}")

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names)
    log.info("\nClassification Report:\n" + report)
    (save_dir / "classification_report.txt").write_text(report)


# ─────────────────────────────────────────────────────────────────────────────
# 5 ▸ Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Fruit Quality Classifier")
    p.add_argument("--dataset", default="dataset/",    help="Path to dataset root")
    p.add_argument("--epochs",  default=20, type=int,  help="Max training epochs")
    p.add_argument("--model",   default="mobilenet",   choices=["mobilenet", "resnet50"])
    p.add_argument("--finetune", action="store_true",  help="Fine-tune backbone after initial training")
    p.add_argument("--finetune_epochs", default=10, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)

    if not dataset_dir.exists():
        log.error(f"Dataset directory not found: {dataset_dir}")
        log.info("Please download the dataset from Kaggle and place it in the 'dataset/' folder.")
        log.info("https://www.kaggle.com/datasets/graviton17/fruit-and-vegetable-quality-levelwise-dataset")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  Fruit Quality Analyzer — Training Pipeline")
    log.info("=" * 60)
    log.info(f"Dataset : {dataset_dir.resolve()}")
    log.info(f"Backbone: {args.model}")
    log.info(f"Epochs  : {args.epochs}")
    log.info(f"GPU     : {tf.config.list_physical_devices('GPU')}")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("\n[1/5] Loading data generators …")
    train_gen, val_gen, class_names = create_data_generators(str(dataset_dir))
    num_classes = len(class_names)
    log.info(f"Classes ({num_classes}): {class_names}")
    log.info(f"Train samples : {train_gen.samples}")
    log.info(f"Val   samples : {val_gen.samples}")

    # ── EDA ───────────────────────────────────────────────────────────────
    log.info("\n[2/5] Generating EDA plots …")
    plot_class_distribution(train_gen, class_names, LOG_DIR)
    plot_sample_images(train_gen, class_names, LOG_DIR)

    # ── Model ─────────────────────────────────────────────────────────────
    log.info("\n[3/5] Building model …")
    model, base_model = build_model(num_classes, backbone=args.model)
    model.summary(print_fn=log.info)

    # ── Training ──────────────────────────────────────────────────────────
    log.info("\n[4/5] Training …")
    callbacks = get_callbacks(MODEL_DIR / "best_checkpoint.h5")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Optional fine-tuning phase
    if args.finetune:
        log.info("\n[4b] Fine-tuning backbone …")
        unfreeze_and_finetune(model, base_model)
        ft_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.finetune_epochs,
            callbacks=get_callbacks(MODEL_DIR / "best_finetuned.h5"),
            verbose=1,
        )
        # Merge histories
        for k in history.history:
            history.history[k].extend(ft_history.history.get(k, []))

    # ── Evaluation ────────────────────────────────────────────────────────
    log.info("\n[5/5] Evaluating …")
    plot_training_curves(history, LOG_DIR)
    evaluate_model(model, val_gen, class_names, LOG_DIR)

    # ── Save artefacts ────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    log.info(f"Model saved → {MODEL_PATH}")

    # Save class names alongside the model
    meta = {
        "class_names": class_names,
        "img_size": list(IMG_SIZE),
        "backbone": args.model,
        "trained_at": datetime.utcnow().isoformat(),
        "num_classes": num_classes,
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Training complete ✓")


if __name__ == "__main__":
    main()
