"""
Data loading, preprocessing, and augmentation.
Expects folder layout:
    data/
      train/
        glioma/  meningioma/  pituitary/  no_tumor/
      test/
        glioma/  meningioma/  pituitary/  no_tumor/
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import BATCH_SIZE, CLASS_NAMES, DATA_DIR, IMG_SIZE, SEED


def _ensure_dirs(data_dir: Path) -> None:
    """Create expected class folders if they don't exist yet."""
    for split in ("train", "test"):
        for cls in CLASS_NAMES:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)


def count_images(data_dir: Path = DATA_DIR) -> dict:
    """Return image counts per split / class for a quick sanity check."""
    counts = {}
    for split in ("train", "test"):
        counts[split] = {}
        for cls in CLASS_NAMES:
            folder = data_dir / split / cls
            n = len(list(folder.glob("*"))) if folder.exists() else 0
            counts[split][cls] = n
    return counts


def create_datasets(
    data_dir: Path = DATA_DIR,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    validation_split: float = 0.15,
):
    """
    Build train / val / test tf.data pipelines from class subfolders.

    Train folder is further split into train + validation.
    Test folder is used as-is for final evaluation.
    """
    _ensure_dirs(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # ── Training set (subset) ──────────────────────────────────────────────
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
    )

    # ── Validation set (subset) ────────────────────────────────────────────
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
    )

    # ── Held-out test set ──────────────────────────────────────────────────
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def get_augmentation_layer() -> tf.keras.Sequential:
    """Light augmentation suitable for medical MRI images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def prepare_datasets(train_ds, val_ds, test_ds, augment: bool = True):
    """Augment on raw pixels, then MobileNetV2 preprocess_input. Prefetch for speed."""
    aug = get_augmentation_layer() if augment else None
    autotune = tf.data.AUTOTUNE

    def _prep(images, labels, training=False):
        images = tf.cast(images, tf.float32)
        if training and aug is not None:
            images = aug(images, training=True)
        images = preprocess_input(images)
        return images, labels

    train_ds = (
        train_ds.map(lambda x, y: _prep(x, y, training=True), num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    val_ds = (
        val_ds.map(lambda x, y: _prep(x, y, training=False), num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    test_ds = (
        test_ds.map(lambda x, y: _prep(x, y, training=False), num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    return train_ds, val_ds, test_ds


def compute_class_weights(train_ds) -> dict:
    """
    Compute sklearn-style class weights from a (possibly batched) train dataset.
    Returns {class_index: weight}.
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    labels = []
    for _, batch_y in train_ds:
        labels.extend(np.argmax(batch_y.numpy(), axis=1).tolist())

    labels = np.array(labels)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}
