"""
FAST training on brain 1 dataset (MobileNetV2, few epochs, light aug).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    BEST_MODEL_PATH,
    EPOCHS_FINETUNE,
    EPOCHS_FROZEN,
    FAST_MODE,
    FINAL_MODEL_PATH,
    FINETUNE_LR,
    LEARNING_RATE,
    MAX_TRAIN_BATCHES,
    MODEL_DIR,
    OUTPUT_DIR,
)
from model import build_model, compile_model, unfreeze_top_layers
from preprocessing import (
    compute_class_weights,
    count_images,
    create_datasets,
    prepare_datasets,
)


def get_callbacks(checkpoint_path: Path = BEST_MODEL_PATH) -> list:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patience = 2 if FAST_MODE else 4
    lr_patience = 1 if FAST_MODE else 2
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def plot_history(history_frozen, history_finetune=None, save_path: Path | None = None):
    acc = list(history_frozen.history["accuracy"])
    val_acc = list(history_frozen.history["val_accuracy"])
    loss = list(history_frozen.history["loss"])
    val_loss = list(history_frozen.history["val_loss"])

    if history_finetune is not None:
        acc += history_finetune.history["accuracy"]
        val_acc += history_finetune.history["val_accuracy"]
        loss += history_finetune.history["loss"]
        val_loss += history_finetune.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    freeze_end = len(history_frozen.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, acc, label="Train Acc")
    axes[0].plot(epochs, val_acc, label="Val Acc")
    if history_finetune is not None:
        axes[0].axvline(freeze_end + 0.5, color="gray", linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss, label="Train Loss")
    axes[1].plot(epochs, val_loss, label="Val Loss")
    if history_finetune is not None:
        axes[1].axvline(freeze_end + 0.5, color="gray", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or (OUTPUT_DIR / "training_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved training curves → {out}")


def train(data_dir=None):
    from config import DATA_DIR

    data_dir = Path(data_dir) if data_dir else DATA_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer GPU if present; mixed precision speeds GPU a lot
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print(f"Using GPU: {gpus}")
        except Exception as e:
            print(f"GPU setup skipped: {e}")
    else:
        print("No GPU detected - using CPU (full accuracy training)")

    counts = count_images(data_dir)
    print("Dataset image counts:")
    for split, classes in counts.items():
        total = sum(classes.values())
        print(f"  {split}: {total} images → {classes}")
        if total == 0:
            raise FileNotFoundError(
                f"No images in {data_dir / split}. Expected train/ and test/ class folders."
            )

    print("\nLoading datasets...")
    train_raw, val_raw, test_raw = create_datasets(data_dir=data_dir)
    # Light / no heavy aug in fast mode
    train_ds, val_ds, test_ds = prepare_datasets(
        train_raw, val_raw, test_raw, augment=not FAST_MODE
    )

    if MAX_TRAIN_BATCHES:
        train_ds = train_ds.take(MAX_TRAIN_BATCHES)
        print(f"FAST subset: limiting to {MAX_TRAIN_BATCHES} train batches/epoch")

    print("Computing class weights...")
    train_for_weights, _, _ = create_datasets(data_dir=data_dir)
    class_weights = compute_class_weights(train_for_weights)
    print(f"  Class weights: {class_weights}")

    callbacks = get_callbacks()

    print("\n=== Phase 1: Frozen MobileNetV2 ===")
    model = build_model(trainable_base=False)
    model = compile_model(model, learning_rate=LEARNING_RATE)

    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    history_finetune = None
    if (not FAST_MODE) and EPOCHS_FINETUNE > 0:
        print("\n=== Phase 2: Fine-tuning last layers ===")
        model = unfreeze_top_layers(model, n_layers=40)
        model = compile_model(model, learning_rate=FINETUNE_LR)
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINETUNE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    model.save(FINAL_MODEL_PATH)
    print(f"\nFinal model saved → {FINAL_MODEL_PATH}")
    plot_history(history_frozen, history_finetune)

    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    print(f"Test results: {results}")
    return model, history_frozen, history_finetune


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train brain tumor classifier (FAST)")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    train(data_dir=args.data_dir)
