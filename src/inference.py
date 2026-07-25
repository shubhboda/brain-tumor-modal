"""
Single-image inference: predict tumor type + confidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import BEST_MODEL_PATH, CLASS_NAMES, FINAL_MODEL_PATH, IMG_SIZE

_MODEL_CACHE: tf.keras.Model | None = None


def load_model(model_path: Path | str | None = None) -> tf.keras.Model:
    """Load and cache the trained model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and model_path is None:
        return _MODEL_CACHE

    path = Path(model_path) if model_path else None
    if path is None:
        # Prefer best checkpoint, fall back to final
        if BEST_MODEL_PATH.exists():
            path = BEST_MODEL_PATH
        elif FINAL_MODEL_PATH.exists():
            path = FINAL_MODEL_PATH
        else:
            raise FileNotFoundError(
                "No trained model found.\n"
                f"Expected: {BEST_MODEL_PATH} or {FINAL_MODEL_PATH}\n"
                "Train first: python src/train.py"
            )

    print(f"Loading model from {path} ...")
    _MODEL_CACHE = tf.keras.models.load_model(path)
    return _MODEL_CACHE


def preprocess_image(image_path: str | Path, img_size: tuple = IMG_SIZE) -> np.ndarray:
    """Load MRI image → resize → EfficientNet preprocess → batch dim."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def predict(
    image_path: str | Path,
    model_path: str | Path | None = None,
    top_k: int = 4,
) -> dict:
    """
    Predict tumor class for a single MRI image.

    Returns:
        {
          "predicted_class": str,
          "confidence": float,          # 0–1
          "all_probabilities": {class: prob, ...}
        }
    """
    model = load_model(model_path)
    batch = preprocess_image(image_path)
    probs = model.predict(batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    result = {
        "image": str(image_path),
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }
    return result


def predict_pretty(image_path: str | Path, model_path: str | Path | None = None) -> None:
    """Print a readable prediction summary."""
    result = predict(image_path, model_path=model_path)
    print(f"\nImage : {result['image']}")
    print(f"Prediction : {result['predicted_class']}")
    print(f"Confidence : {result['confidence']*100:.2f}%")
    print("\nAll class probabilities:")
    for cls, p in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  {cls:12s}  {p*100:6.2f}%  {bar}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict brain tumor type from MRI")
    parser.add_argument("image", type=str, help="Path to MRI image")
    parser.add_argument("--model", type=str, default=None, help="Optional model path")
    args = parser.parse_args()
    predict_pretty(args.image, model_path=args.model)
