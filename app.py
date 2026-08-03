"""
Brain Tumor Classifier — local web app.
Run: python app.py  →  http://127.0.0.1:5000
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import BEST_MODEL_PATH, CLASS_NAMES, FINAL_MODEL_PATH, IMG_SIZE
from inference import load_model

# Keep TF quieter / a bit snappier for single-image inference
tf.get_logger().setLevel("ERROR")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
        # Warm-up so first real upload is not slow
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = _model(dummy, training=False)
    return _model


def predict_pil(img: Image.Image) -> dict:
    model = get_model()
    img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = preprocess_input(np.array(img, dtype=np.float32))
    batch = np.expand_dims(arr, axis=0)
    # Direct call is faster than model.predict() for single images
    probs = model(batch, training=False).numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
    }


@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400
    try:
        img = Image.open(io.BytesIO(file.read()))
        return jsonify(predict_pil(img))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    ready = _model is not None and (
        BEST_MODEL_PATH.exists() or FINAL_MODEL_PATH.exists()
    )
    return jsonify({"ok": True, "model_ready": ready})


if __name__ == "__main__":
    print("Loading + warming up model...")
    get_model()
    print("Ready. Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
