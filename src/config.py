"""
High-accuracy config for Brain Tumor Classification.
Dataset: brain 1/train|test/<class>/
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "brain 1"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASS_NAMES)

# Balanced: 90%+ target, but faster on CPU
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 6
EPOCHS_FINETUNE = 4
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-5
SEED = 42
FAST_MODE = False
MAX_TRAIN_BATCHES = None

BEST_MODEL_PATH = MODEL_DIR / "best_mobilenetv2.keras"
FINAL_MODEL_PATH = MODEL_DIR / "final_mobilenetv2.keras"
