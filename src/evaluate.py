"""
Evaluation: confusion matrix, classification report, sensitivity / specificity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import BEST_MODEL_PATH, CLASS_NAMES, DATA_DIR, OUTPUT_DIR
from preprocessing import create_datasets, prepare_datasets


def _collect_predictions(model, test_ds):
    """Return y_true, y_pred (int labels) and confidence scores."""
    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probs, axis=1))
        y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def sensitivity_specificity(cm: np.ndarray) -> dict:
    """
    Per-class sensitivity (recall) and specificity from confusion matrix.
    Sensitivity_i = TP_i / (TP_i + FN_i)
    Specificity_i = TN_i / (TN_i + FP_i)
    """
    metrics = {}
    n = cm.shape[0]
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        metrics[name] = {"sensitivity": sens, "specificity": spec}
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    out = save_path or (OUTPUT_DIR / "confusion_matrix.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {out}")


def evaluate(
    model_path: Path | str = BEST_MODEL_PATH,
    data_dir: Path | str | None = None,
):
    """Load best model and run full evaluation on the test set."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun training first: python src/train.py"
        )

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    train_raw, val_raw, test_raw = create_datasets(data_dir=data_dir)
    _, _, test_ds = prepare_datasets(train_raw, val_raw, test_raw, augment=False)

    y_true, y_pred, _ = _collect_predictions(model, test_ds)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"{'='*50}\n")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm)

    sens_spec = sensitivity_specificity(cm)
    print("\nSensitivity / Specificity per class:")
    for cls, m in sens_spec.items():
        print(f"  {cls:12s}  sens={m['sensitivity']:.4f}  spec={m['specificity']:.4f}")

    # Save text report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
        f.write("\n\nSensitivity / Specificity:\n")
        for cls, m in sens_spec.items():
            f.write(f"  {cls}: sens={m['sensitivity']:.4f}, spec={m['specificity']:.4f}\n")
    print(f"\nReport saved → {report_path}")

    return {"accuracy": acc, "report": report, "cm": cm, "sens_spec": sens_spec}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate brain tumor classifier")
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    evaluate(model_path=args.model, data_dir=args.data_dir)
