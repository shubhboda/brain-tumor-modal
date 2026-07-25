"""
Brain Tumor Detection & Classification — CLI entry point.

Usage:
  python src/main.py train [--data_dir PATH]
  python src/main.py evaluate [--model PATH] [--data_dir PATH]
  python src/main.py predict PATH/TO/image.jpg [--model PATH]
  python src/main.py info [--data_dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASS_NAMES, DATA_DIR, BEST_MODEL_PATH, FINAL_MODEL_PATH


def cmd_info(args):
    from preprocessing import count_images

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    print(f"Data directory : {data_dir}")
    print(f"Classes        : {CLASS_NAMES}")
    print(f"Best model     : {BEST_MODEL_PATH}  (exists={BEST_MODEL_PATH.exists()})")
    print(f"Final model    : {FINAL_MODEL_PATH} (exists={FINAL_MODEL_PATH.exists()})")
    print("\nImage counts:")
    counts = count_images(data_dir)
    for split, classes in counts.items():
        total = sum(classes.values())
        print(f"  {split} ({total}):")
        for cls, n in classes.items():
            print(f"    {cls:12s}  {n}")


def cmd_train(args):
    from train import train

    train(data_dir=args.data_dir)


def cmd_evaluate(args):
    from evaluate import evaluate

    evaluate(model_path=args.model or BEST_MODEL_PATH, data_dir=args.data_dir)


def cmd_predict(args):
    from inference import predict_pretty

    predict_pretty(args.image, model_path=args.model)


def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI Classifier (EfficientNetB0)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show dataset / model status")
    p_info.add_argument("--data_dir", type=str, default=None)
    p_info.set_defaults(func=cmd_info)

    p_train = sub.add_parser("train", help="Train the classification model")
    p_train.add_argument("--data_dir", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate on test set")
    p_eval.add_argument("--model", type=str, default=None)
    p_eval.add_argument("--data_dir", type=str, default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Predict a single MRI image")
    p_pred.add_argument("image", type=str)
    p_pred.add_argument("--model", type=str, default=None)
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
