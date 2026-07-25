"""
MobileNetV2 transfer-learning model for 4-class brain tumor classification.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

from config import IMG_SIZE, NUM_CLASSES


def build_model(
    img_size: tuple = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
    trainable_base: bool = False,
) -> Model:
    inputs = layers.Input(shape=(*img_size, 3), name="input_image")

    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base.trainable = trainable_base

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=inputs, outputs=outputs, name="BrainTumorMobileNetV2")


def unfreeze_top_layers(model: Model, n_layers: int = 40) -> Model:
    """Unfreeze last n backbone layers; keep BatchNorm frozen."""
    skip = {"gap", "fc1", "dropout", "predictions", "input_image"}
    backbone_layers = [layer for layer in model.layers if layer.name not in skip]
    for layer in backbone_layers:
        layer.trainable = False
    for layer in backbone_layers[-n_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
