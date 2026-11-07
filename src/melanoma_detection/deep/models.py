"""Keras model builders for the melanoma deep-learning line.

Two architectures are supported:

- ``custom_cnn``: the small convolutional network from the legacy notebook.
- ``vgg16``: frozen ImageNet VGG16 features plus a small classification head.

Both models consume RGB images in ``[0, 255]`` and include their own input
preprocessing. Every model exposes a layer named ``embedding`` so the same
extractor works for late fusion.
"""

from __future__ import annotations

from typing import Any

import keras
import tensorflow as tf
from keras import layers

from .. import config

INPUT_SHAPE = (*config.DEEP_IMAGE_SIZE, 3)
VGG16_MEAN_BGR = (103.939, 116.779, 123.68)


@keras.saving.register_keras_serializable(package="melanoma_detection")
class VGG16Preprocess(layers.Layer):
    """Reproduce ``keras.applications.vgg16.preprocess_input`` as a layer.

    A named layer keeps `.keras` artifacts loadable without ``safe_mode=False``.
    """

    def call(self, inputs: Any) -> Any:
        bgr = tf.reverse(inputs, axis=[-1])
        mean = tf.constant(VGG16_MEAN_BGR, dtype=inputs.dtype)
        return bgr - mean


def build_custom_cnn(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    *,
    dropout: float = 0.5,
    base_filters: int = 32,
) -> keras.Model:
    """Small CNN from the legacy notebook, with an explicit embedding layer."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="image"),
            layers.Rescaling(1.0 / 255.0, name="rescale"),
            layers.Conv2D(base_filters, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(base_filters * 2, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(base_filters * 4, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(name="flatten"),
            layers.Dense(base_filters * 4, activation="relu", name="embedding"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid", name="prediction"),
        ],
        name="custom_cnn",
    )
    return model


def build_vgg16_transfer(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    *,
    weights: str | None = "imagenet",
    dense_units: int = 256,
    dropout: float = 0.5,
    unfreeze_last: int = 0,
) -> keras.Model:
    """Frozen VGG16 base with a binary classification head.

    ``unfreeze_last`` optionally makes the last N convolutional layers trainable
    for a light fine-tuning pass. The default keeps the base frozen, matching the
    legacy transfer-learning setup.
    """
    base_model = keras.applications.VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False
    if unfreeze_last > 0:
        for layer in base_model.layers[-unfreeze_last:]:
            layer.trainable = True
    # The base is frozen, so keep batch-normalisation statistics fixed even if
    # a future backbone contains BN layers.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    x = VGG16Preprocess(name="vgg16_preprocess")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(dense_units, activation="relu", name="embedding")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    return keras.Model(inputs, outputs, name="vgg16_transfer")


def build_vgg16_embedding_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    *,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Return the 512-dimensional pooled VGG16 features (no classification head)."""
    base_model = keras.applications.VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    x = VGG16Preprocess(name="vgg16_preprocess")(inputs)
    x = base_model(x, training=False)
    outputs = layers.GlobalAveragePooling2D(name="embedding")(x)
    return keras.Model(inputs, outputs, name="vgg16_embedding")


def build_embedding_extractor(
    model: keras.Model,
    layer_name: str = "embedding",
) -> keras.Model:
    """Wrap a trained model so it returns the named penultimate layer."""
    try:
        layer = model.get_layer(layer_name)
    except ValueError as error:
        raise ValueError(f"model has no layer named {layer_name!r}") from error
    return keras.Model(model.inputs, layer.output, name=f"{model.name}_embedding")


def compile_model(
    model: keras.Model,
    *,
    learning_rate: float = config.DEEP_LEARNING_RATE,
    metrics: list[str] | None = None,
) -> keras.Model:
    """Compile a binary model with Adam and binary cross-entropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics or ["accuracy"],
    )
    return model


def build_model(
    name: str,
    *,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    weights: str | None = "imagenet",
    dropout: float = 0.5,
    base_filters: int = 32,
    dense_units: int = 256,
    unfreeze_last: int = 0,
) -> keras.Model:
    """Build one of :data:`config.DEEP_MODELS` by name."""
    if name == "custom_cnn":
        return build_custom_cnn(
            input_shape,
            dropout=dropout,
            base_filters=base_filters,
        )
    if name == "vgg16":
        return build_vgg16_transfer(
            input_shape,
            weights=weights,
            dense_units=dense_units,
            dropout=dropout,
            unfreeze_last=unfreeze_last,
        )
    raise ValueError(f"unknown deep model {name!r}; choose from {config.DEEP_MODELS}")
