"""Image loading and batching for the deep-learning line.

The loader is pure NumPy/Pillow so it can be tested without TensorFlow. Keras
datasets are created lazily by :func:`make_tf_dataset`.

Images are returned as float32 RGB in ``[0, 255]``. Models include their own
rescaling/preprocessing layer, so the same arrays feed the custom CNN and the
VGG16 transfer model.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .. import config

DEFAULT_IMAGE_SIZE = config.DEEP_IMAGE_SIZE
DEFAULT_BATCH_SIZE = config.DEEP_BATCH_SIZE


def resolve_image_path(image_id: str, image_dir: str | Path) -> Path:
    return Path(image_dir) / f"{image_id}.jpg"


def load_rgb_image(
    path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Load one image as a resized ``H x W x 3`` float32 RGB array in ``[0, 255]``."""
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return one split, validating that it is not empty."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split {split!r}")
    if "split" not in frame.columns:
        raise ValueError("frame is missing the 'split' column")
    subset = frame.loc[frame["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"split {split!r} is empty")
    return subset


def missing_images(frame: pd.DataFrame, image_dir: str | Path) -> list[str]:
    """Return the image_ids that do not have a file on disk."""
    return [
        str(image_id)
        for image_id in frame["image_id"]
        if not resolve_image_path(str(image_id), image_dir).exists()
    ]


def assert_images_exist(
    frame: pd.DataFrame,
    image_dir: str | Path,
    *,
    examples: int = 5,
) -> None:
    missing = missing_images(frame, image_dir)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing under {image_dir}; "
            f"examples: {missing[:examples]}. See data/README.md."
        )


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply gentle geometric/photometric augmentation to one image."""
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
    rotations = int(rng.integers(0, 4))
    if rotations:
        image = np.rot90(image, k=rotations, axes=(0, 1))
    if rng.random() < 0.5:
        image = np.clip(image * float(rng.uniform(0.9, 1.1)), 0.0, 255.0)
    return np.ascontiguousarray(image)


def iter_batches(
    frame: pd.DataFrame,
    image_dir: str | Path,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    seed: int = config.SEED,
    augment: bool = False,
    drop_last: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(images, targets)`` batches for a frame.

    ``target`` must be the binary 0/1 column from the Phase 1 split assignment.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if "target" not in frame.columns:
        raise ValueError("frame is missing the 'target' column")

    frame = frame.reset_index(drop=True)
    order = np.arange(len(frame))
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)

    rng = np.random.default_rng(seed)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        if drop_last and len(indices) < batch_size:
            break

        images = np.empty((len(indices), image_size[0], image_size[1], 3), dtype=np.float32)
        targets = np.empty((len(indices),), dtype=np.float32)
        for position, index in enumerate(indices):
            row = frame.iloc[int(index)]
            image = load_rgb_image(
                resolve_image_path(str(row["image_id"]), image_dir), image_size
            )
            if augment:
                image = augment_image(image, rng)
            images[position] = image
            targets[position] = float(row["target"])
        yield images, targets


def make_tf_dataset(
    frame: pd.DataFrame,
    image_dir: str | Path,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    seed: int = config.SEED,
    augment: bool = False,
    drop_last: bool = False,
    repeat: bool = False,
) -> Any:
    """Build a batched ``tf.data.Dataset`` from a frame.

    TensorFlow is imported lazily so the rest of the package stays usable in a
    tabular-only environment. Augmentation is applied in NumPy before batching,
    which keeps the pipeline identical to :func:`iter_batches`.
    """
    import tensorflow as tf

    assert_images_exist(frame, image_dir)
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    image_size = (int(image_size[0]), int(image_size[1]))

    def generator() -> Iterator[tuple[np.ndarray, np.float32]]:
        for index in range(len(frame)):
            row = frame.iloc[index]
            image = load_rgb_image(
                resolve_image_path(str(row["image_id"]), image_dir), image_size
            )
            if augment:
                image = augment_image(image, rng)
            yield image, np.float32(row["target"])

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(len(frame), 1),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=drop_last)
    if not repeat:
        # ``from_generator`` has unknown cardinality; without this Keras cannot
        # tell where an epoch ends and warns about running out of data.
        if drop_last:
            batches = len(frame) // batch_size
        else:
            batches = math.ceil(len(frame) / batch_size)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(batches))
    return dataset.prefetch(tf.data.AUTOTUNE)
