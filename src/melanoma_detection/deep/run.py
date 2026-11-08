"""Image loading, Keras training, and evaluation for the deep-learning line.

TensorFlow is imported lazily so the tabular environment never needs it. The
image loader and batching helpers are pure NumPy/Pillow and are unit tested
without TensorFlow.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .. import config
from ..data import sha256_file
from ..experiment import (
    evaluate_at_threshold,
    relative_path,
    sanitise_json,
    threshold_bundle,
)

DEFAULT_IMAGE_SIZE = config.DEEP_IMAGE_SIZE
DEFAULT_BATCH_SIZE = config.DEEP_BATCH_SIZE


# ---------------------------------------------------------------------------
# Images and batching
# ---------------------------------------------------------------------------

def resolve_image_path(image_id: str, image_dir: str | Path) -> Path:
    return Path(image_dir) / f"{image_id}.jpg"


def load_rgb_image(
    path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Load one image as a resized ``H x W x 3`` float32 RGB array in ``[0, 255]``."""
    with Image.open(path) as handle:
        # ``image_size`` is (height, width); Pillow's resize expects (width, height).
        size = (int(image_size[1]), int(image_size[0]))
        image = handle.convert("RGB").resize(size, Image.Resampling.BILINEAR)
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
            f"examples: {missing[:examples]}"
        )


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply gentle geometric and photometric augmentation to one image."""
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
    """Yield ``(images, targets)`` batches for a frame."""
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

        images = np.empty(
            (len(indices), image_size[0], image_size[1], 3), dtype=np.float32
        )
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
    """Build a batched ``tf.data.Dataset`` from a frame."""
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
        if drop_last:
            batches = len(frame) // batch_size
        else:
            batches = math.ceil(len(frame) / batch_size)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(batches))
    return dataset.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Class weights and training
# ---------------------------------------------------------------------------

def balanced_class_weight(y: np.ndarray | list[int]) -> dict[int, float]:
    """Return ``{0: n / (2 * n_negative), 1: n / (2 * n_positive)}``."""
    y_array = np.asarray(y, dtype=int).ravel()
    if not np.isin(y_array, (0, 1)).all():
        raise ValueError("target must contain only 0 and 1")
    counts = np.bincount(y_array, minlength=2)
    if (counts == 0).any():
        raise ValueError("both classes must be present to compute class weights")
    total = counts.sum()
    return {
        0: float(total / (2.0 * counts[0])),
        1: float(total / (2.0 * counts[1])),
    }


def class_weight_for(
    frame_or_target: Any, mode: str = "balanced"
) -> dict[int, float] | None:
    """Resolve a class-weight argument from a frame or an array of targets."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(
            f"unknown class-weight mode {mode!r}; choose 'balanced' or 'none'"
        )
    if hasattr(frame_or_target, "columns"):
        target = frame_or_target["target"].to_numpy()
    else:
        target = np.asarray(frame_or_target)
    return balanced_class_weight(target)


def default_callbacks(
    checkpoint_path: str | Path | None = None,
    *,
    patience: int = 8,
    min_delta: float = 1e-4,
) -> list[Any]:
    """Early stopping, learning-rate reduction, and an optional checkpoint."""
    import keras

    callbacks: list[Any] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )
    return callbacks


def train_model(
    model: Any,
    train_dataset: Any,
    validation_dataset: Any,
    *,
    epochs: int = config.DEEP_EPOCHS,
    class_weight: dict[int, float] | None = None,
    callbacks: list[Any] | None = None,
    steps_per_epoch: int | None = None,
    verbose: int = 2,
) -> Any:
    """Fit a compiled Keras model and return its history."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        shuffle=False,
        verbose=verbose,
    )


def history_to_dict(history: Any) -> dict[str, list[float]]:
    """Convert a Keras history to plain JSON-friendly lists."""
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def history_summary(history: dict[str, list[float]]) -> dict[str, Any]:
    """Return the best epoch and its validation metrics."""
    if "val_loss" not in history or not history["val_loss"]:
        return {}
    best_index = int(np.argmin(history["val_loss"]))
    summary: dict[str, Any] = {"best_epoch": best_index + 1}
    for key, values in history.items():
        if best_index < len(values):
            summary[f"best_{key}"] = float(values[best_index])
    return summary


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def deep_train_report_path(model_name: str) -> Path:
    return config.METRICS_DIR / f"deep_train_{model_name}.json"


def save_deep_artifact(
    model: Any,
    metadata: dict[str, Any],
    path: str | Path,
) -> tuple[Path, Path]:
    """Save a ``.keras`` model plus a JSON sidecar with its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

    payload = dict(metadata)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["artifact"] = relative_path(path)
    payload["artifact_sha256"] = sha256_file(path)

    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(
        json_payload(payload), encoding="utf-8"
    )
    return path, sidecar_path


def load_deep_artifact(path: str | Path) -> tuple[Any, dict[str, Any]]:
    import json

    import keras

    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"deep model artifact not found: {path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"deep model sidecar not found: {sidecar_path}")
    model = keras.models.load_model(path)
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return model, metadata


def resolve_deep_model_path(artifact: str | Path | None = None) -> Path:
    """Resolve an explicit artifact or the frozen deep-model pointer."""
    if artifact is not None:
        path = Path(artifact)
        return path if path.is_absolute() else config.PROJECT_ROOT / path

    pointer_path = config.deep_pointer_path()
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"frozen deep model pointer not found: {pointer_path}; "
            "run 04_deep_learning.py train first"
        )
    import json

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    path = Path(pointer["artifact"])
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def write_pointer(
    model_name: str, path: str | Path, metadata: dict[str, Any]
) -> Path:
    """Write the frozen deep-model pointer used by evaluation."""
    pointer_path = config.deep_pointer_path()
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": config.TARGET_DEFINITION,
        "model": model_name,
        "artifact": relative_path(Path(path)),
        "artifact_sha256": sha256_file(path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{
            key: value
            for key, value in metadata.items()
            if key in {"image_size", "seed"}
        },
    }
    pointer_path.write_text(json_payload(payload), encoding="utf-8")
    return pointer_path


def json_payload(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(sanitise_json(payload), indent=2) + "\n"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def predict_scores(model: Any, dataset: Any) -> np.ndarray:
    """Return the positive-class score for every example in a batched dataset."""
    scores = model.predict(dataset, verbose=0)
    return np.asarray(scores, dtype=float).reshape(-1)


def select_thresholds(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    min_sensitivity: float = 0.90,
) -> dict[str, float | None]:
    """Select the F2 and high-sensitivity thresholds from validation scores."""
    return threshold_bundle(np.asarray(y_true), np.asarray(scores), min_sensitivity=min_sensitivity)


def evaluate_frame_scores(
    frame: pd.DataFrame,
    scores: np.ndarray | list[float],
    threshold: float,
    *,
    bootstrap_reps: int = 2_000,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Evaluate scores against a frame's target using lesion-clustered CIs."""
    if len(frame) != len(scores):
        raise ValueError(
            f"frame has {len(frame)} rows but {len(scores)} scores were supplied"
        )
    return evaluate_at_threshold(
        frame["target"].to_numpy(),
        np.asarray(scores, dtype=float),
        frame["lesion_id"].to_numpy(),
        threshold=threshold,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )


def deep_test_paths(
    model_name: str,
    *,
    metrics_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> tuple[Path, Path]:
    default_metrics = config.METRICS_DIR / f"deep_final_test_{model_name}.json"
    default_markdown = config.METRICS_DIR / f"deep_final_test_{model_name}.md"
    return (
        Path(metrics_path) if metrics_path is not None else default_metrics,
        Path(markdown_path) if markdown_path is not None else default_markdown,
    )


def assert_deep_test_not_yet_evaluated(
    model_name: str,
    *,
    force: bool = False,
    metrics_path: str | Path | None = None,
) -> Path:
    """Refuse to overwrite an existing deep test report unless forced."""
    path, _ = deep_test_paths(model_name, metrics_path=metrics_path)
    if path.exists() and not force:
        raise FileExistsError(
            f"deep test report already exists: {path}. "
            "The test split must be evaluated once. Pass --force only if you "
            "intentionally need to overwrite the frozen result."
        )
    return path
