"""Training, class weighting, and artifact persistence for Keras models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..data import sha256_file


def balanced_class_weight(y: np.ndarray | list[int]) -> dict[int, float]:
    """Return balanced class weights for a binary target.

    ``{0: n / (2 * n_negative), 1: n / (2 * n_positive)}`` matches scikit-learn's
    ``class_weight='balanced'`` without adding a dependency.
    """
    y_array = np.asarray(y, dtype=int).ravel()
    if not np.isin(y_array, (0, 1)).all():
        raise ValueError("target must contain only 0 and 1")
    counts = np.bincount(y_array, minlength=2)
    if (counts == 0).any():
        raise ValueError("both classes must be present to compute class weights")
    total = counts.sum()
    return {0: float(total / (2.0 * counts[0])), 1: float(total / (2.0 * counts[1]))}


def class_weight_for(frame_or_target: Any, mode: str = "balanced") -> dict[int, float] | None:
    """Resolve a class-weight argument from a frame or an array of targets."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"unknown class-weight mode {mode!r}; choose 'balanced' or 'none'")

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
) -> list[keras.callbacks.Callback]:
    """Early stopping, LR reduction, and an optional best-checkpoint save."""
    import keras

    callbacks: list[keras.callbacks.Callback] = [
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
    model: keras.Model,
    train_dataset: Any,
    validation_dataset: Any,
    *,
    epochs: int = config.DEEP_EPOCHS,
    class_weight: dict[int, float] | None = None,
    callbacks: list[keras.callbacks.Callback] | None = None,
    steps_per_epoch: int | None = None,
    verbose: int = 2,
) -> keras.callbacks.History:
    """Fit a compiled Keras model and return its history.

    Pass ``steps_per_epoch`` when ``train_dataset`` repeats; an infinite dataset
    has no cardinality, so Keras otherwise never ends the epoch.
    """
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


def history_to_dict(history: keras.callbacks.History) -> dict[str, list[float]]:
    """Convert a Keras history to plain JSON-friendly lists."""
    result: dict[str, list[float]] = {}
    for key, values in history.history.items():
        result[key] = [float(value) for value in values]
    return result


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


def deep_model_path(target: str, model_name: str) -> Path:
    return config.MODELS_DIR / f"deep_{target}_{model_name}.keras"


def deep_pointer_path(target: str) -> Path:
    return config.MODELS_DIR / f"deep_{target}_frozen.json"


def save_deep_artifact(
    model: keras.Model,
    metadata: dict[str, Any],
    path: str | Path,
) -> tuple[Path, Path]:
    """Save a ``.keras`` model plus a JSON sidecar with its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

    payload = dict(metadata)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["artifact"] = _relative(path)
    payload["artifact_sha256"] = sha256_file(path)

    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(_sanitise_json(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    return path, sidecar_path


def load_deep_artifact(path: str | Path) -> tuple[keras.Model, dict[str, Any]]:
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


def resolve_deep_model_path(
    target: str,
    artifact: str | Path | None = None,
) -> Path:
    """Resolve an explicit artifact or the target's frozen deep-model pointer."""
    if artifact is not None:
        path = Path(artifact)
        return path if path.is_absolute() else config.PROJECT_ROOT / path

    pointer_path = deep_pointer_path(target)
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"frozen deep model pointer not found: {pointer_path}; "
            "run scripts/train_deep.py first"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    path = Path(pointer["artifact"])
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def write_pointer(target: str, model_name: str, path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write the per-target pointer used by evaluation and fusion."""
    pointer_path = deep_pointer_path(target)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "model": model_name,
        "artifact": _relative(Path(path)),
        "artifact_sha256": sha256_file(path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{key: value for key, value in metadata.items() if key in {"image_size", "seed"}},
    }
    pointer_path.write_text(
        json.dumps(_sanitise_json(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    return pointer_path


def _sanitise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)
