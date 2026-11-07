"""Penultimate feature extraction for late fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import keras

from .. import config
from .datasets import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, make_tf_dataset
from .models import build_embedding_extractor


def extract_embeddings(
    model: keras.Model,
    frame: pd.DataFrame,
    image_dir: str | Path,
    *,
    layer_name: str = "embedding",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Return one embedding row per frame row, in frame order."""
    extractor = build_embedding_extractor(model, layer_name=layer_name)
    dataset = make_tf_dataset(
        frame,
        image_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    embeddings = extractor.predict(dataset, verbose=0)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) != len(frame):
        raise RuntimeError(
            f"expected {len(frame)} embeddings, got {len(embeddings)}"
        )
    return embeddings


def extract_embeddings_with_artifact(
    artifact: str | Path,
    frame: pd.DataFrame,
    image_dir: str | Path,
    *,
    layer_name: str = "embedding",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, keras.Model]:
    """Load a saved deep artifact and extract embeddings with it."""
    artifact = Path(artifact)
    if not artifact.is_absolute():
        artifact = config.PROJECT_ROOT / artifact
    model = keras.models.load_model(artifact)
    embeddings = extract_embeddings(
        model,
        frame,
        image_dir,
        layer_name=layer_name,
        image_size=image_size,
        batch_size=batch_size,
    )
    return embeddings, model
