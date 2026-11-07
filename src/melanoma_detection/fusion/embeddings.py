"""Embedding table storage and fusion-frame construction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..data import DataValidationError, sha256_file

EMBEDDING_PREFIX = "emb_"


def embedding_feature_names(n_features: int) -> tuple[str, ...]:
    return tuple(f"{EMBEDDING_PREFIX}{index:04d}" for index in range(n_features))


def embeddings_path(
    model_name: str = config.FUSION_EMBEDDING_MODEL,
    target: str = config.TARGET_DEFINITION,
) -> Path:
    return config.INTERIM_DIR / f"embeddings_{target}_{model_name}.npz"


def embeddings_manifest_path(
    model_name: str = config.FUSION_EMBEDDING_MODEL,
    target: str = config.TARGET_DEFINITION,
) -> Path:
    return config.METRICS_DIR / f"embeddings_{target}_{model_name}.json"


def save_embeddings(
    path: str | Path,
    image_ids: np.ndarray | list[str],
    embeddings: np.ndarray,
    *,
    model_name: str,
    target: str,
    metadata: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save an ``image_id`` + embedding matrix as compressed ``.npz`` plus manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_ids = np.asarray(image_ids, dtype=str)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(image_ids) != len(embeddings):
        raise ValueError(
            f"got {len(image_ids)} image_ids but {len(embeddings)} embedding rows"
        )
    if len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError("image_ids contain duplicates")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")

    np.savez_compressed(
        path,
        image_ids=image_ids,
        embeddings=embeddings,
        model_name=np.asarray(model_name),
        target=np.asarray(target),
    )

    manifest_path = Path(
        manifest_path or embeddings_manifest_path(model_name, target)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "image_embeddings",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "model_name": model_name,
        "rows": int(len(image_ids)),
        "dimensions": int(embeddings.shape[1]),
        "artifact": _relative(path),
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
    }
    if metadata:
        payload.update(metadata)
    manifest_path.write_text(
        json.dumps(_sanitise_json(payload), indent=2) + "\n", encoding="utf-8"
    )
    return path, manifest_path


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(image_ids, embeddings)`` from an embedding table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"embedding table not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        image_ids = data["image_ids"].astype(str)
        embeddings = data["embeddings"].astype(np.float32)
    if len(image_ids) != len(embeddings):
        raise DataValidationError(
            f"embedding table has {len(image_ids)} ids and {len(embeddings)} rows"
        )
    return image_ids, embeddings


def merge_embeddings(
    frame: pd.DataFrame,
    image_ids: np.ndarray | list[str],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Attach embedding columns to a frame, preserving frame order."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    table = pd.DataFrame(
        embeddings, columns=list(embedding_feature_names(embeddings.shape[1]))
    )
    collisions = [column for column in table.columns if column in frame.columns]
    if collisions:
        raise DataValidationError(
            "frame already contains embedding columns; pass the raw feature "
            f"frame instead: {collisions[:5]}"
        )
    table.insert(0, "image_id", np.asarray(image_ids, dtype=str))

    merged = frame.merge(table, on="image_id", how="left", validate="one_to_one")
    missing = merged[EMBEDDING_PREFIX + "0000"].isna()
    if missing.any():
        examples = merged.loc[missing, "image_id"].head().tolist()
        raise DataValidationError(
            f"{int(missing.sum())} frame rows have no embedding; examples: {examples}"
        )
    return merged


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


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
