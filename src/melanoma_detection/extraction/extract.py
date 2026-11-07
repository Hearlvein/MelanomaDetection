"""Orchestration for the handcrafted feature extraction.

The public entry point is :func:`extract_features_for_frame`, which takes the
official metadata frame and returns one row per image with the metadata columns
preserved (including ``lesion_id``) plus the extracted features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from .color import extract_dominant_colors
from .mask import horizontal_asymmetry, shape_features, vertical_asymmetry
from .schema import EXTRACTED_FEATURES

IMAGE_DIR_NAME = "HAM10000"
MASK_DIR_NAME = "HAM10000_segmentations_lesion_tschandl"

METADATA_COLUMNS = ("image_id", "lesion_id", "dx", "age", "sex", "localization")


def _image_paths(
    image_id: str,
    image_dir: Path,
    mask_dir: Path,
) -> tuple[Path, Path]:
    return (
        image_dir / f"{image_id}.jpg",
        mask_dir / f"{image_id}_segmentation.png",
    )


def extract_image_features(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    n_colors: int = 3,
) -> dict[str, float]:
    """Extract the colour, asymmetry, and shape features for one image."""
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L")) > 0

    colors, proportions = extract_dominant_colors(image, mask, n_colors=n_colors)

    row: dict[str, float] = {}
    for index in range(n_colors):
        row[f"color_{index + 1}_R"] = float(colors[index, 0])
        row[f"color_{index + 1}_G"] = float(colors[index, 1])
        row[f"color_{index + 1}_B"] = float(colors[index, 2])
        row[f"color_{index + 1}_proportion"] = float(proportions[index])

    row["asym_vert_%"] = vertical_asymmetry(mask)
    row["asym_horiz_%"] = horizontal_asymmetry(mask)
    row.update(shape_features(mask))
    return row


def _iter_progress(items: Iterable[Any], description: str):
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is optional
        yield from items
        return
    yield from tqdm(items, desc=description)


def extract_features_for_frame(
    metadata: pd.DataFrame,
    external_dir: str | Path,
    *,
    n_colors: int = 3,
    limit: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Extract features for every row of ``metadata``.

    ``metadata`` must contain the official columns ``image_id``, ``lesion_id``,
    ``dx``, ``age``, ``sex`` and ``localization``. Rows are returned in the same
    order and metadata columns are preserved.
    """
    missing = [column for column in METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")

    external_dir = Path(external_dir)
    image_dir = external_dir / IMAGE_DIR_NAME
    mask_dir = external_dir / MASK_DIR_NAME

    frame = metadata.copy()
    if limit is not None:
        frame = frame.head(limit).copy()

    rows: list[dict[str, Any]] = []
    items = frame[list(METADATA_COLUMNS)].itertuples(index=False)
    for record in _iter_progress(items, "Extracting features") if progress else items:
        image_id = str(record.image_id)
        image_path, mask_path = _image_paths(image_id, image_dir, mask_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"mask not found: {mask_path}")

        features = extract_image_features(image_path, mask_path, n_colors=n_colors)
        row = record._asdict()
        row["image_id"] = image_id
        row.update(features)
        rows.append(row)

    result = pd.DataFrame(rows)
    ordered = list(METADATA_COLUMNS) + list(EXTRACTED_FEATURES)
    # Preserve any extra metadata columns after the known schema.
    extra = [column for column in result.columns if column not in ordered]
    return result[ordered + extra]


def write_manifest(
    output_path: str | Path,
    *,
    metadata_path: str | Path,
    schema: str,
    rows: int,
    lesions: int,
    n_colors: int,
    command: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest payload for an extraction run."""
    import hashlib
    import sys

    import sklearn

    def _sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    output_path = Path(output_path)
    manifest: dict[str, Any] = {
        "stage": "feature_extraction",
        "schema": schema,
        "n_colors": n_colors,
        "rows": int(rows),
        "lesions": int(lesions),
        "inputs": {str(metadata_path): _sha256(metadata_path)},
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "bytes": output_path.stat().st_size,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "command": command,
    }
    if extra:
        manifest.update(extra)
    return manifest
