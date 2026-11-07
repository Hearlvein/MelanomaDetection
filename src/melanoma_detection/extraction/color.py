"""Deterministic dominant-colour extraction with KMeans.

Only pixels inside the lesion mask are used. Colours are returned in the same
order as the KMeans cluster labels, matching the legacy extract so that the
generated table is schema-compatible with ``HAM10000_features_couleurs_v2.csv``.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image, 0.0, 1.0)
    return np.clip(image, 0, 255) / 255.0


def extract_dominant_colors(
    image: np.ndarray,
    mask: np.ndarray,
    n_colors: int = 3,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(colors, proportions)`` for pixels inside ``mask``.

    ``colors`` has shape ``(n_colors, 3)`` in RGB 0-255 and ``proportions`` sums
    to 1. If the mask contains fewer distinct pixels than ``n_colors``, the
    missing rows are padded with ``nan`` and a zero proportion.
    """
    image = _as_float_rgb(image)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {image.shape[:2]}"
        )

    pixels = image[mask]
    if pixels.size == 0:
        raise ValueError("mask contains no pixels")

    effective_colors = min(int(n_colors), len(pixels))
    if effective_colors < 1:
        raise ValueError("n_colors must be at least 1")

    kmeans = KMeans(
        n_clusters=effective_colors,
        random_state=random_state,
        n_init=10,
    ).fit(pixels * 255.0)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / counts.sum()

    colors = np.full((n_colors, 3), np.nan)
    proportions_full = np.zeros(n_colors)
    colors[: len(kmeans.cluster_centers_)] = kmeans.cluster_centers_
    proportions_full[: len(proportions)] = proportions
    return colors, proportions_full
