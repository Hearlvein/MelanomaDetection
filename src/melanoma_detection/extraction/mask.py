"""Mask-derived asymmetry and shape features.

The functions mirror the legacy notebook but guard against empty masks and
division by zero.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import perimeter, regionprops


def _as_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    return mask


def vertical_asymmetry(mask: np.ndarray) -> float:
    """Percentage of mask pixels that differ after a left/right mirror."""
    mask = _as_mask(mask)
    area = int(mask.sum())
    if area == 0:
        return float("nan")
    cols = mask.shape[1] // 2
    if cols == 0:
        return 0.0
    left = mask[:, :cols]
    right = np.fliplr(mask[:, -cols:])
    diff = np.bitwise_xor(left, right)
    return float(diff.sum() / area * 100.0)


def horizontal_asymmetry(mask: np.ndarray) -> float:
    """Percentage of mask pixels that differ after a top/bottom mirror."""
    mask = _as_mask(mask)
    area = int(mask.sum())
    if area == 0:
        return float("nan")
    rows = mask.shape[0] // 2
    if rows == 0:
        return 0.0
    top = mask[:rows, :]
    bottom = np.flipud(mask[-rows:, :])
    diff = np.bitwise_xor(top, bottom)
    return float(diff.sum() / area * 100.0)


def shape_features(mask: np.ndarray) -> dict[str, float]:
    """Return ``area``, ``Perimeter``, ``circularity`` and ``eccentricity``.

    The keys match the columns used by the legacy extract.
    """
    mask = _as_mask(mask)
    area = int(mask.sum())
    if area == 0:
        return {
            "Area": 0.0,
            "Perimeter": 0.0,
            "circularity": float("nan"),
            "eccentricity": float("nan"),
        }

    boundary = perimeter(mask)
    circularity = 4.0 * np.pi * area / (boundary**2 + 1e-8)

    props = regionprops(mask.astype(np.uint8))
    eccentricity = float(props[0].eccentricity) if props else float("nan")

    return {
        "Area": float(area),
        "Perimeter": float(boundary),
        "circularity": float(circularity),
        "eccentricity": eccentricity,
    }
