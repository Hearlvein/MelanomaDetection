"""Handcrafted feature extraction from HAM10000 images and lesion masks.

Ported from the legacy ``FeaturesExtraction.ipynb``. Every function is pure and
operates on a single image/mask pair so it can be tested without the full
dataset.
"""

from __future__ import annotations

from .color import extract_dominant_colors
from .extract import extract_features_for_frame, extract_image_features
from .mask import (
    horizontal_asymmetry,
    shape_features,
    vertical_asymmetry,
)
from .schema import (
    EXTRACTED_FEATURES,
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V2_LEGACY,
    FEATURE_SCHEMA_VERSIONS,
)

__all__ = [
    "EXTRACTED_FEATURES",
    "FEATURE_COLUMNS_V1",
    "FEATURE_COLUMNS_V2",
    "FEATURE_COLUMNS_V2_LEGACY",
    "FEATURE_SCHEMA_VERSIONS",
    "extract_dominant_colors",
    "extract_features_for_frame",
    "extract_image_features",
    "horizontal_asymmetry",
    "shape_features",
    "vertical_asymmetry",
]
