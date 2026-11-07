"""Backward-compatible re-export of :mod:`melanoma_detection.schema`.

The canonical schema lives at the package level so tabular-only code can use it
without importing the extraction dependencies. ``extraction.schema`` remains a
valid import path for existing code and tests.
"""

from __future__ import annotations

from ..schema import (  # noqa: F401
    ALL_FEATURES_V1,
    ALL_FEATURES_V2,
    ASYMMETRY_COLUMNS,
    BASE_SHAPE_COLUMNS,
    CATEGORICAL_FEATURES,
    COLOR_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    EXTRACTED_FEATURES,
    EXTENDED_SHAPE_COLUMNS,
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V2_LEGACY,
    FEATURE_SCHEMA_VERSIONS,
    IDENTIFIER_COLUMNS,
    NUMERIC_FEATURES_V1,
    NUMERIC_FEATURES_V2,
)

__all__ = [
    "ALL_FEATURES_V1",
    "ALL_FEATURES_V2",
    "ASYMMETRY_COLUMNS",
    "BASE_SHAPE_COLUMNS",
    "CATEGORICAL_FEATURES",
    "COLOR_COLUMNS",
    "DEMOGRAPHIC_COLUMNS",
    "EXTRACTED_FEATURES",
    "EXTENDED_SHAPE_COLUMNS",
    "FEATURE_COLUMNS_V1",
    "FEATURE_COLUMNS_V2",
    "FEATURE_COLUMNS_V2_LEGACY",
    "FEATURE_SCHEMA_VERSIONS",
    "IDENTIFIER_COLUMNS",
    "NUMERIC_FEATURES_V1",
    "NUMERIC_FEATURES_V2",
]
