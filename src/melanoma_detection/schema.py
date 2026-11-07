"""Canonical feature-table schemas.

This module is deliberately dependency-free so both configuration and the data
loader can import it without pulling in the extraction stack (scikit-learn,
scikit-image, Pillow).

Two feature-table versions are supported:

- ``v1``: the 21-column workbook used by the legacy notebooks and by the frozen
  Phase 5 melanoma model (no ``circularity`` or ``eccentricity``).
- ``v2``: the extract produced by the newer extraction line. The archived
  upstream CSV has 23 columns (no ``lesion_id``); the supported
  ``scripts/extract_features.py`` output has 24 columns and keeps
  ``lesion_id`` so the lesion-grouped split can be enforced.
"""

from __future__ import annotations

IDENTIFIER_COLUMNS = ("image_id", "lesion_id", "dx")
DEMOGRAPHIC_COLUMNS = ("age", "sex", "localization")
COLOR_COLUMNS = tuple(
    f"color_{index}_{channel}"
    for index in range(1, 4)
    for channel in ("R", "G", "B", "proportion")
)
ASYMMETRY_COLUMNS = ("asym_vert_%", "asym_horiz_%")
BASE_SHAPE_COLUMNS = ("Perimeter", "Area")
EXTENDED_SHAPE_COLUMNS = ("circularity", "eccentricity")

EXTRACTED_FEATURES = (
    COLOR_COLUMNS + ASYMMETRY_COLUMNS + BASE_SHAPE_COLUMNS + EXTENDED_SHAPE_COLUMNS
)

# v1 has no lesion_id and no circularity/eccentricity.
FEATURE_COLUMNS_V1 = (
    ("image_id", "dx")
    + DEMOGRAPHIC_COLUMNS
    + COLOR_COLUMNS
    + ASYMMETRY_COLUMNS
    + BASE_SHAPE_COLUMNS
)

# The legacy V2 CSV has 23 columns: it has no ``lesion_id`` because the
# notebook dropped it before extraction.
FEATURE_COLUMNS_V2_LEGACY = (
    ("image_id", "dx")
    + DEMOGRAPHIC_COLUMNS
    + COLOR_COLUMNS
    + ASYMMETRY_COLUMNS
    + BASE_SHAPE_COLUMNS
    + EXTENDED_SHAPE_COLUMNS
)

# The supported extraction output keeps ``lesion_id`` (24 columns).
FEATURE_COLUMNS_V2 = (
    ("image_id", "lesion_id", "dx")
    + DEMOGRAPHIC_COLUMNS
    + COLOR_COLUMNS
    + ASYMMETRY_COLUMNS
    + BASE_SHAPE_COLUMNS
    + EXTENDED_SHAPE_COLUMNS
)

FEATURE_SCHEMA_VERSIONS = {
    "v1": FEATURE_COLUMNS_V1,
    "v2": FEATURE_COLUMNS_V2,
    "v2_legacy": FEATURE_COLUMNS_V2_LEGACY,
}

# Columns that are safe to use as model inputs. Identifiers and labels live
# outside these tuples.
NUMERIC_FEATURES_V1 = (
    ("age",) + COLOR_COLUMNS + ASYMMETRY_COLUMNS + BASE_SHAPE_COLUMNS
)
NUMERIC_FEATURES_V2 = NUMERIC_FEATURES_V1 + EXTENDED_SHAPE_COLUMNS
CATEGORICAL_FEATURES = ("sex", "localization")

ALL_FEATURES_V1 = NUMERIC_FEATURES_V1 + CATEGORICAL_FEATURES
ALL_FEATURES_V2 = NUMERIC_FEATURES_V2 + CATEGORICAL_FEATURES
