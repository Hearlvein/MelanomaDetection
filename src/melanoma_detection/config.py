from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_PATH = RAW_DIR / "HAM10000_features_couleurs.xlsx"
METADATA_PATH = RAW_DIR / "HAM10000_metadata.csv"

# Contract for the current HAM10000 extract.
EXPECTED_ROWS = 10_015
EXPECTED_LESIONS = 7_470
EXPECTED_DX = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

# Feature schema. Keep label/audit columns out of this list.
NUMERIC_FEATURES = (
    "age",
    "color_1_R",
    "color_1_G",
    "color_1_B",
    "color_1_proportion",
    "color_2_R",
    "color_2_G",
    "color_2_B",
    "color_2_proportion",
    "color_3_R",
    "color_3_G",
    "color_3_B",
    "color_3_proportion",
    "asym_vert_%",
    "asym_horiz_%",
    "Perimeter",
    "Area",
)
CATEGORICAL_FEATURES = ("sex", "localization")
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL_COLUMNS = (
    "image_id",
    "lesion_id",
    "dx_original",
    "dx_current",
    "dx_type",
    "target",
    "target_label",
    "fold",
    "split",
)
GROUP_COLUMN = "lesion_id"

# Splitting defaults.
SEED = 42
N_SPLITS = 5
TEST_FOLD = 0
VAL_FOLD = 1
DEFAULT_SELECT_K = 15

# The current notebooks use "melanoma vs all non-melanoma". Keep that as the
# Phase 1 default so the split can be compared with the existing work. The CLI
# supports the other definitions without editing code.
TARGET_DEFINITION = "melanoma"

TARGET_SPECS = {
    "melanoma": {
        "description": "Melanoma vs all non-melanoma diagnoses (current notebook task).",
        "positive": ("mel",),
        "negative": ("nv", "bkl", "bcc", "akiec", "vasc", "df"),
        "positive_label": "melanoma",
        "negative_label": "non_melanoma",
    },
    "malignant": {
        "description": "Malignant vs benign lesions (melanoma + BCC + AKIEC vs the rest).",
        "positive": ("mel", "bcc", "akiec"),
        "negative": ("nv", "bkl", "df", "vasc"),
        "positive_label": "malignant",
        "negative_label": "benign",
    },
    "multiclass": {
        "description": "Original seven-class HAM10000 diagnosis task.",
        "positive": None,
        "negative": None,
        "positive_label": None,
        "negative_label": None,
    },
}
