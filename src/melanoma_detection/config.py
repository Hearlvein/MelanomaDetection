"""Project-wide constants: paths, feature schema, targets, and defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Raw inputs.
FEATURES_PATH = RAW_DIR / "HAM10000_features_couleurs.xlsx"
METADATA_PATH = RAW_DIR / "HAM10000_metadata.csv"

# Deep learning uses the official HAM10000 image archive.
EXTERNAL_DATA_DIR = DATA_DIR / "external" / "HAM10000_Dataset"
IMAGE_DIR_NAME = "HAM10000"

# Data contract.
EXPECTED_ROWS = 10_015
EXPECTED_LESIONS = 7_470
EXPECTED_DX = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

# Feature schema. The frozen model was trained on exactly these columns.
# Identifiers and labels are deliberately excluded.
DEMOGRAPHIC_COLUMNS = ("age", "sex", "localization")
COLOR_COLUMNS = tuple(
    f"color_{index}_{channel}"
    for index in range(1, 4)
    for channel in ("R", "G", "B", "proportion")
)
ASYMMETRY_COLUMNS = ("asym_vert_%", "asym_horiz_%")
SHAPE_COLUMNS = ("Perimeter", "Area")

FEATURE_COLUMNS = (
    ("image_id", "dx")
    + DEMOGRAPHIC_COLUMNS
    + COLOR_COLUMNS
    + ASYMMETRY_COLUMNS
    + SHAPE_COLUMNS
)
NUMERIC_FEATURES = ("age",) + COLOR_COLUMNS + ASYMMETRY_COLUMNS + SHAPE_COLUMNS
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

# The only task kept in this project.
TARGET_DEFINITION = "melanoma"
TARGET_SPECS = {
    "melanoma": {
        "description": "Melanoma vs all non-melanoma diagnoses.",
        "positive": ("mel",),
        "negative": ("nv", "bkl", "bcc", "akiec", "vasc", "df"),
        "positive_label": "melanoma",
        "negative_label": "non_melanoma",
    },
}

# Deep-learning defaults (Keras).
DEEP_IMAGE_SIZE = (128, 128)
DEEP_BATCH_SIZE = 32
DEEP_EPOCHS = 50
DEEP_LEARNING_RATE = 1e-4
DEEP_MODELS = ("custom_cnn", "vgg16")


def interim_dataset_path() -> Path:
    return INTERIM_DIR / "ham10000_with_lesions.csv"


def split_assignment_path(target: str = TARGET_DEFINITION) -> Path:
    return PROCESSED_DIR / f"split_assignments_{target}.csv"


def dataset_report_path() -> Path:
    return METRICS_DIR / "dataset.json"


def model_comparison_path() -> Path:
    return METRICS_DIR / "model_comparison.json"


def final_test_paths() -> tuple[Path, Path]:
    return METRICS_DIR / "final_test.json", METRICS_DIR / "final_test.md"


def artifact_path(model: str = "extra_trees", target: str = TARGET_DEFINITION) -> Path:
    return MODELS_DIR / f"{model}_{target}.joblib"


def frozen_pointer_path(
    model: str = "extra_trees", target: str = TARGET_DEFINITION
) -> Path:
    return MODELS_DIR / f"{model}_{target}_frozen.json"


def deep_model_path(model: str, target: str = TARGET_DEFINITION) -> Path:
    return MODELS_DIR / f"deep_{target}_{model}.keras"


def deep_pointer_path(target: str = TARGET_DEFINITION) -> Path:
    return MODELS_DIR / f"deep_{target}_frozen.json"
