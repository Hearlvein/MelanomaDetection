from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from melanoma_detection import config
from melanoma_detection.data import load_modeling_frame
from melanoma_detection.features import build_preprocessor, build_selector
from melanoma_detection.pipeline import build_pipeline, selected_feature_count


def _toy_frame(
    age: list[float],
    sex: list[str],
    localization: list[str],
) -> pd.DataFrame:
    n_rows = len(age)
    data: dict[str, object] = {"image_id": [f"img_{i}" for i in range(n_rows)]}
    for offset, column in enumerate(config.NUMERIC_FEATURES):
        data[column] = np.arange(n_rows, dtype=float) + offset
    data["age"] = age
    data["sex"] = sex
    data["localization"] = localization
    return pd.DataFrame(data)


def test_preprocessor_statistics_are_learned_from_train_only() -> None:
    train = _toy_frame(
        age=[10.0, 20.0, np.nan, 30.0],
        sex=["male", "female", "male", "female"],
        localization=["back", "trunk", "back", "trunk"],
    )
    test = _toy_frame(
        age=[1000.0, 2000.0],
        sex=["unknown_sex", "male"],
        localization=["unknown_location", "back"],
    )

    preprocessor = build_preprocessor(scale_numeric=True)
    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    numeric_pipeline = preprocessor.named_transformers_["num"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    # Median and mean of age must come from train, not from the extreme test ages.
    assert imputer.statistics_[0] == pytest.approx(20.0)
    assert scaler.mean_[0] == pytest.approx(20.0)

    # Unknown categories seen only in test must not enter the encoder's category list.
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    assert "unknown_sex" not in encoder.categories_[0]
    assert "unknown_location" not in encoder.categories_[1]

    # Unknown categories are encoded as all zeros, with no exception.
    assert transformed.shape == (2, len(config.NUMERIC_FEATURES) + 4)
    assert np.allclose(transformed[0, len(config.NUMERIC_FEATURES) :], 0.0)


def test_preprocessor_can_skip_scaling_for_tree_models() -> None:
    train = _toy_frame(
        age=[10.0, 20.0, np.nan, 30.0],
        sex=["male", "female", "male", "female"],
        localization=["back", "trunk", "back", "trunk"],
    )
    preprocessor = build_preprocessor(scale_numeric=False)
    preprocessor.fit(train)

    numeric_steps = preprocessor.named_transformers_["num"].named_steps
    assert "scaler" not in numeric_steps


def test_preprocessor_can_use_feature_subsets() -> None:
    train = _toy_frame(
        age=[10.0, 20.0, np.nan, 30.0],
        sex=["male", "female", "male", "female"],
        localization=["back", "trunk", "back", "trunk"],
    )
    preprocessor = build_preprocessor(
        scale_numeric=True,
        numeric_features=("age",),
        categorical_features=(),
    )
    transformed = preprocessor.fit_transform(train)

    assert transformed.shape == (4, 1)
    assert np.isfinite(transformed).all()


def test_selector_can_be_disabled() -> None:
    assert build_selector(None) == "passthrough"


def test_pipeline_fits_and_selects_inside_the_pipeline() -> None:
    train = _toy_frame(
        age=[10.0, 20.0, np.nan, 30.0, 40.0, 50.0],
        sex=["male", "male", "female", "female", "male", "female"],
        localization=["back", "trunk", "trunk", "back", "back", "trunk"],
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    pipeline = build_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        scale_numeric=True,
        k=2,
    )
    pipeline.fit(train, y)

    predictions = pipeline.predict(train)
    assert len(predictions) == len(train)
    assert selected_feature_count(pipeline) == 2


def test_feature_and_label_columns_are_disjoint() -> None:
    assert set(config.ALL_FEATURES).isdisjoint(config.LABEL_COLUMNS)
    assert config.GROUP_COLUMN in config.LABEL_COLUMNS


@pytest.mark.integration
def test_real_modeling_frame_can_be_loaded() -> None:
    interim_path = config.INTERIM_DIR / "ham10000_with_lesions.csv"
    split_path = config.PROCESSED_DIR / "split_assignments_melanoma.csv"
    if not interim_path.exists() or not split_path.exists():
        pytest.skip("Phase 1 artifacts are not present")

    frame = load_modeling_frame("melanoma")

    assert len(frame) == config.EXPECTED_ROWS
    assert set(config.ALL_FEATURES).issubset(frame.columns)
    assert set(frame["split"].unique()) == {"train", "validation", "test"}
    assert int(frame["target"].sum()) == 1_113
