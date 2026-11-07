from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from melanoma_detection.extraction import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V2_LEGACY,
    extract_dominant_colors,
    extract_features_for_frame,
    extract_image_features,
    horizontal_asymmetry,
    shape_features,
    vertical_asymmetry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rectangle_mask(
    shape: tuple[int, int] = (40, 40),
    y: int = 10,
    x: int = 10,
    height: int = 20,
    width: int = 20,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y : y + height, x : x + width] = True
    return mask


def _write_image(path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _write_mask(path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def test_schema_versions_have_the_expected_width() -> None:
    assert len(FEATURE_COLUMNS_V1) == 21
    assert len(FEATURE_COLUMNS_V2_LEGACY) == 23
    assert len(FEATURE_COLUMNS_V2) == 24
    assert "lesion_id" not in FEATURE_COLUMNS_V1
    assert "lesion_id" not in FEATURE_COLUMNS_V2_LEGACY
    assert "lesion_id" in FEATURE_COLUMNS_V2
    assert "circularity" in FEATURE_COLUMNS_V2
    assert "eccentricity" in FEATURE_COLUMNS_V2


def test_reference_v2_csv_matches_the_legacy_schema() -> None:
    reference = (
        PROJECT_ROOT / "data" / "reference" / "HAM10000_features_couleurs_v2.csv"
    )
    if not reference.exists():
        pytest.skip("reference V2 CSV is not present")
    header = pd.read_csv(reference, nrows=0).columns.tolist()
    assert header == list(FEATURE_COLUMNS_V2_LEGACY)


def test_vertical_asymmetry_is_zero_for_a_centered_rectangle() -> None:
    assert vertical_asymmetry(_rectangle_mask()) == pytest.approx(0.0)


def test_vertical_asymmetry_detects_an_offset_rectangle() -> None:
    mask = _rectangle_mask(x=0, width=10)
    # The mask area is 200 pixels and the XOR is 200 pixels.
    assert vertical_asymmetry(mask) == pytest.approx(100.0)


def test_horizontal_asymmetry_is_zero_for_a_centered_rectangle() -> None:
    assert horizontal_asymmetry(_rectangle_mask()) == pytest.approx(0.0)


def test_shape_features_of_a_rectangle() -> None:
    features = shape_features(_rectangle_mask())

    assert features["Area"] == pytest.approx(400.0)
    assert features["Perimeter"] > 0
    assert 0 < features["circularity"] <= 1
    assert features["eccentricity"] == pytest.approx(0.0, abs=1e-6)


def test_empty_mask_returns_nan_shape_features() -> None:
    features = shape_features(np.zeros((20, 20), dtype=bool))

    assert features["Area"] == 0.0
    assert np.isnan(features["circularity"])
    assert np.isnan(features["eccentricity"])


def test_dominant_colors_recover_known_colors() -> None:
    image = np.zeros((30, 30, 3), dtype=np.float32)
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True

    image[10:20, 10:13] = (1.0, 0.0, 0.0)
    image[10:20, 13:16] = (0.0, 1.0, 0.0)
    image[10:20, 16:20] = (0.0, 0.0, 1.0)

    colors, proportions = extract_dominant_colors(image, mask, n_colors=3)

    found = {tuple(np.round(color, 3)) for color in colors}
    assert found == {(255.0, 0.0, 0.0), (0.0, 255.0, 0.0), (0.0, 0.0, 255.0)}
    assert proportions.sum() == pytest.approx(1.0)


def test_dominant_colors_pad_when_fewer_pixels_than_requested() -> None:
    image = np.zeros((20, 20, 3), dtype=np.float32)
    image[5, 5] = (0.2, 0.4, 0.6)
    image[5, 6] = (0.8, 0.1, 0.3)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5, 5:7] = True

    colors, proportions = extract_dominant_colors(image, mask, n_colors=3)

    assert colors.shape == (3, 3)
    assert proportions.shape == (3,)
    assert proportions.sum() == pytest.approx(1.0)
    assert np.isnan(colors[2]).all()


def test_extract_image_features_returns_the_full_schema(tmp_path) -> None:
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[8:22, 8:22] = (120, 60, 200)
    mask = np.zeros((30, 30), dtype=bool)
    mask[8:22, 8:22] = True

    image_path = tmp_path / "ISIC_test.jpg"
    mask_path = tmp_path / "ISIC_test_segmentation.png"
    _write_image(image_path, image)
    _write_mask(mask_path, mask)

    row = extract_image_features(image_path, mask_path)

    expected = {
        "color_1_R", "color_1_G", "color_1_B", "color_1_proportion",
        "color_2_R", "color_2_G", "color_2_B", "color_2_proportion",
        "color_3_R", "color_3_G", "color_3_B", "color_3_proportion",
        "asym_vert_%", "asym_horiz_%", "Perimeter", "Area",
        "circularity", "eccentricity",
    }
    assert set(row) == expected
    assert all(np.isfinite(value) for value in row.values())


def test_extract_features_for_frame_preserves_metadata_and_lesions(tmp_path) -> None:
    external = tmp_path / "HAM10000_Dataset"
    image_dir = external / "HAM10000"
    mask_dir = external / "HAM10000_segmentations_lesion_tschandl"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rows = []
    for index in range(2):
        image_id = f"ISIC_{index:04d}"
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        image[6:18, 6:18] = (40 * (index + 1), 80, 120)
        mask = np.zeros((24, 24), dtype=bool)
        mask[6:18, 6:18] = True
        _write_image(image_dir / f"{image_id}.jpg", image)
        _write_mask(mask_dir / f"{image_id}_segmentation.png", mask)
        rows.append(
            {
                "image_id": image_id,
                "lesion_id": f"HAM_{index:07d}",
                "dx": "mel" if index == 0 else "nv",
                "age": 50.0 + index,
                "sex": "male",
                "localization": "back",
            }
        )

    metadata = pd.DataFrame(rows)
    result = extract_features_for_frame(metadata, external, progress=False)

    assert len(result) == 2
    assert result["lesion_id"].tolist() == ["HAM_0000000", "HAM_0000001"]
    assert result["dx"].tolist() == ["mel", "nv"]
    for column in FEATURE_COLUMNS_V2:
        assert column in result.columns
    assert np.isfinite(result["Area"]).all()
