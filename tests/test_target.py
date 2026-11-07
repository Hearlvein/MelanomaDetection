from __future__ import annotations

import pandas as pd
import pytest

from melanoma_detection.data import add_target


@pytest.fixture
def diagnosis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dx_original": ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"],
        }
    )


def test_melanoma_target_is_binary_and_marks_only_mel(diagnosis_frame: pd.DataFrame) -> None:
    result, info = add_target(diagnosis_frame, "melanoma")

    assert result["target"].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert result["target_label"].tolist() == [
        "melanoma",
        "non_melanoma",
        "non_melanoma",
        "non_melanoma",
        "non_melanoma",
        "non_melanoma",
        "non_melanoma",
    ]
    assert info["positive_classes"] == ["mel"]
    assert set(info["negative_classes"]) == {"nv", "bkl", "bcc", "akiec", "vasc", "df"}


def test_malignant_target_includes_melanoma_bcc_and_akiec(diagnosis_frame: pd.DataFrame) -> None:
    result, info = add_target(diagnosis_frame, "malignant")

    assert result["target"].tolist() == [1, 0, 0, 1, 1, 0, 0]
    assert result["target_label"].tolist() == [
        "malignant",
        "benign",
        "benign",
        "malignant",
        "malignant",
        "benign",
        "benign",
    ]
    assert set(info["positive_classes"]) == {"mel", "bcc", "akiec"}


def test_multiclass_target_preserves_original_diagnosis(diagnosis_frame: pd.DataFrame) -> None:
    result, info = add_target(diagnosis_frame, "multiclass")

    assert result["target"].tolist() == diagnosis_frame["dx_original"].tolist()
    assert result["target_label"].tolist() == diagnosis_frame["dx_original"].tolist()
    assert info["positive_classes"] is None
    assert info["negative_classes"] is None


def test_unknown_target_is_rejected(diagnosis_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown target definition"):
        add_target(diagnosis_frame, "not-a-real-target")


def test_target_requires_original_diagnosis() -> None:
    with pytest.raises(ValueError, match="dx_original"):
        add_target(pd.DataFrame({"dx_current": ["mel"]}), "melanoma")
