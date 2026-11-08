from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from melanoma_detection.deep import run  # noqa: E402


def _make_images(tmp_path, count: int = 6, size: int = 24):
    image_dir = tmp_path / "HAM10000"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for index in range(count):
        image_id = f"ISIC_{index:04d}"
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(image_dir / f"{image_id}.jpg")
        rows.append(
            {
                "image_id": image_id,
                "lesion_id": f"HAM_{index:07d}",
                "target": index % 2,
                "split": "train" if index < count - 2 else "validation",
            }
        )
    return pd.DataFrame(rows), image_dir


def test_load_rgb_image_returns_the_requested_shape_and_range(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path, size=32)
    path = run.resolve_image_path(frame.iloc[0]["image_id"], image_dir)

    image = run.load_rgb_image(path, (16, 20))

    assert image.shape == (16, 20, 3)
    assert image.dtype == np.float32
    assert 0 <= image.min() <= image.max() <= 255


def test_iter_batches_covers_every_row(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path, count=7)
    batches = list(
        run.iter_batches(
            frame, image_dir, image_size=(16, 16), batch_size=3, shuffle=True
        )
    )

    assert sum(len(images) for images, _ in batches) == 7
    assert all(images.shape[1:] == (16, 16, 3) for images, _ in batches)
    assert all(targets.shape == (len(images),) for images, targets in batches)


def test_iter_batches_rejects_a_bad_batch_size(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        list(run.iter_batches(frame, image_dir, batch_size=0))


def test_split_frame_validates_the_split_name(tmp_path) -> None:
    frame, _ = _make_images(tmp_path)

    assert len(run.split_frame(frame, "train")) == 4
    with pytest.raises(ValueError, match="unknown split"):
        run.split_frame(frame, "holdout")


def test_missing_images_and_assert_images_exist(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path, count=2)
    (image_dir / f"{frame.iloc[0]['image_id']}.jpg").unlink()

    missing = run.missing_images(frame, image_dir)
    assert missing == [frame.iloc[0]["image_id"]]

    with pytest.raises(FileNotFoundError, match="images are missing"):
        run.assert_images_exist(frame, image_dir)


def test_augment_image_keeps_shape_and_range(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path, count=1)
    image = run.load_rgb_image(
        run.resolve_image_path(frame.iloc[0]["image_id"], image_dir), (16, 16)
    )

    augmented = run.augment_image(image, np.random.default_rng(0))

    assert augmented.shape == image.shape
    assert augmented.dtype == np.float32
    assert 0 <= augmented.min() <= augmented.max() <= 255


def test_balanced_class_weight_matches_the_definition() -> None:
    weights = run.balanced_class_weight([0, 0, 0, 1])

    assert weights[0] == pytest.approx(4 / (2 * 3))
    assert weights[1] == pytest.approx(4 / (2 * 1))


def test_class_weight_modes() -> None:
    frame = pd.DataFrame({"target": [0, 0, 1, 1]})

    assert run.class_weight_for(frame, "none") is None
    assert run.class_weight_for(frame, "balanced") == {0: 1.0, 1: 1.0}
    with pytest.raises(ValueError, match="unknown class-weight"):
        run.class_weight_for(frame, "mystery")


def test_history_summary_returns_the_best_epoch() -> None:
    history = {"loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.7, 0.9]}

    summary = run.history_summary(history)

    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == pytest.approx(0.7)


def test_deep_test_guard_refuses_a_second_run(tmp_path) -> None:
    metrics = tmp_path / "deep_final_test.json"

    path = run.assert_deep_test_not_yet_evaluated(
        "custom_cnn", metrics_path=metrics
    )
    assert path == metrics

    metrics.write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        run.assert_deep_test_not_yet_evaluated("custom_cnn", metrics_path=metrics)
    assert (
        run.assert_deep_test_not_yet_evaluated(
            "custom_cnn", force=True, metrics_path=metrics
        )
        == metrics
    )


def test_custom_cnn_builds_and_predicts() -> None:
    tf = pytest.importorskip("tensorflow")
    from melanoma_detection.deep import models

    model = models.build_custom_cnn((32, 32, 3), base_filters=4)
    models.compile_model(model)
    output = model(tf.zeros((2, 32, 32, 3)))

    assert tuple(output.shape) == (2, 1)


def test_vgg16_transfer_builds_without_pretrained_weights() -> None:
    tf = pytest.importorskip("tensorflow")
    from melanoma_detection.deep import models

    model = models.build_vgg16_transfer(
        (64, 64, 3), weights=None, dense_units=8
    )
    output = model(tf.zeros((1, 64, 64, 3)))

    assert tuple(output.shape) == (1, 1)
