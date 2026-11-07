"""Evaluation utilities for the binary melanoma/malignant task.

All functions here are deliberately independent of any particular split file or
model. They take explicit labels, scores, thresholds, and lesion groups so the
caller cannot accidentally evaluate the wrong positive class or resample images
instead of lesions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# Continuous metrics that are meaningful to bootstrap. Counts and the threshold
# itself are reported separately.
BOOTSTRAP_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "f2",
    "accuracy",
    "balanced_accuracy",
    "average_precision",
    "roc_auc",
    "brier_score",
    "predicted_positive_rate",
)


def _as_binary_arrays(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true, dtype=int).ravel()
    y_score_array = np.asarray(y_score, dtype=float).ravel()

    if y_true_array.shape != y_score_array.shape:
        raise ValueError(
            f"y_true and y_score must have the same length; "
            f"got {len(y_true_array)} and {len(y_score_array)}"
        )
    if len(y_true_array) == 0:
        raise ValueError("y_true and y_score must not be empty")
    if not np.isin(y_true_array, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1 for the binary task")
    if not np.isfinite(y_score_array).all():
        raise ValueError("y_score contains non-finite values")

    return y_true_array, y_score_array


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float | int | dict[str, int]]:
    """Compute binary metrics with class 1 as the positive class.

    ``average_precision`` and ``roc_auc`` are threshold-independent. The other
    metrics use ``prediction = score >= threshold``.
    """
    y_true_array, y_score_array = _as_binary_arrays(y_true, y_score)
    threshold = float(threshold)

    y_pred = (y_score_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()

    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    ppv = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    accuracy = _safe_ratio(tp + tn, len(y_true_array))
    balanced_accuracy = float(np.nanmean([sensitivity, specificity]))
    predicted_positive_rate = _safe_ratio(tp + fp, len(y_true_array))
    prevalence = _safe_ratio(tp + fn, len(y_true_array))

    if np.isfinite(ppv) and np.isfinite(sensitivity) and (ppv + sensitivity) > 0:
        f1 = float(2 * ppv * sensitivity / (ppv + sensitivity))
    else:
        f1 = 0.0

    if np.isfinite(ppv) and np.isfinite(sensitivity) and (4 * ppv + sensitivity) > 0:
        f2 = float(5 * ppv * sensitivity / (4 * ppv + sensitivity))
    else:
        f2 = 0.0

    has_two_classes = len(np.unique(y_true_array)) == 2
    average_precision = (
        float(average_precision_score(y_true_array, y_score_array))
        if has_two_classes
        else float("nan")
    )
    roc_auc = (
        float(roc_auc_score(y_true_array, y_score_array))
        if has_two_classes
        else float("nan")
    )
    brier_score = float(np.mean((y_score_array - y_true_array) ** 2))

    return {
        "threshold": threshold,
        "n": int(len(y_true_array)),
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "prevalence": float(prevalence),
        "predicted_positive_rate": float(predicted_positive_rate),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "f2": float(f2),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "average_precision": float(average_precision),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier_score),
    }


def _precision_recall_thresholds(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_array, y_score_array = _as_binary_arrays(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true_array, y_score_array)
    # The final precision/recall entry corresponds to "predict everything
    # positive" and has no threshold.
    return precision[:-1], recall[:-1], thresholds


def _f2_from_precision_recall(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denominator = 4 * precision + recall
    return np.divide(
        5 * precision * recall,
        denominator,
        out=np.zeros_like(precision, dtype=float),
        where=denominator > 0,
    )


def select_f2_threshold(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
) -> float:
    """Choose the threshold that maximises F2 on the supplied scores.

    Ties are broken toward the highest threshold, which is the more specific
    operating point. The caller must pass training-only or OOF scores.
    """
    precision, recall, thresholds = _precision_recall_thresholds(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    f2 = _f2_from_precision_recall(precision, recall)
    best_value = float(np.max(f2))
    candidates = np.flatnonzero(np.isclose(f2, best_value))
    return float(thresholds[candidates[-1]])


def select_sensitivity_threshold(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    min_sensitivity: float = 0.90,
) -> float | None:
    """Choose the highest threshold that reaches ``min_sensitivity``.

    Among thresholds meeting the sensitivity constraint, the highest threshold
    gives the most specific / highest-precision operating point. Returns
    ``None`` when the constraint cannot be met.
    """
    if not 0 <= min_sensitivity <= 1:
        raise ValueError("min_sensitivity must be in [0, 1]")

    precision, recall, thresholds = _precision_recall_thresholds(y_true, y_score)
    if len(thresholds) == 0:
        return None

    valid = recall >= min_sensitivity
    if not valid.any():
        return None

    valid_indices = np.flatnonzero(valid)
    best_precision = float(np.max(precision[valid_indices]))
    candidates = valid_indices[np.isclose(precision[valid_indices], best_precision)]
    return float(thresholds[candidates[-1]])


def _group_indices(groups: np.ndarray | list[object]) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    groups_array = np.asarray(groups)
    if len(groups_array) == 0:
        raise ValueError("groups must not be empty")

    index_by_group: dict[object, list[int]] = {}
    for index, group in enumerate(groups_array):
        index_by_group.setdefault(group, []).append(index)

    unique_groups = np.asarray(list(index_by_group))
    if len(unique_groups) < 2:
        raise ValueError("at least two groups are required for cluster bootstrap")

    return unique_groups, {
        group: np.asarray(indices, dtype=int) for group, indices in index_by_group.items()
    }


def cluster_bootstrap_metric_cis(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    groups: np.ndarray | list[object],
    threshold: float,
    *,
    n_boot: int = 2_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs resampling whole lesions, not individual images.

    HAM10000 contains several images per lesion, so image-level bootstrap would
    pretend the validation set contains more independent cases than it does.
    """
    y_true_array, y_score_array = _as_binary_arrays(y_true, y_score)
    groups_array = np.asarray(groups)
    if len(groups_array) != len(y_true_array):
        raise ValueError("groups must have the same length as y_true")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    unique_groups, index_by_group = _group_indices(groups_array)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([index_by_group[group] for group in sampled_groups])
        replicate = binary_metrics(
            y_true_array[indices],
            y_score_array[indices],
            threshold=threshold,
        )
        for metric in BOOTSTRAP_METRICS:
            value = replicate[metric]
            if np.isfinite(value):
                samples[metric].append(float(value))

    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    minimum_samples = max(20, int(0.5 * n_boot))
    intervals: dict[str, dict[str, float]] = {}

    for metric, values in samples.items():
        if len(values) < minimum_samples:
            continue
        lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
        intervals[metric] = {"lower": float(lower), "upper": float(upper)}

    return intervals
