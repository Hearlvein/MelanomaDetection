"""Feature pipelines, imbalance strategies, and model factories.

Everything here is a *specification*: no label is ever touched. The pipeline
objects must be fitted inside a training fold so preprocessing, feature
selection, and resampling never see validation or test data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "extra_trees",
)
TREE_MODELS = ("random_forest", "extra_trees")

CLASS_WEIGHT_STRATEGIES = ("none", "balanced", "balanced_subsample")
SAMPLER_STRATEGIES = ("random_over", "random_under", "smote")
ALL_STRATEGIES = CLASS_WEIGHT_STRATEGIES + SAMPLER_STRATEGIES
# The small comparison used by ``02_compare_models.py``: no correction, class
# weights, and one resampling method. Three strategies x four models keeps the
# experiment short enough to read and rerun.
DEFAULT_STRATEGIES = ("none", "balanced", "smote")

SELECT_K_VALUES = (10, 15, 20, 25, 30)
COLOUR_FEATURES = tuple(
    feature for feature in config.NUMERIC_FEATURES if feature.startswith("color_")
)
SHAPE_FEATURES = tuple(
    feature
    for feature in config.NUMERIC_FEATURES
    if feature in {"asym_vert_%", "asym_horiz_%", "Perimeter", "Area"}
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_numeric_pipeline(*, scale_numeric: bool = True) -> Pipeline:
    """Impute numeric features and optionally standardise them."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_categorical_pipeline() -> Pipeline:
    """Impute categorical features and one-hot encode them."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor(
    *,
    scale_numeric: bool = True,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build the leak-free feature preprocessor.

    ``numeric_features`` and ``categorical_features`` default to the full schema
    and can be overridden by the feature-subset baselines.
    """
    numeric = tuple(
        config.NUMERIC_FEATURES if numeric_features is None else numeric_features
    )
    categorical = tuple(
        config.CATEGORICAL_FEATURES
        if categorical_features is None
        else categorical_features
    )
    if not numeric and not categorical:
        raise ValueError("at least one numeric or categorical feature is required")

    transformers = []
    if numeric:
        transformers.append(
            ("num", build_numeric_pipeline(scale_numeric=scale_numeric), list(numeric))
        )
    if categorical:
        transformers.append(
            ("cat", build_categorical_pipeline(), list(categorical))
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_selector(
    k: int | str | None = config.DEFAULT_SELECT_K,
) -> SelectKBest | str:
    """Return a univariate selector or passthrough."""
    if k is None:
        return "passthrough"
    return SelectKBest(score_func=f_classif, k=k)


def build_pipeline(
    classifier: Any,
    *,
    scale_numeric: bool = True,
    k: int | str | None = config.DEFAULT_SELECT_K,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> Pipeline:
    """Build preprocess -> select -> classify as one pipeline."""
    steps: list[tuple[str, Any]] = [
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=scale_numeric,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            ),
        )
    ]
    if k is not None:
        steps.append(("selector", build_selector(k)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def selected_feature_count(pipeline: Pipeline) -> int | None:
    """Number of features kept by the selector, or ``None`` without a selector."""
    if "selector" not in pipeline.named_steps:
        return None
    selector = pipeline.named_steps["selector"]
    if not hasattr(selector, "get_support"):
        return None
    return int(selector.get_support().sum())


# ---------------------------------------------------------------------------
# Imbalance strategies
# ---------------------------------------------------------------------------

def class_weight_for(strategy: str, model_name: str) -> str | None:
    """Map a strategy name to a valid ``class_weight`` for the model."""
    if strategy == "none":
        return None
    if strategy == "balanced":
        return "balanced"
    if strategy == "balanced_subsample":
        if model_name not in TREE_MODELS:
            raise ValueError(
                f"balanced_subsample is not supported for {model_name!r}; "
                "use 'balanced' or a resampling strategy"
            )
        return "balanced_subsample"
    if strategy in SAMPLER_STRATEGIES:
        return None
    raise ValueError(f"unknown imbalance strategy: {strategy!r}")


def make_sampler(strategy: str, *, seed: int = config.SEED) -> Any | None:
    """Build a training-fold-only sampler, or ``None`` for weight strategies."""
    if strategy in CLASS_WEIGHT_STRATEGIES:
        return None
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=seed)
    if strategy == "smote":
        return SMOTE(random_state=seed)
    raise ValueError(f"unknown imbalance strategy: {strategy!r}")


def strategies_for_model(model_name: str) -> tuple[str, ...]:
    """Return the strategies that are valid for a model family."""
    if model_name in TREE_MODELS:
        return ALL_STRATEGIES
    return tuple(
        strategy for strategy in ALL_STRATEGIES if strategy != "balanced_subsample"
    )


def build_imbalance_pipeline(
    classifier: Any,
    *,
    strategy: str,
    scale_numeric: bool = True,
    k: int | str | None = config.DEFAULT_SELECT_K,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    seed: int = config.SEED,
) -> Pipeline | ImbalancedPipeline:
    """Build preprocess -> select -> optional sampler -> classify.

    The sampler only runs during ``fit``. Validation and test rows are never
    resampled.
    """
    sampler = make_sampler(strategy, seed=seed)
    steps: list[tuple[str, Any]] = [
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=scale_numeric,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            ),
        )
    ]
    if k is not None:
        steps.append(("selector", build_selector(k)))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", classifier))

    if sampler is not None:
        return ImbalancedPipeline(steps)
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def build_logistic_regression(class_weight: str | None) -> LogisticRegression:
    return LogisticRegression(
        max_iter=5_000,
        solver="liblinear",
        class_weight=class_weight,
        random_state=config.SEED,
    )


def build_random_forest(class_weight: str | None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=config.SEED,
        n_jobs=-1,
    )


def build_hist_gradient_boosting(
    class_weight: str | None,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=False,
        class_weight=class_weight,
        random_state=config.SEED,
    )


def build_extra_trees(class_weight: str | None) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=config.SEED,
        n_jobs=-1,
    )


def candidate_specs() -> list[dict[str, Any]]:
    """Return the four model families with their tuning search spaces."""
    selector_space = list(SELECT_K_VALUES)
    return [
        {
            "name": "logistic_regression",
            "label": "Logistic Regression",
            "factory": build_logistic_regression,
            "scale_numeric": True,
            "param_distributions": {
                "classifier__C": loguniform(1e-3, 1e2),
                "classifier__penalty": ["l1", "l2"],
                "selector__k": selector_space,
            },
        },
        {
            "name": "random_forest",
            "label": "Random Forest",
            "factory": build_random_forest,
            "scale_numeric": False,
            "param_distributions": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", "log2", None],
                "selector__k": selector_space,
            },
        },
        {
            "name": "hist_gradient_boosting",
            "label": "HistGradientBoosting",
            "factory": build_hist_gradient_boosting,
            "scale_numeric": False,
            "param_distributions": {
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_iter": [100, 200, 300],
                "classifier__max_leaf_nodes": [15, 31, 63],
                "classifier__min_samples_leaf": [10, 20, 40],
                "classifier__l2_regularization": [0.0, 0.1, 1.0],
                "selector__k": selector_space,
            },
        },
        {
            "name": "extra_trees",
            "label": "ExtraTrees",
            "factory": build_extra_trees,
            "scale_numeric": False,
            "param_distributions": {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", None],
                "selector__k": selector_space,
            },
        },
    ]


def build_candidate_pipeline(spec: dict[str, Any], strategy: str):
    """Build the pipeline for one candidate and imbalance strategy."""
    classifier = spec["factory"](class_weight_for(strategy, spec["name"]))
    return build_imbalance_pipeline(
        classifier,
        strategy=strategy,
        scale_numeric=spec["scale_numeric"],
        k=config.DEFAULT_SELECT_K,
        seed=config.SEED,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_specs() -> list[dict[str, Any]]:
    """Return the constant baselines and single-feature-group ablations."""
    return [
        {
            "name": "majority",
            "description": "Constant zero score; always predicts non-melanoma.",
            "kind": "constant",
            "score": 0.0,
            "numeric_features": (),
            "categorical_features": (),
        },
        {
            "name": "random",
            "description": "Uniform random scores; empirical chance level.",
            "kind": "random",
            "numeric_features": (),
            "categorical_features": (),
        },
        {
            "name": "age_only",
            "description": "Logistic regression using age only.",
            "kind": "fitted",
            "numeric_features": ("age",),
            "categorical_features": (),
        },
        {
            "name": "metadata_only",
            "description": "Age, sex, and anatomical localization only.",
            "kind": "fitted",
            "numeric_features": ("age",),
            "categorical_features": ("sex", "localization"),
        },
        {
            "name": "colour_only",
            "description": "RGB channels and colour proportions only.",
            "kind": "fitted",
            "numeric_features": COLOUR_FEATURES,
            "categorical_features": (),
        },
        {
            "name": "shape_only",
            "description": "Asymmetry, perimeter, and area only.",
            "kind": "fitted",
            "numeric_features": SHAPE_FEATURES,
            "categorical_features": (),
        },
        {
            "name": "all_features",
            "description": "All tabular features, logistic regression, k=15.",
            "kind": "fitted",
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
        },
    ]


def build_baseline_pipeline(spec: dict[str, Any]) -> Pipeline:
    """Build the logistic-regression pipeline for a fitted baseline spec."""
    if spec["kind"] != "fitted":
        raise ValueError(f"baseline {spec['name']!r} is not a fitted model")
    # The small feature-subset ablations keep all their columns; the
    # all-feature reference uses the same selector as the main candidates.
    k = config.DEFAULT_SELECT_K if spec["name"] == "all_features" else None
    return build_pipeline(
        build_logistic_regression(class_weight="balanced"),
        scale_numeric=True,
        k=k,
        numeric_features=spec["numeric_features"],
        categorical_features=spec["categorical_features"],
    )
