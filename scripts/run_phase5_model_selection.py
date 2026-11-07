#!/usr/bin/env python3
"""Phase 5: tune candidates, compare them on validation, and freeze one model.

Hyperparameters are tuned with grouped CV inside ``split == "train"``. The
threshold is selected from training out-of-fold predictions. The validation
split is used for model comparison. The test split is not used.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import load_modeling_frame  # noqa: E402
from melanoma_detection.splits import make_grouped_cv, split_fingerprint  # noqa: E402
from melanoma_detection.training import (  # noqa: E402
    assert_split_isolation,
    split_counts,
)
from melanoma_detection.train import (  # noqa: E402
    CANDIDATE_ORDER,
    artifact_path_for,
    candidate_specs,
    evaluate_tuned_candidate,
    frozen_pointer_path,
    public_candidate_result,
    save_artifact,
    select_final_candidate,
    tune_candidate,
)

BINARY_TARGETS = ("melanoma", "malignant")


def _sanitise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:.3f}" if np.isfinite(number) else "—"


def _format_with_ci(result: dict[str, Any], metric: str) -> str:
    value = _format_value(result["validation"]["metrics"].get(metric))
    interval = result["validation"]["confidence_intervals"].get(metric)
    if interval is None or value == "—":
        return value
    return f"{value} [{interval['lower']:.3f}–{interval['upper']:.3f}]"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_params(params: dict[str, Any]) -> str:
    pieces = []
    for key, value in sorted(params.items()):
        short_key = key.split("__")[-1]
        if isinstance(value, float):
            pieces.append(f"{short_key}={value:.4g}")
        else:
            pieces.append(f"{short_key}={value}")
    return ", ".join(pieces)


def _load_phase4_strategies(target: str) -> tuple[dict[str, str], str | None]:
    path = config.METRICS_DIR / f"phase4_imbalance_{target}.json"
    if not path.exists():
        return {}, None

    payload = json.loads(path.read_text(encoding="utf-8"))
    best_per_model = (
        payload.get("recommendations", {}).get("best_per_model", {})
    )
    strategies = {
        model: details["strategy"]
        for model, details in best_per_model.items()
        if isinstance(details, dict) and "strategy" in details
    }
    return strategies, _relative(path)


def _render_summary(
    *,
    target_info: dict[str, Any],
    protocol: dict[str, Any],
    data: dict[str, dict[str, Any]],
    phase4_path: str | None,
    strategies: dict[str, str],
    results: list[dict[str, Any]],
    selected: dict[str, Any],
    artifact: Path,
    sidecar: Path,
    pointer: Path,
    command: str,
) -> str:
    successful = [result for result in results if "error" not in result]
    failed = [result for result in results if "error" in result]

    lines = [
        "# Phase 5 — Model selection and freezing",
        "",
        f"- Generated at: `{protocol['created_at']}`",
        f"- Target: **{target_info['definition']}**",
        f"- Positive class: **{target_info['labels'][1]}** (`target = 1`)",
        f"- Negative class: **{target_info['labels'][0]}** (`target = 0`)",
        f"- Tuning: {protocol['n_iter']} randomized candidates per model, grouped CV",
        f"- CV scoring: `average_precision`",
        f"- Bootstrap: {protocol['bootstrap_reps']} lesion-clustered replicates",
        f"- Threshold source: `{protocol['threshold_source']}`",
        f"- Phase 4 imbalance input: `{phase4_path or 'not found; defaults used'}`",
        "",
        "## Protocol",
        "",
        "- Hyperparameters are tuned with `RandomizedSearchCV` and `StratifiedGroupKFold` on `split == \"train\"`.",
        "- Preprocessing, feature selection, and any resampling are fitted inside each CV fold.",
        "- The decision threshold is selected from training OOF scores and then frozen.",
        "- The validation split is used only to compare already-tuned candidates.",
        "- The test split is not used in this report or for model selection.",
        "",
        "## Data roles",
        "",
        _markdown_table(
            ["Split", "Rows", "Lesions", "Positive", "Negative", "Prevalence", "Use"],
            [
                [
                    "train",
                    data["train"]["rows"],
                    data["train"]["lesions"],
                    data["train"]["positive"],
                    data["train"]["negative"],
                    _format_value(data["train"]["prevalence"]),
                    "tuning + OOF threshold selection",
                ],
                [
                    "validation",
                    data["validation"]["rows"],
                    data["validation"]["lesions"],
                    data["validation"]["positive"],
                    data["validation"]["negative"],
                    _format_value(data["validation"]["prevalence"]),
                    "model comparison",
                ],
            ],
        ),
        "",
        "## Imbalance strategies",
        "",
        _markdown_table(
            ["Model", "Strategy"],
            [[model, strategy] for model, strategy in strategies.items()],
        ),
        "",
        "## Tuned candidates",
        "",
        _markdown_table(
            [
                "Model",
                "Strategy",
                "CV AP (mean ± std)",
                "OOF AP",
                "Val AP",
                "Threshold",
                "Sensitivity",
                "Specificity",
                "PPV",
                "F2",
                "Brier",
            ],
            [
                [
                    result["model_label"],
                    result["strategy"],
                    f"{_format_value(result['cv']['mean_test_score'])} ± "
                    f"{_format_value(result['cv']['std_test_score'])}",
                    _format_value(result["oof"]["average_precision"]),
                    _format_with_ci(result, "average_precision"),
                    _format_value(result["threshold_selection"]["f2_threshold"]),
                    _format_with_ci(result, "sensitivity"),
                    _format_with_ci(result, "specificity"),
                    _format_with_ci(result, "ppv"),
                    _format_with_ci(result, "f2"),
                    _format_with_ci(result, "brier_score"),
                ]
                for result in successful
            ],
        ),
        "",
        *(
            [
                "## Failed candidates",
                "",
                _markdown_table(
                    ["Model", "Strategy", "Error"],
                    [
                        [result["model_label"], result["strategy"], result["error"]]
                        for result in failed
                    ],
                ),
                "",
            ]
            if failed
            else []
        ),
        "## Selected model",
        "",
        f"**{selected['model_label']}** with imbalance strategy "
        f"`{selected['strategy']}`.",
        "",
        f"- Best parameters: `{_format_params(selected['best_params'])}`",
        f"- CV average precision: {_format_value(selected['cv']['mean_test_score'])} "
        f"± {_format_value(selected['cv']['std_test_score'])}",
        f"- Frozen F2 threshold: {_format_value(selected['threshold_selection']['f2_threshold'])}",
        f"- Validation AP: {_format_with_ci(selected, 'average_precision')}",
        f"- Validation sensitivity: {_format_with_ci(selected, 'sensitivity')}",
        f"- Validation specificity: {_format_with_ci(selected, 'specificity')}",
        f"- Validation PPV: {_format_with_ci(selected, 'ppv')}",
        f"- Validation F2: {_format_with_ci(selected, 'f2')}",
        f"- Validation Brier score: {_format_with_ci(selected, 'brier_score')}",
        "",
        "Confusion matrix at the frozen threshold:",
        "",
        _markdown_table(
            ["TP", "FP", "FN", "TN"],
            [
                [
                    selected["validation"]["confusion_matrix"]["tp"],
                    selected["validation"]["confusion_matrix"]["fp"],
                    selected["validation"]["confusion_matrix"]["fn"],
                    selected["validation"]["confusion_matrix"]["tn"],
                ]
            ],
        ),
        "",
        "## Frozen artifact",
        "",
        f"- Model: `{_relative(artifact)}`",
        f"- Sidecar: `{_relative(sidecar)}`",
        f"- Target pointer: `{_relative(pointer)}`",
        "",
        "The test split has **not** been evaluated. Run the one-shot final test only after",
        "the target and operating point are confirmed:",
        "",
        "```bash",
        f"uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 "
        f"--with imbalanced-learn --with matplotlib python scripts/evaluate_final_test.py "
        f"--target {target_info['definition']}",
        "```",
        "",
        "The final-test script refuses to overwrite an existing result unless `--force` is passed.",
        "",
        "## Reproduce model selection",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Interpretation notes",
        "",
        "- Validation is the model-selection split, so the selected model's validation estimate is mildly optimistic.",
        "- The test estimate is the unbiased estimate and must be reported once.",
        "- Confidence intervals resample whole lesions, not individual images.",
        "- These results are not clinical evidence and must not be presented as a diagnostic device.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=BINARY_TARGETS, default=config.TARGET_DEFINITION)
    parser.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--sensitivity", type=float, default=0.90)
    parser.add_argument(
        "--strategy",
        default=None,
        help="Override the Phase 4 imbalance strategy for every candidate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CANDIDATE_ORDER,
        default=list(CANDIDATE_ORDER),
        help="Candidate model families to tune.",
    )
    parser.add_argument(
        "--select-model",
        choices=CANDIDATE_ORDER,
        default=None,
        help="Override automatic selection with a specific candidate.",
    )
    args = parser.parse_args()

    # scikit-learn 1.9 deprecates LogisticRegression's ``penalty``. Keep the
    # search space compatible with the supported range and silence only these
    # specific warnings, which otherwise repeat on every inner fit.
    warnings.filterwarnings(
        "ignore", message="'penalty' was deprecated.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message="Inconsistent values: penalty=.*", category=UserWarning
    )

    frame = load_modeling_frame(args.target)
    assert_split_isolation(frame)

    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise RuntimeError("training and validation splits must both contain rows")

    spec = config.TARGET_SPECS[args.target]
    target_info = {
        "definition": args.target,
        "description": spec["description"],
        "positive_classes": list(spec["positive"]),
        "negative_classes": list(spec["negative"]),
        "labels": {1: spec["positive_label"], 0: spec["negative_label"]},
    }

    phase4_strategies, phase4_path = _load_phase4_strategies(args.target)
    strategies = {
        model: args.strategy or phase4_strategies.get(model, "balanced")
        for model in CANDIDATE_ORDER
    }
    specs = [
        candidate
        for candidate in candidate_specs(strategies)
        if candidate["name"] in args.models
    ]

    cv = make_grouped_cv(n_splits=args.n_splits, seed=config.SEED)
    feature_columns = list(config.ALL_FEATURES)
    X_train = train[feature_columns]
    y_train = train["target"].to_numpy()
    groups_train = train[config.GROUP_COLUMN].to_numpy()
    X_validation = validation[feature_columns]
    y_validation = validation["target"].to_numpy()
    groups_validation = validation[config.GROUP_COLUMN].to_numpy()

    results: list[dict[str, Any]] = []
    pipelines: dict[str, Any] = {}

    for candidate in specs:
        name = candidate["name"]
        strategy = candidate["strategy"]
        print(f"Tuning {candidate['label']} / {strategy}...")
        try:
            search = tune_candidate(
                candidate,
                strategy,
                X_train,
                y_train,
                groups_train,
                cv,
                n_iter=args.n_iter,
                seed=config.SEED,
            )
            result = evaluate_tuned_candidate(
                search,
                candidate,
                strategy,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                X_validation=X_validation,
                y_validation=y_validation,
                groups_validation=groups_validation,
                cv=cv,
                bootstrap_reps=args.bootstrap_reps,
                min_sensitivity=args.sensitivity,
            )
            pipelines[name] = result["_pipeline"]
            print(
                f"  best CV AP={result['cv']['mean_test_score']:.4f}, "
                f"validation AP={result['validation']['metrics']['average_precision']:.4f}"
            )
        except Exception as error:  # noqa: BLE001 - report per-candidate failures
            result = {
                "model": name,
                "model_label": candidate["label"],
                "strategy": strategy,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  failed: {result['error']}")
        results.append(result)

    successful = [result for result in results if "error" not in result]
    if not successful:
        raise RuntimeError("all Phase 5 candidates failed")

    if args.select_model is not None:
        matches = [result for result in successful if result["model"] == args.select_model]
        if not matches:
            raise RuntimeError(
                f"selected model {args.select_model!r} was not evaluated successfully"
            )
        selected = matches[0]
    else:
        selected = select_final_candidate(results)

    selected_pipeline = pipelines[selected["model"]]
    artifact = artifact_path_for(args.target, selected["model"])
    data = split_counts(frame)
    fingerprint = split_fingerprint(frame, args.target, config.SEED)
    created_at = datetime.now(timezone.utc).isoformat()

    command = (
        "uv run --python 3.13 "
        "--with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn "
        f"python scripts/run_phase5_model_selection.py --target {args.target} "
        f"--n-splits {args.n_splits} --n-iter {args.n_iter} "
        f"--bootstrap-reps {args.bootstrap_reps} --sensitivity {args.sensitivity}"
    )

    metadata = {
        "phase": 5,
        "artifact_kind": "frozen_model",
        "target": target_info,
        "model": {
            "name": selected["model"],
            "label": selected["model_label"],
            "scale_numeric": selected["scale_numeric"],
        },
        "imbalance_strategy": selected["strategy"],
        "best_params": selected["best_params"],
        "cv": selected["cv"],
        "oof": selected["oof"],
        "threshold": selected["threshold_selection"],
        "validation": selected["validation"],
        "validation_at_sensitivity_threshold": selected[
            "validation_at_sensitivity_threshold"
        ],
        "feature_columns": feature_columns,
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')",
            "selector": "SelectKBest(f_classif)",
            "sampler": (
                selected["strategy"]
                if selected["strategy"]
                in {"random_over", "random_under", "smote"}
                else None
            ),
        },
        "training": data["train"],
        "validation_data": data["validation"],
        "split": {
            "fingerprint": fingerprint,
            "seed": config.SEED,
            "n_splits": args.n_splits,
            "test_fold": config.TEST_FOLD,
            "val_fold": config.VAL_FOLD,
            "group_column": config.GROUP_COLUMN,
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "command": command,
        "notes": [
            "Threshold selected from grouped-CV OOF predictions on split == 'train'.",
            "Validation used for model comparison only.",
            "Test split not used during model selection.",
            "Not clinical evidence; research use only.",
        ],
    }

    artifact, sidecar = save_artifact(selected_pipeline, metadata, artifact)

    pointer = frozen_pointer_path(args.target)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer_payload = {
        "target": args.target,
        "model": selected["model"],
        "artifact": _relative(artifact),
        "sidecar": _relative(sidecar),
        "threshold": selected["threshold_selection"],
        "validation": selected["validation"]["metrics"],
        "split_fingerprint": fingerprint,
        "created_at": created_at,
    }
    pointer.write_text(
        json.dumps(_sanitise_json(pointer_payload), indent=2) + "\n",
        encoding="utf-8",
    )

    protocol = {
        "created_at": created_at,
        "seed": config.SEED,
        "n_splits": args.n_splits,
        "n_iter": args.n_iter,
        "bootstrap_reps": args.bootstrap_reps,
        "sensitivity_target": args.sensitivity,
        "threshold_source": "grouped CV OOF predictions on split == train",
        "test_used": False,
        "selection_rule": (
            "highest validation average precision; F2 tie-break within tolerance; "
            "manual override" if args.select_model else
            "highest validation average precision; F2 tie-break within 0.005"
        ),
    }

    metrics_path = config.METRICS_DIR / f"phase5_model_selection_{args.target}.json"
    summary_path = config.METRICS_DIR / f"phase5_model_selection_{args.target}.md"
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": 5,
        "target": target_info,
        "protocol": protocol,
        "data": {
            "train": data["train"],
            "validation": data["validation"],
            "test_used": False,
        },
        "phase4": {
            "path": phase4_path,
            "strategies": strategies,
        },
        "candidates": [public_candidate_result(result) for result in results],
        "selected": public_candidate_result(selected),
        "artifact": {
            "path": _relative(artifact),
            "sidecar": _relative(sidecar),
            "pointer": _relative(pointer),
            "sha256": json.loads(sidecar.read_text(encoding="utf-8"))["artifact_sha256"],
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }

    summary = _render_summary(
        target_info=target_info,
        protocol=protocol,
        data=data,
        phase4_path=phase4_path,
        strategies={
            candidate["name"]: candidate["strategy"] for candidate in specs
        },
        results=results,
        selected=selected,
        artifact=artifact,
        sidecar=sidecar,
        pointer=pointer,
        command=command,
    )

    metrics_path.write_text(
        json.dumps(_sanitise_json(payload), indent=2) + "\n", encoding="utf-8"
    )
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Froze model: {artifact}")
    print(
        "Selected: "
        f"{selected['model_label']} / {selected['strategy']} "
        f"(validation AP={selected['validation']['metrics']['average_precision']:.3f}, "
        f"F2={selected['validation']['metrics']['f2']:.3f}, "
        f"threshold={selected['threshold_selection']['f2_threshold']:.3f})"
    )
    print("Test split was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
