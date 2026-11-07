# Phase 5 — Model selection, freezing, and one-shot test evaluation

## Goal

Phase 5 tunes a small set of classifiers, compares the tuned candidates on the
validation split, freezes one model and one threshold, and evaluates the test
split exactly once.

The test split must not be used for hyperparameter tuning, threshold selection,
model comparison, or repeated reporting.

## Pipeline

```text
raw feature columns
  -> median/most-frequent imputation
  -> optional scaling
  -> one-hot encoding
  -> SelectKBest(f_classif, k)
  -> optional sampler (imbalance strategy)
  -> classifier
```

Preprocessing, selection, and any resampling are fitted inside each CV fold.
The sampler, when present, runs only during `fit`.

## Candidates

| Model | Main search dimensions |
|---|---|
| Logistic Regression | `C` (log-uniform), `penalty` (`l1`, `l2`) |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| HistGradientBoosting | `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` |
| ExtraTrees | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |

All candidates also search `selector__k ∈ {10, 15, 20, 25, 30}`.

The imbalance strategy for each model is read from the Phase 4 report
(`recommendations.best_per_model`). If the report is missing, `balanced` is
used. `--strategy` overrides every candidate.

ExtraTrees was not part of the Phase 4 comparison, so it uses the default
`balanced` strategy. Treat that choice as untested for this model family rather
than as a Phase 4 result.

## Tuning protocol

1. Run `RandomizedSearchCV` with `StratifiedGroupKFold` on `split == "train"`.
2. Use `scoring="average_precision"`.
3. Use 30 randomized candidates per model by default (`--n-iter`).
4. Keep the search inside the training split only.
5. Record best parameters, mean/std CV average precision, and rank.

## Threshold and evaluation protocol

For each tuned candidate:

1. Generate grouped out-of-fold positive-class scores on the training split.
2. Select the F2-maximising threshold from those OOF scores.
3. Freeze the threshold.
4. Refit the tuned pipeline on the full training split.
5. Evaluate the validation split once at the frozen threshold.
6. Compute lesion-clustered 95% confidence intervals.

Validation is the model-selection split in Phase 5, so the selected model's
validation estimate is mildly optimistic. The test estimate is the unbiased
one.

## Selection rule

Automatic selection:

1. find the highest validation average precision;
2. consider candidates within `0.005` of that value;
3. among them, prefer the highest validation F2;
4. break remaining ties toward the simpler model order
   (Logistic Regression, Random Forest, HistGradientBoosting, ExtraTrees).

Use `--select-model` to override the rule intentionally.

## Frozen artifacts

The selected pipeline is saved with `joblib` and a JSON sidecar:

- `models/phase5_<target>_<model>.joblib`
- `models/phase5_<target>_<model>.json`
- `models/phase5_<target>_frozen.json` — pointer to the canonical frozen artifact

The sidecar records:

- target definition and class labels;
- model family and imbalance strategy;
- best hyperparameters;
- frozen threshold and sensitivity threshold;
- OOF and validation metrics with confidence intervals;
- feature columns and preprocessing description;
- split fingerprint, seed, fold configuration;
- training and validation counts;
- package versions and the reproduce command;
- non-clinical-use notes.

The final-test script verifies the target and split fingerprint before loading
the test labels.

## One-shot test guard

`scripts/evaluate_final_test.py` refuses to run if
`reports/metrics/phase5_final_test_<target>.json` already exists. This prevents
accidental repeated test evaluation.

`--force` exists only for intentional overwrites. Do not use it to “try another
threshold” or to compare repeated test runs.

## Usage

### 1. Select and freeze a model (validation only)

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn \
  python scripts/run_phase5_model_selection.py --target melanoma --n-iter 30
```

Outputs:

- `reports/metrics/phase5_model_selection_<target>.json`
- `reports/metrics/phase5_model_selection_<target>.md`
- `models/phase5_<target>_<model>.joblib`
- `models/phase5_<target>_<model>.json`
- `models/phase5_<target>_frozen.json`

### 2. Evaluate the test split once

Only after the target and operating point are confirmed:

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 \
  --with imbalanced-learn --with matplotlib \
  python scripts/evaluate_final_test.py --target melanoma
```

Outputs:

- `reports/metrics/phase5_final_test_<target>.json`
- `reports/metrics/phase5_final_test_<target>.md`
- `reports/figures/phase5_final_test_<target>_*.png`

## Code

- `src/melanoma_detection/train.py` — candidate specs, tuning, OOF threshold freezing, artifact persistence, one-shot guard.
- `scripts/run_phase5_model_selection.py` — tuning orchestration, validation comparison, selection, artifact freezing.
- `scripts/evaluate_final_test.py` — one-shot test evaluation from a frozen artifact.
- `tests/test_train.py` — candidate specs, tuning/freezing consistency, selection rule, artifact roundtrip, one-shot guard.

## Limitations

- One validation estimate per candidate is noisy; overlapping confidence
  intervals are expected.
- The selection rule is a documented heuristic, not a formal model-selection
  procedure.
- Class weighting and resampling can degrade probability calibration.
- HAM10000 has no patient identifier; lesion grouping is the strongest available
  leakage control.
- These results are not clinical evidence and must not be presented as a
  diagnostic device.
