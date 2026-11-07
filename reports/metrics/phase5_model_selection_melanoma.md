# Phase 5 — Model selection and freezing

- Generated at: `2026-11-06T21:49:56.942166+00:00`
- Target: **melanoma**
- Positive class: **melanoma** (`target = 1`)
- Negative class: **non_melanoma** (`target = 0`)
- Tuning: 30 randomized candidates per model, grouped CV
- CV scoring: `average_precision`
- Bootstrap: 2000 lesion-clustered replicates
- Threshold source: `grouped CV OOF predictions on split == train`
- Phase 4 imbalance input: `reports/metrics/phase4_imbalance_melanoma.json`

## Protocol

- Hyperparameters are tuned with `RandomizedSearchCV` and `StratifiedGroupKFold` on `split == "train"`.
- Preprocessing, feature selection, and any resampling are fitted inside each CV fold.
- The decision threshold is selected from training OOF scores and then frozen.
- The validation split is used only to compare already-tuned candidates.
- The test split is not used in this report or for model selection.

## Data roles

| Split | Rows | Lesions | Positive | Negative | Prevalence | Use |
| --- | --- | --- | --- | --- | --- | --- |
| train | 6009 | 4482 | 668 | 5341 | 0.111 | tuning + OOF threshold selection |
| validation | 2003 | 1494 | 222 | 1781 | 0.111 | model comparison |

## Imbalance strategies

| Model | Strategy |
| --- | --- |
| logistic_regression | none |
| random_forest | random_under |
| hist_gradient_boosting | balanced |
| extra_trees | balanced |

## Tuned candidates

| Model | Strategy | CV AP (mean ± std) | OOF AP | Val AP | Threshold | Sensitivity | Specificity | PPV | F2 | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | none | 0.341 ± 0.034 | 0.332 | 0.305 [0.243–0.387] | 0.090 | 0.824 [0.762–0.884] | 0.646 [0.618–0.672] | 0.225 [0.185–0.264] | 0.538 [0.480–0.590] | 0.088 [0.075–0.101] |
| Random Forest | random_under | 0.316 ± 0.044 | 0.309 | 0.355 [0.287–0.429] | 0.422 | 0.824 [0.764–0.879] | 0.579 [0.553–0.605] | 0.196 [0.162–0.231] | 0.502 [0.448–0.554] | 0.186 [0.177–0.195] |
| HistGradientBoosting | balanced | 0.368 ± 0.024 | 0.359 | 0.353 [0.285–0.431] | 0.153 | 0.770 [0.704–0.830] | 0.695 [0.668–0.719] | 0.239 [0.197–0.282] | 0.533 [0.476–0.587] | 0.102 [0.092–0.113] |
| ExtraTrees | balanced | 0.348 ± 0.034 | 0.337 | 0.359 [0.289–0.439] | 0.220 | 0.784 [0.714–0.850] | 0.681 [0.654–0.705] | 0.234 [0.194–0.276] | 0.533 [0.474–0.586] | 0.095 [0.086–0.104] |

## Selected model

**ExtraTrees** with imbalance strategy `balanced`.

- Best parameters: `max_depth=30, max_features=None, min_samples_leaf=4, n_estimators=300, k=30`
- CV average precision: 0.348 ± 0.034
- Frozen F2 threshold: 0.220
- Validation AP: 0.359 [0.289–0.439]
- Validation sensitivity: 0.784 [0.714–0.850]
- Validation specificity: 0.681 [0.654–0.705]
- Validation PPV: 0.234 [0.194–0.276]
- Validation F2: 0.533 [0.474–0.586]
- Validation Brier score: 0.095 [0.086–0.104]

Confusion matrix at the frozen threshold:

| TP | FP | FN | TN |
| --- | --- | --- | --- |
| 174 | 569 | 48 | 1212 |

## Frozen artifact

- Model: `models/phase5_melanoma_extra_trees.joblib`
- Sidecar: `models/phase5_melanoma_extra_trees.json`
- Target pointer: `models/phase5_melanoma_frozen.json`

The test split has **not** been evaluated. Run the one-shot final test only after
the target and operating point are confirmed:

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn --with matplotlib python scripts/evaluate_final_test.py --target melanoma
```

The final-test script refuses to overwrite an existing result unless `--force` is passed.

## Reproduce model selection

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn python scripts/run_phase5_model_selection.py --target melanoma --n-splits 5 --n-iter 30 --bootstrap-reps 2000 --sensitivity 0.9
```

## Interpretation notes

- Validation is the model-selection split, so the selected model's validation estimate is mildly optimistic.
- The test estimate is the unbiased estimate and must be reported once.
- Confidence intervals resample whole lesions, not individual images.
- These results are not clinical evidence and must not be presented as a diagnostic device.
