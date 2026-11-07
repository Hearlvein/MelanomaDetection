# Phase 4 — Class imbalance handling

## Goal

Phase 4 answers a narrow question: **does explicit imbalance handling improve
melanoma detection under the existing grouped-CV and threshold protocol?**

It is a strategy comparison, not the final model selection. Phase 5 tunes and
freezes the final model.

## Why imbalance handling matters

HAM10000 is imbalanced. In the melanoma split, only about 11% of images are
positive. A model can reach high accuracy by predicting the majority class, so
accuracy is not a useful selection metric. The relevant metrics are:

- average precision / PR-AUC;
- F2 at the out-of-fold (OOF) selected threshold;
- sensitivity, specificity, and PPV;
- Brier score and calibration.

Class weighting changes the predicted-probability distribution. A strategy can
improve ranking while worsening calibration, so both threshold-independent and
threshold-selected metrics are reported.

## Strategies

| Strategy | Implementation | Models |
|---|---|---|
| `none` | `class_weight=None` | All |
| `balanced` | `class_weight="balanced"` | All |
| `balanced_subsample` | `class_weight="balanced_subsample"` | Tree models only |
| `random_over` | `RandomOverSampler` inside the pipeline | All |
| `random_under` | `RandomUnderSampler` inside the pipeline | All |
| `smote` | `SMOTE` inside the pipeline | All |

`balanced_subsample` is rejected for estimators that do not support it.

## Resampling rules

Resampling is implemented with an imbalanced-learn `Pipeline`. The sampler is a
pipeline step placed **after** preprocessing and feature selection and **before**
the classifier.

Consequences:

- the sampler runs only during `Pipeline.fit`;
- `predict` and `predict_proba` skip the sampler;
- validation and test rows are never resampled or expanded;
- preprocessing and feature selection are fitted on the original training fold,
  before any synthetic samples are created.

The script never resamples the full dataset before splitting.

## Models

- Logistic Regression;
- Random Forest;
- HistGradientBoosting.

This is enough to test whether a strategy behaves consistently across a linear
model and tree models without turning Phase 4 into a model zoo.

## Protocol

For every model × strategy combination:

1. Build the Phase 2 pipeline (`preprocess -> SelectKBest -> [sampler] -> classify`).
2. Run grouped 5-fold CV (`StratifiedGroupKFold` by `lesion_id`) on `split == "train"`.
3. Generate OOF positive-class probabilities.
4. Select the F2-maximising threshold on the OOF scores.
5. Refit on the full training split.
6. Evaluate the validation split once at the frozen threshold.
7. Compute lesion-clustered 95% confidence intervals.

The test split is not used.

## Outputs

- `reports/metrics/phase4_imbalance_<target>.json` — all results, thresholds, metrics, CIs, failures.
- `reports/metrics/phase4_imbalance_<target>.md` — human-readable comparison and recommendation.

The recommendation selects the best strategy per model by validation average
precision, with validation F2 as a tie-breaker. Phase 5 uses that conclusion as
its starting point; it does not inherit every strategy.

## Usage

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn \
  python scripts/run_phase4_imbalance.py --target melanoma
```

Options:

```text
--target {melanoma,malignant}
--n-splits 5
--k 15
--bootstrap-reps 2000
--sensitivity 0.90
```

## Code

- `src/melanoma_detection/imbalance.py` — class-weight mapping, sampler factories, imbalanced pipeline builder.
- `src/melanoma_detection/training.py` — shared OOF scores, threshold bundle, evaluation, split helpers.
- `scripts/run_phase4_imbalance.py` — comparison orchestration and report.
- `tests/test_imbalance.py` — strategy mapping, sampler placement, no prediction-time resampling, preprocessing statistics.
- `tests/test_training.py` — shared helper correctness and split isolation.

## Current melanoma result

`reports/metrics/phase4_imbalance_melanoma.md` compares all strategies on the
validation split. The main observations were:

- Logistic Regression: `none` had the best validation AP (0.305) and F2 (0.530);
  `balanced` was close (AP 0.299, F2 0.537).
- Random Forest: `random_under` had the best AP (0.317), but its F2 (0.482) was
  lower than `balanced` (0.514) and `balanced_subsample` (0.503).
- HistGradientBoosting: `balanced` had the best AP (0.306); F2 (0.478) was
  similar across strategies.
- No resampling strategy produced a clear improvement in both AP and F2.
  Differences were small relative to the bootstrap intervals.

Phase 5 therefore tunes each model with its Phase 4 strategy recommendation
rather than forcing one global strategy. The `none` versus `balanced` choice for
Logistic Regression remains close and should be revisited if that model family
is selected.

## Limitations

- A single validation estimate per combination is noisy; confidence intervals
  overlap is expected.
- SMOTE on one-hot encoded tabular data is of limited theoretical value.
- Phase 4 does not tune hyperparameters.
- These results are not clinical evidence and must not be presented as a
  diagnostic device.
