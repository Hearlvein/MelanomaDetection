# Model card — HAM10000 melanoma detection (research use only)

This model card describes the Phase 5 frozen model framework. The concrete
numbers for a run are recorded in the generated artifacts and reports, not in
this static document.

## Model details

| Field | Value |
|---|---|
| Task | Binary melanoma detection on HAM10000 |
| Default target | `melanoma` vs all non-melanoma diagnoses |
| Alternative target | `malignant` vs benign |
| Input | Pre-extracted tabular metadata, colour, and shape features |
| Output | Positive-class probability and a thresholded prediction |
| Threshold | Frozen from grouped-CV out-of-fold predictions on the training split |
| Artifact | `models/phase5_<target>_<model>.joblib` plus JSON sidecar |

The exact model family, imbalance strategy, hyperparameters, threshold, and
metrics are stored in the sidecar and in
`reports/metrics/phase5_model_selection_<target>.json`.

## Intended use

- Research and teaching on the HAM10000 dataset.
- Reproducible comparison of leakage-free pipelines.
- Demonstration of lesion-grouped evaluation and one-shot test discipline.

## Out-of-scope use

- Clinical diagnosis, triage, screening, or patient communication.
- Use as a medical device or as part of a regulated workflow.
- Use on populations, imaging devices, or skin tones not represented in HAM10000.
- Use on raw images without the required feature-extraction step.

**This model is not a diagnostic device and must not guide patient care.**

## Training data

HAM10000 provides dermatoscopic images and metadata. The pipeline uses:

- age, sex, and anatomical localization;
- three dominant colour channels and proportions;
- asymmetry, perimeter, and area.

Rows are grouped by `lesion_id`. Train, validation, and test splits are
lesion-disjoint. The split fingerprint is recorded in the artifact sidecar.

## Evaluation protocol

- Hyperparameters are tuned with grouped CV inside the training split.
- The decision threshold is selected from training OOF scores.
- The validation split is used to compare already-tuned candidates.
- The test split is evaluated once, after the model and threshold are frozen.
- Confidence intervals resample whole lesions, not individual images.

## Recorded melanoma run (Phase 5)

| Field | Value |
|---|---|
| Model | ExtraTrees, `class_weight="balanced"` |
| Frozen threshold | 0.220 |
| Test rows / lesions | 2,003 / 1,494 |
| Sensitivity | 0.834 [0.784–0.884] |
| Specificity | 0.696 [0.671–0.721] |
| PPV | 0.256 [0.212–0.299] |
| F2 | 0.574 [0.517–0.622] |
| Average precision | 0.391 [0.311–0.470] |
| ROC-AUC | 0.835 [0.806–0.862] |

The test split has been evaluated once. See
`reports/metrics/phase5_final_test_melanoma.md` for the full report and
`reports/metrics/phase5_model_selection_melanoma.md` for the validation
comparison.

## Metrics

The primary reporting metrics are:

- average precision / PR-AUC;
- sensitivity and specificity at the frozen threshold;
- PPV and NPV;
- F2, which weights recall more heavily than precision;
- Brier score for probability calibration.

Accuracy is not a headline metric because the positive prevalence is low.

## Limitations

- HAM10000 is a curated dermatoscopic dataset, not a clinical population.
- Pre-extracted tabular features discard most image information.
- The dataset lacks a patient identifier; lesion grouping is the strongest
  available leakage control.
- Skin-tone, age, sex, and anatomical coverage are limited and uneven.
- Class weighting or resampling can worsen calibration.
- Validation was used for model selection, so validation metrics are mildly
  optimistic; test metrics are the unbiased estimate for this dataset.
- Performance is not sufficient for clinical deployment.

## Ethical considerations

- Do not present research metrics as evidence of clinical readiness.
- Do not use the model to defer, deny, or delay dermatological care.
- Report subgroup performance before any broader evaluation.
- Keep the test split untouched outside the documented one-shot evaluation.

## Artifacts

| Artifact | Purpose |
|---|---|
| `models/phase5_<target>_<model>.joblib` | Frozen fitted pipeline |
| `models/phase5_<target>_<model>.json` | Sidecar with parameters, threshold, metrics, and split fingerprint |
| `models/phase5_<target>_frozen.json` | Pointer to the canonical frozen artifact |
| `reports/metrics/phase5_model_selection_<target>.json` | Candidate comparison and selection |
| `reports/metrics/phase5_model_selection_<target>.md` | Human-readable model selection report |
| `reports/metrics/phase5_final_test_<target>.json` | One-shot test evaluation |
| `reports/metrics/phase5_final_test_<target>.md` | Human-readable final test report |
