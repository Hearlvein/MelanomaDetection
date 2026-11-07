# Phase 3 — Evaluation protocol and baselines

- Generated at: `2026-11-06T13:54:09.500510+00:00`
- Target: **melanoma**
- Positive class: **melanoma** (`target = 1`)
- Negative class: **non_melanoma** (`target = 0`)
- Description: Melanoma vs all non-melanoma diagnoses (current notebook task).
- Validation threshold source: `grouped CV OOF predictions on split == train`
- Bootstrap: 2000 lesion-clustered replicates, 95% percentile intervals
- Secondary sensitivity target: 0.90

## Protocol

- Model fitting, preprocessing, feature selection, and threshold selection use **training data only**.
- Threshold selection uses grouped cross-validation out-of-fold predictions inside `split == "train"`.
- The validation split is evaluated at the frozen threshold; it is not used for repeated tuning.
- The test split is **not used for fitting, threshold selection, or metric reporting**.
- Confidence intervals resample whole `lesion_id` groups, not individual images.

## Data roles

| Split | Rows | Lesions | Positive | Negative | Prevalence | Used for |
| --- | --- | --- | --- | --- | --- | --- |
| train | 6009 | 4482 | 668 | 5341 | 0.111 | fit + grouped-CV threshold selection |
| validation | 2003 | 1494 | 222 | 1781 | 0.111 | frozen-threshold evaluation |
| test | 2003 | 1494 | 223 | 1780 | 0.111 | not used for modelling |

## Models and feature sets

| Model | Numeric features | Categorical features | Description |
| --- | --- | --- | --- |
| majority | 0 | 0 | Constant zero score; classifies everything as negative. |
| random | 0 | 0 | Uniform random scores; empirical chance baseline. |
| age_only | 1 | 0 | Logistic regression using age only. |
| metadata_only | 1 | 2 | Age, sex, and anatomical localization only. |
| colour_only | 12 | 0 | RGB channels and colour proportions only. |
| shape_only | 4 | 0 | Asymmetry, perimeter, and area only. |
| all_features | 17 | 2 | Phase 2 reference: all features, logistic regression, k=15. |

## Threshold selection

| Model | F2 threshold (OOF) | Sensitivity ≥ 0.90 threshold (OOF) | Validation prevalence |
| --- | --- | --- | --- |
| majority | 0.500 | — | 0.111 |
| random | 0.500 | — | 0.111 |
| age_only | 0.448 | 0.368 | 0.111 |
| metadata_only | 0.419 | 0.334 | 0.111 |
| colour_only | 0.393 | 0.377 | 0.111 |
| shape_only | 0.502 | 0.360 | 0.111 |
| all_features | 0.423 | 0.338 | 0.111 |

## Validation metrics at the frozen F2 threshold

| Model | Threshold | Sensitivity | Specificity | PPV | NPV | F1 | F2 | Predicted positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.500 | 0.000 [0.000–0.000] | 1.000 [1.000–1.000] | — | 0.889 [0.869–0.908] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 |
| random | 0.500 | 0.482 [0.415–0.545] | 0.493 [0.470–0.516] | 0.106 [0.083–0.131] | 0.884 [0.859–0.907] | 0.174 [0.139–0.209] | 0.282 [0.234–0.327] | 0.504 |
| age_only | 0.448 | 0.802 [0.722–0.881] | 0.466 [0.437–0.495] | 0.158 [0.129–0.189] | 0.950 [0.926–0.971] | 0.264 [0.222–0.308] | 0.441 [0.384–0.497] | 0.564 |
| metadata_only | 0.419 | 0.802 [0.716–0.883] | 0.500 [0.471–0.529] | 0.167 [0.137–0.199] | 0.953 [0.931–0.973] | 0.276 [0.232–0.322] | 0.455 [0.398–0.513] | 0.534 |
| colour_only | 0.393 | 0.901 [0.856–0.940] | 0.506 [0.478–0.535] | 0.185 [0.153–0.220] | 0.976 [0.964–0.986] | 0.307 [0.260–0.355] | 0.508 [0.452–0.561] | 0.539 |
| shape_only | 0.502 | 0.752 [0.688–0.815] | 0.519 [0.495–0.544] | 0.163 [0.136–0.194] | 0.944 [0.925–0.961] | 0.268 [0.228–0.310] | 0.437 [0.385–0.486] | 0.511 |
| all_features | 0.423 | 0.838 [0.776–0.899] | 0.630 [0.601–0.657] | 0.220 [0.181–0.259] | 0.969 [0.955–0.981] | 0.349 [0.297–0.397] | 0.537 [0.479–0.590] | 0.422 |

## Threshold-independent metrics

| Model | Average precision | ROC-AUC | Brier score |
| --- | --- | --- | --- |
| majority | 0.111 [0.092–0.131] | 0.500 [0.500–0.500] | 0.111 [0.092–0.131] |
| random | 0.107 [0.087–0.132] | 0.495 [0.454–0.535] | 0.334 [0.321–0.348] |
| age_only | 0.220 [0.166–0.288] | 0.690 [0.630–0.749] | 0.227 [0.219–0.235] |
| metadata_only | 0.255 [0.192–0.337] | 0.731 [0.682–0.780] | 0.215 [0.205–0.224] |
| colour_only | 0.165 [0.134–0.208] | 0.703 [0.671–0.736] | 0.222 [0.211–0.233] |
| shape_only | 0.188 [0.149–0.245] | 0.673 [0.631–0.716] | 0.235 [0.227–0.242] |
| all_features | 0.299 [0.237–0.383] | 0.792 [0.756–0.827] | 0.190 [0.179–0.202] |

## Confusion matrices at the frozen F2 threshold

| Model | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| majority | 0 | 0 | 222 | 1781 |
| random | 107 | 903 | 115 | 878 |
| age_only | 178 | 951 | 44 | 830 |
| metadata_only | 178 | 891 | 44 | 890 |
| colour_only | 200 | 880 | 22 | 901 |
| shape_only | 167 | 856 | 55 | 925 |
| all_features | 186 | 659 | 36 | 1122 |

## Validation at the OOF high-sensitivity threshold (target ≥ 0.90)

The threshold is selected on training OOF predictions. Validation sensitivity can differ.

| Model | Threshold | Sensitivity | Specificity | PPV | NPV | Predicted positive |
| --- | --- | --- | --- | --- | --- | --- |
| age_only | 0.368 | 0.887 | 0.206 | 0.122 | 0.936 | 0.804 |
| metadata_only | 0.334 | 0.883 | 0.348 | 0.144 | 0.960 | 0.678 |
| colour_only | 0.377 | 0.928 | 0.491 | 0.185 | 0.982 | 0.556 |
| shape_only | 0.360 | 0.910 | 0.291 | 0.138 | 0.963 | 0.731 |
| all_features | 0.338 | 0.901 | 0.521 | 0.190 | 0.977 | 0.526 |

## Figures

- `reports/figures/phase3_melanoma_precision_recall.png`
- `reports/figures/phase3_melanoma_roc.png`
- `reports/figures/phase3_melanoma_calibration.png`
- `reports/figures/phase3_melanoma_threshold_sweep.png`
- `reports/figures/phase3_melanoma_confusion_matrix.png`

## Interpretation notes

- The all-features model is the untuned Phase 2 reference, not a final selected model.
- Baselines use the same split and metrics; they are not tuned on validation.
- A PR-AUC close to the validation prevalence means the model is near chance.
- The test split must remain untouched until the model and threshold are frozen after model selection.
- These results are not clinical evidence and must not be presented as a diagnostic device.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 --with matplotlib python scripts/run_phase3_evaluation.py --target melanoma --n-splits 5 --k 15 --bootstrap-reps 2000 --sensitivity 0.9
```
