# Phase 4 — Imbalance strategy comparison

- Generated at: `2026-11-06T21:39:58.288468+00:00`
- Target: **melanoma**
- Positive class: **melanoma** (`target = 1`)
- Negative class: **non_melanoma** (`target = 0`)
- Bootstrap: 2000 lesion-clustered replicates, 95% percentile intervals
- Threshold source: `grouped CV OOF predictions on split == train`

## Protocol

- Every model/strategy combination uses the Phase 2 preprocessor and `SelectKBest`.
- Resampling happens inside the training fold only, after preprocessing and feature selection.
- Thresholds are selected from grouped-CV OOF scores inside `split == "train"`.
- The validation split is evaluated once per combination at the frozen threshold.
- The test split is not used.

## Data roles

| Split | Rows | Lesions | Positive | Negative | Prevalence |
| --- | --- | --- | --- | --- | --- |
| train | 6009 | 4482 | 668 | 5341 | 0.111 |
| validation | 2003 | 1494 | 222 | 1781 | 0.111 |
| test | 2003 | 1494 | 223 | 1780 | 0.111 |

## Results

| Model | Strategy | Threshold | Sensitivity | Specificity | PPV | NPV | F1 | F2 | AP | ROC-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | none | 0.102 | 0.784 [0.714–0.848] | 0.674 [0.647–0.701] | 0.231 [0.190–0.273] | 0.962 [0.947–0.974] | 0.357 [0.305–0.408] | 0.530 [0.472–0.583] | 0.305 [0.243–0.386] | 0.793 [0.756–0.828] | 0.088 [0.075–0.101] |
| Logistic Regression | balanced | 0.423 | 0.838 [0.776–0.899] | 0.630 [0.601–0.657] | 0.220 [0.181–0.259] | 0.969 [0.955–0.981] | 0.349 [0.297–0.397] | 0.537 [0.479–0.590] | 0.299 [0.237–0.383] | 0.792 [0.756–0.827] | 0.190 [0.179–0.202] |
| Logistic Regression | random_over | 0.415 | 0.842 [0.784–0.900] | 0.627 [0.598–0.654] | 0.219 [0.182–0.258] | 0.970 [0.956–0.981] | 0.348 [0.298–0.397] | 0.537 [0.480–0.590] | 0.303 [0.241–0.385] | 0.793 [0.755–0.828] | 0.188 [0.177–0.200] |
| Logistic Regression | random_under | 0.448 | 0.815 [0.751–0.878] | 0.637 [0.610–0.664] | 0.219 [0.180–0.258] | 0.965 [0.951–0.978] | 0.345 [0.293–0.393] | 0.527 [0.469–0.578] | 0.296 [0.232–0.378] | 0.788 [0.751–0.823] | 0.196 [0.184–0.208] |
| Logistic Regression | smote | 0.413 | 0.829 [0.768–0.890] | 0.609 [0.581–0.637] | 0.209 [0.172–0.246] | 0.966 [0.952–0.979] | 0.334 [0.285–0.381] | 0.520 [0.464–0.571] | 0.293 [0.233–0.373] | 0.789 [0.751–0.824] | 0.191 [0.179–0.203] |
| Random Forest | none | 0.104 | 0.752 [0.682–0.815] | 0.634 [0.609–0.660] | 0.204 [0.169–0.242] | 0.954 [0.937–0.968] | 0.321 [0.272–0.369] | 0.489 [0.431–0.542] | 0.304 [0.242–0.377] | 0.776 [0.738–0.812] | 0.088 [0.075–0.101] |
| Random Forest | balanced | 0.260 | 0.752 [0.682–0.818] | 0.681 [0.655–0.706] | 0.227 [0.187–0.267] | 0.957 [0.941–0.970] | 0.349 [0.297–0.398] | 0.514 [0.455–0.567] | 0.300 [0.241–0.372] | 0.783 [0.744–0.819] | 0.108 [0.099–0.117] |
| Random Forest | balanced_subsample | 0.126 | 0.802 [0.737–0.866] | 0.605 [0.576–0.631] | 0.202 [0.166–0.238] | 0.961 [0.945–0.974] | 0.322 [0.274–0.369] | 0.503 [0.446–0.554] | 0.307 [0.247–0.381] | 0.783 [0.744–0.820] | 0.089 [0.078–0.101] |
| Random Forest | random_over | 0.169 | 0.752 [0.683–0.820] | 0.651 [0.624–0.676] | 0.212 [0.174–0.250] | 0.955 [0.939–0.969] | 0.331 [0.280–0.379] | 0.498 [0.441–0.552] | 0.296 [0.237–0.369] | 0.777 [0.739–0.813] | 0.094 [0.084–0.106] |
| Random Forest | random_under | 0.464 | 0.743 [0.674–0.809] | 0.631 [0.606–0.657] | 0.200 [0.164–0.239] | 0.952 [0.935–0.966] | 0.316 [0.267–0.364] | 0.482 [0.425–0.535] | 0.317 [0.253–0.390] | 0.775 [0.735–0.811] | 0.192 [0.183–0.202] |
| Random Forest | smote | 0.213 | 0.775 [0.704–0.843] | 0.636 [0.608–0.662] | 0.210 [0.172–0.248] | 0.958 [0.942–0.972] | 0.330 [0.281–0.378] | 0.504 [0.445–0.557] | 0.287 [0.229–0.355] | 0.778 [0.740–0.814] | 0.107 [0.097–0.117] |
| HistGradientBoosting | none | 0.042 | 0.779 [0.709–0.844] | 0.580 [0.553–0.607] | 0.188 [0.154–0.224] | 0.955 [0.938–0.970] | 0.303 [0.257–0.349] | 0.478 [0.421–0.531] | 0.268 [0.212–0.338] | 0.755 [0.714–0.793] | 0.093 [0.079–0.108] |
| HistGradientBoosting | balanced | 0.203 | 0.712 [0.635–0.783] | 0.660 [0.633–0.686] | 0.207 [0.169–0.246] | 0.948 [0.931–0.963] | 0.320 [0.270–0.370] | 0.478 [0.418–0.533] | 0.306 [0.244–0.378] | 0.765 [0.724–0.803] | 0.123 [0.112–0.135] |
| HistGradientBoosting | random_over | 0.163 | 0.752 [0.681–0.821] | 0.608 [0.581–0.635] | 0.193 [0.158–0.229] | 0.952 [0.935–0.967] | 0.307 [0.260–0.355] | 0.476 [0.421–0.530] | 0.270 [0.215–0.338] | 0.757 [0.716–0.795] | 0.125 [0.113–0.137] |
| HistGradientBoosting | random_under | 0.207 | 0.838 [0.781–0.888] | 0.498 [0.471–0.526] | 0.172 [0.142–0.204] | 0.961 [0.945–0.975] | 0.286 [0.242–0.329] | 0.473 [0.418–0.522] | 0.276 [0.219–0.344] | 0.758 [0.719–0.793] | 0.206 [0.192–0.221] |
| HistGradientBoosting | smote | 0.137 | 0.698 [0.620–0.771] | 0.665 [0.638–0.691] | 0.206 [0.168–0.246] | 0.946 [0.928–0.963] | 0.319 [0.268–0.366] | 0.473 [0.414–0.526] | 0.247 [0.198–0.307] | 0.754 [0.713–0.794] | 0.106 [0.094–0.119] |

## High-sensitivity operating points

| Model | Strategy | Threshold | Sensitivity | Specificity | PPV | NPV |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | none | 0.063 | 0.887 | 0.514 | 0.185 | 0.973 |
| Logistic Regression | balanced | 0.338 | 0.901 | 0.521 | 0.190 | 0.977 |
| Logistic Regression | random_over | 0.335 | 0.892 | 0.523 | 0.189 | 0.975 |
| Logistic Regression | random_under | 0.323 | 0.892 | 0.472 | 0.174 | 0.972 |
| Logistic Regression | smote | 0.299 | 0.901 | 0.482 | 0.178 | 0.975 |
| Random Forest | none | 0.067 | 0.905 | 0.470 | 0.176 | 0.976 |
| Random Forest | balanced | 0.165 | 0.878 | 0.506 | 0.182 | 0.971 |
| Random Forest | balanced_subsample | 0.090 | 0.869 | 0.496 | 0.177 | 0.968 |
| Random Forest | random_over | 0.103 | 0.883 | 0.492 | 0.178 | 0.971 |
| Random Forest | random_under | 0.366 | 0.838 | 0.497 | 0.172 | 0.961 |
| Random Forest | smote | 0.133 | 0.892 | 0.481 | 0.176 | 0.973 |
| HistGradientBoosting | none | 0.022 | 0.874 | 0.446 | 0.164 | 0.966 |
| HistGradientBoosting | balanced | 0.065 | 0.896 | 0.433 | 0.165 | 0.971 |
| HistGradientBoosting | random_over | 0.060 | 0.905 | 0.428 | 0.165 | 0.973 |
| HistGradientBoosting | random_under | 0.167 | 0.865 | 0.458 | 0.166 | 0.964 |
| HistGradientBoosting | smote | 0.046 | 0.892 | 0.426 | 0.162 | 0.969 |

## Recommendation

Best strategy per model by validation average precision, with F2 as a tie-breaker:

| Model | Strategy | Validation AP | Validation F2 |
| --- | --- | --- | --- |
| logistic_regression | none | 0.305 | 0.530 |
| random_forest | random_under | 0.317 | 0.482 |
| hist_gradient_boosting | balanced | 0.306 | 0.478 |

Global best: **random_forest** with **random_under** (validation AP 0.317, F2 0.482).

This is a strategy comparison, not the final model selection. Phase 5 re-tunes a small model set using the chosen strategy and evaluates the test split only after freezing.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn python scripts/run_phase4_imbalance.py --target melanoma --n-splits 5 --k 15 --bootstrap-reps 2000 --sensitivity 0.9
```
