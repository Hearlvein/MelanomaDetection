# Data dictionary

## Raw inputs

| File | Description |
|---|---|
| `data/raw/HAM10000_features_couleurs.xlsx` | 10,015 images with handcrafted colour, asymmetry, perimeter, and area features. Current `dx` is collapsed to `mel` / `nomel`. |
| `data/raw/HAM10000_metadata.csv` | Official HAM10000 metadata (MD5 `70bb67f71bd23dde0bd2eda4bdcc98e5`). Source of `lesion_id`, original seven-class `dx`, and `dx_type`. |

## Merged/audit columns

| Column | Type | Use |
|---|---|---|
| `image_id` | string | Image identifier; join key only. Not a feature. |
| `lesion_id` | string | Lesion grouping key for split/CV. **Never a feature.** |
| `dx_original` | category | Original HAM10000 diagnosis (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). **Never a feature.** |
| `dx_current` | category | Collapsed label from the workbook (`mel` / `nomel`). **Never a feature.** |
| `dx_type` | category | Diagnosis confirmation method (`histo`, `follow_up`, `consensus`, `confocal`). **Never a feature.** |
| `target` | int or string | Modelling target produced by `add_target`. **Never a feature.** |
| `target_label` | string | Human-readable target label. **Never a feature.** |
| `fold` | int | Global 5-fold assignment from `StratifiedGroupKFold`: train rows are folds 2–4, validation is fold 1, test is fold 0. Run CV only inside `split == "train"`. **Never a feature.** |
| `split` | string | `train`, `validation`, or `test`. **Never a feature.** |

## Candidate model features

| Column(s) | Type | Notes |
|---|---|---|
| `age` | numeric | 57 missing values in the current data; impute inside the training pipeline. |
| `sex` | category | `male`, `female`, `unknown`. |
| `localization` | category | 15 anatomical sites, including `unknown`. |
| `color_1_R`, `color_1_G`, `color_1_B` | numeric | Dominant colour 1 channels. |
| `color_1_proportion` | numeric | Proportion of colour 1. The three proportions sum to 1, so one is redundant. |
| `color_2_R`, `color_2_G`, `color_2_B`, `color_2_proportion` | numeric | Dominant colour 2. |
| `color_3_R`, `color_3_G`, `color_3_B`, `color_3_proportion` | numeric | Dominant colour 3. |
| `asym_vert_%` | numeric | Vertical asymmetry. |
| `asym_horiz_%` | numeric | Horizontal asymmetry. |
| `Perimeter` | numeric | Lesion perimeter; highly correlated with `Area` (r ≈ 0.956). |
| `Area` | numeric | Lesion area; consider `log(Area)`. |

## Leakage rules

- Split and cross-validate by `lesion_id`, never by `image_id` alone.
- Fit imputers, encoders, scalers, and feature selectors inside the training fold only.
- Do not use `dx_original`, `dx_current`, `dx_type`, `target`, `target_label`, `fold`, `split`, `image_id`, or `lesion_id` as features.
- Do not use the test set for model or threshold selection.
