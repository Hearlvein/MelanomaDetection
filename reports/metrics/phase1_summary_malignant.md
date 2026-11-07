# Phase 1 — Data integrity and grouped split

- Generated at: `2026-11-06T10:55:04.524539+00:00`
- Target definition: **malignant**
- Description: Malignant vs benign lesions (melanoma + BCC + AKIEC vs the rest).
- Positive classes: ['mel', 'bcc', 'akiec']
- Negative classes: ['nv', 'bkl', 'df', 'vasc']
- Random seed: `42`
- Split fingerprint: `1a34cd75276b4550dbca35099c57e0fea9981e0dba262832af6f104e329e835a`

## Inputs

| File | SHA-256 |
| --- | --- |
| data/raw/HAM10000_features_couleurs.xlsx | cb2b56ce02c0869000b3766c8b9abf2bfea6bdf7f6a465f02242b313c041a3b4 |
| data/raw/HAM10000_metadata.csv | 06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5 |

## Dataset checks

| Check | Result |
| --- | --- |
| Rows | 10015 |
| Unique image IDs | 10015 |
| Unique lesions | 7470 |
| Original diagnosis classes | akiec, bcc, bkl, df, mel, nv, vasc |
| Missing values | {'age': 57} |

## Split summary

| Split | Rows | Lesions | Share |
| --- | --- | --- | --- |
| train | 6009 | 4482 | 60.000% |
| validation | 2003 | 1494 | 20.000% |
| test | 2003 | 1494 | 20.000% |

### Target distribution by split

| Split | benign | malignant |
| --- | --- | --- |
| train | 4837 | 1172 |
| validation | 1612 | 391 |
| test | 1612 | 391 |

### Original diagnosis distribution by split

| Split | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 182 | 337 | 652 | 74 | 653 | 4033 | 78 |
| validation | 70 | 93 | 211 | 19 | 228 | 1346 | 36 |
| test | 75 | 84 | 236 | 22 | 232 | 1326 | 28 |

## Leakage checks

| Check | Result |
| --- | --- |
| Lesion overlap train/validation | 0 |
| Lesion overlap train/test | 0 |
| Lesion overlap validation/test | 0 |
| Max target prevalence difference | 0.0001 |
| Prevalence tolerance | 0.0300 |

## Outputs

| Artifact | Path |
| --- | --- |
| interim_dataset | data/interim/ham10000_with_lesions.csv |
| split_assignments | data/processed/split_assignments_malignant.csv |
| manifest | reports/metrics/phase1_manifest_malignant.json |
| summary | reports/metrics/phase1_summary_malignant.md |

## Notes for later phases

- Use `split` for the final holdout.
- Run grouped cross-validation **inside `split == 'train'`** (recompute `StratifiedGroupKFold` on the training rows, or use the precomputed training folds 2, 3, and 4 as a 3-fold grouped CV). Never include validation or test rows in CV.
- `lesion_id` is a grouping key only. It must never be a model feature.
- `dx_original`, `dx_current`, `dx_type`, `target`, `target_label`, `fold`, and `split` are label/audit columns. They must never be model features.
- The Phase 1 default target is `melanoma` (melanoma vs all non-melanoma), matching the current notebooks. Use `--target malignant` or `--target multiclass` to regenerate the split for another task.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 python scripts/build_phase1_dataset.py --target malignant --seed 42 --n-splits 5 --test-fold 0 --val-fold 1
```
