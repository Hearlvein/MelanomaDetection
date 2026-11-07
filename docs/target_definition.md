# Target definition

## Current Phase 1 default

Phase 1 uses **Option A: melanoma vs all non-melanoma** by default. This matches
the existing notebooks and the current `HAM10000_features_couleurs.xlsx` labels.

| Role | Original HAM10000 diagnoses |
|---|---|
| Positive (`target = 1`) | `mel` |
| Negative (`target = 0`) | `nv`, `bkl`, `bcc`, `akiec`, `vasc`, `df` |

This means **BCC (514 images) and AKIEC (327 images) are treated as negative**,
even though they are malignant. That is acceptable only if the project is
explicitly scoped as "melanoma vs everything else". It is not the same as
"malignant vs benign".

## Available options

| Option | CLI value | Positive class | Negative class | Intended use |
|---|---|---|---|---|
| A. Melanoma vs all non-melanoma | `melanoma` | `mel` | `nv`, `bkl`, `bcc`, `akiec`, `vasc`, `df` | Melanoma-specific detector; must document that other malignancies are negatives |
| B. Malignant vs benign | `malignant` | `mel`, `bcc`, `akiec` | `nv`, `bkl`, `df`, `vasc` | Clinical triage / malignancy screening |
| C. Original seven-class | `multiclass` | all seven classes | — | Multi-class diagnosis; report per-class melanoma recall |

Option B is often the most defensible clinical framing. Option C is appropriate
if the assignment requires the original HAM10000 taxonomy. The choice must be
made before modelling and recorded here.

## How to regenerate the split

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 \
  python scripts/build_phase1_dataset.py --target melanoma
```

Use `--target malignant` or `--target multiclass` for the other definitions.
Each run writes a target-specific split file and manifest, so the options can be
compared without overwriting one another.

## Decision status

**Pending final confirmation.** Phase 1 and Phase 3 are parameterised so that
changing the target does not require rewriting the split or evaluation code.
Until the target is confirmed, the default `melanoma` split is the one used by
the Phase 1 and Phase 3 scripts. `malignant` is also supported by both; the
multiclass target needs a separate evaluation protocol.
