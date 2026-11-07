# Late fusion

Late fusion combines the V2 handcrafted feature table with image embeddings:

```text
V2 tabular features ── preprocess ──────────────┐
                                                ├─ ExtraTrees ── probability
image embeddings ── scale ── PCA(32) ────────────┘
```

The embedding model is pretrained or trained once (VGG16 by default). Fusion
training never fits the CNN; it treats embeddings as fixed image features.

## Why late fusion

- Tabular features capture colour, shape, and demographic context.
- Embeddings capture texture and appearance the handcrafted features miss.
- Keeping the CNN frozen means fusion cannot leak labels through a jointly
  trained network, and embeddings can be regenerated independently.

## Leakage rules

- The tabular and embedding inputs use the same lesion-grouped split.
- Median imputation, `StandardScaler`, and `PCA` all live **inside** the
  scikit-learn pipeline. They are fitted on training folds only.
- The F2 threshold is selected from grouped out-of-fold scores inside
  `split == "train"`.
- Validation is scored once for the training report; it is not used for
  repeated tuning.
- The test split is scored once by `scripts/evaluate_fusion.py` and protected by
  a report-existence guard.

## Commands

1. Build the V2 dataset and split (the frozen V1 artifact is untouched):

```bash
python scripts/build_phase1_dataset.py --target melanoma --features v2
```

2. Extract embeddings. Pretrained VGG16 features need no HAM10000 training:

```bash
python scripts/extract_embeddings.py \
  --target melanoma \
  --model vgg16 \
  --features v2 \
  --weights imagenet
```

Alternatively, reuse a trained deep artifact:

```bash
python scripts/extract_embeddings.py \
  --target melanoma \
  --model custom_cnn \
  --artifact models/deep_melanoma_custom_cnn.keras
```

3. Train the fusion model:

```bash
python scripts/train_fusion.py --target melanoma --features v2
```

4. Evaluate the test split once:

```bash
python scripts/evaluate_fusion.py --target melanoma --features v2
```

## Defaults

| Setting | Value |
|---|---|
| Tabular features | V2 (adds `circularity` and `eccentricity`) |
| Embedding model | VGG16, 512 pooled features |
| PCA components | 32 |
| Classifier | ExtraTrees, `class_weight="balanced"` |
| CV | `StratifiedGroupKFold`, 5 folds, grouped by `lesion_id` |

All defaults are in `src/melanoma_detection/config.py` and can be overridden on
the CLI.

## Outputs

| Path | Purpose |
|---|---|
| `data/interim/embeddings_<target>_<model>.npz` | `image_id` + embedding matrix |
| `reports/metrics/embeddings_<target>_<model>.json` | Embedding manifest and hashes |
| `models/fusion_<target>_extra_trees.joblib` | Fitted fusion pipeline |
| `models/fusion_<target>_extra_trees.json` | Sidecar: features, threshold, validation |
| `models/fusion_<target>_frozen.json` | Pointer to the canonical fusion artifact |
| `reports/metrics/fusion_train_<target>.json` / `.md` | Training and validation report |
| `reports/metrics/fusion_final_test_<target>.json` / `.md` | One-shot test report |

## Caveats

- Embeddings must be extracted from the same image preprocessing used by the
  frozen embedding model.
- The embedding table keeps one row per image; the split is still grouped by
  lesion, so multiple images from one lesion stay together.
- Only `melanoma` is officially supported. Other targets print a warning and
  should be treated as experiments.
- Real embeddings require HAM10000 images. Synthetic tests use
  `--weights none` and are not model results.
