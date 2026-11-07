# Deep-learning line

The deep-learning line is the Keras/TensorFlow alternative to the tabular
pipeline. It consumes raw HAM10000 images on the same lesion-grouped Phase 1
split. Only `melanoma` is officially supported (`config.DEEP_TARGETS`), although
other binary targets remain configurable for experiments.

## Models

| Name | Description | Embedding layer |
|---|---|---|
| `custom_cnn` | Small three-block CNN from the legacy notebook | `Dense` named `embedding` |
| `vgg16` | Frozen ImageNet VGG16 base plus a small binary head | `Dense` named `embedding` |

Both models accept RGB images as float32 in `[0, 255]` and contain their own
preprocessing. `custom_cnn` rescales by `1/255`; `vgg16` reproduces Keras'
VGG16 mean-BGR preprocessing as a serialisable layer. Every model exposes a
layer named `embedding`, so one extractor works for both.

## Leakage rules

- The train/validation/test split is the lesion-grouped Phase 1 assignment.
- `scripts/train_deep.py` loads only train and validation rows.
- Early stopping monitors validation loss; the test split is not loaded.
- `scripts/evaluate_deep.py` scores validation and test once, using thresholds
  selected on validation.
- A completed test report is immutable unless `--force` is passed.

## Commands

Train:

```bash
python scripts/train_deep.py \
  --target melanoma \
  --model custom_cnn \
  --image-dir data/external/HAM10000_Dataset/HAM10000
```

Use `--model vgg16 --weights imagenet` for transfer learning, or
`--weights none` for offline/synthetic checks. Training a repeating Keras
dataset requires `steps_per_epoch`; the CLI computes it from the training frame.

One-shot evaluation:

```bash
python scripts/evaluate_deep.py \
  --target melanoma \
  --model custom_cnn \
  --image-dir data/external/HAM10000_Dataset/HAM10000
```

Extract penultimate embeddings for late fusion:

```bash
python scripts/extract_embeddings.py \
  --target melanoma \
  --model vgg16 \
  --weights imagenet \
  --image-dir data/external/HAM10000_Dataset/HAM10000
```

Pass `--artifact models/deep_melanoma_custom_cnn.keras` to use a trained model
instead of pretrained VGG16 features.

## Outputs

| Path | Purpose |
|---|---|
| `models/deep_<target>_<model>.keras` | Trained Keras model |
| `models/deep_<target>_<model>.json` | Sidecar: split fingerprint, history, hashes |
| `models/deep_<target>_frozen.json` | Pointer to the canonical deep artifact |
| `reports/metrics/deep_train_<target>_<model>.json` | Training report |
| `reports/metrics/deep_final_test_<target>_<model>.json` | One-shot test metrics |
| `reports/metrics/deep_final_test_<target>_<model>.md` | Human-readable test report |

Model binaries are git-ignored; sidecars and reports are tracked.

## Data requirement

HAM10000 images are not distributed with this repository. Place them at
`data/external/HAM10000_Dataset/HAM10000/{image_id}.jpg`; see `data/README.md`.
Without them, only synthetic or reduced-data smoke checks are possible.
