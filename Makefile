PYTHON ?= python
TARGET ?= melanoma
TABULAR_EXTRAS = tabular,extraction,dev

.PHONY: help install install-deep test compile dataset-v1 dataset-v2 features embeddings deep-train deep-evaluate fusion-train fusion-evaluate final-test clean

help:
	@echo "Targets:"
	@echo "  install        Install editable package with tabular/extraction/test extras"
	@echo "  install-deep   Same, plus TensorFlow"
	@echo "  test           Run the full test suite (deep tests skip without TensorFlow)"
	@echo "  compile        Byte-compile scripts, package, and tests"
	@echo "  dataset-v1     Build the frozen V1 dataset and melanoma split"
	@echo "  dataset-v2     Build the V2 dataset and melanoma split"
	@echo "  features       Extract V2 image features (requires images and masks)"
	@echo "  embeddings     Extract VGG16 embeddings (requires images)"
	@echo "  deep-train     Train a Keras classifier (MODEL=custom_cnn|vgg16)"
	@echo "  deep-evaluate  One-shot deep test evaluation"
	@echo "  fusion-train   Train the V2 + embedding fusion model"
	@echo "  fusion-evaluate  One-shot fusion test evaluation"
	@echo "  final-test     One-shot Phase 5 tabular test evaluation (guard applies)"
	@echo "  clean          Remove caches"

install:
	$(PYTHON) -m pip install -e ".[$(TABULAR_EXTRAS)]"

install-deep:
	$(PYTHON) -m pip install -e ".[tabular,extraction,deep,dev]"

test:
	$(PYTHON) -m pytest -q

compile:
	$(PYTHON) -m compileall -q scripts src tests

dataset-v1:
	$(PYTHON) scripts/build_phase1_dataset.py --target $(TARGET) --features v1

dataset-v2:
	$(PYTHON) scripts/build_phase1_dataset.py --target $(TARGET) --features v2

features:
	$(PYTHON) scripts/extract_features.py --overwrite

embeddings:
	$(PYTHON) scripts/extract_embeddings.py --target $(TARGET) --features v2 --overwrite

MODEL ?= custom_cnn

deep-train:
	$(PYTHON) scripts/train_deep.py --target $(TARGET) --model $(MODEL)

deep-evaluate:
	$(PYTHON) scripts/evaluate_deep.py --target $(TARGET) --model $(MODEL)

fusion-train:
	$(PYTHON) scripts/train_fusion.py --target $(TARGET) --features v2

fusion-evaluate:
	$(PYTHON) scripts/evaluate_fusion.py --target $(TARGET) --features v2

final-test:
	$(PYTHON) scripts/evaluate_final_test.py --target $(TARGET)

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
