PYTHON ?= python
MODEL ?= custom_cnn

.PHONY: help install install-deep test dataset compare deep-train deep-evaluate clean

help:
	@echo "Targets:"
	@echo "  install        Install the tabular/test dependencies"
	@echo "  install-deep   Install TensorFlow and Pillow as well"
	@echo "  test           Run the test suite (deep tests skip without TensorFlow)"
	@echo "  dataset        Build the merged dataset and grouped split"
	@echo "  compare        Compare baselines and tuned models"
	@echo "  deep-train     Train a Keras model (MODEL=custom_cnn|vgg16)"
	@echo "  deep-evaluate  One-shot deep test evaluation"
	@echo "  clean          Remove caches"

install:
	$(PYTHON) -m pip install -e ".[tabular,dev]"

install-deep:
	$(PYTHON) -m pip install -e ".[tabular,deep,dev]"

test:
	$(PYTHON) -m pytest -q

dataset:
	$(PYTHON) scripts/01_dataset.py

compare:
	$(PYTHON) scripts/02_compare_models.py

deep-train:
	$(PYTHON) scripts/04_deep_learning.py train --model $(MODEL)

deep-evaluate:
	$(PYTHON) scripts/04_deep_learning.py evaluate --model $(MODEL)

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
