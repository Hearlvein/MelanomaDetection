"""Deep-learning line (Keras/TensorFlow).

Submodules are imported explicitly so tabular-only environments never import
TensorFlow:

- :mod:`melanoma_detection.deep.datasets` — pure NumPy/Pillow image loading.
- :mod:`melanoma_detection.deep.models` — Keras CNN builders.
- :mod:`melanoma_detection.deep.train` — training and artifact helpers.
- :mod:`melanoma_detection.deep.evaluate` — prediction and metrics.
- :mod:`melanoma_detection.deep.embeddings` — penultimate feature extraction.
"""

from __future__ import annotations

__all__: list[str] = []
