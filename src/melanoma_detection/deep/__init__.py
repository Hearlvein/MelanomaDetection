"""Deep-learning line (Keras/TensorFlow).

Submodules are imported explicitly so tabular-only environments never import
TensorFlow:

- :mod:`melanoma_detection.deep.run` — NumPy/Pillow image loading, training,
  artifact helpers, and evaluation.
- :mod:`melanoma_detection.deep.models` — Keras CNN and VGG16 builders.
"""

from __future__ import annotations

__all__: list[str] = []
