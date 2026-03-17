from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from uao_neumonia.utils.paths import repo_root


_MODEL = None


def get_model(model_path: str | Path | None = None):
    """Load and cache the trained CNN model.

    The model is loaded once and stored in a module-level cache to avoid
    re-loading it on every prediction.

    Args:
        model_path: Optional path to a `.h5` model file. If not provided, the
            default is resolved relative to the repository root:
            `model/conv_MLP_84.h5`.

    Returns:
        A TensorFlow/Keras model instance.

    Raises:
        FileNotFoundError: If the resolved model path does not exist.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if model_path is None:
        model_path = repo_root() / "model" / "conv_MLP_84.h5"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    _MODEL = tf.keras.models.load_model(str(model_path), compile=False)
    return _MODEL
