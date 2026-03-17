from __future__ import annotations

import numpy as np

from uao_neumonia.core.ml.grad_cam import grad_cam
from uao_neumonia.core.ml.load_model import get_model
from uao_neumonia.core.ml.preprocess_img import preprocess


def predict(array):
    """Run the end-to-end inference pipeline for a single image.

    This function orchestrates the high-level flow used by the GUI:

    - Preprocess the input image into the model input tensor.
    - Load (or reuse) the CNN model.
    - Predict class probabilities.
    - Generate a Grad-CAM heatmap for explainability.

    Args:
        array: Input image as a NumPy array (typically RGB/BGR).

    Returns:
        A tuple `(label, proba, heatmap)` where:

        - `label` is one of: "bacteriana", "normal", "viral".
        - `proba` is the confidence percentage (0-100) for the predicted class.
        - `heatmap` is an RGB image (NumPy array) with the Grad-CAM overlay.
    """
    batch_array_img = preprocess(array)
    model = get_model()

    preds = model.predict(batch_array_img)
    prediction = int(np.argmax(preds))
    proba = float(np.max(preds) * 100)

    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"

    heatmap = grad_cam(array, model=model)
    return label, proba, heatmap
