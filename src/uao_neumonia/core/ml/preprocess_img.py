from __future__ import annotations

import cv2
import numpy as np


def preprocess(array: np.ndarray) -> np.ndarray:
    """Preprocess an input X-ray image into the CNN expected tensor.

    The pipeline performs:

    - Resize to 512x512.
    - Convert to grayscale.
    - Apply CLAHE histogram equalization.
    - Normalize to the [0, 1] range.
    - Expand dimensions to obtain a batch tensor.

    Args:
        array: Input image as a NumPy array (typically BGR as read by OpenCV).

    Returns:
        A NumPy array with shape (1, 512, 512, 1) and float values in [0, 1].
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
