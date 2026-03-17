from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image


def read_dicom(path: str | Path):
    """Read a DICOM file and return an RGB array plus a PIL image for display.

    Args:
        path: Path to a DICOM file (`.dcm`).

    Returns:
        A tuple `(img_rgb, img2show)` where:

        - `img_rgb` is a NumPy RGB array.
        - `img2show` is a PIL Image instance for GUI rendering.
    """
    ds = pydicom.dcmread(str(path))
    img_array = ds.pixel_array
    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)

    img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_rgb, img2show


def read_image(path: str | Path):
    """Read a standard image file (jpg/png/jpeg) using OpenCV.

    Args:
        path: Path to an image file.

    Returns:
        A tuple `(img_array, img2show)` where `img_array` is a NumPy array and
        `img2show` is a PIL Image for GUI rendering.

    Raises:
        FileNotFoundError: If OpenCV cannot read the provided image.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def load_image(path: str | Path):
    """Load an image from disk, automatically handling DICOM and common formats.

    Args:
        path: Path to a `.dcm` file or a standard image file.

    Returns:
        The same tuple returned by `read_dicom` or `read_image` depending on the
        extension.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".dcm":
        return read_dicom(path)
    return read_image(path)
