"""
Image loading utilities for MyKid pipeline.
"""
from pathlib import Path
from typing import Union
import logging
import cv2
import numpy as np

from packages.shared.errors import ImageProcessingError
from packages.shared.logger import get_logger, log_event, LogEvent

logger = get_logger("mykid.image_processing.loader")


def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """
    Load an image from various sources and return a standard RGB numpy array.

    Args:
        source: A file path, byte string, or existing numpy array.

    Returns:
        RGB numpy array (H, W, 3).

    Raises:
        ImageProcessingError: If the image cannot be loaded or is invalid.
    """
    image = None

    try:
        # 1. From Path or String
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            # cv2.imread loads in BGR format
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"OpenCV could not decode image: {path}")
                
            log_event(logger, logging.INFO, LogEvent.IMAGE_LOADED, "Loaded image from path", metadata={"source_type": "path", "path": str(path)})

        # 2. From Bytes (e.g. uploaded file, network response)
        elif isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not decode byte stream")
                
            log_event(logger, logging.INFO, LogEvent.IMAGE_LOADED, "Loaded image from bytes", metadata={"source_type": "bytes", "size": len(source)})

        # 3. Already a Numpy Array (assume BGR if 3 channels from OpenCV, but we enforce RGB output)
        # Note: If it's already an array, we assume the caller knows its format.
        # But to be safe and consistent with cv2, if it's 3-channel we assume BGR input.
        # In practice, if a caller passes RGB, they should avoid this loader or we need a flag.
        # For simplicity, if it's an array, we assume it's BGR from a previous cv2 operation.
        elif isinstance(source, np.ndarray):
            image = source
            log_event(logger, logging.INFO, LogEvent.IMAGE_LOADED, "Loaded image from array", metadata={"source_type": "ndarray", "shape": image.shape})

        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

        # Validate loaded image
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Loaded image is empty or invalid")

        # Handle grayscale (1 channel) or RGBA (4 channels)
        if len(image.shape) == 2:
            # Grayscale to BGR first so the BGR2RGB step works consistently
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # Convert BGR to RGB (AI models expect RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    except Exception as e:
        error_msg = f"Failed to load image: {str(e)}"
        log_event(logger, logging.ERROR, LogEvent.ERROR_IMAGE, error_msg, metadata={"error": str(e)})
        raise ImageProcessingError(error_msg) from e
