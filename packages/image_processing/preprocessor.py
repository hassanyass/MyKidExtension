"""
Image preprocessing utilities for MyKid pipeline.
"""
import logging
import cv2
import numpy as np

from packages.shared.config import get_config
from packages.shared.errors import ImageProcessingError
from packages.shared.logger import get_logger, log_event, LogEvent


logger = get_logger("mykid.image_processing.preprocessor")


def resize_preserve_aspect_ratio(image: np.ndarray, max_dimension: int = None) -> np.ndarray:
    """
    Resize an image such that its longest side equals max_dimension,
    preserving the aspect ratio. If the image is already smaller, it is not enlarged.

    Args:
        image: RGB numpy array (H, W, 3).
        max_dimension: Maximum allowed size for width or height.
                       Defaults to config.image.max_size.

    Returns:
        Resized RGB numpy array.
        
    Raises:
        ImageProcessingError: If the image is invalid.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ImageProcessingError("Cannot resize an empty or invalid image")

    if max_dimension is None:
        config = get_config()
        max_dimension = config.image.max_size

    h, w = image.shape[:2]
    
    # If the image is smaller than max_dimension, return as is
    if max(h, w) <= max_dimension:
        return image

    # Calculate scale factor
    scale = max_dimension / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)

    try:
        # INTER_AREA is recommended for decimation (shrinking)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        log_event(
            logger, 
            logging.INFO,
            LogEvent.IMAGE_PROCESSED, 
            "Resized image",
            metadata={
                "action": "resize", 
                "original_shape": (h, w), 
                "new_shape": (new_h, new_w)
            }
        )
        return resized
        
    except Exception as e:
        error_msg = f"Failed to resize image: {str(e)}"
        log_event(logger, logging.ERROR, LogEvent.ERROR_IMAGE, error_msg, metadata={"error": str(e)})
        raise ImageProcessingError(error_msg) from e


def preprocess_for_inference(image: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing pipeline applied before model inference.
    
    Args:
        image: RGB numpy array.
        
    Returns:
        Preprocessed RGB numpy array ready for the model wrapper.
    """
    return resize_preserve_aspect_ratio(image)
