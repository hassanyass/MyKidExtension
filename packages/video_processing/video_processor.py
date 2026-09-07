"""
Video processing utilities for MyKid pipeline.
"""
import logging
from typing import Dict, Any
import cv2

from packages.shared.errors import VideoProcessingError
from packages.shared.logger import get_logger, log_event, LogEvent


logger = get_logger("mykid.video_processing.processor")


def open_video(path: str) -> cv2.VideoCapture:
    """
    Open a video file using OpenCV.
    
    Args:
        path: Path to the video file.
        
    Returns:
        An open cv2.VideoCapture object.
        
    Raises:
        VideoProcessingError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        error_msg = f"Failed to open video file: {path}"
        log_event(logger, logging.ERROR, LogEvent.ERROR_VIDEO, error_msg, metadata={"path": path})
        raise VideoProcessingError(error_msg)
        
    return cap


def get_metadata(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """
    Extract metadata from an open VideoCapture object.
    
    Args:
        cap: An open cv2.VideoCapture object.
        
    Returns:
        Dictionary containing fps, width, height, and frame_count.
    """
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }


def create_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Create an OpenCV VideoWriter.
    
    Args:
        output_path: Path where the video will be saved.
        fps: Frames per second.
        width: Video width.
        height: Video height.
        
    Returns:
        An initialized cv2.VideoWriter object.
    """
    # mp4v is a standard encoder that works well cross-platform for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        error_msg = f"Failed to initialize VideoWriter for {output_path}"
        log_event(logger, logging.ERROR, LogEvent.ERROR_VIDEO, error_msg)
        raise VideoProcessingError(error_msg)
        
    return writer
