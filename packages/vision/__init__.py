"""
MyKid Vision Package

AI model abstraction layer.
"""

from packages.vision.base_model import VisionModel
from packages.vision.yolo_model import YoloModel

__all__ = [
    "VisionModel",
    "YoloModel",
]
