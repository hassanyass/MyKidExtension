"""
MyKid Shared Package

Core types, configuration, logging, and error handling shared across all MyKid components.
"""

from packages.shared.types import (
    BoundingBox,
    Detection,
    SceneRisk,
    AnalysisResult,
    ProtectionRegion,
    VideoFrameResult,
    RiskLevel,
    ProtectionAction,
    ProtectionMode,
    ProcessingState,
)
from packages.shared.config import MyKidConfig, load_config, get_config
from packages.shared.logger import get_logger, LogEvent
from packages.shared.errors import (
    MyKidError,
    ModelError,
    ImageProcessingError,
    VideoProcessingError,
    InferenceError,
    ConfigurationError,
    ProtectionError,
)

__all__ = [
    # Types
    "BoundingBox",
    "Detection",
    "SceneRisk",
    "AnalysisResult",
    "ProtectionRegion",
    "VideoFrameResult",
    "RiskLevel",
    "ProtectionAction",
    "ProtectionMode",
    "ProcessingState",
    # Config
    "MyKidConfig",
    "load_config",
    "get_config",
    # Logging
    "get_logger",
    "LogEvent",
    # Errors
    "MyKidError",
    "ModelError",
    "ImageProcessingError",
    "VideoProcessingError",
    "InferenceError",
    "ConfigurationError",
    "ProtectionError",
]
