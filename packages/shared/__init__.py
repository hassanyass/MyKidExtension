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
from packages.shared.labels import (
    SafetyCategory,
    is_safety_relevant,
    get_safety_category,
    get_base_risk,
    get_all_safety_labels,
    get_coco_label,
    SAFETY_CATEGORIES,
    COCO_LABELS,
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
    # Labels
    "SafetyCategory",
    "is_safety_relevant",
    "get_safety_category",
    "get_base_risk",
    "get_all_safety_labels",
    "get_coco_label",
    "SAFETY_CATEGORIES",
    "COCO_LABELS",
]
