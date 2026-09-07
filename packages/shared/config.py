"""
MyKid Configuration System

Centralized configuration loaded from YAML with environment variable overrides.
All configurable thresholds and settings live here — no magic numbers elsewhere.

Environment variable override format:
    MYKID_<SECTION>_<KEY> (e.g. MYKID_DETECTION_OBJECT_CONFIDENCE_THRESHOLD)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from packages.shared.errors import ConfigurationError


# Default config path relative to project root
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"

# Singleton instance
_config_instance: Optional["MyKidConfig"] = None


@dataclass
class DetectionConfig:
    """Detection-related thresholds."""
    object_confidence_threshold: float = 0.5
    scene_risk_threshold: float = 0.6


@dataclass
class ProtectionConfig:
    """Protection engine settings."""
    blur_padding: float = 0.1
    mode: str = "gaussian"


@dataclass
class VideoConfig:
    """Video processing settings."""
    inference_fps: int = 8
    temporal_persistence_frames: int = 10


@dataclass
class ImageConfig:
    """Image processing settings."""
    max_size: int = 1920


@dataclass
class ModelConfig:
    """Model settings."""
    path: str = "models/"


@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str = "INFO"


@dataclass
class MyKidConfig:
    """
    Root configuration for the MyKid system.
    All thresholds and settings are centralized here.
    """
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __repr__(self) -> str:
        return (
            f"MyKidConfig(\n"
            f"  detection={self.detection},\n"
            f"  protection={self.protection},\n"
            f"  video={self.video},\n"
            f"  image={self.image},\n"
            f"  model={self.model},\n"
            f"  logging={self.logging}\n"
            f")"
        )


def _apply_env_overrides(config: MyKidConfig) -> None:
    """
    Apply environment variable overrides to the config.
    Format: MYKID_<SECTION>_<KEY> (uppercase, underscores).

    Examples:
        MYKID_DETECTION_OBJECT_CONFIDENCE_THRESHOLD=0.7
        MYKID_PROTECTION_MODE=pixelate
        MYKID_VIDEO_INFERENCE_FPS=10
    """
    env_mappings = {
        "MYKID_DETECTION_OBJECT_CONFIDENCE_THRESHOLD": (config.detection, "object_confidence_threshold", float),
        "MYKID_DETECTION_SCENE_RISK_THRESHOLD": (config.detection, "scene_risk_threshold", float),
        "MYKID_PROTECTION_BLUR_PADDING": (config.protection, "blur_padding", float),
        "MYKID_PROTECTION_MODE": (config.protection, "mode", str),
        "MYKID_VIDEO_INFERENCE_FPS": (config.video, "inference_fps", int),
        "MYKID_VIDEO_TEMPORAL_PERSISTENCE_FRAMES": (config.video, "temporal_persistence_frames", int),
        "MYKID_IMAGE_MAX_SIZE": (config.image, "max_size", int),
        "MYKID_MODEL_PATH": (config.model, "path", str),
        "MYKID_LOGGING_LEVEL": (config.logging, "level", str),
    }

    for env_key, (obj, attr, type_fn) in env_mappings.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                setattr(obj, attr, type_fn(env_val))
            except (ValueError, TypeError) as e:
                raise ConfigurationError(
                    f"Invalid environment variable {env_key}={env_val}: {e}"
                )


def _parse_yaml(data: dict) -> MyKidConfig:
    """Parse a YAML dictionary into a MyKidConfig."""
    config = MyKidConfig()

    if "detection" in data and isinstance(data["detection"], dict):
        for key, value in data["detection"].items():
            if hasattr(config.detection, key):
                setattr(config.detection, key, value)

    if "protection" in data and isinstance(data["protection"], dict):
        for key, value in data["protection"].items():
            if hasattr(config.protection, key):
                setattr(config.protection, key, value)

    if "video" in data and isinstance(data["video"], dict):
        for key, value in data["video"].items():
            if hasattr(config.video, key):
                setattr(config.video, key, value)

    if "image" in data and isinstance(data["image"], dict):
        for key, value in data["image"].items():
            if hasattr(config.image, key):
                setattr(config.image, key, value)

    if "model" in data and isinstance(data["model"], dict):
        for key, value in data["model"].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    if "logging" in data and isinstance(data["logging"], dict):
        for key, value in data["logging"].items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)

    return config


def load_config(path: Optional[str] = None) -> MyKidConfig:
    """
    Load configuration from a YAML file, then apply environment variable overrides.

    Args:
        path: Path to YAML config file. Defaults to configs/default.yaml.

    Returns:
        MyKidConfig with all values resolved.

    Raises:
        ConfigurationError: If the YAML file cannot be parsed.
    """
    global _config_instance

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    config = MyKidConfig()

    if config_path.exists():
        if yaml is None:
            raise ConfigurationError(
                "PyYAML is required to load config files. Install with: pip install pyyaml"
            )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                config = _parse_yaml(data)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse config file {config_path}: {e}")

    _apply_env_overrides(config)

    _config_instance = config
    return config


def get_config() -> MyKidConfig:
    """
    Get the current config singleton.
    If not loaded yet, loads from the default path.

    Returns:
        The current MyKidConfig instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def reset_config() -> None:
    """Reset the config singleton (useful for testing)."""
    global _config_instance
    _config_instance = None
