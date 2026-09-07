"""
MyKid Structured Logger

JSON-structured logging with component-named loggers.
Privacy-conscious: never logs image data, browsing content, or personal information.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class LogEvent:
    """Structured log event constants matching the SKILL.md specification."""

    # Model lifecycle
    MODEL_LOADED = "MODEL_LOADED"
    MODEL_ERROR = "MODEL_ERROR"

    # Image processing
    IMAGE_ANALYSIS_STARTED = "IMAGE_ANALYSIS_STARTED"
    IMAGE_ANALYSIS_COMPLETED = "IMAGE_ANALYSIS_COMPLETED"

    # Video processing
    VIDEO_STARTED = "VIDEO_STARTED"
    VIDEO_COMPLETED = "VIDEO_COMPLETED"
    FRAME_ANALYZED = "FRAME_ANALYZED"

    # Detection & protection
    RISK_DETECTED = "RISK_DETECTED"
    PROTECTION_APPLIED = "PROTECTION_APPLIED"

    # Errors
    PROCESSING_ERROR = "PROCESSING_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"

    # General
    PIPELINE_STARTED = "PIPELINE_STARTED"
    PIPELINE_COMPLETED = "PIPELINE_COMPLETED"


class StructuredFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.

    Output format:
    {
        "timestamp": "2025-01-01T00:00:00.000Z",
        "level": "INFO",
        "component": "mykid.vision",
        "event": "MODEL_LOADED",
        "message": "Model loaded successfully",
        "metadata": { ... }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        # Include event if set via extra
        if hasattr(record, "event"):
            log_entry["event"] = record.event

        # Include metadata if set via extra
        if hasattr(record, "metadata"):
            log_entry["metadata"] = record.metadata

        return json.dumps(log_entry, default=str)


def get_logger(
    component: str,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Get a structured logger for a MyKid component.

    Args:
        component: Component name (e.g., "vision", "risk", "protection").
                   Will be prefixed with "mykid." automatically.
        level: Override log level. If None, uses the config or defaults to INFO.

    Returns:
        A configured Logger instance.
    """
    logger_name = f"mykid.{component}" if not component.startswith("mykid.") else component
    logger = logging.getLogger(logger_name)

    # Only add handler if the logger doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    # Set level
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    elif not logger.level:
        logger.setLevel(logging.INFO)

    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a structured event with optional metadata.

    Args:
        logger: The logger instance.
        level: Logging level (e.g., logging.INFO).
        event: Event constant from LogEvent.
        message: Human-readable message.
        metadata: Optional dictionary of additional data.
    """
    extra: Dict[str, Any] = {"event": event}
    if metadata:
        extra["metadata"] = metadata
    logger.log(level, message, extra=extra)
