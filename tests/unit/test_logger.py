"""
Tests for MyKid structured logging.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.logger import get_logger, log_event, LogEvent, StructuredFormatter


class TestLogEvent:
    """Test that all expected event constants exist."""

    def test_model_events(self):
        assert LogEvent.MODEL_LOADED == "MODEL_LOADED"
        assert LogEvent.MODEL_ERROR == "MODEL_ERROR"

    def test_image_events(self):
        assert LogEvent.IMAGE_ANALYSIS_STARTED == "IMAGE_ANALYSIS_STARTED"
        assert LogEvent.IMAGE_ANALYSIS_COMPLETED == "IMAGE_ANALYSIS_COMPLETED"

    def test_video_events(self):
        assert LogEvent.VIDEO_STARTED == "VIDEO_STARTED"
        assert LogEvent.VIDEO_COMPLETED == "VIDEO_COMPLETED"
        assert LogEvent.FRAME_ANALYZED == "FRAME_ANALYZED"

    def test_detection_events(self):
        assert LogEvent.RISK_DETECTED == "RISK_DETECTED"
        assert LogEvent.PROTECTION_APPLIED == "PROTECTION_APPLIED"

    def test_error_events(self):
        assert LogEvent.PROCESSING_ERROR == "PROCESSING_ERROR"
        assert LogEvent.CONFIGURATION_ERROR == "CONFIGURATION_ERROR"


class TestGetLogger:
    """Test logger creation and configuration."""

    def test_creates_named_logger(self):
        logger = get_logger("vision")
        assert logger.name == "mykid.vision"

    def test_prefixed_name_not_doubled(self):
        logger = get_logger("mykid.risk")
        assert logger.name == "mykid.risk"

    def test_different_components_different_loggers(self):
        logger1 = get_logger("vision")
        logger2 = get_logger("risk")
        assert logger1.name != logger2.name

    def test_custom_level(self):
        logger = get_logger("test_level", level="DEBUG")
        assert logger.level == logging.DEBUG


class TestStructuredFormatter:
    """Test that the formatter produces valid JSON."""

    def test_format_produces_json(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="mykid.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["component"] == "mykid.test"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_format_includes_event(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="mykid.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Model loaded",
            args=None,
            exc_info=None,
        )
        record.event = LogEvent.MODEL_LOADED  # type: ignore
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["event"] == "MODEL_LOADED"

    def test_format_includes_metadata(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="mykid.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Detection found",
            args=None,
            exc_info=None,
        )
        record.metadata = {"label": "knife", "confidence": 0.93}  # type: ignore
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["metadata"]["label"] == "knife"
        assert parsed["metadata"]["confidence"] == 0.93


class TestLogEventFunction:
    """Test the log_event helper."""

    def test_log_event_executes_without_error(self):
        logger = get_logger("test_event_fn")
        # Should not raise
        log_event(
            logger,
            logging.INFO,
            LogEvent.IMAGE_ANALYSIS_STARTED,
            "Starting analysis",
            metadata={"image_size": "1920x1080"},
        )

    def test_log_event_without_metadata(self):
        logger = get_logger("test_event_no_meta")
        log_event(
            logger,
            logging.WARNING,
            LogEvent.PROCESSING_ERROR,
            "Something went wrong",
        )
