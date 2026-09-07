"""
Tests for MyKid exception hierarchy.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.errors import (
    MyKidError,
    ModelError,
    ImageProcessingError,
    VideoProcessingError,
    InferenceError,
    ConfigurationError,
    ProtectionError,
)


class TestExceptionHierarchy:
    """All custom exceptions should inherit from MyKidError."""

    def test_model_error_is_mykid_error(self):
        assert issubclass(ModelError, MyKidError)

    def test_image_processing_error_is_mykid_error(self):
        assert issubclass(ImageProcessingError, MyKidError)

    def test_video_processing_error_is_mykid_error(self):
        assert issubclass(VideoProcessingError, MyKidError)

    def test_inference_error_is_mykid_error(self):
        assert issubclass(InferenceError, MyKidError)

    def test_configuration_error_is_mykid_error(self):
        assert issubclass(ConfigurationError, MyKidError)

    def test_protection_error_is_mykid_error(self):
        assert issubclass(ProtectionError, MyKidError)

    def test_all_are_exceptions(self):
        for exc_class in [
            MyKidError, ModelError, ImageProcessingError,
            VideoProcessingError, InferenceError,
            ConfigurationError, ProtectionError,
        ]:
            assert issubclass(exc_class, Exception)


class TestErrorCodes:
    """Each exception should carry a default error code."""

    def test_mykid_error_code(self):
        err = MyKidError("test")
        assert err.error_code == "MYKID_ERROR"

    def test_model_error_code(self):
        err = ModelError("model not found")
        assert err.error_code == "MODEL_ERROR"

    def test_image_processing_error_code(self):
        err = ImageProcessingError("invalid format")
        assert err.error_code == "IMAGE_PROCESSING_ERROR"

    def test_video_processing_error_code(self):
        err = VideoProcessingError("corrupt file")
        assert err.error_code == "VIDEO_PROCESSING_ERROR"

    def test_inference_error_code(self):
        err = InferenceError("timeout")
        assert err.error_code == "INFERENCE_ERROR"

    def test_configuration_error_code(self):
        err = ConfigurationError("missing key")
        assert err.error_code == "CONFIGURATION_ERROR"

    def test_protection_error_code(self):
        err = ProtectionError("blur failed")
        assert err.error_code == "PROTECTION_ERROR"


class TestErrorCustomCode:
    """Errors should accept custom error codes."""

    def test_custom_code(self):
        err = ModelError("failed", error_code="MODEL_LOAD_FAILED")
        assert err.error_code == "MODEL_LOAD_FAILED"


class TestErrorSerialization:
    """Errors should be serializable to dict."""

    def test_to_dict(self):
        err = ModelError("Model file not found")
        d = err.to_dict()
        assert d["error_code"] == "MODEL_ERROR"
        assert d["message"] == "Model file not found"
        assert d["type"] == "ModelError"

    def test_repr(self):
        err = InferenceError("timeout reached")
        r = repr(err)
        assert "InferenceError" in r
        assert "INFERENCE_ERROR" in r
        assert "timeout reached" in r


class TestErrorRaising:
    """Errors should be catchable at multiple levels."""

    def test_catch_specific(self):
        with pytest.raises(ModelError):
            raise ModelError("test")

    def test_catch_as_base(self):
        with pytest.raises(MyKidError):
            raise ImageProcessingError("test")

    def test_catch_as_exception(self):
        with pytest.raises(Exception):
            raise VideoProcessingError("test")

    def test_message_accessible(self):
        try:
            raise InferenceError("inference timed out")
        except MyKidError as e:
            assert e.message == "inference timed out"
            assert str(e) == "inference timed out"
