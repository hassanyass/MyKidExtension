"""
MyKid Exception Hierarchy

Custom exceptions for every major failure mode in the pipeline.
Each exception carries a structured error code and message.
No silent `except: pass` — errors are either handled intentionally or surfaced clearly.
"""


class MyKidError(Exception):
    """
    Base exception for all MyKid errors.

    Attributes:
        message: Human-readable error description.
        error_code: Machine-readable error identifier.
    """

    def __init__(self, message: str, error_code: str = "MYKID_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code!r}, message={self.message!r})"


class ModelError(MyKidError):
    """Raised when the AI model cannot be loaded or is unavailable."""

    def __init__(self, message: str, error_code: str = "MODEL_ERROR"):
        super().__init__(message, error_code)


class ImageProcessingError(MyKidError):
    """Raised for invalid images, unsupported formats, or image processing failures."""

    def __init__(self, message: str, error_code: str = "IMAGE_PROCESSING_ERROR"):
        super().__init__(message, error_code)


class VideoProcessingError(MyKidError):
    """Raised for corrupt videos, unsupported codecs, or video processing failures."""

    def __init__(self, message: str, error_code: str = "VIDEO_PROCESSING_ERROR"):
        super().__init__(message, error_code)


class InferenceError(MyKidError):
    """Raised when AI inference fails, times out, or produces invalid results."""

    def __init__(self, message: str, error_code: str = "INFERENCE_ERROR"):
        super().__init__(message, error_code)


class ConfigurationError(MyKidError):
    """Raised for missing or invalid configuration values."""

    def __init__(self, message: str, error_code: str = "CONFIGURATION_ERROR"):
        super().__init__(message, error_code)


class ProtectionError(MyKidError):
    """Raised when the blur/protection engine fails to process content."""

    def __init__(self, message: str, error_code: str = "PROTECTION_ERROR"):
        super().__init__(message, error_code)
