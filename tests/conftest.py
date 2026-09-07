"""
MyKid Test Configuration

Shared pytest fixtures for the entire test suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so 'packages' can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.shared.types import (
    BoundingBox,
    Detection,
    SceneRisk,
    AnalysisResult,
    RiskLevel,
    ProtectionAction,
)
from packages.shared.config import load_config, reset_config


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the config singleton before each test to ensure isolation."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def default_config():
    """Load and return the default configuration."""
    return load_config()


@pytest.fixture
def sample_bbox():
    """A sample bounding box for testing."""
    return BoundingBox(x=100.0, y=80.0, width=200.0, height=160.0)


@pytest.fixture
def sample_detection(sample_bbox):
    """A sample detection for testing."""
    return Detection(
        label="knife",
        confidence=0.93,
        bbox=sample_bbox,
        risk=RiskLevel.MEDIUM,
    )


@pytest.fixture
def sample_safe_detection():
    """A sample safe detection (low confidence, no risk)."""
    return Detection(
        label="spoon",
        confidence=0.85,
        bbox=BoundingBox(x=50.0, y=50.0, width=80.0, height=40.0),
        risk=RiskLevel.LOW,
    )


@pytest.fixture
def sample_analysis_result(sample_detection):
    """A sample analysis result with one detection."""
    return AnalysisResult(
        content_type="image",
        detections=[sample_detection],
        scene_risk=SceneRisk(violence=0.05, graphic=0.01),
        overall_risk=RiskLevel.MEDIUM,
        action=ProtectionAction.BLUR_REGION,
    )


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="mykid_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_path():
    """Path to the default config file."""
    return str(PROJECT_ROOT / "configs" / "default.yaml")
