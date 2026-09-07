"""
Tests for MyKid core types.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_risk_level_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"

    def test_protection_action_values(self):
        assert ProtectionAction.ALLOW.value == "allow"
        assert ProtectionAction.BLUR_REGION.value == "blur_region"
        assert ProtectionAction.BLUR_FRAME.value == "blur_frame"
        assert ProtectionAction.BLOCK.value == "block"

    def test_protection_mode_values(self):
        assert ProtectionMode.GAUSSIAN.value == "gaussian"
        assert ProtectionMode.PIXELATE.value == "pixelate"
        assert ProtectionMode.COVER.value == "cover"

    def test_processing_state_values(self):
        assert ProcessingState.UNPROCESSED.value == "unprocessed"
        assert ProcessingState.PROCESSING.value == "processing"
        assert ProcessingState.PROCESSED.value == "processed"
        assert ProcessingState.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# BoundingBox Tests
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_creation(self):
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=80.0)
        assert bbox.x == 10.0
        assert bbox.y == 20.0
        assert bbox.width == 100.0
        assert bbox.height == 80.0

    def test_to_dict(self):
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=80.0)
        d = bbox.to_dict()
        assert d == {"x": 10.0, "y": 20.0, "width": 100.0, "height": 80.0}

    def test_area(self):
        bbox = BoundingBox(x=0, y=0, width=100.0, height=50.0)
        assert bbox.area() == 5000.0

    def test_area_zero(self):
        bbox = BoundingBox(x=0, y=0, width=0, height=50.0)
        assert bbox.area() == 0.0

    def test_padded(self):
        bbox = BoundingBox(x=100.0, y=100.0, width=200.0, height=100.0)
        padded = bbox.padded(0.1, image_width=800.0, image_height=600.0)
        # Padding: 20px horizontal, 10px vertical
        assert padded.x == 80.0
        assert padded.y == 90.0
        assert padded.width == 240.0
        assert padded.height == 120.0

    def test_padded_clamped_to_image(self):
        bbox = BoundingBox(x=5.0, y=5.0, width=100.0, height=100.0)
        padded = bbox.padded(0.2, image_width=120.0, image_height=120.0)
        assert padded.x == 0.0
        assert padded.y == 0.0
        # Width and height clamped to image size
        assert padded.width <= 120.0
        assert padded.height <= 120.0


# ---------------------------------------------------------------------------
# Detection Tests
# ---------------------------------------------------------------------------

class TestDetection:
    def test_creation(self):
        det = Detection(label="knife", confidence=0.93)
        assert det.label == "knife"
        assert det.confidence == 0.93
        assert det.bbox is None
        assert det.risk == RiskLevel.LOW

    def test_creation_with_bbox(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        det = Detection(label="gun", confidence=0.88, bbox=bbox, risk=RiskLevel.HIGH)
        assert det.bbox is not None
        assert det.risk == RiskLevel.HIGH

    def test_to_dict_without_bbox(self):
        det = Detection(label="knife", confidence=0.93)
        d = det.to_dict()
        assert "bbox" not in d
        assert d["label"] == "knife"
        assert d["confidence"] == 0.93
        assert d["risk"] == "low"

    def test_to_dict_with_bbox(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        det = Detection(label="knife", confidence=0.93, bbox=bbox)
        d = det.to_dict()
        assert "bbox" in d
        assert d["bbox"]["x"] == 10


# ---------------------------------------------------------------------------
# SceneRisk Tests
# ---------------------------------------------------------------------------

class TestSceneRisk:
    def test_defaults(self):
        sr = SceneRisk()
        assert sr.violence == 0.0
        assert sr.graphic == 0.0

    def test_max_score(self):
        sr = SceneRisk(violence=0.82, graphic=0.45)
        assert sr.max_score() == 0.82

    def test_to_dict(self):
        sr = SceneRisk(violence=0.5, graphic=0.3)
        d = sr.to_dict()
        assert d == {"violence": 0.5, "graphic": 0.3}


# ---------------------------------------------------------------------------
# AnalysisResult Tests
# ---------------------------------------------------------------------------

class TestAnalysisResult:
    def test_defaults(self):
        result = AnalysisResult()
        assert result.content_type == "image"
        assert result.detections == []
        assert result.overall_risk == RiskLevel.LOW
        assert result.action == ProtectionAction.ALLOW

    def test_to_dict(self, sample_analysis_result):
        d = sample_analysis_result.to_dict()
        assert d["content_type"] == "image"
        assert len(d["detections"]) == 1
        assert d["overall_risk"] == "medium"
        assert d["action"] == "blur_region"


# ---------------------------------------------------------------------------
# ProtectionRegion Tests
# ---------------------------------------------------------------------------

class TestProtectionRegion:
    def test_creation(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        region = ProtectionRegion(bbox=bbox, mode=ProtectionMode.GAUSSIAN, source_label="knife")
        assert region.source_label == "knife"

    def test_to_dict(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        region = ProtectionRegion(bbox=bbox, mode=ProtectionMode.PIXELATE, source_label="gun")
        d = region.to_dict()
        assert d["mode"] == "pixelate"
        assert d["source_label"] == "gun"


# ---------------------------------------------------------------------------
# VideoFrameResult Tests
# ---------------------------------------------------------------------------

class TestVideoFrameResult:
    def test_creation(self):
        result = VideoFrameResult(frame_index=42, timestamp_ms=1400.0)
        assert result.frame_index == 42
        assert result.timestamp_ms == 1400.0
        assert result.action == ProtectionAction.ALLOW

    def test_to_dict(self):
        result = VideoFrameResult(
            frame_index=10,
            timestamp_ms=333.3,
            overall_risk=RiskLevel.HIGH,
            action=ProtectionAction.BLUR_FRAME,
        )
        d = result.to_dict()
        assert d["frame_index"] == 10
        assert d["overall_risk"] == "high"
        assert d["action"] == "blur_frame"
