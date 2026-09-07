"""
Unit tests for the Risk Engine.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.config import MyKidConfig, DetectionConfig, ProtectionConfig
from packages.shared.types import (
    Detection, 
    BoundingBox, 
    RiskLevel, 
    SceneRisk,
    ProtectionAction,
    ProtectionMode
)
from packages.risk.risk_engine import RiskEngine


@pytest.fixture
def config():
    c = MyKidConfig()
    c.detection = DetectionConfig(
        object_confidence_threshold=0.5,
        scene_risk_threshold=0.6,
        max_allowed_object_area_ratio=0.6
    )
    c.protection = ProtectionConfig(
        blur_padding=0.1,
        mode="gaussian"
    )
    return c


@pytest.fixture
def engine(config):
    return RiskEngine(config)


@pytest.fixture
def image_shape():
    # height, width
    return (1000, 1000)


class TestRiskEngine:
    
    def test_empty_detections(self, engine, image_shape):
        result = engine.evaluate([], image_shape=image_shape)
        
        assert result.overall_risk == RiskLevel.LOW
        assert result.action == ProtectionAction.ALLOW
        assert len(result.protection_regions) == 0

    def test_single_low_risk_detection(self, engine, image_shape):
        det = Detection(
            label="person", 
            confidence=0.9, 
            bbox=BoundingBox(0, 0, 100, 100), 
            risk=RiskLevel.LOW
        )
        result = engine.evaluate([det], image_shape=image_shape)
        
        assert result.overall_risk == RiskLevel.LOW
        assert result.action == ProtectionAction.ALLOW
        assert len(result.protection_regions) == 0

    def test_single_medium_risk_detection(self, engine, image_shape):
        det = Detection(
            label="knife", 
            confidence=0.9, 
            bbox=BoundingBox(100, 100, 100, 100), # 100x100 area
            risk=RiskLevel.MEDIUM
        )
        result = engine.evaluate([det], image_shape=image_shape)
        
        assert result.overall_risk == RiskLevel.MEDIUM
        assert result.action == ProtectionAction.BLUR_REGION
        assert len(result.protection_regions) == 1
        
        # Test padding (10%)
        # Original: x=100, y=100, w=100, h=100
        # pad = 10 -> new_x=90, new_y=90, new_w=120, new_h=120
        region = result.protection_regions[0]
        assert region.bbox.x == 90
        assert region.bbox.y == 90
        assert region.bbox.width == 120
        assert region.bbox.height == 120
        assert region.mode == ProtectionMode.GAUSSIAN
        assert region.source_label == "knife"

    def test_multiple_detections_max_risk(self, engine, image_shape):
        dets = [
            Detection("person", 0.9, BoundingBox(0,0,10,10), RiskLevel.LOW),
            Detection("knife", 0.8, BoundingBox(10,10,10,10), RiskLevel.MEDIUM),
            Detection("gun", 0.9, BoundingBox(20,20,10,10), RiskLevel.HIGH),
        ]
        
        result = engine.evaluate(dets, image_shape=image_shape)
        
        # Should take the max risk
        assert result.overall_risk == RiskLevel.HIGH
        assert result.action == ProtectionAction.BLUR_REGION
        
        # Should only return protection regions for MEDIUM/HIGH
        assert len(result.protection_regions) == 2
        labels = [r.source_label for r in result.protection_regions]
        assert "knife" in labels
        assert "gun" in labels
        assert "person" not in labels

    def test_high_scene_risk_escalates_to_frame_blur(self, engine, image_shape):
        # Empty detections, but high scene risk
        scene_risk = SceneRisk(violence=0.8, graphic=0.1)
        
        result = engine.evaluate([], scene_risk=scene_risk, image_shape=image_shape)
        
        assert result.overall_risk == RiskLevel.HIGH
        assert result.action == ProtectionAction.BLUR_FRAME
        assert len(result.protection_regions) == 0

    def test_spatial_coverage_escalates_to_frame_blur(self, engine, image_shape):
        # Image is 1000x1000 = 1,000,000 area. Threshold is 0.6 (600,000)
        # Bounding box is 800x800 = 640,000 area
        det = Detection(
            label="knife", 
            confidence=0.9, 
            bbox=BoundingBox(0, 0, 800, 800),
            risk=RiskLevel.MEDIUM
        )
        
        result = engine.evaluate([det], image_shape=image_shape)
        
        assert result.overall_risk == RiskLevel.MEDIUM
        assert result.action == ProtectionAction.BLUR_FRAME
        
        # Depending on design, we might still generate regions, but action is BLUR_FRAME
        # Our implementation generates regions but they might be ignored by the policy
        # However, for BLUR_FRAME action, our engine does NOT generate regions
        assert len(result.protection_regions) == 0

    def test_spatial_coverage_ignores_low_risk(self, engine, image_shape):
        # Massive LOW risk object
        det = Detection(
            label="person", 
            confidence=0.9, 
            bbox=BoundingBox(0, 0, 800, 800),
            risk=RiskLevel.LOW
        )
        
        result = engine.evaluate([det], image_shape=image_shape)
        
        assert result.overall_risk == RiskLevel.LOW
        assert result.action == ProtectionAction.ALLOW
