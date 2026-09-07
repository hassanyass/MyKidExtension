"""
Unit tests for the Protection Engine.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.config import MyKidConfig, ProtectionConfig
from packages.shared.types import (
    AnalysisResult, 
    ProtectionAction, 
    ProtectionRegion, 
    ProtectionMode,
    BoundingBox
)
from packages.protection.protection_engine import ProtectionEngine


@pytest.fixture
def config():
    c = MyKidConfig()
    c.protection = ProtectionConfig(
        blur_padding=0.1,
        mode="gaussian"
    )
    return c


@pytest.fixture
def engine(config):
    return ProtectionEngine(config)


@pytest.fixture
def dummy_image():
    # 100x100 RGB image (all zeros/black)
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestProtectionEngine:
    
    def test_allow_action_returns_unaltered(self, engine, dummy_image):
        analysis = AnalysisResult(action=ProtectionAction.ALLOW)
        
        # Modify dummy image slightly
        dummy_image[50, 50] = [255, 255, 255]
        
        result = engine.protect(dummy_image, analysis)
        
        # Should be the exact same array (or identical contents)
        assert np.array_equal(result, dummy_image)
        
    def test_blur_frame_action(self, engine, dummy_image):
        # Set a white pixel
        dummy_image[50, 50] = [255, 255, 255]
        
        analysis = AnalysisResult(action=ProtectionAction.BLUR_FRAME)
        
        result = engine.protect(dummy_image, analysis)
        
        # After Gaussian blur, the bright pixel at 50,50 should be dimmed
        # and surrounding pixels should no longer be exactly 0
        assert not np.array_equal(result, dummy_image)
        
    def test_blur_region_action(self, engine, dummy_image):
        # Image is entirely 255 (white)
        dummy_image.fill(255)
        
        region = ProtectionRegion(
            bbox=BoundingBox(10, 10, 20, 20),
            mode=ProtectionMode.COVER, # Solid black
            source_label="test"
        )
        analysis = AnalysisResult(
            action=ProtectionAction.BLUR_REGION,
            protection_regions=[region]
        )
        
        result = engine.protect(dummy_image, analysis)
        
        # The region [10:30, 10:30] should be 0 (solid cover)
        roi = result[10:30, 10:30]
        assert np.all(roi == 0)
        
        # Pixels outside the region should still be 255
        assert np.all(result[0:9, 0:9] == 255)
        assert np.all(result[31:99, 31:99] == 255)
        
    def test_blur_region_out_of_bounds(self, engine, dummy_image):
        dummy_image.fill(255)
        
        # Bbox extends outside the 100x100 image
        region = ProtectionRegion(
            bbox=BoundingBox(80, 80, 50, 50), # Goes up to 130x130
            mode=ProtectionMode.COVER,
            source_label="test"
        )
        analysis = AnalysisResult(
            action=ProtectionAction.BLUR_REGION,
            protection_regions=[region]
        )
        
        # Should not crash, should clamp to 100x100
        result = engine.protect(dummy_image, analysis)
        
        # The clamped region [80:100, 80:100] should be 0
        roi = result[80:100, 80:100]
        assert np.all(roi == 0)
        
    def test_pixelation_mode(self, engine, dummy_image):
        # Draw a pattern
        dummy_image[10:20, 10:20] = [255, 0, 0]
        dummy_image[20:30, 20:30] = [0, 255, 0]
        
        region = ProtectionRegion(
            bbox=BoundingBox(10, 10, 20, 20),
            mode=ProtectionMode.PIXELATE,
            source_label="test"
        )
        analysis = AnalysisResult(
            action=ProtectionAction.BLUR_REGION,
            protection_regions=[region]
        )
        
        result = engine.protect(dummy_image, analysis)
        
        # Should be modified, but not solid cover (some color should remain)
        assert not np.array_equal(result, dummy_image)
        roi = result[10:30, 10:30]
        assert np.any(roi > 0)
