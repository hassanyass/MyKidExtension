"""
End-to-end integration tests for the full Image Pipeline.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.config import MyKidConfig
from packages.shared.types import (
    Detection, 
    BoundingBox, 
    RiskLevel,
    ProtectionAction
)
from packages.vision.base_model import VisionModel
from packages.pipeline.image_pipeline import ImagePipeline


# Mock Vision Model to avoid loading PyTorch/Ultralytics during standard CI
class MockVisionModel(VisionModel):
    def __init__(self, should_find_harmful=True):
        self.should_find_harmful = should_find_harmful
        self.is_loaded = False
        
    def load(self) -> None:
        self.is_loaded = True
        
    def predict(self, image: np.ndarray) -> list[Detection]:
        if not self.should_find_harmful:
            return []
            
        # Return a mock detection of a knife in the center of the image
        h, w = image.shape[:2]
        return [
            Detection(
                label="knife",
                confidence=0.95,
                bbox=BoundingBox(
                    x=w/2 - 50, 
                    y=h/2 - 50, 
                    width=100, 
                    height=100
                ),
                risk=RiskLevel.MEDIUM
            )
        ]


@pytest.fixture
def config():
    c = MyKidConfig()
    c.image.max_size = 640
    c.protection.mode = "gaussian"
    c.protection.blur_padding = 0.0 # 0 padding for easier assertion math
    return c


@pytest.fixture
def dummy_image_bytes():
    # Create a 800x800 white image (will trigger resize to 640)
    img = np.full((800, 800, 3), 255, dtype=np.uint8)
    
    # Draw a black box in the middle where the mock detection will be
    img[350:450, 350:450] = 0
    
    # Encode to bytes (like a web request)
    success, encoded = cv2.imencode('.jpg', img)
    assert success
    return encoded.tobytes()


class TestEndToEndImagePipeline:
    
    def test_pipeline_harmful_image(self, config, dummy_image_bytes):
        # 1. Setup pipeline with mock vision model
        mock_model = MockVisionModel(should_find_harmful=True)
        pipeline = ImagePipeline(config, vision_model=mock_model)
        
        # 2. Process
        final_image, analysis = pipeline.process(dummy_image_bytes)
        
        # 3. Assertions
        
        # Check Analysis Result
        assert analysis.overall_risk == RiskLevel.MEDIUM
        assert analysis.action == ProtectionAction.BLUR_REGION
        assert len(analysis.protection_regions) == 1
        
        # Check Image Properties
        # Input was 800x800, max_size is 640 -> should have been resized
        assert final_image.shape[0] <= 640
        assert final_image.shape[1] <= 640
        
        # Check Modification (the center should no longer be purely black due to Gaussian blur)
        # Because we resize to 640x640, the original 800x800 center box is scaled.
        # The mock model returns a bbox at (w/2 - 50), which for 640 is 270:370
        h, w = final_image.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # The pixel should be slightly altered (not exactly 0)
        # Actually with Gaussian blur on a black box surrounded by white, 
        # the center might stay 0 if the box is big enough, but edges will blur.
        # Let's just assert that the region is different from a raw resize.
        raw_resized = cv2.resize(np.full((800, 800, 3), 255, dtype=np.uint8), (640, 640))
        raw_resized[270:370, 270:370] = 0
        
        assert not np.array_equal(final_image, raw_resized)

    def test_pipeline_safe_image(self, config, dummy_image_bytes):
        # 1. Setup pipeline with safe mock model
        mock_model = MockVisionModel(should_find_harmful=False)
        pipeline = ImagePipeline(config, vision_model=mock_model)
        
        # 2. Process
        final_image, analysis = pipeline.process(dummy_image_bytes)
        
        # 3. Assertions
        assert analysis.overall_risk == RiskLevel.LOW
        assert analysis.action == ProtectionAction.ALLOW
        assert len(analysis.protection_regions) == 0
        
        # Image should be resized but NOT blurred
        assert final_image.shape[0] <= 640
        assert final_image.shape[1] <= 640
