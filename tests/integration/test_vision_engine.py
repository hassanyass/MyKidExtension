"""
Integration tests for the AI Vision Engine Abstraction.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.config import load_config
from packages.shared.types import Detection, RiskLevel
from packages.vision.yolo_model import YoloModel


@pytest.fixture(scope="module")
def config():
    """Load configuration for tests."""
    return load_config()


@pytest.fixture(scope="module")
def vision_model(config):
    """
    Initialize and load the YOLO model once for all tests.
    Uses the default yolo11n.pt model.
    """
    model = YoloModel(config)
    model.load()
    return model


@pytest.fixture
def safe_image():
    """Create a mock safe RGB image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(200)  # Light gray background
    return img


class TestVisionEngineIntegration:
    """End-to-end tests for the vision abstraction layer."""

    def test_model_initialization(self, vision_model):
        """Test that the model initializes correctly."""
        assert vision_model is not None
        assert vision_model.model is not None
        
    def test_predict_safe_image(self, vision_model, safe_image):
        """Test that a blank/safe image returns NO safety-relevant detections."""
        detections = vision_model.predict(safe_image)
        
        # We expect 0 detections because the image is blank
        assert isinstance(detections, list)
        assert len(detections) == 0

    def test_predict_mocked_unsafe_detection(self, vision_model, monkeypatch, safe_image):
        """
        Since we can't easily generate an image that perfectly triggers YOLO's 'knife' class
        without external data, we mock the ultralytics model output to verify our 
        wrapper's filtering logic works correctly.
        """
        # Create a mock for ultralytics YOLO result
        class MockTensor:
            def __init__(self, val):
                self.val = val
            def item(self):
                return self.val
            def tolist(self):
                return self.val
                
        class MockBox:
            def __init__(self, cls_id, conf, xyxy):
                self.cls = [MockTensor(cls_id)]
                self.conf = [MockTensor(conf)]
                self.xyxy = [MockTensor(xyxy)]
                
        class MockResult:
            def __init__(self):
                # We mock: 1 safe object (person: 0), 1 unsafe object (knife: 43)
                self.boxes = [
                    MockBox(cls_id=0, conf=0.9, xyxy=[10, 10, 100, 100]),
                    MockBox(cls_id=43, conf=0.85, xyxy=[150, 150, 200, 200])
                ]
                
        # Patch the internal model call. YOLO uses `predict` internally when called.
        def mock_predict(*args, **kwargs):
            return [MockResult()]
            
        # Also patch `__call__` by replacing the instance method just in case
        monkeypatch.setattr(vision_model.model, "predict", mock_predict)
        
        # Or better yet, we can just replace the model completely for the test
        class MockYOLO:
            def __init__(self):
                self.names = {0: "person", 43: "knife"}
            def __call__(self, *args, **kwargs):
                return [MockResult()]
                
        monkeypatch.setattr(vision_model, "model", MockYOLO())
        
        # Run prediction
        detections = vision_model.predict(safe_image)
        
        # Verify filtering: should ONLY return the knife, not the person
        assert len(detections) == 1
        
        # Verify the detection properties
        det = detections[0]
        assert isinstance(det, Detection)
        assert det.label == "knife"
        assert det.confidence == 0.85
        assert det.risk == RiskLevel.MEDIUM
        
        # Verify bounding box mapping
        assert det.bbox.x == 150.0
        assert det.bbox.y == 150.0
        assert det.bbox.width == 50.0
        assert det.bbox.height == 50.0
