"""
Tests for MyKid image processing loader and preprocessor.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.image_processing.loader import load_image
from packages.image_processing.preprocessor import resize_preserve_aspect_ratio, preprocess_for_inference
from packages.shared.errors import ImageProcessingError
from packages.shared.config import get_config


@pytest.fixture
def sample_rgb_image():
    """Create a simple 100x200 RGB image."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Red channel
    return img


@pytest.fixture
def sample_bgr_image():
    """Create a simple 100x200 BGR image (OpenCV default)."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # Red channel in BGR
    return img


@pytest.fixture
def sample_image_path(tmp_path, sample_bgr_image):
    """Save a sample image and return its path."""
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), sample_bgr_image)
    return img_path


class TestImageLoader:
    
    def test_load_from_path(self, sample_image_path):
        img_rgb = load_image(sample_image_path)
        assert isinstance(img_rgb, np.ndarray)
        assert img_rgb.shape == (100, 200, 3)
        # Check if it was correctly converted from BGR to RGB
        # (JPEG compression might change 255 to ~254, so we check > 250)
        assert img_rgb[0, 0, 0] > 250
        assert img_rgb[0, 0, 2] < 5

    def test_load_from_string_path(self, sample_image_path):
        img_rgb = load_image(str(sample_image_path))
        assert isinstance(img_rgb, np.ndarray)
        assert img_rgb.shape == (100, 200, 3)

    def test_load_from_bytes(self, sample_image_path):
        with open(sample_image_path, "rb") as f:
            img_bytes = f.read()
            
        img_rgb = load_image(img_bytes)
        assert isinstance(img_rgb, np.ndarray)
        assert img_rgb.shape == (100, 200, 3)

    def test_load_from_ndarray(self, sample_bgr_image):
        img_rgb = load_image(sample_bgr_image)
        assert isinstance(img_rgb, np.ndarray)
        assert img_rgb.shape == (100, 200, 3)
        # Should have converted BGR to RGB
        assert img_rgb[0, 0, 0] == 255

    def test_load_grayscale_image(self, tmp_path):
        gray_img = np.zeros((100, 200), dtype=np.uint8)
        gray_img.fill(128)
        
        img_path = tmp_path / "gray.jpg"
        cv2.imwrite(str(img_path), gray_img)
        
        img_rgb = load_image(img_path)
        assert img_rgb.shape == (100, 200, 3)
        assert img_rgb[0, 0, 0] == 128
        assert img_rgb[0, 0, 1] == 128
        assert img_rgb[0, 0, 2] == 128

    def test_load_rgba_image(self, tmp_path):
        rgba_img = np.zeros((100, 200, 4), dtype=np.uint8)
        rgba_img[:, :, 0] = 255 # B
        rgba_img[:, :, 3] = 255 # A
        
        img_path = tmp_path / "rgba.png"
        cv2.imwrite(str(img_path), rgba_img)
        
        img_rgb = load_image(img_path)
        assert img_rgb.shape == (100, 200, 3)

    def test_load_nonexistent_path(self):
        with pytest.raises(ImageProcessingError):
            load_image("does_not_exist.jpg")

    def test_load_invalid_bytes(self):
        with pytest.raises(ImageProcessingError):
            load_image(b"not an image")

    def test_load_invalid_type(self):
        with pytest.raises(ImageProcessingError):
            load_image(12345)


class TestPreprocessor:
    
    def test_resize_smaller_image(self, sample_rgb_image):
        # Image is 100x200
        # If max_dimension is 500, it should not resize
        resized = resize_preserve_aspect_ratio(sample_rgb_image, max_dimension=500)
        assert resized.shape == (100, 200, 3)

    def test_resize_larger_image(self, sample_rgb_image):
        # Image is 100x200 (H=100, W=200). Longest side is W=200.
        # Max dimension is 100. Should scale by 0.5.
        # New size should be H=50, W=100.
        resized = resize_preserve_aspect_ratio(sample_rgb_image, max_dimension=100)
        assert resized.shape == (50, 100, 3)

    def test_resize_tall_image(self):
        # Image is 200x100 (H=200, W=100). Longest side is H=200.
        # Max dimension is 100. Should scale by 0.5.
        tall_img = np.zeros((200, 100, 3), dtype=np.uint8)
        resized = resize_preserve_aspect_ratio(tall_img, max_dimension=100)
        assert resized.shape == (100, 50, 3)

    def test_resize_uses_config_default(self, sample_rgb_image):
        config = get_config()
        # Create a huge image
        huge_h = config.image.max_size + 100
        huge_w = config.image.max_size + 200
        huge_img = np.zeros((huge_h, huge_w, 3), dtype=np.uint8)
        
        resized = preprocess_for_inference(huge_img)
        
        assert max(resized.shape[:2]) == config.image.max_size

    def test_resize_invalid_image(self):
        with pytest.raises(ImageProcessingError):
            resize_preserve_aspect_ratio(np.array([]))
