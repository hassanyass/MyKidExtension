"""
MyKid Image Protection Engine.

Applies visual obfuscation to images based on the Risk Engine's decision.
"""
from typing import Tuple
import logging
import cv2
import numpy as np

from packages.shared.types import (
    AnalysisResult,
    ProtectionAction,
    ProtectionRegion,
    ProtectionMode
)
from packages.shared.config import MyKidConfig
from packages.shared.logger import get_logger, log_event, LogEvent
from packages.shared.errors import ProtectionError


logger = get_logger("mykid.protection_engine")


class ProtectionEngine:
    """
    Applies visual protection (blur, pixelation, etc.) to an image.
    """
    
    def __init__(self, config: MyKidConfig):
        """
        Initialize the Protection Engine.
        
        Args:
            config: Global configuration.
        """
        self.config = config

    def protect(self, image: np.ndarray, analysis: AnalysisResult) -> np.ndarray:
        """
        Apply the protection policy to the image.
        
        Args:
            image: Original RGB image array.
            analysis: The analysis result dictating the action.
            
        Returns:
            The protected RGB image array.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProtectionError("Invalid or empty image array passed to ProtectionEngine")
            
        # 1. ALLOW -> Return unaltered
        if analysis.action == ProtectionAction.ALLOW:
            return image
            
        # 2. BLUR_FRAME -> Protect the entire image
        if analysis.action == ProtectionAction.BLUR_FRAME:
            try:
                # Use the configured mode for the full frame
                mode = ProtectionMode(self.config.protection.mode)
                protected_image = self._apply_protection(image.copy(), mode)
                
                log_event(
                    logger,
                    logging.INFO,
                    LogEvent.PROTECTION_APPLIED,
                    "Full frame protection applied",
                    metadata={"action": "BLUR_FRAME", "mode": mode.value}
                )
                return protected_image
            except Exception as e:
                error_msg = f"Failed to apply full-frame protection: {str(e)}"
                log_event(logger, logging.ERROR, LogEvent.PROCESSING_ERROR, error_msg, metadata={"error": str(e)})
                raise ProtectionError(error_msg) from e
                
        # 3. BLUR_REGION -> Protect specific regions
        if analysis.action == ProtectionAction.BLUR_REGION:
            if not analysis.protection_regions:
                # Edge case: action is BLUR_REGION but no regions defined. Safe fallback.
                return image
                
            protected_image = image.copy()
            img_h, img_w = protected_image.shape[:2]
            
            try:
                for region in analysis.protection_regions:
                    # Convert float bbox to integer coordinates clamped to image dimensions
                    x1 = max(0, int(region.bbox.x))
                    y1 = max(0, int(region.bbox.y))
                    x2 = min(img_w, int(region.bbox.x + region.bbox.width))
                    y2 = min(img_h, int(region.bbox.y + region.bbox.height))
                    
                    # Ensure valid slice
                    if x1 >= x2 or y1 >= y2:
                        continue
                        
                    roi = protected_image[y1:y2, x1:x2]
                    protected_roi = self._apply_protection(roi, region.mode)
                    protected_image[y1:y2, x1:x2] = protected_roi
                    
                log_event(
                    logger,
                    logging.INFO,
                    LogEvent.PROTECTION_APPLIED,
                    f"Region protection applied to {len(analysis.protection_regions)} regions",
                    metadata={
                        "action": "BLUR_REGION", 
                        "regions_count": len(analysis.protection_regions)
                    }
                )
                return protected_image
                
            except Exception as e:
                error_msg = f"Failed to apply region protection: {str(e)}"
                log_event(logger, logging.ERROR, LogEvent.PROCESSING_ERROR, error_msg, metadata={"error": str(e)})
                raise ProtectionError(error_msg) from e
                
        # Fallback for unknown actions
        return image

    def _apply_protection(self, roi: np.ndarray, mode: ProtectionMode) -> np.ndarray:
        """Route to the specific protection implementation based on mode."""
        if mode == ProtectionMode.GAUSSIAN:
            return self._apply_gaussian_blur(roi)
        elif mode == ProtectionMode.PIXELATE:
            return self._apply_pixelation(roi)
        elif mode == ProtectionMode.COVER:
            return self._apply_solid_cover(roi)
        else:
            # Fallback to Gaussian
            return self._apply_gaussian_blur(roi)

    def _apply_gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        """Apply a heavy Gaussian blur to the region."""
        h, w = roi.shape[:2]
        
        # Calculate kernel size based on ROI size (roughly 15% of the shortest side)
        # Ensure it's an odd number and at least 15
        k_size = int(min(h, w) * 0.15)
        if k_size % 2 == 0:
            k_size += 1
        k_size = max(15, k_size)
        
        # In extreme cases where ROI is very small, cap the kernel
        k_size = min(k_size, 99)
        
        return cv2.GaussianBlur(roi, (k_size, k_size), 0)

    def _apply_pixelation(self, roi: np.ndarray) -> np.ndarray:
        """Apply pixelation (downscale then upscale) to the region."""
        h, w = roi.shape[:2]
        
        # Avoid pixelating extremely small regions
        if h < 10 or w < 10:
            return self._apply_solid_cover(roi)
            
        # Target pixel size (e.g., 16x16 blocks)
        blocks = 16
        
        # Downscale
        small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
        # Upscale back to original size
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _apply_solid_cover(self, roi: np.ndarray) -> np.ndarray:
        """Apply a solid black cover to the region."""
        cover = np.zeros_like(roi)
        # Alternatively, a dark gray: cover.fill(50)
        return cover
