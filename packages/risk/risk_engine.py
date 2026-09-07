"""
MyKid Image Risk Engine.

Evaluates raw detections and scene context to determine the overall risk
and appropriate protection action.
"""
from typing import List, Optional, Tuple
import logging

from packages.shared.types import (
    Detection, 
    SceneRisk, 
    RiskLevel, 
    ProtectionAction, 
    ProtectionRegion, 
    ProtectionMode,
    AnalysisResult
)
from packages.shared.config import MyKidConfig
from packages.shared.logger import get_logger, log_event, LogEvent


logger = get_logger("mykid.risk_engine")


class RiskEngine:
    """
    Evaluates visual detections against safety rules to produce
    a final AnalysisResult with ProtectionRegions.
    """
    
    # Simple hierarchy for risk maxing
    _RISK_WEIGHTS = {
        RiskLevel.LOW: 0,
        RiskLevel.MEDIUM: 1,
        RiskLevel.HIGH: 2
    }
    
    def __init__(self, config: MyKidConfig):
        """
        Initialize the Risk Engine.
        
        Args:
            config: Global configuration (thresholds, modes).
        """
        self.config = config
        
    def evaluate(
        self, 
        detections: List[Detection], 
        scene_risk: Optional[SceneRisk] = None, 
        image_shape: Optional[Tuple[int, int]] = None
    ) -> AnalysisResult:
        """
        Evaluate risk and formulate an action plan.
        
        Args:
            detections: List of safety-relevant objects detected.
            scene_risk: Optional scene-level risk scores.
            image_shape: (height, width) of the image. Needed to calculate coverage ratio.
            
        Returns:
            AnalysisResult containing overall risk and protection regions.
        """
        scene_risk = scene_risk or SceneRisk()
        
        # 1. Determine overall risk
        overall_risk = self._calculate_overall_risk(detections, scene_risk)
        
        # 2. Determine base action
        action = ProtectionAction.ALLOW
        if overall_risk in (RiskLevel.MEDIUM, RiskLevel.HIGH):
            action = ProtectionAction.BLUR_REGION
            
        # 3. Check scene threshold (escalate to frame blur)
        if scene_risk.max_score() >= self.config.detection.scene_risk_threshold:
            action = ProtectionAction.BLUR_FRAME
            
        # 4. Check spatial coverage (escalate to frame blur)
        if action == ProtectionAction.BLUR_REGION and image_shape is not None:
            if self._exceeds_coverage_threshold(detections, image_shape):
                action = ProtectionAction.BLUR_FRAME
                
        # 5. Generate protection regions
        protection_regions = []
        if action == ProtectionAction.BLUR_REGION:
            protection_mode = ProtectionMode(self.config.protection.mode)
            padding = self.config.protection.blur_padding
            img_h, img_w = image_shape if image_shape else (9999, 9999) # fallback if not provided
            
            for det in detections:
                if det.risk in (RiskLevel.MEDIUM, RiskLevel.HIGH) and det.bbox is not None:
                    padded_bbox = det.bbox.padded(padding, image_width=img_w, image_height=img_h)
                    
                    region = ProtectionRegion(
                        bbox=padded_bbox,
                        mode=protection_mode,
                        source_label=det.label
                    )
                    protection_regions.append(region)
                    
        result = AnalysisResult(
            detections=detections,
            scene_risk=scene_risk,
            overall_risk=overall_risk,
            action=action,
            protection_regions=protection_regions
        )
        
        if overall_risk != RiskLevel.LOW:
            log_event(
                logger, 
                logging.INFO,
                LogEvent.RISK_DETECTED, 
                f"Risk evaluated as {overall_risk.value.upper()}",
                metadata={
                    "overall_risk": overall_risk.value,
                    "action": action.value,
                    "regions_count": len(protection_regions)
                }
            )
            
        return result

    def _calculate_overall_risk(self, detections: List[Detection], scene_risk: SceneRisk) -> RiskLevel:
        """Find the highest risk across all objects and scene context."""
        max_weight = self._RISK_WEIGHTS[RiskLevel.LOW]
        
        # Check objects
        for det in detections:
            weight = self._RISK_WEIGHTS[det.risk]
            if weight > max_weight:
                max_weight = weight
                
        # Check scene against threshold
        if scene_risk.max_score() >= self.config.detection.scene_risk_threshold:
            max_weight = max(max_weight, self._RISK_WEIGHTS[RiskLevel.HIGH])
            
        # Reverse map weight to enum
        for level, weight in self._RISK_WEIGHTS.items():
            if weight == max_weight:
                return level
                
        return RiskLevel.LOW

    def _exceeds_coverage_threshold(self, detections: List[Detection], image_shape: Tuple[int, int]) -> bool:
        """
        Check if the combined area of harmful objects exceeds the maximum allowed ratio.
        """
        if not detections:
            return False
            
        img_h, img_w = image_shape
        image_area = img_h * img_w
        if image_area == 0:
            return False
            
        total_harmful_area = 0.0
        for det in detections:
            if det.risk in (RiskLevel.MEDIUM, RiskLevel.HIGH) and det.bbox is not None:
                total_harmful_area += det.bbox.area()
                
        ratio = total_harmful_area / image_area
        return ratio > self.config.detection.max_allowed_object_area_ratio
