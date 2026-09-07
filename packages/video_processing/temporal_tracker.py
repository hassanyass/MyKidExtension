"""
Temporal tracking for the Video Engine.

Uses OpenCV trackers to track detected objects across frames without running full AI inference.
"""
import logging
from dataclasses import dataclass
import numpy as np
import cv2

from packages.shared.types import AnalysisResult, ProtectionRegion, BoundingBox, ProtectionAction, RiskLevel
from packages.shared.logger import get_logger


logger = get_logger("mykid.video_processing.tracker")


@dataclass
class TrackedObject:
    """State of a single tracked object."""
    tracker: cv2.Tracker
    region: ProtectionRegion
    frames_since_last_seen: int = 0
    is_lost: bool = False


class TemporalTracker:
    """
    Tracks harmful regions across video frames to reduce AI inference frequency.
    Implements a decay buffer to persist protections when tracking temporarily fails.
    """
    
    def __init__(self, max_persistence_frames: int = 10):
        """
        Initialize the temporal tracker.
        
        Args:
            max_persistence_frames: How many frames to keep protecting an object
                                    after the tracker loses it.
        """
        self.max_persistence_frames = max_persistence_frames
        self.active_trackers: list[TrackedObject] = []
        # We also need to store the overarching properties from the AnalysisResult
        # so we can reconstruct it on tracking frames.
        self.last_action = ProtectionAction.ALLOW
        self.last_risk = RiskLevel.LOW

    def initialize(self, frame: np.ndarray, analysis: AnalysisResult) -> None:
        """
        Reset trackers based on fresh AI inference.
        
        Args:
            frame: The current RGB frame (numpy array).
            analysis: The fresh AnalysisResult from the ImagePipeline.
        """
        self.active_trackers = []
        self.last_action = analysis.action
        self.last_risk = analysis.overall_risk
        
        # We only need to track if we are blurring specific regions.
        # If the action is ALLOW or BLUR_FRAME, we don't strictly need bounding box tracking,
        # but we persist the state.
        if analysis.action != ProtectionAction.BLUR_REGION:
            return
            
        for region in analysis.protection_regions:
            # OpenCV trackers expect (x, y, w, h)
            bbox = (
                int(region.bbox.x),
                int(region.bbox.y),
                int(region.bbox.width),
                int(region.bbox.height)
            )
            
            try:
                # MIL is lightweight and generally available
                tracker = cv2.TrackerMIL_create()
                # OpenCV trackers work best with BGR or Grayscale, but RGB works as well.
                # However, our frame is RGB, which is fine structurally for MIL.
                tracker.init(frame, bbox)
                
                self.active_trackers.append(
                    TrackedObject(
                        tracker=tracker,
                        region=region
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to initialize tracker for region {bbox}: {e}")

    def update(self, frame: np.ndarray) -> AnalysisResult:
        """
        Update trackers to find objects in the new frame.
        
        Args:
            frame: The current RGB frame.
            
        Returns:
            A reconstructed AnalysisResult containing updated tracking coordinates.
        """
        updated_regions = []
        surviving_trackers = []
        
        for tracked in self.active_trackers:
            if tracked.is_lost:
                # Keep decaying
                tracked.frames_since_last_seen += 1
                if tracked.frames_since_last_seen <= self.max_persistence_frames:
                    surviving_trackers.append(tracked)
                    updated_regions.append(tracked.region)
                continue
                
            # Attempt to update
            success, bbox = tracked.tracker.update(frame)
            
            if success:
                # Update region box
                tracked.region.bbox = BoundingBox(
                    x=bbox[0],
                    y=bbox[1],
                    width=bbox[2],
                    height=bbox[3]
                )
                tracked.frames_since_last_seen = 0
                surviving_trackers.append(tracked)
                updated_regions.append(tracked.region)
            else:
                # Tracking failed, start decaying
                tracked.is_lost = True
                tracked.frames_since_last_seen = 1
                if tracked.frames_since_last_seen <= self.max_persistence_frames:
                    surviving_trackers.append(tracked)
                    updated_regions.append(tracked.region)
                    
        self.active_trackers = surviving_trackers
        
        return AnalysisResult(
            overall_risk=self.last_risk,
            action=self.last_action,
            protection_regions=updated_regions
        )
