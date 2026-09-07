"""
Integration tests for the Temporal Tracker.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.types import AnalysisResult, ProtectionRegion, BoundingBox, ProtectionAction, RiskLevel
from packages.video_processing.temporal_tracker import TemporalTracker


class TestTemporalTracker:
    
    def test_tracker_initialization(self):
        tracker = TemporalTracker(max_persistence_frames=5)
        
        # Create a mock frame
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        
        # Create a mock analysis result with a region to track
        analysis = AnalysisResult(
            overall_risk=RiskLevel.MEDIUM,
            action=ProtectionAction.BLUR_REGION,
            protection_regions=[
                ProtectionRegion(
                    source_label="knife",
                    bbox=BoundingBox(x=10, y=10, width=20, height=20)
                )
            ]
        )
        
        # Initialize
        tracker.initialize(frame, analysis)
        
        # Should have 1 active tracker
        assert len(tracker.active_trackers) == 1
        assert tracker.last_action == ProtectionAction.BLUR_REGION
        assert tracker.last_risk == RiskLevel.MEDIUM

    def test_tracker_update_and_decay(self):
        tracker = TemporalTracker(max_persistence_frames=2)
        
        # 1. Initialize
        frame1 = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Draw a clear black box for MIL to track easily
        frame1[10:30, 10:30] = 0
        
        analysis = AnalysisResult(
            overall_risk=RiskLevel.MEDIUM,
            action=ProtectionAction.BLUR_REGION,
            protection_regions=[
                ProtectionRegion(
                    source_label="knife",
                    bbox=BoundingBox(x=10, y=10, width=20, height=20)
                )
            ]
        )
        tracker.initialize(frame1, analysis)
        
        # 2. Update with the same frame (should track perfectly)
        new_analysis = tracker.update(frame1)
        assert len(new_analysis.protection_regions) == 1
        assert new_analysis.overall_risk == RiskLevel.MEDIUM
        assert new_analysis.action == ProtectionAction.BLUR_REGION
        
        # 3. Force tracker to fail to test decay logic
        for t in tracker.active_trackers:
            t.is_lost = True
            t.frames_since_last_seen = 0
            
        frame_blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        
        decay_analysis_1 = tracker.update(frame_blank)
        # Should still persist (persistence = 1)
        assert len(decay_analysis_1.protection_regions) == 1
        assert tracker.active_trackers[0].is_lost == True
        assert tracker.active_trackers[0].frames_since_last_seen == 1
        
        decay_analysis_2 = tracker.update(frame_blank)
        # Should still persist (persistence = 2 == max)
        assert len(decay_analysis_2.protection_regions) == 1
        assert tracker.active_trackers[0].frames_since_last_seen == 2
        
        decay_analysis_3 = tracker.update(frame_blank)
        # Should now vanish (persistence = 3 > 2)
        assert len(decay_analysis_3.protection_regions) == 0
        assert len(tracker.active_trackers) == 0
