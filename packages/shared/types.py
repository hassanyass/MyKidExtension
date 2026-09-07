"""
MyKid Core Types

All shared data models for the MyKid visual safety pipeline.
Uses standard library dataclasses and enums — no external dependencies.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """Risk level assigned after detection + context analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProtectionAction(Enum):
    """Action decided by the policy engine."""
    ALLOW = "allow"
    BLUR_REGION = "blur_region"
    BLUR_FRAME = "blur_frame"
    BLOCK = "block"


class ProtectionMode(Enum):
    """Visual protection method applied by the protection engine."""
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    COVER = "cover"


class ProcessingState(Enum):
    """Tracks the processing lifecycle of a content item."""
    UNPROCESSED = "unprocessed"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box for a detected region."""
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return max(0.0, self.width) * max(0.0, self.height)

    def padded(self, padding_ratio: float, image_width: float, image_height: float) -> "BoundingBox":
        """
        Return a new BoundingBox expanded by the given padding ratio,
        clamped to the image boundaries.

        Args:
            padding_ratio: Fraction of bbox dimensions to add (e.g. 0.1 = 10%).
            image_width: Width of the containing image.
            image_height: Height of the containing image.

        Returns:
            A new BoundingBox with padding applied.
        """
        pad_w = self.width * padding_ratio
        pad_h = self.height * padding_ratio

        new_x = max(0.0, self.x - pad_w)
        new_y = max(0.0, self.y - pad_h)
        new_w = min(image_width - new_x, self.width + 2 * pad_w)
        new_h = min(image_height - new_y, self.height + 2 * pad_h)

        return BoundingBox(x=new_x, y=new_y, width=new_w, height=new_h)


@dataclass
class Detection:
    """A single detection from the AI vision engine."""
    label: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    risk: RiskLevel = RiskLevel.LOW

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "confidence": self.confidence,
            "risk": self.risk.value,
        }
        if self.bbox is not None:
            result["bbox"] = self.bbox.to_dict()
        return result


@dataclass
class SceneRisk:
    """Scene-level risk scores (not object-specific)."""
    violence: float = 0.0
    graphic: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def max_score(self) -> float:
        """Return the highest scene risk score."""
        return max(self.violence, self.graphic)


@dataclass
class ProtectionRegion:
    """A region to be visually protected, with its protection parameters."""
    bbox: BoundingBox
    mode: ProtectionMode = ProtectionMode.GAUSSIAN
    source_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "mode": self.mode.value,
            "source_label": self.source_label,
        }


@dataclass
class AnalysisResult:
    """Complete result from the visual safety pipeline for a single image/frame."""
    content_type: str = "image"
    detections: List[Detection] = field(default_factory=list)
    scene_risk: SceneRisk = field(default_factory=SceneRisk)
    overall_risk: RiskLevel = RiskLevel.LOW
    action: ProtectionAction = ProtectionAction.ALLOW
    protection_regions: List[ProtectionRegion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type,
            "detections": [d.to_dict() for d in self.detections],
            "scene_risk": self.scene_risk.to_dict(),
            "overall_risk": self.overall_risk.value,
            "action": self.action.value,
            "protection_regions": [r.to_dict() for r in self.protection_regions],
        }


@dataclass
class VideoFrameResult:
    """Analysis result for a single video frame."""
    frame_index: int
    timestamp_ms: float
    detections: List[Detection] = field(default_factory=list)
    scene_risk: SceneRisk = field(default_factory=SceneRisk)
    overall_risk: RiskLevel = RiskLevel.LOW
    action: ProtectionAction = ProtectionAction.ALLOW
    protection_regions: List[ProtectionRegion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "detections": [d.to_dict() for d in self.detections],
            "scene_risk": self.scene_risk.to_dict(),
            "overall_risk": self.overall_risk.value,
            "action": self.action.value,
            "protection_regions": [r.to_dict() for r in self.protection_regions],
        }
