"""
MyKid Safety Label Mapping

Maps AI model output labels to MyKid safety categories.
This is the bridge between raw model detections and the risk engine.

When models are swapped or upgraded, only this mapping needs to change.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from packages.shared.types import RiskLevel


class SafetyCategory(Enum):
    """Categories of potentially harmful visual content."""
    WEAPONS = "weapons"
    VIOLENCE = "violence"
    GRAPHIC = "graphic"


@dataclass
class SafetyCategoryConfig:
    """Configuration for a single safety category."""
    labels: List[str]
    base_risk: RiskLevel
    description: str

    def contains(self, label: str) -> bool:
        """Check if a label belongs to this category."""
        return label.lower() in {l.lower() for l in self.labels}


# ---------------------------------------------------------------------------
# COCO-80 labels relevant to safety
# ---------------------------------------------------------------------------

# Full list of COCO-80 class names (index → label)
COCO_LABELS: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

# Reverse lookup: label → COCO index
COCO_LABEL_TO_INDEX: Dict[str, int] = {v: k for k, v in COCO_LABELS.items()}


# ---------------------------------------------------------------------------
# Safety category definitions
# ---------------------------------------------------------------------------

SAFETY_CATEGORIES: Dict[SafetyCategory, SafetyCategoryConfig] = {
    SafetyCategory.WEAPONS: SafetyCategoryConfig(
        labels=["knife", "scissors"],
        base_risk=RiskLevel.MEDIUM,
        description="Objects that could be used as weapons. "
                    "Risk depends on visual context (e.g., kitchen vs violence).",
    ),
    # Future categories (not yet supported by COCO model):
    # SafetyCategory.VIOLENCE: SafetyCategoryConfig(
    #     labels=[],  # Requires scene classifier
    #     base_risk=RiskLevel.HIGH,
    #     description="Scenes depicting physical violence or aggression.",
    # ),
    # SafetyCategory.GRAPHIC: SafetyCategoryConfig(
    #     labels=[],  # Requires scene classifier
    #     base_risk=RiskLevel.HIGH,
    #     description="Graphic content including blood, injury, disturbing imagery.",
    # ),
}


# ---------------------------------------------------------------------------
# Label → Safety Category mapping
# ---------------------------------------------------------------------------

def _build_label_to_category_map() -> Dict[str, SafetyCategory]:
    """Build a flat lookup from model label → safety category."""
    mapping: Dict[str, SafetyCategory] = {}
    for category, config in SAFETY_CATEGORIES.items():
        for label in config.labels:
            mapping[label.lower()] = category
    return mapping


# Pre-built lookup for fast access
LABEL_TO_SAFETY_CATEGORY: Dict[str, SafetyCategory] = _build_label_to_category_map()

# Set of all labels considered safety-relevant
SAFETY_RELEVANT_LABELS: Set[str] = set(LABEL_TO_SAFETY_CATEGORY.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_safety_relevant(label: str) -> bool:
    """
    Check if a detection label is relevant to visual safety.

    Args:
        label: The model detection label (e.g., "knife", "person", "car").

    Returns:
        True if the label maps to a safety category.
    """
    return label.lower() in SAFETY_RELEVANT_LABELS


def get_safety_category(label: str) -> Optional[SafetyCategory]:
    """
    Get the safety category for a detection label.

    Args:
        label: The model detection label.

    Returns:
        The SafetyCategory if safety-relevant, else None.
    """
    return LABEL_TO_SAFETY_CATEGORY.get(label.lower())


def get_base_risk(label: str) -> RiskLevel:
    """
    Get the base risk level for a detection label.

    The base risk is the starting point before context/confidence adjustment.

    Args:
        label: The model detection label.

    Returns:
        The base RiskLevel for the label, or LOW if not safety-relevant.
    """
    category = get_safety_category(label)
    if category is None:
        return RiskLevel.LOW
    return SAFETY_CATEGORIES[category].base_risk


def get_all_safety_labels() -> List[str]:
    """Return all labels that are considered safety-relevant."""
    return sorted(SAFETY_RELEVANT_LABELS)


def get_coco_label(index: int) -> str:
    """
    Get the COCO label name for a class index.

    Args:
        index: COCO class index (0-79).

    Returns:
        The label string, or "unknown" if the index is invalid.
    """
    return COCO_LABELS.get(index, "unknown")
