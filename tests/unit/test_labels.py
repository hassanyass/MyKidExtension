"""
Tests for MyKid safety label mapping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.labels import (
    SafetyCategory,
    SafetyCategoryConfig,
    COCO_LABELS,
    COCO_LABEL_TO_INDEX,
    SAFETY_CATEGORIES,
    LABEL_TO_SAFETY_CATEGORY,
    SAFETY_RELEVANT_LABELS,
    is_safety_relevant,
    get_safety_category,
    get_base_risk,
    get_all_safety_labels,
    get_coco_label,
)
from packages.shared.types import RiskLevel


class TestCOCOLabels:
    """Test COCO label lookup."""

    def test_coco_has_80_classes(self):
        assert len(COCO_LABELS) == 80

    def test_knife_is_class_43(self):
        assert COCO_LABELS[43] == "knife"

    def test_scissors_is_class_76(self):
        assert COCO_LABELS[76] == "scissors"

    def test_reverse_lookup(self):
        assert COCO_LABEL_TO_INDEX["knife"] == 43
        assert COCO_LABEL_TO_INDEX["scissors"] == 76

    def test_get_coco_label_valid(self):
        assert get_coco_label(0) == "person"
        assert get_coco_label(43) == "knife"
        assert get_coco_label(79) == "toothbrush"

    def test_get_coco_label_invalid(self):
        assert get_coco_label(80) == "unknown"
        assert get_coco_label(-1) == "unknown"


class TestSafetyCategories:
    """Test safety category definitions."""

    def test_weapons_category_exists(self):
        assert SafetyCategory.WEAPONS in SAFETY_CATEGORIES

    def test_weapons_includes_knife(self):
        config = SAFETY_CATEGORIES[SafetyCategory.WEAPONS]
        assert "knife" in config.labels

    def test_weapons_includes_scissors(self):
        config = SAFETY_CATEGORIES[SafetyCategory.WEAPONS]
        assert "scissors" in config.labels

    def test_weapons_base_risk_is_medium(self):
        config = SAFETY_CATEGORIES[SafetyCategory.WEAPONS]
        assert config.base_risk == RiskLevel.MEDIUM

    def test_category_config_contains(self):
        config = SAFETY_CATEGORIES[SafetyCategory.WEAPONS]
        assert config.contains("knife")
        assert config.contains("Knife")  # case insensitive
        assert not config.contains("car")


class TestIsSafetyRelevant:
    """Test safety relevance checks."""

    def test_knife_is_relevant(self):
        assert is_safety_relevant("knife") is True

    def test_scissors_is_relevant(self):
        assert is_safety_relevant("scissors") is True

    def test_case_insensitive(self):
        assert is_safety_relevant("Knife") is True
        assert is_safety_relevant("SCISSORS") is True

    def test_person_not_relevant(self):
        assert is_safety_relevant("person") is False

    def test_car_not_relevant(self):
        assert is_safety_relevant("car") is False

    def test_empty_not_relevant(self):
        assert is_safety_relevant("") is False

    def test_unknown_not_relevant(self):
        assert is_safety_relevant("spaceship") is False


class TestGetSafetyCategory:
    """Test category lookup."""

    def test_knife_is_weapons(self):
        assert get_safety_category("knife") == SafetyCategory.WEAPONS

    def test_scissors_is_weapons(self):
        assert get_safety_category("scissors") == SafetyCategory.WEAPONS

    def test_person_has_no_category(self):
        assert get_safety_category("person") is None

    def test_case_insensitive(self):
        assert get_safety_category("Knife") == SafetyCategory.WEAPONS


class TestGetBaseRisk:
    """Test base risk lookup."""

    def test_knife_base_risk(self):
        assert get_base_risk("knife") == RiskLevel.MEDIUM

    def test_scissors_base_risk(self):
        assert get_base_risk("scissors") == RiskLevel.MEDIUM

    def test_safe_label_is_low(self):
        assert get_base_risk("person") == RiskLevel.LOW
        assert get_base_risk("car") == RiskLevel.LOW

    def test_unknown_label_is_low(self):
        assert get_base_risk("unknown_object") == RiskLevel.LOW


class TestGetAllSafetyLabels:
    """Test safety label listing."""

    def test_returns_list(self):
        labels = get_all_safety_labels()
        assert isinstance(labels, list)

    def test_includes_knife_and_scissors(self):
        labels = get_all_safety_labels()
        assert "knife" in labels
        assert "scissors" in labels

    def test_is_sorted(self):
        labels = get_all_safety_labels()
        assert labels == sorted(labels)

    def test_does_not_include_safe_labels(self):
        labels = get_all_safety_labels()
        assert "person" not in labels
        assert "car" not in labels


class TestLabelMappingConsistency:
    """Ensure mappings are consistent with each other."""

    def test_all_safety_labels_in_coco(self):
        """All safety-relevant labels should exist in COCO."""
        for label in SAFETY_RELEVANT_LABELS:
            assert label in COCO_LABEL_TO_INDEX, f"{label} not in COCO"

    def test_label_to_category_matches_categories(self):
        """The flat mapping should match the category definitions."""
        for category, config in SAFETY_CATEGORIES.items():
            for label in config.labels:
                assert LABEL_TO_SAFETY_CATEGORY[label.lower()] == category
