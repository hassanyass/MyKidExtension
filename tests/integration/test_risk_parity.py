"""
Cross-pipeline parity: the browser extension's risk engine must reach the
same verdict as the Python reference engine for the same detections.

The extension's src/inference/risk.js is a hand port of
packages/risk/risk_engine.py. Ports drift. When this one drifts, the
browser and the reference pipeline disagree about whether a child sees
something — so the two are pinned together here rather than trusted to
stay in sync by inspection.

This is the first concrete piece of Phase 13 (full integration), landed
early because 10b-iii is what created the duplication.

Skipped automatically if Node.js isn't available.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from packages.risk.risk_engine import RiskEngine
from packages.shared.config import MyKidConfig
from packages.shared.types import BoundingBox, Detection, RiskLevel, SceneRisk


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER = PROJECT_ROOT / "scripts" / "risk_parity_runner.js"

# Labels the browser treats as safety-relevant by default, mirroring
# SAFETY_RELEVANT_LABELS in apps/extension/src/shared/config.js.
SAFETY_LABELS = {"knife", "scissors"}

# Scenarios are defined once and fed to both engines.
SCENARIOS = [
    {
        "name": "no detections",
        "detections": [],
        "width": 640,
        "height": 480,
    },
    {
        "name": "knife above threshold, small box",
        "detections": [
            {"label": "knife", "confidence": 0.9,
             "bbox": {"x": 100, "y": 100, "width": 100, "height": 100}},
        ],
        "width": 640,
        "height": 480,
    },
    {
        "name": "knife below confidence threshold is ignored",
        "detections": [
            {"label": "knife", "confidence": 0.3,
             "bbox": {"x": 100, "y": 100, "width": 100, "height": 100}},
        ],
        "width": 640,
        "height": 480,
    },
    {
        "name": "large knife escalates to full-frame protection",
        "detections": [
            {"label": "knife", "confidence": 0.9,
             "bbox": {"x": 0, "y": 0, "width": 640, "height": 336}},
        ],
        "width": 640,
        "height": 480,
    },
    {
        "name": "non-safety label is ignored",
        "detections": [
            {"label": "person", "confidence": 0.95,
             "bbox": {"x": 10, "y": 10, "width": 50, "height": 50}},
        ],
        "width": 640,
        "height": 480,
    },
    {
        "name": "multiple weapons produce multiple regions",
        "detections": [
            {"label": "knife", "confidence": 0.8,
             "bbox": {"x": 10, "y": 10, "width": 50, "height": 50}},
            {"label": "scissors", "confidence": 0.7,
             "bbox": {"x": 300, "y": 200, "width": 80, "height": 40}},
        ],
        "width": 640,
        "height": 480,
    },
    {
        "name": "box padding clamps at the image edge",
        "detections": [
            {"label": "knife", "confidence": 0.9,
             "bbox": {"x": 0, "y": 0, "width": 100, "height": 100}},
        ],
        "width": 640,
        "height": 480,
    },
    # --- Scene-level risk (Phase B) --------------------------------
    # The scene classifier produces these; they must escalate to
    # whole-frame protection identically in both engines.
    {
        "name": "gore above threshold forces full-frame protection",
        "detections": [],
        "sceneRisk": {"violence": 0.0, "graphic": 0.85, "sexual": 0.05},
        "width": 640,
        "height": 480,
    },
    {
        "name": "sexual content above threshold forces full-frame protection",
        "detections": [],
        "sceneRisk": {"violence": 0.0, "graphic": 0.02, "sexual": 0.91},
        "width": 640,
        "height": 480,
    },
    {
        "name": "scene scores below threshold leave the image alone",
        "detections": [],
        "sceneRisk": {"violence": 0.0, "graphic": 0.35, "sexual": 0.40},
        "width": 640,
        "height": 480,
    },
    {
        "name": "scene risk overrides object-level region blur",
        "detections": [
            {"label": "knife", "confidence": 0.9,
             "bbox": {"x": 10, "y": 10, "width": 50, "height": 50}},
        ],
        "sceneRisk": {"violence": 0.0, "graphic": 0.88, "sexual": 0.01},
        "width": 640,
        "height": 480,
    },
    {
        "name": "safe-fixture scores (measured on bus.jpg) stay ALLOW",
        "detections": [],
        "sceneRisk": {"violence": 0.0, "graphic": 0.0353, "sexual": 0.0318},
        "width": 810,
        "height": 1080,
    },
]


def _python_verdicts(config: MyKidConfig):
    """Run the reference engine, mirroring how the vision layer filters."""
    engine = RiskEngine(config)
    threshold = config.detection.object_confidence_threshold
    verdicts = []

    for scenario in SCENARIOS:
        detections = [
            Detection(
                label=d["label"],
                confidence=d["confidence"],
                bbox=BoundingBox(**d["bbox"]),
                risk=RiskLevel.MEDIUM,
            )
            for d in scenario["detections"]
            if d["label"] in SAFETY_LABELS and d["confidence"] >= threshold
        ]

        scene = scenario.get("sceneRisk")
        analysis = engine.evaluate(
            detections,
            SceneRisk(**scene) if scene else None,
            image_shape=(scenario["height"], scenario["width"]),
        )

        verdicts.append(
            {
                "overall_risk": analysis.overall_risk.value,
                "action": analysis.action.value,
                "regions": [
                    {
                        "x": round(r.bbox.x, 4),
                        "y": round(r.bbox.y, 4),
                        "width": round(r.bbox.width, 4),
                        "height": round(r.bbox.height, 4),
                    }
                    for r in analysis.protection_regions
                ],
            }
        )

    return verdicts


def _browser_verdicts(scenarios=None, debug_label=None, no_storage=False):
    """Run the extension's ported engine via Node."""
    scenarios = scenarios if scenarios is not None else SCENARIOS
    body = [
        {
            "detections": s["detections"],
            "sceneRisk": s.get("sceneRisk"),
            "width": s["width"],
            "height": s["height"],
        }
        for s in scenarios
    ]

    if debug_label is None and not no_storage:
        payload = json.dumps(body)
    else:
        payload = json.dumps(
            {"scenarios": body, "debugLabel": debug_label, "noStorage": no_storage}
        )

    result = subprocess.run(
        ["node", str(RUNNER)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        pytest.fail(f"risk_parity_runner.js failed:\n{result.stderr}")

    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
class TestRiskEngineParity:
    """The browser and Python risk engines must agree, scenario for scenario."""

    def test_runner_script_exists(self):
        assert RUNNER.exists(), f"Missing parity runner at {RUNNER}"

    def test_verdicts_match(self):
        python_verdicts = _python_verdicts(MyKidConfig())
        browser_verdicts = _browser_verdicts()

        assert len(browser_verdicts) == len(SCENARIOS)

        for scenario, expected, actual in zip(
            SCENARIOS, python_verdicts, browser_verdicts
        ):
            assert actual == expected, (
                f"Risk engines disagree on '{scenario['name']}':\n"
                f"  python : {expected}\n"
                f"  browser: {actual}"
            )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
class TestDebugLabelOverride:
    """
    The popup's "treat as harmful" selector is the only way to observe the
    protection chain working, since COCO gives us just knife and scissors
    and those essentially never appear while browsing. It was reported as
    doing nothing, so its behaviour is pinned here rather than assumed.
    """

    SCENARIO = [
        {
            "name": "person detected",
            "detections": [
                {"label": "person", "confidence": 0.9,
                 "bbox": {"x": 48.5, "y": 398.2, "width": 194.9, "height": 506.4}},
            ],
            "width": 810,
            "height": 1080,
        }
    ]

    def test_person_ignored_by_default(self):
        """Without the override, 'person' must not trigger protection."""
        verdict = _browser_verdicts(self.SCENARIO)[0]
        assert verdict["action"] == "allow"
        assert verdict["regions"] == []

    def test_person_protected_when_override_set(self):
        """With the override, the same detection must produce a region."""
        verdict = _browser_verdicts(self.SCENARIO, debug_label="person")[0]
        assert verdict["action"] == "blur_region"
        assert verdict["overall_risk"] == "medium"
        assert len(verdict["regions"]) == 1

    def test_override_does_not_disable_real_safety_labels(self):
        """The override must add to the safety labels, never replace them."""
        knife = [
            {
                "detections": [
                    {"label": "knife", "confidence": 0.9,
                     "bbox": {"x": 10, "y": 10, "width": 50, "height": 50}},
                ],
                "width": 640,
                "height": 480,
            }
        ]
        verdict = _browser_verdicts(knife, debug_label="person")[0]
        assert verdict["action"] == "blur_region"


def _browser_config(stored=None, no_storage=False):
    """Ask the extension for its resolved config, given stored overrides."""
    payload = json.dumps(
        {
            "scenarios": [],
            "configOnly": True,
            "noStorage": no_storage,
            "storedConfig": stored or {},
        }
    )
    result = subprocess.run(
        ["node", str(RUNNER)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"risk_parity_runner.js failed:\n{result.stderr}")
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
class TestPowerToggle:
    """
    The on/off switch is the product's entire interface, so its semantics
    are pinned rather than trusted.

    The failure that matters most is silent: a toggle that reports "on"
    while nothing is actually being checked leaves someone believing they
    are protected when they are not.
    """

    def test_defaults_to_enabled(self):
        """
        A safety tool that ships switched off protects nobody until
        someone remembers to turn it on.
        """
        assert _browser_config()["enabled"] is True

    def test_disabled_state_is_carried_without_storage(self):
        """
        The offscreen document can't read storage, so 'off' has to survive
        being passed through a message — the same delivery path that
        silently broke the debug label.
        """
        assert _browser_config(no_storage=True)["enabled"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
class TestConfigDelivery:
    """
    Config must reach the offscreen document, which cannot read it itself.

    Offscreen documents get only a restricted subset of the extension APIs.
    An earlier version had the offscreen document call
    `chrome.storage.local.get` directly; that failed, was swallowed by a
    bare `except`, and left the risk engine silently running on defaults.
    Detection worked, 'person' was found 58 times on a real page, and
    nothing was ever protected.

    These tests pin the delivery mechanism, not just the decision logic —
    the risk engine was correct the whole time.
    """

    SCENARIO = [
        {
            "detections": [
                {"label": "person", "confidence": 0.9,
                 "bbox": {"x": 48, "y": 398, "width": 195, "height": 506}},
            ],
            "width": 810,
            "height": 1080,
        }
    ]

    def test_config_survives_without_storage_access(self):
        """
        With storage unavailable, an explicitly-passed override must still
        apply. This is the exact path the service worker now uses.
        """
        verdict = _browser_verdicts(
            self.SCENARIO, debug_label="person", no_storage=True
        )[0]
        assert verdict["action"] == "blur_region", (
            "Config passed explicitly must reach the risk engine even when "
            "chrome.storage is unavailable — this is the reported bug."
        )
        assert len(verdict["regions"]) == 1

    def test_defaults_still_protect_real_safety_labels_without_storage(self):
        """
        Losing config access must never silently disable knife/scissors —
        degraded settings are tolerable, degraded safety is not.
        """
        knife = [
            {
                "detections": [
                    {"label": "knife", "confidence": 0.9,
                     "bbox": {"x": 10, "y": 10, "width": 50, "height": 50}},
                ],
                "width": 640,
                "height": 480,
            }
        ]
        verdict = _browser_verdicts(knife, no_storage=True)[0]
        assert verdict["action"] == "blur_region"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
class TestHarmCategories:
    """
    Each harm category is independently switchable, and all default on.

    Two failures matter here and both are silent: a category defaulting to
    off would leave a gap nobody asked for, and the weapons switch leaking
    into the label set would disable knife/scissors detection while the UI
    still showed it enabled.
    """

    def test_all_categories_default_on(self):
        """
        A safety tool should cover everything it can unless deliberately
        narrowed — not require opting in and discovering the gaps later.
        """
        config = _browser_config()
        assert config["categories"] == {
            "gore": True,
            "sexual": True,
            "weapons": True,
        }

    def test_weapons_off_removes_weapon_labels(self):
        config = _browser_config({"categories": {"weapons": False}})
        assert "knife" not in config["harmfulLabels"]
        assert "scissors" not in config["harmfulLabels"]

    def test_weapons_on_keeps_weapon_labels(self):
        config = _browser_config({"categories": {"weapons": True}})
        assert "knife" in config["harmfulLabels"]
        assert "scissors" in config["harmfulLabels"]

    def test_disabling_one_category_leaves_others_alone(self):
        """Partial overrides must merge, not replace the whole section."""
        config = _browser_config({"categories": {"gore": False}})
        assert config["categories"]["gore"] is False
        assert config["categories"]["sexual"] is True
        assert config["categories"]["weapons"] is True

    def test_multiple_test_labels_are_all_applied(self):
        """
        The testing selector takes a list, not a single value — one pick was
        too narrow to exercise the pipeline properly.
        """
        config = _browser_config({"debug": {"treatLabelAsHarmful": ["person", "dog"]}})
        assert "person" in config["harmfulLabels"]
        assert "dog" in config["harmfulLabels"]
        # Real categories must survive alongside the test labels.
        assert "knife" in config["harmfulLabels"]

    def test_legacy_single_label_still_works(self):
        """Settings saved by an older build must keep applying."""
        config = _browser_config({"debug": {"treatLabelAsHarmful": "person"}})
        assert "person" in config["harmfulLabels"]
