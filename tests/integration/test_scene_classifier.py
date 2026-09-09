"""
Scene classifier integration tests (Phase B).

What these can and cannot establish is worth stating plainly, because the
difference matters for a child-safety tool:

  - They CAN show the model loads, runs, produces well-formed scores, and
    does not fire on ordinary images. False positives are a real risk: an
    extension that blurs everything gets switched off, protecting nobody.

  - They CANNOT show it detects gore, violence or sexual content. That
    needs a harmful evaluation set, which this project does not have (see
    docs/11, Open Decisions — sourcing it is the project owner's call).

So: no test here asserts detection accuracy, because none can. Claiming
otherwise would be exactly the overclaiming SKILL.md rule 10 forbids.
"""
from pathlib import Path

import numpy as np
import pytest

from packages.shared.config import MyKidConfig
from packages.shared.errors import ModelError
from packages.shared.types import SceneRisk
from packages.vision.scene_classifier import SCENE_LABELS, SceneClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = (
    PROJECT_ROOT / "models" / "candidates" / "image-safety-classifier-xs.onnx"
)

SAFE_FIXTURES = [
    PROJECT_ROOT / "test-data" / "images" / "detection-samples" / "bus.jpg",
    PROJECT_ROOT / "test-data" / "images" / "detection-samples" / "zidane.jpg",
    PROJECT_ROOT / "test-data" / "images" / "safe" / "test_safe.jpg",
]

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Scene classifier not downloaded — run scripts/benchmark_safety_models.py",
)


@pytest.fixture(scope="module")
def classifier():
    instance = SceneClassifier(MyKidConfig())
    instance.load()
    return instance


class TestSceneClassifierContract:
    """Behaviour that must hold regardless of whether weights are present."""

    def test_label_order_is_pinned(self):
        """
        Order comes from the model's own config, and the mapping to
        SceneRisk depends on it. If it silently changed, gore scores would
        land in the sexual field and vice versa — wrong protection for the
        wrong reason, with nothing obviously broken.
        """
        assert SCENE_LABELS == ["NSFL", "NSFW", "SFW"]

    def test_predict_before_load_raises(self):
        instance = SceneClassifier(MyKidConfig())
        with pytest.raises(ModelError, match="not loaded"):
            instance.predict(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_missing_weights_give_an_actionable_error(self):
        instance = SceneClassifier(MyKidConfig(), model_path="does/not/exist.onnx")
        with pytest.raises(ModelError, match="benchmark_safety_models"):
            instance.load()


@requires_model
class TestSceneClassifierOnSafeContent:
    def test_scores_are_well_formed(self, classifier):
        import cv2

        image = cv2.cvtColor(cv2.imread(str(SAFE_FIXTURES[0])), cv2.COLOR_BGR2RGB)
        risk = classifier.predict(image)

        assert isinstance(risk, SceneRisk)
        assert 0.0 <= risk.graphic <= 1.0
        assert 0.0 <= risk.sexual <= 1.0

    def test_violence_is_left_unset(self, classifier):
        """
        This model folds violence into NSFL rather than scoring it
        separately. Populating `violence` would put a number there we never
        measured; the field stays 0 until a dedicated model fills it.
        """
        import cv2

        image = cv2.cvtColor(cv2.imread(str(SAFE_FIXTURES[0])), cv2.COLOR_BGR2RGB)
        assert classifier.predict(image).violence == 0.0

    @pytest.mark.parametrize("fixture", SAFE_FIXTURES, ids=lambda p: p.name)
    def test_safe_images_stay_below_the_protection_threshold(
        self, classifier, fixture
    ):
        """
        Ordinary images must not trip protection. This is the false-positive
        guard: it says nothing about whether harmful content is caught.
        """
        import cv2

        if not fixture.exists():
            pytest.skip(f"fixture missing: {fixture.name}")

        image = cv2.cvtColor(cv2.imread(str(fixture)), cv2.COLOR_BGR2RGB)
        risk = classifier.predict(image)
        threshold = MyKidConfig().detection.scene_risk_threshold

        assert risk.max_score() < threshold, (
            f"{fixture.name} scored {risk.max_score():.3f} against a threshold of "
            f"{threshold} — a safe image would be blurred."
        )

    def test_rejects_invalid_input(self, classifier):
        from packages.shared.errors import InferenceError

        with pytest.raises(InferenceError):
            classifier.predict(np.array([]))
