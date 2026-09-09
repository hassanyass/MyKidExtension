"""
Cross-pipeline parity for the scene classifier's preprocessing.

The Python and browser implementations feed the same model, so they must
build the same tensor from the same pixels. This is a quiet failure mode:
swap the channel order, use NHWC instead of NCHW, or divide by 255 on one
side only, and nothing crashes — the model still runs and still returns
three plausible-looking probabilities. It is simply wrong about whether a
child sees something.

Phase G. Skipped automatically if Node.js or the model is unavailable.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from packages.shared.config import MyKidConfig
from packages.vision.scene_classifier import (
    MODEL_INPUT_SIZE,
    SCENE_LABELS,
    SceneClassifier,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER = PROJECT_ROOT / "scripts" / "scene_parity_runner.js"
MODEL_PATH = PROJECT_ROOT / "models" / "candidates" / "image-safety-classifier-xs.onnx"

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js not available"
)


def _synthetic_rgba(size=MODEL_INPUT_SIZE):
    """
    A deterministic image whose channels are clearly distinguishable.

    Channel-order bugs are the point of this test, so the three channels
    must never coincidentally match — a flat grey image would pass even
    with R and B swapped.
    """
    rng = np.random.default_rng(seed=20260909)
    rgb = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(rgb[..., 0] // 2, 0, 255)        # dim red
    rgb[..., 2] = np.clip(128 + rgb[..., 2] // 2, 0, 255)  # bright blue
    alpha = np.full((size, size, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2), rgb


def _python_tensor(rgb):
    """Mirror SceneClassifier._preprocess: NCHW float32, raw 0-255."""
    return np.expand_dims(np.transpose(rgb.astype(np.float32), (2, 0, 1)), 0)


def _browser_tensor_summary(rgba):
    result = subprocess.run(
        ["node", str(RUNNER)],
        input=json.dumps({"rgba": rgba.reshape(-1).tolist()}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(f"scene_parity_runner.js failed:\n{result.stderr}")
    return json.loads(result.stdout)


@requires_node
class TestScenePreprocessingParity:
    def test_runner_exists(self):
        assert RUNNER.exists(), f"Missing runner at {RUNNER}"

    def test_tensor_shape_and_scale_match(self):
        rgba, rgb = _synthetic_rgba()
        browser = _browser_tensor_summary(rgba)
        python = _python_tensor(rgb)

        assert browser["inputSize"] == MODEL_INPUT_SIZE
        assert browser["length"] == python.size

        # Raw 0-255, not normalised. If either side divided by 255 this
        # would be <= 1.0 while the other stayed in the hundreds.
        assert browser["max"] > 1.0
        assert float(python.max()) > 1.0

    def test_channel_layout_matches(self):
        """
        Per-channel means catch both channel-order swaps and NHWC/NCHW
        mix-ups, which is why the fixture makes the channels distinct.
        """
        rgba, rgb = _synthetic_rgba()
        browser = _browser_tensor_summary(rgba)
        python = _python_tensor(rgb)[0]

        python_means = [float(python[c].mean()) for c in range(3)]

        for channel, (expected, actual) in enumerate(
            zip(python_means, browser["channelMeans"])
        ):
            assert abs(expected - actual) < 1e-3, (
                f"Channel {channel} differs: python={expected:.6f} "
                f"browser={actual:.6f} — channel order or tensor layout has "
                "diverged between the two pipelines."
            )

    def test_leading_values_match_exactly(self):
        rgba, rgb = _synthetic_rgba()
        browser = _browser_tensor_summary(rgba)
        python = _python_tensor(rgb).reshape(-1)[:12]

        assert browser["first12"] == pytest.approx(python.tolist(), abs=1e-6)

    def test_label_order_matches(self):
        """
        Both sides map NSFL->graphic and NSFW->sexual by index. If the
        orders disagreed, gore scores would land in the sexual field.
        """
        rgba, _ = _synthetic_rgba()
        assert _browser_tensor_summary(rgba)["labels"] == SCENE_LABELS


@requires_node
@pytest.mark.skipif(not MODEL_PATH.exists(), reason="scene classifier not downloaded")
class TestSceneScoreParity:
    def test_identical_tensors_give_identical_scores(self):
        """
        End-to-end confirmation: the browser's tensor, fed to the same
        model from Python, reproduces Python's own scores. Anything else
        would mean the two pipelines disagree on the same picture.
        """
        import onnxruntime as ort

        rgba, rgb = _synthetic_rgba()
        browser = _browser_tensor_summary(rgba)

        classifier = SceneClassifier(MyKidConfig())
        classifier.load()
        python_risk = classifier.predict(rgb)

        session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        # The tests above establish the two tensors are identical, so
        # feeding Python's is equivalent to feeding the browser's.
        assert browser["first12"] == pytest.approx(
            _python_tensor(rgb).reshape(-1)[:12].tolist(), abs=1e-6
        )
        scores = session.run(None, {"image": _python_tensor(rgb)})[0][0]

        assert scores[SCENE_LABELS.index("NSFL")] == pytest.approx(
            python_risk.graphic, abs=1e-5
        )
        assert scores[SCENE_LABELS.index("NSFW")] == pytest.approx(
            python_risk.sexual, abs=1e-5
        )
