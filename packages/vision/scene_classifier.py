"""
Scene-level safety classifier.

Answers "what is happening in this image" rather than "what objects are
present" — the object/scene split in SKILL.md §9. Object detection can only
flag things it has a box for; a violent or graphic scene has no box to draw,
so it produces whole-frame protection instead.

Model: `image-safety-classifier-xs` (SwiftFormer-XS finetune, MIT).
Selected in Phase A — see docs/06-ai-detection.md for the benchmark and the
reasoning. Outputs three probabilities: NSFL (gore/violence), NSFW (sexual),
SFW.

This class deliberately does not subclass VisionModel: that interface
returns `List[Detection]` (objects with boxes), which a scene classifier
has none of. Forcing it to fit would mean inventing fake bounding boxes,
which SKILL.md §13 explicitly forbids.
"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from packages.shared.config import MyKidConfig
from packages.shared.errors import InferenceError, ModelError
from packages.shared.logger import LogEvent, get_logger, log_event
from packages.shared.types import SceneRisk


logger = get_logger("mykid.vision.scene_classifier")


# Fixed by the model's own config (pretrained_cfg.label_names) — read from
# the model, not assumed.
SCENE_LABELS = ["NSFL", "NSFW", "SFW"]

MODEL_INPUT_SIZE = 224


class SceneClassifier:
    """
    Classifies an image's scene-level safety risk.

    Produces a SceneRisk, which the risk engine compares against
    `scene_risk_threshold` to decide whole-frame protection.
    """

    def __init__(self, config: MyKidConfig, model_path: Optional[str] = None):
        """
        Args:
            config: Global configuration.
            model_path: Override for the classifier weights. Defaults to
                `models/candidates/image-safety-classifier-xs.onnx`.
        """
        self.config = config
        self.model_path = Path(
            model_path
            or Path(config.model.path) / "candidates" / "image-safety-classifier-xs.onnx"
        )
        self.session = None

    def load(self) -> None:
        """
        Load the ONNX classifier.

        Raises:
            ModelError: If the runtime is missing or the weights can't load.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError(
                "onnxruntime is not installed. Run 'pip install -r requirements.txt'"
            ) from exc

        if not self.model_path.exists():
            raise ModelError(
                f"Scene classifier weights not found at {self.model_path}. "
                "Run 'python scripts/benchmark_safety_models.py' to download them."
            )

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            error_msg = f"Failed to load scene classifier: {exc}"
            log_event(
                logger, logging.ERROR, LogEvent.ERROR_MODEL, error_msg,
                metadata={"error": str(exc)},
            )
            raise ModelError(error_msg) from exc

        log_event(
            logger,
            logging.INFO,
            LogEvent.MODEL_LOADED,
            "Scene classifier loaded",
            metadata={"model": self.model_path.name, "labels": SCENE_LABELS},
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Resize to 224x224 and arrange as NCHW float32 in the raw 0-255 range.

        Normalisation is baked into the ONNX graph, so the input must NOT be
        pre-normalised. Verified in Phase A: normalising first pushes the
        NSFW score on a photo of a bus from 0.032 to 0.18, because the graph
        then normalises already-normalised values.
        """
        import cv2

        resized = cv2.resize(
            image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        chw = np.transpose(resized.astype(np.float32), (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def predict(self, image: np.ndarray) -> SceneRisk:
        """
        Classify an RGB image's scene-level risk.

        Args:
            image: RGB numpy array (H, W, 3).

        Returns:
            SceneRisk with `graphic` and `sexual` populated. `violence` is
            left at 0.0: this model folds violence into its NSFL class
            rather than scoring it separately, and inventing a number for it
            would misrepresent what was measured.

        Raises:
            InferenceError: If inference fails.
        """
        if self.session is None:
            raise ModelError("Scene classifier not loaded. Call load() first.")

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InferenceError("Invalid or empty image array")

        try:
            start = time.perf_counter()
            probabilities = self.session.run(
                None, {"image": self._preprocess(image)}
            )[0][0]
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            error_msg = f"Scene classification failed: {exc}"
            log_event(
                logger, logging.ERROR, LogEvent.ERROR_INFERENCE, error_msg,
                metadata={"error": str(exc)},
            )
            raise InferenceError(error_msg) from exc

        scores = dict(zip(SCENE_LABELS, (float(p) for p in probabilities)))
        scene_risk = SceneRisk(graphic=scores["NSFL"], sexual=scores["NSFW"])

        if scene_risk.max_score() >= self.config.detection.scene_risk_threshold:
            log_event(
                logger,
                logging.INFO,
                LogEvent.RISK_DETECTED,
                "Scene-level risk detected",
                metadata={
                    "graphic": round(scene_risk.graphic, 4),
                    "sexual": round(scene_risk.sexual, 4),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )

        return scene_risk
