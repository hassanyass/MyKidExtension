"""
Parity reference for the browser detection pipeline.

Runs the *same* ONNX file the extension uses (models/yolo11n.onnx) through
onnxruntime with explicit, hand-written letterbox preprocessing and YOLO
output decoding — deliberately mirroring what the browser implementation
does in JavaScript, step for step.

Why not just use the Ultralytics pipeline as the reference? Because
Ultralytics does preprocessing and NMS internally, so comparing against it
would tell us "the browser disagrees" without telling us *which* stage
disagrees. This script makes each stage explicit and comparable.

Use it to validate the browser's numbers (Phase 10b-ii), and later as the
basis for the cross-pipeline consistency tests in Phase 13.

Usage:
    python scripts/parity_reference.py <image-path> [--conf 0.25]
    python scripts/parity_reference.py test-data/images/safe/test_safe.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make `packages` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.shared.labels import get_coco_label, is_safety_relevant  # noqa: E402


MODEL_INPUT_SIZE = 640
LETTERBOX_PAD_VALUE = 114  # gray, matching Ultralytics' default
DEFAULT_IOU_THRESHOLD = 0.7  # Ultralytics' default NMS IoU


def letterbox(image: np.ndarray, size: int = MODEL_INPUT_SIZE):
    """
    Resize an RGB image into a square `size`x`size` canvas, preserving
    aspect ratio and padding the remainder with flat gray.

    Returns:
        (canvas, scale, pad_x, pad_y) — the padded image plus the values
        needed to map model-space coordinates back to the original image.
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def to_input_tensor(canvas: np.ndarray) -> np.ndarray:
    """RGB HWC uint8 -> normalized NCHW float32, exactly as the browser does."""
    chw = canvas.astype(np.float32) / 255.0
    chw = np.transpose(chw, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(chw, axis=0)  # CHW -> NCHW


def iou(box_a, box_b) -> float:
    """Intersection-over-union for two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """Greedy per-class NMS over detections sorted by confidence."""
    kept = []
    by_confidence = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    while by_confidence:
        best = by_confidence.pop(0)
        kept.append(best)
        by_confidence = [
            d
            for d in by_confidence
            if d["class_id"] != best["class_id"]
            or iou(d["_xyxy"], best["_xyxy"]) < iou_threshold
        ]

    return kept


def decode_output(output: np.ndarray, scale, pad_x, pad_y, orig_w, orig_h, conf_threshold):
    """
    Decode a YOLOv8/v11 [1, 84, 8400] output tensor into detections in
    original-image coordinates.

    Layout: 84 channels = 4 box coords (cx, cy, w, h) + 80 class scores.
    There is no separate objectness score — confidence is the max class score.
    """
    predictions = output[0]  # [84, 8400]
    num_channels, num_anchors = predictions.shape
    num_classes = num_channels - 4

    class_scores = predictions[4:, :]  # [80, 8400]
    best_class_ids = np.argmax(class_scores, axis=0)  # [8400]
    best_scores = class_scores[best_class_ids, np.arange(num_anchors)]  # [8400]

    candidates = np.where(best_scores >= conf_threshold)[0]

    detections = []
    for anchor in candidates:
        cx, cy, w, h = predictions[0:4, anchor]

        # Model space (640x640, letterboxed) -> original image space
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        # Clamp to image bounds
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(orig_w), float(x2)), min(float(orig_h), float(y2))

        class_id = int(best_class_ids[anchor])
        label = get_coco_label(class_id)

        detections.append(
            {
                "label": label,
                "class_id": class_id,
                "confidence": round(float(best_scores[anchor]), 4),
                "bbox": {
                    "x": round(x1, 1),
                    "y": round(y1, 1),
                    "width": round(x2 - x1, 1),
                    "height": round(y2 - y1, 1),
                },
                "safety_relevant": is_safety_relevant(label),
                "_xyxy": (x1, y1, x2, y2),
            }
        )

    kept = non_max_suppression(detections)
    for detection in kept:
        detection.pop("_xyxy", None)

    return sorted(kept, key=lambda d: d["confidence"], reverse=True)


def run(image_path: str, conf_threshold: float, model_path: str):
    import cv2
    import onnxruntime as ort

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    canvas, scale, pad_x, pad_y = letterbox(image_rgb)
    input_tensor = to_input_tensor(canvas)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]

    detections = decode_output(
        output, scale, pad_x, pad_y, orig_w, orig_h, conf_threshold
    )

    return {
        "image": image_path,
        "original_size": {"width": orig_w, "height": orig_h},
        "letterbox": {
            "scale": round(float(scale), 6),
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
        },
        "output_shape": list(output.shape),
        "conf_threshold": conf_threshold,
        "detection_count": len(detections),
        "safety_relevant_count": sum(1 for d in detections if d["safety_relevant"]),
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the image to run detection on")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default 0.25, lower than the pipeline's "
        "0.5 so parity checks have more boxes to compare)",
    )
    parser.add_argument(
        "--model",
        default="models/yolo11n.onnx",
        help="Path to the ONNX model (default: models/yolo11n.onnx)",
    )
    args = parser.parse_args()

    print(json.dumps(run(args.image, args.conf, args.model), indent=2))


if __name__ == "__main__":
    main()
