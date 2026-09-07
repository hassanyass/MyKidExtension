"""
MyKid Model Benchmark Script

Downloads YOLOv11n pretrained weights, runs inference on test images,
measures performance, and outputs structured benchmark results.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --image path/to/image.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.shared.config import load_config
from packages.shared.labels import (
    is_safety_relevant,
    get_safety_category,
    get_base_risk,
    get_coco_label,
    COCO_LABELS,
)
from packages.shared.logger import get_logger, log_event, LogEvent
import logging


logger = get_logger("benchmark", level="INFO")


def benchmark_yolo(image_paths: list, model_name: str = "yolo11n.pt"):
    """
    Benchmark a YOLO model on a set of images.

    Args:
        image_paths: List of image file paths to test.
        model_name: YOLO model variant to benchmark.

    Returns:
        Dictionary of benchmark results.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    config = load_config()

    # Load model
    print(f"\n{'='*60}")
    print(f"  MyKid Model Benchmark — {model_name}")
    print(f"{'='*60}\n")

    print(f"[1/4] Loading model: {model_name}...")
    load_start = time.time()
    model = YOLO(model_name)
    load_time = time.time() - load_start
    print(f"       Model loaded in {load_time:.2f}s")

    # Model info
    model_info = {
        "model_name": model_name,
        "task": str(model.task),
        "num_classes": len(model.names) if hasattr(model, 'names') else 0,
        "load_time_s": round(load_time, 3),
    }

    print(f"       Task: {model_info['task']}")
    print(f"       Classes: {model_info['num_classes']}")

    # Identify safety-relevant classes
    safety_classes = {}
    if hasattr(model, 'names'):
        for idx, name in model.names.items():
            if is_safety_relevant(name):
                category = get_safety_category(name)
                safety_classes[name] = {
                    "index": idx,
                    "category": category.value if category else "unknown",
                    "base_risk": get_base_risk(name).value,
                }

    print(f"\n[2/4] Safety-relevant classes in model:")
    if safety_classes:
        for label, info in safety_classes.items():
            print(f"       • {label} (index={info['index']}, "
                  f"category={info['category']}, risk={info['base_risk']})")
    else:
        print("       ⚠ No safety-relevant classes found in model!")

    # Run inference on images
    print(f"\n[3/4] Running inference on {len(image_paths)} image(s)...")
    confidence_threshold = config.detection.object_confidence_threshold

    image_results = []
    total_inference_time = 0

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"       ⚠ Skipping {img_path} (not found)")
            continue

        print(f"\n       Image: {img_path.name}")

        # Run inference
        inference_start = time.time()
        results = model(str(img_path), conf=confidence_threshold, verbose=False)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time

        result = results[0]
        detections = []
        safety_detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                xyxy = box.xyxy[0].tolist()

                detection = {
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(xyxy[0], 1),
                        "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1),
                        "y2": round(xyxy[3], 1),
                    },
                    "safety_relevant": is_safety_relevant(label),
                }
                detections.append(detection)

                if is_safety_relevant(label):
                    safety_detections.append(detection)

        img_result = {
            "image": img_path.name,
            "inference_time_ms": round(inference_time * 1000, 1),
            "total_detections": len(detections),
            "safety_detections": len(safety_detections),
            "detections": detections,
        }
        image_results.append(img_result)

        print(f"       Inference: {img_result['inference_time_ms']}ms")
        print(f"       Detections: {img_result['total_detections']} total, "
              f"{img_result['safety_detections']} safety-relevant")

        for det in detections:
            marker = "⚠" if det["safety_relevant"] else "✓"
            print(f"         {marker} {det['label']} ({det['confidence']:.2f})")

    # Summary
    avg_inference = (total_inference_time / len(image_results) * 1000) if image_results else 0

    print(f"\n[4/4] Benchmark Summary")
    print(f"{'='*60}")

    benchmark_summary = {
        "model": model_info,
        "safety_classes": safety_classes,
        "benchmark": {
            "images_tested": len(image_results),
            "avg_inference_ms": round(avg_inference, 1),
            "confidence_threshold": confidence_threshold,
        },
        "image_results": image_results,
    }

    print(f"       Model: {model_name}")
    print(f"       Images tested: {len(image_results)}")
    print(f"       Avg inference: {avg_inference:.1f}ms")
    print(f"       Confidence threshold: {confidence_threshold}")
    print(f"{'='*60}\n")

    return benchmark_summary


def create_test_images(output_dir: Path):
    """
    Create simple test images for benchmarking if no images are available.
    Uses solid color images with text as placeholders.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed, skipping test image creation")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    images = []

    # Create a simple test image (solid color)
    for name, color in [("test_safe", (100, 180, 100)), ("test_neutral", (180, 180, 180))]:
        img_path = output_dir / f"{name}.jpg"
        if not img_path.exists():
            img = Image.new("RGB", (640, 480), color)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Test Image: {name}", fill=(255, 255, 255))
            img.save(str(img_path))
        images.append(str(img_path))

    return images


def main():
    parser = argparse.ArgumentParser(description="MyKid Model Benchmark")
    parser.add_argument(
        "--image", "-i",
        nargs="*",
        help="Path(s) to image(s) to test. If not provided, creates test images.",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        help="YOLO model variant (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save benchmark results JSON",
    )
    args = parser.parse_args()

    # Determine test images
    if args.image:
        image_paths = args.image
    else:
        print("No images specified, creating test images...")
        test_dir = PROJECT_ROOT / "test-data" / "images" / "safe"
        image_paths = create_test_images(test_dir)

    if not image_paths:
        print("ERROR: No images available for benchmarking")
        sys.exit(1)

    # Run benchmark
    results = benchmark_yolo(image_paths, model_name=args.model)

    # Save results
    output_path = args.output or str(PROJECT_ROOT / "models" / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
