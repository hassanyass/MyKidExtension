"""
Phase A — benchmark candidate models for *genuinely harmful* content.

The COCO-trained YOLOv11n selected in Phase 2 detects 80 everyday object
classes, of which exactly two (knife, scissors) are safety-relevant. It has
no concept of blood, gore, violence or nudity — the categories the product
actually exists to catch. This script benchmarks candidates that do.

What it measures: model size, load time, inference latency, and output on
safe fixtures. What it deliberately does **not** measure: detection accuracy
on harmful content. That requires a harmful-content evaluation set, which
this project does not have and which is a sourcing decision for the project
owner (see docs/11, Open Decisions). Reporting accuracy we haven't measured
would be exactly the overclaiming SKILL.md rule 10 forbids.

A model scoring SFW on safe images proves it runs and isn't wildly
trigger-happy. It says nothing about whether it catches gore.

Usage:
    python scripts/benchmark_safety_models.py
    python scripts/benchmark_safety_models.py --runs 20
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

# Pinned so results are comparable across runs and machines. Left to
# default, each session grabs every core; with several models alive they
# oversubscribe and latency inflates several-fold (observed: 11ms -> 253ms).
INTRA_OP_THREADS = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates"
RESULTS_PATH = PROJECT_ROOT / "models" / "safety_benchmark_results.json"

SAFETY_CLASSIFIER_URL = (
    "https://huggingface.co/OwenElliott/image-safety-classifier-xs/"
    "resolve/main/onnx/image-safety-classifier-xs.onnx"
)
SAFETY_CLASSIFIER_PATH = CANDIDATE_DIR / "image-safety-classifier-xs.onnx"

# Output order is fixed by the model's config (pretrained_cfg.label_names).
SAFETY_LABELS = ["NSFL", "NSFW", "SFW"]

FIXTURES = [
    "test-data/images/detection-samples/bus.jpg",
    "test-data/images/detection-samples/zidane.jpg",
    "test-data/images/safe/test_safe.jpg",
    "test-data/images/safe/test_neutral.jpg",
]


def available_fixtures():
    return [p for p in FIXTURES if (PROJECT_ROOT / p).exists()]


def ensure_safety_classifier():
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    if not SAFETY_CLASSIFIER_PATH.exists():
        print(f"  downloading {SAFETY_CLASSIFIER_PATH.name}...")
        urllib.request.urlretrieve(SAFETY_CLASSIFIER_URL, SAFETY_CLASSIFIER_PATH)
    return SAFETY_CLASSIFIER_PATH


def benchmark_safety_classifier(runs: int):
    """SwiftFormer-XS finetune: NSFL (gore) / NSFW (sexual) / SFW."""
    import cv2
    import onnxruntime as ort

    path = ensure_safety_classifier()

    load_start = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_THREADS
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    load_ms = (time.perf_counter() - load_start) * 1000

    def preprocess(image_path):
        """
        Raw 0-255 RGB, NCHW. Preprocessing (resize/normalise) is baked into
        the ONNX graph — verified empirically: feeding pre-normalised input
        instead pushes NSFW on a photo of a bus from 0.03 to 0.18, i.e. it
        double-normalises and degrades the output.
        """
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(np.transpose(image.astype(np.float32), (2, 0, 1)), 0)

    fixture_results = {}
    latencies = []

    for fixture in available_fixtures():
        tensor = preprocess(PROJECT_ROOT / fixture)

        for _ in range(runs):
            start = time.perf_counter()
            probabilities = session.run(None, {"image": tensor})[0][0]
            latencies.append((time.perf_counter() - start) * 1000)

        fixture_results[fixture] = {
            "predicted": SAFETY_LABELS[int(np.argmax(probabilities))],
            "scores": {
                label: round(float(score), 4)
                for label, score in zip(SAFETY_LABELS, probabilities)
            },
        }

    return {
        "name": "image-safety-classifier-xs",
        "source": "OwenElliott/image-safety-classifier-xs",
        "architecture": "swiftformer_xs (3.5M params)",
        "license": "MIT (base SwiftFormer: Apache-2.0)",
        "categories": ["NSFL (gore/violence)", "NSFW (sexual)", "SFW"],
        "output": "3-class probabilities — scene level, no bounding boxes",
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "input_size": 224,
        "load_ms": round(load_ms),
        "min_inference_ms": round(float(np.min(latencies)), 1),
        "median_inference_ms": round(float(np.median(latencies)), 1),
        "mean_inference_ms": round(float(np.mean(latencies)), 1),
        "fixtures": fixture_results,
    }


def benchmark_nudenet(runs: int):
    """NudeNet 320n: 18 anatomical classes, with bounding boxes."""
    try:
        import nudenet
        from nudenet import NudeDetector
    except ImportError:
        return {"name": "nudenet-320n", "error": "not installed (pip install nudenet)"}

    model_path = Path(nudenet.__file__).parent / "320n.onnx"

    load_start = time.perf_counter()
    detector = NudeDetector()
    load_ms = (time.perf_counter() - load_start) * 1000

    fixture_results = {}
    latencies = []

    for fixture in available_fixtures():
        for _ in range(runs):
            start = time.perf_counter()
            detections = detector.detect(str(PROJECT_ROOT / fixture))
            latencies.append((time.perf_counter() - start) * 1000)

        fixture_results[fixture] = {
            "detection_count": len(detections),
            "classes": sorted({d["class"] for d in detections}),
        }

    return {
        "name": "nudenet-320n",
        "source": "notAI-tech/NudeNet",
        "architecture": "yolov8n finetune @ 320x320",
        "license": "MIT (package) — but the model derives from Ultralytics "
        "YOLOv8n, which is AGPL-3.0; treat the weights as AGPL-encumbered",
        "categories": ["18 anatomical classes, covered/exposed"],
        "output": "bounding boxes — supports region-level blur",
        "size_mb": round(model_path.stat().st_size / 1024 / 1024, 2)
        if model_path.exists()
        else None,
        "input_size": 320,
        "load_ms": round(load_ms),
        "min_inference_ms": round(float(np.min(latencies)), 1),
        "median_inference_ms": round(float(np.median(latencies)), 1),
        "mean_inference_ms": round(float(np.mean(latencies)), 1),
        "fixtures": fixture_results,
    }


def benchmark_incumbent(runs: int):
    """The current YOLOv11n, for comparison against the candidates."""
    import cv2
    import onnxruntime as ort

    path = PROJECT_ROOT / "models" / "yolo11n.onnx"
    if not path.exists():
        return {"name": "yolo11n", "error": "models/yolo11n.onnx not found"}

    load_start = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_THREADS
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    load_ms = (time.perf_counter() - load_start) * 1000

    fixtures = available_fixtures()
    latencies = []

    if fixtures:
        image = cv2.cvtColor(cv2.imread(str(PROJECT_ROOT / fixtures[0])), cv2.COLOR_BGR2RGB)
        canvas = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = np.expand_dims(
            np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1)), 0
        )
        input_name = session.get_inputs()[0].name

        for _ in range(runs):
            start = time.perf_counter()
            session.run(None, {input_name: tensor})
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "name": "yolo11n (incumbent)",
        "source": "Ultralytics YOLOv11n, COCO",
        "license": "AGPL-3.0",
        "categories": ["80 COCO classes — only knife and scissors are safety-relevant"],
        "output": "bounding boxes",
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "input_size": 640,
        "load_ms": round(load_ms),
        "min_inference_ms": round(float(np.min(latencies)), 1) if latencies else None,
        "median_inference_ms": round(float(np.median(latencies)), 1)
        if latencies
        else None,
        "mean_inference_ms": round(float(np.mean(latencies)), 1) if latencies else None,
        "note": "Cannot detect blood, gore, violence or nudity — those are not "
        "COCO categories.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="Runs per fixture")
    args = parser.parse_args()

    fixtures = available_fixtures()
    if not fixtures:
        raise SystemExit(
            "No fixtures found. Run: python scripts/fetch_detection_samples.py"
        )

    print(f"Benchmarking against {len(fixtures)} safe fixture(s), {args.runs} runs each.\n")

    results = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_per_fixture": args.runs,
        "fixtures": fixtures,
        "caveat": (
            "Latency, size and licence are measured. Accuracy on harmful content "
            "is NOT — no harmful evaluation set exists in this project. Safe "
            "fixtures only demonstrate the model runs and does not fire "
            "indiscriminately."
        ),
        "models": [],
    }

    for label, fn in [
        ("safety classifier", benchmark_safety_classifier),
        ("nudenet", benchmark_nudenet),
        ("incumbent yolo", benchmark_incumbent),
    ]:
        print(f"- {label}...")
        try:
            results["models"].append(fn(args.runs))
        except Exception as exc:  # noqa: BLE001 - report, don't mask
            results["models"].append({"name": label, "error": str(exc)})
            print(f"    failed: {exc}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'model':32s} {'size':>8s} {'load':>8s} {'median':>9s}  license")
    print("-" * 88)
    for model in results["models"]:
        if "error" in model:
            print(f"{model['name']:32s} {'-':>8s} {'-':>8s} {'-':>9s}  {model['error']}")
            continue
        print(
            f"{model['name']:32s} "
            f"{str(model.get('size_mb', '-')) + ' MB':>8s} "
            f"{str(model.get('load_ms', '-')) + ' ms':>8s} "
            f"{str(model.get('median_inference_ms', '-')) + ' ms':>9s}  "
            f"{model.get('license', '-')[:40]}"
        )

    print(f"\nResults written to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
