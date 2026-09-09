"""
Phase F — quantize the shipped models and verify they still behave.

Motivation: the extension carries ~14MB of wasm runtime plus two models,
and every image analysed pays their inference cost. INT8 dynamic
quantization typically cuts model size ~4x and speeds inference up, at
some cost in numeric fidelity.

That cost is the whole risk. A quantized classifier that scores gore
slightly lower is not a neutral trade — it is a safety regression, and one
that would be invisible without checking. So this script does not just
quantize: it re-runs both models on the fixtures and reports how far the
outputs moved, so adoption is a decision made on evidence.

Usage:
    python scripts/quantize_models.py
"""
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = PROJECT_ROOT / "models" / "candidates"
RESULTS = PROJECT_ROOT / "models" / "quantization_results.json"

INTRA_OP_THREADS = 2

FIXTURES = [
    PROJECT_ROOT / "test-data" / "images" / "detection-samples" / "bus.jpg",
    PROJECT_ROOT / "test-data" / "images" / "detection-samples" / "zidane.jpg",
    PROJECT_ROOT / "test-data" / "images" / "safe" / "test_safe.jpg",
    PROJECT_ROOT / "test-data" / "images" / "safe" / "test_neutral.jpg",
]

SCENE_LABELS = ["NSFL", "NSFW", "SFW"]


def _session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_THREADS
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def _strip_stale_shapes(src: Path) -> Path:
    """
    Drop the graph's cached intermediate shape annotations before quantizing.

    The safety classifier ships value_info entries that disagree with what
    ONNX re-infers (220 features vs 3 classes on the head), which makes
    quantize_dynamic abort during its shape-inference pass. The annotations
    are only a cache — they are re-derived from the graph — so clearing
    them changes no computation, and the quantized model is verified
    against fp32 outputs afterwards regardless.
    """
    import onnx

    model = onnx.load(str(src))
    if not model.graph.value_info:
        return src

    del model.graph.value_info[:]
    cleaned = src.with_suffix(".noshape.onnx")
    onnx.save(model, str(cleaned))
    return cleaned


def quantize(src: Path, dst: Path):
    """Dynamic INT8 quantization — no calibration data needed."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if dst.exists():
        dst.unlink()

    prepared = _strip_stale_shapes(src)
    try:
        quantize_dynamic(
            model_input=str(prepared),
            model_output=str(dst),
            weight_type=QuantType.QUInt8,
        )
    finally:
        if prepared != src and prepared.exists():
            prepared.unlink()
    return dst


def time_session(session, feeds, runs=15):
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feeds)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "min_ms": round(float(np.min(latencies)), 1),
        "median_ms": round(float(np.median(latencies)), 1),
    }


def scene_inputs():
    import cv2

    tensors = {}
    for fixture in FIXTURES:
        if not fixture.exists():
            continue
        image = cv2.cvtColor(cv2.imread(str(fixture)), cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensors[fixture.name] = np.expand_dims(
            np.transpose(resized.astype(np.float32), (2, 0, 1)), 0
        )
    return tensors


def yolo_input():
    import cv2

    image = cv2.cvtColor(cv2.imread(str(FIXTURES[0])), cv2.COLOR_BGR2RGB)
    canvas = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(
        np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1)), 0
    )


def evaluate_scene_classifier():
    src = CANDIDATES / "image-safety-classifier-xs.onnx"
    if not src.exists():
        return {"error": "run scripts/benchmark_safety_models.py first"}

    dst = quantize(src, CANDIDATES / "image-safety-classifier-xs.int8.onnx")
    inputs = scene_inputs()

    fp32, int8 = _session(src), _session(dst)
    comparisons = {}
    worst_shift = 0.0

    for name, tensor in inputs.items():
        a = fp32.run(None, {"image": tensor})[0][0]
        b = int8.run(None, {"image": tensor})[0][0]
        shift = float(np.max(np.abs(a - b)))
        worst_shift = max(worst_shift, shift)
        comparisons[name] = {
            "fp32": {l: round(float(v), 4) for l, v in zip(SCENE_LABELS, a)},
            "int8": {l: round(float(v), 4) for l, v in zip(SCENE_LABELS, b)},
            "max_abs_shift": round(shift, 4),
            "verdict_changed": SCENE_LABELS[int(np.argmax(a))]
            != SCENE_LABELS[int(np.argmax(b))],
        }

    sample = next(iter(inputs.values()))
    return {
        "model": "image-safety-classifier-xs",
        "fp32_mb": round(src.stat().st_size / 1024 / 1024, 2),
        "int8_mb": round(dst.stat().st_size / 1024 / 1024, 2),
        "fp32_latency": time_session(fp32, {"image": sample}),
        "int8_latency": time_session(int8, {"image": sample}),
        "max_abs_score_shift": round(worst_shift, 4),
        "any_verdict_changed": any(c["verdict_changed"] for c in comparisons.values()),
        "per_fixture": comparisons,
    }


def evaluate_yolo():
    src = PROJECT_ROOT / "models" / "yolo11n.onnx"
    if not src.exists():
        return {"error": "models/yolo11n.onnx not found"}

    dst = quantize(src, CANDIDATES / "yolo11n.int8.onnx")
    tensor = yolo_input()

    fp32, int8 = _session(src), _session(dst)
    name_fp32 = fp32.get_inputs()[0].name
    name_int8 = int8.get_inputs()[0].name

    a = fp32.run(None, {name_fp32: tensor})[0]
    b = int8.run(None, {name_int8: tensor})[0]

    # Compare the class-score block only; raw box regression values are not
    # directly meaningful without decoding.
    scores_fp32 = a[0][4:, :]
    scores_int8 = b[0][4:, :]

    return {
        "model": "yolo11n",
        "fp32_mb": round(src.stat().st_size / 1024 / 1024, 2),
        "int8_mb": round(dst.stat().st_size / 1024 / 1024, 2),
        "fp32_latency": time_session(fp32, {name_fp32: tensor}),
        "int8_latency": time_session(int8, {name_int8: tensor}),
        "max_class_score_shift": round(
            float(np.max(np.abs(scores_fp32 - scores_int8))), 4
        ),
        "peak_confidence_fp32": round(float(np.max(scores_fp32)), 4),
        "peak_confidence_int8": round(float(np.max(scores_int8)), 4),
    }


def main():
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    print("Quantizing (INT8 dynamic) and comparing against fp32...\n")

    results = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "dynamic INT8 (QUInt8 weights)",
        "scene_classifier": evaluate_scene_classifier(),
        "yolo": evaluate_yolo(),
    }
    RESULTS.write_text(json.dumps(results, indent=2))

    for key in ("scene_classifier", "yolo"):
        entry = results[key]
        if "error" in entry:
            print(f"{key}: {entry['error']}")
            continue
        print(f"{entry['model']}")
        print(
            f"  size    {entry['fp32_mb']} MB -> {entry['int8_mb']} MB "
            f"({100 - entry['int8_mb'] / entry['fp32_mb'] * 100:.0f}% smaller)"
        )
        print(
            f"  median  {entry['fp32_latency']['median_ms']} ms -> "
            f"{entry['int8_latency']['median_ms']} ms"
        )
        if key == "scene_classifier":
            print(f"  max score shift      {entry['max_abs_score_shift']}")
            print(f"  any verdict changed  {entry['any_verdict_changed']}")
        else:
            print(f"  max class-score shift {entry['max_class_score_shift']}")
            print(
                f"  peak conf  {entry['peak_confidence_fp32']} -> "
                f"{entry['peak_confidence_int8']}"
            )
        print()

    print(f"Written to {RESULTS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
