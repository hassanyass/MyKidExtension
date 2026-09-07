"""
MyKid ONNX Export Script

Exports YOLOv11n to ONNX format for future browser deployment via ONNX Runtime Web.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model yolo11s.pt --output models/yolo11s.onnx
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_to_onnx(model_name: str = "yolo11n.pt", output_path: str = None):
    """
    Export a YOLO model to ONNX format.

    Args:
        model_name: YOLO model variant to export.
        output_path: Where to save the ONNX file. Defaults to models/ directory.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  MyKid ONNX Export — {model_name}")
    print(f"{'='*60}\n")

    # Load model
    print(f"[1/3] Loading model: {model_name}...")
    model = YOLO(model_name)
    print(f"       Model loaded successfully")

    # Export to ONNX
    print(f"\n[2/3] Exporting to ONNX...")
    export_start = time.time()

    onnx_path = model.export(
        format="onnx",
        simplify=True,
        imgsz=640,
    )
    export_time = time.time() - export_start

    onnx_file = Path(onnx_path)
    onnx_size_mb = onnx_file.stat().st_size / (1024 * 1024)

    print(f"       Export completed in {export_time:.1f}s")
    print(f"       ONNX file: {onnx_path}")
    print(f"       ONNX size: {onnx_size_mb:.1f} MB")

    # Move to target location if specified
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if onnx_file != target:
            import shutil
            shutil.move(str(onnx_file), str(target))
            onnx_path = str(target)
            print(f"       Moved to: {onnx_path}")

    # Validate ONNX with onnxruntime
    print(f"\n[3/3] Validating ONNX model...")
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)

        # Get model input info
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape

        print(f"       Input name: {input_name}")
        print(f"       Input shape: {input_shape}")

        # Run test inference with dummy data
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        inference_start = time.time()
        outputs = session.run(None, {input_name: dummy_input})
        inference_time = (time.time() - inference_start) * 1000

        print(f"       Output shapes: {[o.shape for o in outputs]}")
        print(f"       Test inference: {inference_time:.1f}ms")
        print(f"\n       ✅ ONNX model validated successfully!")

    except ImportError:
        print("       ⚠ onnxruntime not installed, skipping validation")
    except Exception as e:
        print(f"       ❌ ONNX validation failed: {e}")

    print(f"\n{'='*60}")
    print(f"  Export complete: {onnx_path}")
    print(f"  Size: {onnx_size_mb:.1f} MB")
    print(f"{'='*60}\n")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="MyKid ONNX Export")
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        help="YOLO model variant to export (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for ONNX file (default: models/<model>.onnx)",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        model_stem = Path(args.model).stem
        output = str(PROJECT_ROOT / "models" / f"{model_stem}.onnx")

    export_to_onnx(model_name=args.model, output_path=output)


if __name__ == "__main__":
    main()
