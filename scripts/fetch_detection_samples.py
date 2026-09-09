"""
Populate test-data/images/detection-samples/ with images that actually
produce detections.

The project's own fixtures in test-data/images/safe/ are blank — useful for
"safe content stays untouched" tests, useless for validating bounding-box
decoding, because a pipeline that returns zero boxes passes them whether or
not its coordinate maths is correct.

This copies the standard sample images bundled with the installed
`ultralytics` package (no network access needed — they ship with the pip
package). They are AGPL-3.0 licensed, like the YOLO model itself, so they
are gitignored rather than committed; regenerate them with this script.

Usage:
    python scripts/fetch_detection_samples.py
"""
import pathlib
import shutil
import sys

SAMPLES = ["bus.jpg", "zidane.jpg"]
DEST_DIR = pathlib.Path("test-data/images/detection-samples")


def main() -> int:
    try:
        import ultralytics
    except ImportError:
        print(
            "ultralytics is not installed — run `pip install -r requirements.txt` first.",
            file=sys.stderr,
        )
        return 1

    assets_dir = pathlib.Path(ultralytics.__file__).parent / "assets"
    if not assets_dir.exists():
        print(f"No assets directory found at {assets_dir}", file=sys.stderr)
        return 1

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in SAMPLES:
        source = assets_dir / name
        if not source.exists():
            print(f"  skipped {name} (not present in this ultralytics version)")
            continue
        shutil.copyfile(source, DEST_DIR / name)
        print(f"  copied {name}")
        copied += 1

    if copied == 0:
        print("No sample images were copied.", file=sys.stderr)
        return 1

    print(f"\n{copied} sample image(s) in {DEST_DIR}")
    print("Next: `node apps/extension/scripts/setup.js` to make them available "
          "to the extension's parity test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
