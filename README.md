# MyKid — AI Visual Safety System

An AI-powered visual safety system designed to reduce a child's exposure to harmful visual content while browsing the web.

## What It Does

MyKid detects potentially harmful visual content in images and videos, assesses contextual risk, and applies visual protection (blur/pixelate/cover) — all locally on the device.

```text
Image/Video → AI Detection → Risk Assessment → Protection Decision → Safe Output
```

**Core principle:** `DETECT ≠ DECIDE ≠ PROTECT`

Detection and action are always separate. A knife in a cooking scene is not the same as a knife in a violent scene.

## Architecture

```text
MyKid
├── Browser Layer (Chrome Extension, Manifest V3)
├── Evaluation Layer (Streamlit dashboard)
└── Visual Safety Engine
    ├── Image Pipeline
    ├── Video Pipeline
    ├── Vision Engine (AI model abstraction)
    ├── Risk Engine (contextual assessment)
    ├── Policy Engine (action decisions)
    └── Protection Engine (blur/pixelate/cover)
```

## Project Structure

```text
MyKidExtension/
├── packages/
│   ├── shared/        # Core types, config, logging, errors
│   ├── vision/        # AI model abstraction
│   ├── risk/          # Risk assessment engine
│   ├── protection/    # Blur/pixelate/cover engine
│   ├── image_processing/  # Image pipeline
│   └── video_processing/  # Video pipeline
├── apps/
│   ├── extension/     # Chrome Extension (Manifest V3)
│   └── dashboard/     # Streamlit evaluation UI
├── models/            # Model weights
├── configs/           # Configuration files
├── tests/             # Test suite
├── test-data/         # Test images and videos
├── docs/              # Documentation
└── scripts/           # Utility scripts
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

## Configuration

All thresholds and settings are centralized in `configs/default.yaml`.

Override any value via environment variables:
```bash
MYKID_DETECTION_OBJECT_CONFIDENCE_THRESHOLD=0.7
MYKID_PROTECTION_MODE=pixelate
MYKID_VIDEO_INFERENCE_FPS=10
```

## Development Status

| Phase | Status |
|---|---|
| Phase 1-8 — Python reference pipeline | ✅ Done |
| Phase 10-11 — Browser extension (images + video) | ✅ Built — video not yet confirmed in-browser |
| Harmful-content model (gore / sexual) | ✅ Done |
| On/off toggle | ✅ Done |
| Weapon detection (firearms) | ⏸️ Blocked — needs an evaluation set |
| Phase 13-14 — Parity, privacy audit, limitations | ✅ Done |

See [docs/11-remaining-implementation-plan.md](docs/11-remaining-implementation-plan.md) for the phase-by-phase breakdown, and
[docs/12-privacy-and-limitations.md](docs/12-privacy-and-limitations.md) for the privacy audit and known limitations.

**Honest status:** the extension detects gore and sexual content (scene-level,
whole-image blur) plus knives and scissors (region blur), runs entirely
on-device, and is controlled by a single on/off toggle. **Detection accuracy
on harmful content has never been measured** — there is no harmful evaluation
set in the project — so treat capability claims accordingly. See
[docs/12](docs/12-privacy-and-limitations.md) §2.

## Browser extension

```bash
python scripts/benchmark_safety_models.py   # downloads the safety classifier
node apps/extension/scripts/setup.js        # stages models + wasm runtime
```

Then load `apps/extension/` unpacked at `chrome://extensions` (Developer mode on).

## Documentation

See the [`docs/`](docs/) directory and [`.agents/skills/mykid/SKILL.md`](.agents/skills/mykid/SKILL.md) for the full specification.

## Privacy

MyKid processes all content **locally**. No child browsing data is sent to external servers.
