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
| Phase 1 — Foundation | 🔨 In Progress |
| Phase 2 — Model Research | ⬜ Not Started |
| Phase 3-6 — Image Pipeline | ⬜ Not Started |
| Phase 7-9 — Video Pipeline | ⬜ Not Started |
| Phase 10-12 — Browser Extension | ⬜ Not Started |
| Phase 13-14 — Integration & Hardening | ⬜ Not Started |

## Documentation

See the [`docs/`](docs/) directory and [`.agents/skills/mykid/SKILL.md`](.agents/skills/mykid/SKILL.md) for the full specification.

## Privacy

MyKid processes all content **locally**. No child browsing data is sent to external servers.
