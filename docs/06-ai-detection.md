# AI Detection — Model Evaluation & Selection

## Decision Summary

| Role | Selected Model | Rationale |
|---|---|---|
| **Object Detection** | YOLOv11n (COCO pretrained) | Smallest, fastest, browser-ready, detects knife/scissors |
| **Scene Classification** | ViT violence classifier (deferred) | Scene-level risk signals, Python-only initially |

---

## 1. Requirements

From SKILL.md, the AI vision engine must:

1. Detect potentially harmful objects and provide bounding boxes
2. Classify scene-level risk (violence, graphic content)
3. Run fast enough for real-time video processing
4. Be small enough for browser deployment via ONNX Runtime Web
5. Support ONNX export
6. Use a permissive or compatible license

---

## 2. Candidates Evaluated

### 2.1 YOLOv11n (COCO) — ✅ SELECTED

| Attribute | Value |
|---|---|
| **Architecture** | YOLOv11 with C3k2 + C2PSA attention |
| **Size** | ~6.4 MB (PyTorch), ~12 MB (ONNX) |
| **Classes** | 80 COCO classes |
| **Safety-relevant classes** | knife (43), scissors (76) |
| **Inference speed** | ~30-80ms CPU, ~5-15ms GPU |
| **ONNX export** | Native via Ultralytics |
| **Browser deployment** | ONNX → ONNX Runtime Web ✅ |
| **License** | AGPL-3.0 (commercial license available) |

**Why selected:**
- Smallest variant with best speed-to-accuracy ratio
- 22% fewer parameters than YOLOv8n with better detection
- Native ONNX export for browser deployment path
- Mature ecosystem, extensive documentation
- Knife detection out-of-the-box validates the pipeline

**Limitations:**
- Only 2 safety-relevant classes from COCO (knife, scissors)
- No gun/firearm/sword detection without custom training
- No scene-level classification

### 2.2 YOLOv11s (COCO)

| Attribute | Value |
|---|---|
| **Size** | ~19 MB |
| **Accuracy** | Higher mAP than nano |
| **Speed** | ~2x slower than nano |

**Verdict:** Not selected for baseline. Available as an upgrade path if nano accuracy is insufficient.

### 2.3 Custom YOLO (Weapon Dataset)

| Attribute | Value |
|---|---|
| **Classes** | gun, pistol, rifle, knife + custom |
| **Data sources** | Roboflow Universe, Kaggle weapon datasets |
| **Training required** | Yes |

**Verdict:** Deferred. Requires dataset curation, training infrastructure, and validation. Will be considered after the baseline pipeline is proven with COCO.

### 2.4 NudeNet v3

| Attribute | Value |
|---|---|
| **Size** | ~26 MB |
| **Categories** | Nudity/exposed body parts only |
| **Runtime** | ONNX-based |
| **License** | MIT |

**Verdict:** Not selected. Focused exclusively on nudity — does not cover weapons, violence, or graphic content. Could be integrated as an additional safety layer in the future.

### 2.5 ViT Violence Classifier

| Attribute | Value |
|---|---|
| **Model** | `jaranohaal/vit-base-violence-detection` |
| **Size** | ~350 MB |
| **Categories** | violent / non-violent (binary) |
| **License** | Apache-2.0 |

**Verdict:** Deferred for browser but useful for the Python reference pipeline. Provides scene-level risk that YOLO cannot. Too heavy for browser deployment initially.

---

## 3. Architecture Decision

### Dual-Model Strategy

```text
Image/Frame
    │
    ├──→ YOLOv11n (Object Detection)
    │       → bounding boxes + labels
    │       → Fast, lightweight
    │
    └──→ [Future] Scene Classifier
            → violence / graphic scores
            → Contextual risk signal
    │
    └──→ Risk Engine
            → Combines both signals
            → Produces action
```

This separates detection (WHAT is there) from classification (WHAT is happening), per SKILL.md §9.

---

## 4. COCO Safety Label Mapping

The following COCO classes are mapped to safety categories:

| COCO Label | Index | Safety Category | Base Risk |
|---|---|---|---|
| knife | 43 | weapons | MEDIUM |
| scissors | 76 | weapons | MEDIUM |

All other 78 COCO classes are considered safe and do not trigger protection.

**Why MEDIUM base risk?**
- A knife/scissors detection is a signal, not an automatic blur trigger
- The risk engine will adjust based on confidence and context
- A knife in a kitchen scene at low confidence → ALLOW
- A knife at high confidence with violence signals → PROTECT

---

## 5. Future Model Upgrade Path

| Phase | Model Change |
|---|---|
| Current | YOLOv11n (COCO) — knife, scissors |
| Short-term | Add ViT scene classifier for violence/graphic scores |
| Medium-term | Fine-tune YOLO on weapon dataset (gun, firearm, sword) |
| Long-term | Custom multi-task model (objects + scene + safety) |

The model abstraction layer (Phase 3) ensures these upgrades are non-breaking.

---

## 6. Known Limitations

1. **Limited weapon coverage**: Only knife and scissors from COCO. No firearms.
2. **No scene classification yet**: Cannot detect "a violent scene" without objects.
3. **COCO knife bias**: Trained mostly on kitchen/dining contexts — may miss tactical/combat knives.
4. **No explicit safety training**: COCO was not designed for safety detection.
5. **AGPL license**: Requires commercial license for distribution.

These are documented honestly per SKILL.md Rule 10.

---

## 7. Benchmark Results

Run `python scripts/benchmark_models.py` to generate results on your hardware.

Results are saved to `models/benchmark_results.json`.
