# AI Detection — Model Evaluation & Selection

## Decision Summary

> **Revised 2026-09-08 (Phase A).** The original selection below was made to prove the pipeline, not to catch harm. Phase 2 optimised for "a model that produces detections in a browser"; the product needs "a model that recognises blood, gore, violence and nudity". YOLOv11n does neither of the latter — those are not COCO categories — so the primary detector has changed. Phase 2's reasoning is preserved below for the record.

| Role | Selected Model | Rationale |
|---|---|---|
| **Harm classification (primary)** | `image-safety-classifier-xs` | Actually covers the target categories: NSFL (gore/violence), NSFW (sexual). 12.5MB, ~13ms, MIT, ONNX with preprocessing baked in. |
| **Object Detection (secondary)** | YOLOv11n (COCO pretrained) | Retained for knife/scissors *regions* — localised blur where the classifier can only blur the whole frame. Now the most expensive model for the least product value; gated rather than always-on. |
| **Nudity regions (optional)** | NudeNet 320n | Boxes instead of full-frame for nudity. Strong capability, but AGPL-encumbered weights — see licensing below. |

---

## Phase A — Harmful-Content Model Evaluation (2026-09-08)

### Why this phase happened

The browser pipeline was proven end-to-end using `person` as a stand-in for "harmful", because COCO offers nothing better. That made the gap unavoidable: **the system worked perfectly and could not detect a single thing it exists to catch.** Blood, gore, violence and nudity are not COCO classes, and no threshold change reaches them.

### Measured results

Reproduce with `python scripts/benchmark_safety_models.py`; raw output in `models/safety_benchmark_results.json`.

| Model | Size | Load | Median inference | Output | Licence |
|---|---|---|---|---|---|
| **image-safety-classifier-xs** | 12.53 MB | 87 ms | **12.8 ms** | 3 scene-level probabilities | MIT (base SwiftFormer Apache-2.0) |
| NudeNet 320n | 11.59 MB | 112 ms | 25.3 ms | 18 anatomical classes, **with boxes** | MIT package, AGPL-derived weights |
| YOLOv11n *(incumbent)* | 10.14 MB | 151 ms | 202.0 ms | 80 COCO classes, with boxes | AGPL-3.0 |

**On the latency figures:** these are native `onnxruntime` on CPU with `intra_op_num_threads=2`, all three sessions alive in one process. Measured in isolation, the classifier runs ~11ms and YOLOv11n ~42ms — so the *absolute* numbers move a lot with load and thread contention (an early unpinned run reported 253ms for an 11ms model). The **ordering is stable and is the finding**: the classifier is several times cheaper than YOLO while covering far more of what matters. Browser figures will be higher again — measured YOLOv11n in-browser is ~230ms against ~42ms native, roughly a 5× WASM penalty.

### Behaviour on safe fixtures

| Fixture | Predicted | NSFL | NSFW | SFW |
|---|---|---|---|---|
| bus.jpg | SFW | 0.035 | 0.032 | 0.933 |
| zidane.jpg | SFW | 0.033 | 0.030 | 0.937 |
| test_safe.jpg | SFW | 0.029 | 0.040 | 0.931 |
| test_neutral.jpg | SFW | 0.031 | 0.082 | 0.887 |

**What this does and does not establish.** It shows the model runs, and that it doesn't fire indiscriminately on ordinary images — a real false-positive risk for a tool that would otherwise blur everything and get switched off. It says **nothing** about whether it catches gore, because we have no harmful evaluation set. The vendor reports 97.76% accuracy on a proprietary 320k-image dataset; that is self-reported, on their own data, and is not independent evidence. **Detection accuracy remains unmeasured** (SKILL.md rule 10) and stays that way until Phase 9c resolves test-data sourcing.

### Integration notes (verified, not assumed)

- **Input:** `image`, shape `[batch, 3, 224, 224]`, float32.
- **Feed raw 0-255 RGB — do not normalise.** Preprocessing is baked into the ONNX graph. Verified empirically: pre-normalising pushes NSFW on a photo of a bus from 0.032 to 0.18, i.e. it double-normalises and degrades output.
- **Output:** `probabilities`, shape `[batch, 3]`, ordered **`[NSFL, NSFW, SFW]`** — from the model's own `pretrained_cfg.label_names`, not guessed.

### Mapping onto the existing risk engine

`SceneRisk` currently carries `violence` and `graphic` (`types.py:110`), and the risk engine escalates to `BLUR_FRAME` when either reaches `scene_risk_threshold` (0.6). The classifier's outputs map onto this almost directly:

| Classifier output | SceneRisk field | Note |
|---|---|---|
| NSFL (gore/violence) | `graphic` | Direct fit. |
| NSFW (sexual) | — | **No field exists.** Needs a `sexual` field adding to `SceneRisk` and `max_score()`, in both the Python and JS engines. |

The `violence` field stays unused for now: the classifier folds violence into NSFL rather than scoring it separately. Keeping the field distinct leaves room for a dedicated violence model later without reshaping the type.

### Recommended architecture: cheap gate first

Running every model on every image is the wrong shape — YOLO alone is ~230ms in-browser, and images are analysed continuously while browsing. Instead:

```text
image ──► safety classifier (~13ms native)  ──► NSFL / NSFW / SFW
             │                                        │
             │ confidently SFW                        │ above threshold
             ▼                                        ▼
        stop — no further work                  BLUR_FRAME (done — no object
                                                detection needed)
             │
             │ (optional) object-level detail wanted
             ▼
        YOLOv11n ──► knife / scissors regions ──► BLUR_REGION
```

This inverts the current pipeline. The expensive model runs least, the cheap model runs always, and the cheap one covers more of the product's actual purpose.

### Licensing consequence, worth deciding deliberately

The classifier is **MIT** over an **Apache-2.0** base. YOLOv11n is **AGPL-3.0**, and NudeNet's weights derive from Ultralytics YOLOv8n so are best treated as AGPL-encumbered too.

That means a stack of *just the safety classifier* is **fully permissive** — which would resolve the AGPL question that has been open since Phase 2, at the cost of losing knife/scissors region blur. Since the classifier already flags gore and sexual content full-frame, and knives are a minor part of the harm surface, dropping AGPL entirely is a genuinely viable option rather than a sacrifice. This is a product/legal call, not an engineering one.

### Not selected, and why

- **`jaranohaal/vit-base-violence-detection`** (~350MB) — 28× the classifier for one category the classifier already covers. Viable in the Python reference pipeline; not in a browser.
- **CLIP zero-shot** against prompts like "blood", "injury" — flexible and needs no training, but the image encoder is ~25-40MB quantized and far slower than 13ms. Worth revisiting only if the classifier proves weak on gore specifically.
- **Fine-tuned weapon YOLO** (gun, rifle) — still deferred, still requires dataset curation and a training run. Phase C.

---

## Phase 2 evaluation (original, retained for the record)

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
