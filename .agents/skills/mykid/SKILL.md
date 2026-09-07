---
name: mykid
description: >
  MyKid AI Visual Safety System — Primary implementation instruction for the coding agent.
  This skill MUST be read before implementing, modifying, or refactoring the MyKid project.
  MyKid is an AI-powered visual safety system designed to reduce a child's exposure to
  harmful visual content while browsing the web. Covers architecture, detection pipeline,
  risk assessment, protection engine, video processing, Chrome extension, and all development phases.
---

# MyKid — AI Visual Safety System

## 1. Purpose of This File

This file is the **primary implementation instruction** for the coding agent working on MyKid.

The agent MUST read and follow this file before implementing, modifying, or refactoring the project.

MyKid is an AI-powered visual safety system designed to reduce a child's exposure to harmful visual content while browsing the web.

The current implementation focuses **ONLY on images and videos**.

The system must:

1. Detect potentially harmful visual content.
2. Identify where the harmful content appears when possible.
3. Estimate the risk of the detected content.
4. Decide whether protection is required.
5. Blur or cover the harmful region.
6. For scene-level harmful content, blur or cover the entire image/frame.
7. Apply the same protection logic to video frames.
8. Maintain stable protection across video frames.
9. Eventually operate as a Chrome browser extension.
10. Prefer local/on-device processing where technically feasible.

The goal is not simply to demonstrate object detection.

The goal is to build a complete working pipeline:

```text
INPUT → VISUAL CONTENT PROCESSING → AI DETECTION → RISK ASSESSMENT → PROTECTION DECISION → BLUR / COVER → SAFE OUTPUT
```

The implementation must be modular so that the AI model, risk logic, image processing, video processing, and browser extension can evolve independently.

---

## 2. Current Scope

### 2.1 In Scope — Images

Supported image inputs: JPG, JPEG, PNG, WebP

The system should be able to: load an image, preprocess it, run visual detection, classify detected content, calculate risk, identify protected regions, blur or cover those regions, produce the protected image, expose detection metadata for debugging/evaluation.

### 2.2 In Scope — Videos

Supported video formats: MP4, WebM

The video pipeline should: open the video, extract/sample frames, run visual detection, evaluate risk, identify protected regions, maintain detection state across frames, blur/cover harmful regions, reconstruct the processed video, preserve video timing as much as reasonably possible.

### 2.3 In Scope — Browser

Chrome Extension using Manifest V3. The extension should: inspect images on web pages, detect dynamically loaded images, identify HTML5 video elements, process visual content, apply protection, avoid interfering with normal browsing when content is safe.

The extension is a **delivery layer**. The visual safety engine must remain independent from the browser-specific implementation.

---

## 3. Explicitly Out of Scope

Do NOT implement unless explicitly requested:
text moderation, OCR, speech recognition, audio moderation, speech-to-text, text-to-speech, chat moderation, website reputation analysis, URL classification, parental accounts, cloud dashboards, user authentication, payment systems, social media monitoring, behavioral tracking, child profiling, facial recognition, gender classification, identity recognition.

The current product is: **Image + Video Visual Safety**. Do not expand scope without explicit instruction.

---

## 4. Product Principle

Detection and action must remain separate.

```text
Visual Content → Detection → Context / Risk Assessment → Protection Policy → Visual Protection
```

A knife does not automatically mean harmful. It may appear in cooking, education, product demos, games, or violent scenes. The system should combine available visual signals to determine risk.

---

## 5. High-Level Architecture

```text
Web Content (Images/Videos)
  → Content Adapter (Image/Video Input)
  → Preprocessing (Resize, Normalize, Frame preparation)
  → AI Vision Engine (Object Detection, Safety Classification)
  → Risk Engine (Confidence, Context, Severity, Temporal state)
  → Policy Engine (SAFE, PROTECT, BLOCK)
  → Protection Engine (Blur, Pixelate, Cover)
  → Safe Output (Image/Video)
```

---

## 6. Core Components

### 6.1 Content Adapter
Receives visual content (image file, video file, HTML image, HTML video, canvas/frame). No AI logic.

### 6.2 Preprocessing Engine
Prepares content for inference: resizing, normalization, color conversion, frame extraction, frame sampling, tensor conversion. Isolated from model implementation.

### 6.3 AI Vision Engine
Produces structured detections only. Example output:
```json
{
  "detections": [
    { "label": "knife", "confidence": 0.91, "bbox": { "x": 320, "y": 180, "width": 120, "height": 80 } }
  ]
}
```
The AI engine should NOT directly blur anything.

### 6.4 Risk Engine
Converts detections into risk levels (LOW, MEDIUM, HIGH). Combines object type, confidence, scene classification, context. Must be configurable — no hardcoded business logic.

### 6.5 Policy Engine
Determines protection action: ALLOW, BLUR_REGION, BLUR_FRAME, BLOCK. Primary actions for current implementation: ALLOW, BLUR_REGION, BLUR_FRAME.

### 6.6 Protection Engine
Modifies visual content given image/frame + protected regions + protection type. Returns protected image/frame. Must not perform AI inference.

---

## 7. AI Detection Strategy

Use pretrained models initially. Do not train from scratch unless explicitly required. Consider: detection accuracy, inference speed, model size, CPU/GPU performance, browser compatibility, ONNX/WebGPU compatibility, licensing, supported classes, ease of deployment.

Potential technologies: YOLO, OpenCV, ONNX Runtime, ONNX Runtime Web, TensorFlow.js, MediaPipe. Benchmark candidates before finalizing.

---

## 8. Detection Categories

### Weapons
knife, gun, firearm, sword, other weapon categories

### Violent Visual Content
physical fighting, assault, violent interaction, dangerous physical scenes

### Graphic / Injury-Related Content
visible blood, severe injury, graphic wounds, disturbing injury scenes

Document actual model labels rather than inventing unsupported labels.

---

## 9. Object Detection vs Scene Classification

**Object-Level**: What object is present and where? → localized protection (blur region)
**Scene-Level**: What is happening visually? → full protection (blur entire image/frame)

Architecture must support both.

---

## 10. Image Processing Pipeline

```text
Image → Load → Validate → Preprocess → AI inference → Detection normalization → Risk assessment → Protection decision → Generate protected regions → Blur/cover → Return protected image + metadata
```

---

## 11. Image Detection Output Schema

```json
{
  "content_type": "image",
  "detections": [
    { "label": "knife", "confidence": 0.93, "bbox": { "x": 120, "y": 80, "width": 200, "height": 160 }, "risk": "medium" }
  ],
  "scene_risk": { "violence": 0.05, "graphic": 0.01 },
  "overall_risk": "medium",
  "action": "BLUR_REGION"
}
```

---

## 12. Image Protection

Bounding boxes should receive configurable padding (e.g., 10%). Do not permanently hardcode the padding value.

---

## 13. Scene-Level Protection

High-risk scene → Blur entire image. Do not produce fake bounding boxes for scene-level classification.

---

## 14. Blur Engine Requirements

Supported methods:
- **Gaussian Blur** — default, natural-looking
- **Pixelation** — alternative method
- **Solid/Overlay Cover** — stronger visual protection

Protection method must be configurable (e.g., `PROTECTION_MODE=gaussian`).

---

## 15. Video Processing Pipeline

```text
Video → Open → Read metadata → Extract/sample frames → Preprocess frame → AI inference → Risk assessment → Temporal consistency → Protection decision → Blur region/frame → Write processed frame → Repeat → Output video
```

---

## 16. Video Frame Sampling

Do not run expensive AI inference on every frame. Use configurable inference strategy (e.g., `INFERENCE_FPS=8` for a 30fps video). Between inference frames, use tracking or state propagation.

---

## 17. Temporal Consistency

Maintain temporal state to prevent flickering. Strategy: Detection → Tracking → Confidence smoothing → Temporal persistence → Protection state. Keep protection active for N frames unless evidence indicates content has disappeared. Algorithm must be configurable.

---

## 18. Video Tracking

Approaches: IoU-based matching, centroid tracking, OpenCV tracking, model-supported tracking, lightweight optical-flow. Start simple.

---

## 19. Video Protection Modes

**Region Protection**: frame + bbox → blur bbox
**Full-Frame Protection**: frame → blur entire frame

---

## 20. Video Output Requirements

Preserve: resolution, frame ordering, frame timing, playable format, acceptable quality. Audio should not be modified. Document limitations clearly.

---

## 21. Browser Extension Architecture

```text
Chrome
 ├── Content Script (Detect images, Detect videos, Observe dynamic DOM)
 ├── Extension Runtime
 └── Local Vision Engine (Detection, Risk, Protection)
```
Use Chrome Manifest V3. Use MutationObserver for dynamic content. Maintain processing state: UNPROCESSED → PROCESSING → PROCESSED → FAILED.

---

## 22. Browser Performance

Do NOT: run large models continuously, process every DOM mutation blindly, repeatedly analyze unchanged images, perform expensive inference on main thread.

Consider: Web Workers, Offscreen Documents, WebGPU, ONNX Runtime Web, model quantization, frame sampling, caching, throttling. Use only when justified by profiling.

---

## 23. Local-First Privacy

Prefer local processing. Avoid sending child browsing content to remote servers unless explicitly required. Do not introduce a backend simply for convenience.

---

## 24. Prototype vs Production

**Reference/Development Pipeline**: Python, OpenCV, YOLO, Streamlit — for evaluation, experimentation, debugging, benchmarking.

**Browser Production Pipeline**: TypeScript, Chrome Extension, ONNX Runtime Web, WebGPU, TensorFlow.js, Canvas, Web Workers — must not depend on Python runtime.

---

## 25. Model Abstraction

Create an abstraction (e.g., `VisionModel`) with: `load()`, `predict_image()`, `predict_frame()`, `get_metadata()`. Implementations: YOLOModel, SafetyClassifier, ONNXModel, BrowserModel. All expose normalized output.

---

## 26. Configuration

All thresholds must be configurable:
```text
OBJECT_CONFIDENCE_THRESHOLD, SCENE_RISK_THRESHOLD, BLUR_PADDING, VIDEO_INFERENCE_FPS,
TEMPORAL_PERSISTENCE_FRAMES, MAX_IMAGE_SIZE, MODEL_PATH, PROTECTION_MODE
```
Use configuration files/environment variables. No scattered magic numbers.

---

## 27. Safety Logic

Avoid simplistic rules. Use contextual risk assessment. Consider both false positives and false negatives. For child-safety, missing genuinely harmful content is critical.

---

## 28. Error Handling

Handle: model unavailable, invalid image, corrupt video, unsupported codec, inference failure, browser API unavailable, WebGPU unavailable, memory error, processing timeout. App must not crash. Define configurable safe fallback behavior.

---

## 29. Logging

Use structured logs (MODEL_LOADED, IMAGE_ANALYSIS_STARTED, RISK_DETECTED, etc.). Do not log private browsing content, image data, or personal information.

---

## 30. Testing Strategy

Testing is mandatory: Unit Tests, Integration Tests, Model Evaluation, Image Tests, Video Tests, Browser Tests, Performance Tests, Regression Tests.

---

## 31. Project Structure

```text
mykid/
├── SKILLS.md
├── README.md
├── docs/ (00-project-overview through 10-implementation-rules)
├── apps/ (extension/, dashboard/)
├── packages/ (vision/, risk/, protection/, image-processing/, video-processing/, shared/)
├── models/
├── tests/ (unit/, integration/, image/, video/, browser/)
├── scripts/
└── configs/
```

---

## 32. Development Phases

**Phase 1** — Repository Foundation: project structure, config, shared types, logging, error handling, test infrastructure
**Phase 2** — AI Model Research: evaluate candidates, document findings, select baseline model
**Phase 3** — Image Detection Engine: image → preprocessing → model → normalized detections
**Phase 4** — Image Risk Engine: detections → risk calculation → action
**Phase 5** — Image Protection Engine: detection → bbox → padding → blur → output
**Phase 6** — End-to-End Image Pipeline: connect all components, create test suite
**Phase 7** — Video Engine: video → frame extraction → sampling → detection → protection → output
**Phase 8** — Temporal Consistency: tracking, state, confidence smoothing, persistence
**Phase 9** — Video Quality and Performance: optimize inference, memory, encoding
**Phase 10** — Browser Image Protection: Manifest V3, discover image → AI → protect
**Phase 11** — Browser Video Protection: HTML5 video → frame sampling → AI → overlay
**Phase 12** — Browser AI Deployment: export model to ONNX Runtime Web / TF.js / WebGPU
**Phase 13** — Full Integration: validate image + video across core engine and browser
**Phase 14** — Testing and Hardening: regression, browser, model, performance, failure, privacy

---

## 33. Definition of Done

### Image
- [ ] Images can be loaded
- [ ] AI detection works
- [ ] Detection results are structured
- [ ] Risk is calculated
- [ ] Safe images remain unchanged
- [ ] Harmful objects can be blurred
- [ ] Full-frame harmful scenes can be protected
- [ ] Multiple detections work
- [ ] Bounding-box padding works
- [ ] Errors are handled
- [ ] Tests exist
- [ ] Metrics are documented

### Video
- [ ] Videos can be loaded
- [ ] Frames can be extracted
- [ ] Frame sampling works
- [ ] AI detection works
- [ ] Risk works
- [ ] Harmful regions are protected
- [ ] Full-frame protection works
- [ ] Temporal consistency works
- [ ] Blur does not excessively flicker
- [ ] Output video is playable
- [ ] Performance is measured
- [ ] Limitations are documented

### Browser
- [ ] Chrome Manifest V3 extension works
- [ ] Images can be discovered
- [ ] Dynamic images are handled
- [ ] Images can be protected
- [ ] HTML5 videos can be discovered
- [ ] Video protection works
- [ ] AI can run locally/browser-side where supported
- [ ] Extension does not unnecessarily break pages
- [ ] Performance is measured
- [ ] Browser limitations are documented

### Engineering
- [ ] Components are modular
- [ ] Configuration is centralized
- [ ] No unnecessary hardcoded thresholds
- [ ] Errors are handled
- [ ] Logs are useful and privacy-conscious
- [ ] Tests exist
- [ ] Documentation matches implementation
- [ ] No unnecessary backend exists
- [ ] No text/audio functionality has been introduced

---

## 34. Agent Rules

1. **Understand Before Coding** — Inspect repo, code, docs, dependencies, architecture before implementing.
2. **Implement Incrementally** — Foundation → Image → Video → Browser → Optimization → Testing. Each stage must remain runnable.
3. **Test Every Major Component** — Test immediately after implementing.
4. **Do Not Hide Errors** — No broad silent exception handling.
5. **Avoid Premature Complexity** — Start simple. No unnecessary microservices, databases, cloud infra.
6. **Keep AI Separate** — AI inference must not mix with UI, DOM, blur, or encoding code.
7. **No Magic Numbers** — All configurable values in one configuration system.
8. **Preserve a Working State** — Run tests and verify after each step.
9. **Document Architectural Decisions** — What, why, alternatives, limitations.
10. **Do Not Pretend the System Is Perfect** — Describe measurable performance and limitations honestly.

---

## 35. Required Agent Workflow

```text
1. Read SKILLS.md
2. Identify the relevant phase
3. Read relevant /docs file
4. Inspect existing implementation
5. Create an implementation plan
6. Implement the smallest complete step
7. Run tests
8. Run the application/demo
9. Verify output
10. Update documentation
11. Report what changed
```

---

## 36. Priority Order

```text
1. Correctness
2. Safety
3. Detection reliability
4. Protection reliability
5. Privacy
6. Performance
7. Maintainability
8. UI polish
```

---

## 37. Final Product Architecture

```text
                         MYKID
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Browser Layer              Evaluation Layer
       (Chrome Extension)            (Streamlit)
             │                           │
             └─────────────┬─────────────┘
                           │
                    Visual Safety API
                           │
             ┌─────────────┼─────────────┐
             │             │             │
        Image Engine   Video Engine   Shared Types
             │             │             │
             └─────────────┼─────────────┘
                           │
                    Vision Engine
                           │
                  ┌────────┴────────┐
                  │                 │
            Object Model      Scene Model
                  │                 │
                  └────────┬────────┘
                           │
                      Risk Engine → Policy Engine → Protection Engine → Safe Visual Output
```

---

## 38. Core Principle

```text
DETECT ≠ DECIDE ≠ PROTECT
```

This separation is fundamental to the MyKid architecture. Always preserve it.
