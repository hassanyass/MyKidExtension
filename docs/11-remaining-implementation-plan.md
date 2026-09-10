# MyKid — Remaining Implementation Plan (Phase-by-Phase)

> Restructured 2026-09-08. Per the project's own protocol (`03-development-phases.md` §Execution Protocol — plan → confirm → implement → test → document → advance), this file lists the **remaining chunks only**. Each chunk gets its own short, concrete implementation plan presented for confirmation immediately before it starts — this document is a map, not a pre-written solution.
>
> **Why phases 10/11 are split into sub-chunks:** the stated end goal is a browser extension that works on real, messy sites — YouTube's SPA navigation, infinite-scroll feeds, a custom video player, Shorts — not just a static demo page. Building that as one giant "Phase 10: browser extension" step would mean discovering YouTube's quirks only after everything else is already wired together. Splitting it lets each layer of complexity get proven and confirmed on its own before the next one is added.
>
> **Chosen order (2026-09-08):** Chunk 0 → Phase 10a → 10b → 10c → 11a → 11b → 11c → 11d → 12 → 13 → 14. Phase 9 (video perf) and 9b/9c (dashboard, test-data) are **parked** — the extension is the priority deliverable — and will be picked up later or interleaved once the browser track needs them (e.g. 9c's harmful fixtures become necessary once 10b needs something real to detect).
>
> ## ⚠️ Verification status — read this before trusting a ✅ below
>
> A ✅ in this document means **written and syntax/unit-checked**, not **observed working in a browser**. That distinction got blurred: 10c, 11a, 11b and 11c were each marked implemented and described as working while resting on a foundation (10b-iii's protection path) that had never been confirmed end-to-end. It hadn't been — a rescan bug meant the debug label silently analysed nothing, so protection was never observable at all.
>
> | Phase | Confirmed working in-browser |
> |---|---|
> | 10a — skeleton | ✅ yes (element counts) |
> | 10b-i — engine loads | ✅ yes (ready + timings) |
> | 10b-ii — detection maths | ✅ yes (matched Python to ~1px) |
> | 10b-iii — protection on page images | ✅ **yes** (2026-09-08 — people blurred on a real page) |
> | 10c — analysis on scroll / rescan | ✅ implied: images are analysed as they enter the viewport, and the rescan path is what made protection appear |
> | 10c — infinite scroll + SPA re-scan | ⚠️ not specifically exercised |
> | 11a / 11b / 11c — video | ⚠️ **logic verified, real sites not yet** (2026-09-10) — `devtest/video-check.html` runs the real module against a live `<video>`: discovery, frame capture, overlay placement, layering below player controls, persistence and expiry all pass. Not yet observed on YouTube itself. |
>
> Anything not ✅ is speculative until tested. Don't build further on it without checking.

---

## 🔄 Amended queue (2026-09-08) — supersedes the phase order below

Direction from the user, after seeing the image pipeline work:

1. **Scope:** browser images + video, YouTube and general browsing. **Cross-site hardening (11d) is deferred** — this is a demo, and breadth matters less than depth right now.
2. **Product goal:** the extension is a single **on/off toggle**. While on, genuinely harmful visual content is blurred — *blood, gore, violence, and the like*. Not `person`.
3. `person` was never a proxy for harm. It was a proxy for "the plumbing works". That job is done.

### The blocker this exposes

**The model cannot detect any of the things the product is for.** YOLOv11n is COCO-trained: 80 everyday object classes, of which exactly two (knife, scissors) are safety-relevant. Blood, gore, violence, nudity and weapons-beyond-knives are **not categories it has**. No threshold change reaches them; this needs different models.

This was documented from the start (`06-ai-detection.md` §6 "Known Limitations", and the scene classifier deferred in §5), but it was a footnote while the browser plumbing was being built. It is now the critical path.

**What already helps:** the risk engine has always accepted a `sceneRisk` with `violence` / `graphic` scores and escalates to full-frame blur at ≥0.6 (`risk_engine.py`, mirrored in `risk.js`). The offscreen document currently passes `null`. **The socket exists — nothing is plugged into it.** Scene-level harm needs no architectural change, only a classifier.

### Amended phase order

| # | Phase | Why here |
|---|---|---|
| **A** | ~~Harmful-content model research & selection~~ ✅ **Done (2026-09-08)** | Benchmarked and decided — see `06-ai-detection.md` "Phase A". Selected **`image-safety-classifier-xs`**: NSFL (gore/violence) + NSFW (sexual), 12.5MB, ~13ms, MIT, ONNX with preprocessing baked in. Reproduce: `python scripts/benchmark_safety_models.py`. |
| **B** | ~~Scene classifier wired into `sceneRisk`~~ ✅ **Done (2026-09-08)**, pending your in-browser check | The extension can now detect gore and sexual content. Details below. |
| **C** | **Expanded object detection** (weapons) ⏸️ **Researched, blocked** | Options found, none adoptable without decisions you own. See below. |
| **D** | ~~On/off toggle UX~~ ✅ **Done (2026-09-08)**, pending your in-browser check | The product control now exists. Details below. |
| **E** | **Verify video end-to-end** (11a–11c) | Video runs the same analysis path; the config bug blocked it too. Confirm before building further. |
| **F** | ~~Phase 12 — perf & quantization~~ ✅ **Done (2026-09-09)** | Quantization **rejected on evidence** — it made things worse. The real lever turned out to be elsewhere. Details below. |
| **G** | ~~Phase 13 — Python↔browser parity~~ ✅ **Done (2026-09-09)** | Extended to the scene classifier: tensor layout, channel order and label order now pinned across both languages. |
| **H** | ~~Phase 14 — hardening, privacy audit, limitations~~ ✅ **Done (2026-09-09)** | Privacy audit clean; `docs/12-privacy-and-limitations.md` and `docs/09-testing-evaluation.md` written. |

**Deferred:** 11d (cross-site), Phase 9 (video perf), 9b (Streamlit dashboard).

**Promoted to blocking:** Phase 9c (real harmful test fixtures). A gore or nudity detector cannot be validated against blank images and a bus photo. This now needs a decision on sourcing — see Open Decisions.

### Phase A — candidate landscape (starting point, not a conclusion)

| Harm category | Off-the-shelf options | Browser viability |
|---|---|---|
| **Nudity / sexual** | NudeNet v3 (ONNX, ~26MB, MIT) — returns **boxes**, so region blur works; NSFW MobileNet variants (~2.5–25MB) — classification only → full-frame | Good. NudeNet is the strongest fit and is already noted in `06-ai-detection.md` §2.4. |
| **Violence** | `jaranohaal/vit-base-violence-detection` (~350MB) — too heavy for browser, usable in the Python reference; smaller fine-tuned MobileNet/EfficientNet classifiers | Weak off-the-shelf. Likely needs a small fine-tune or a distilled model. |
| **Blood / gore** | Thinnest coverage of the three. Few credible pretrained models. | **The hard one.** Realistic routes: (a) CLIP zero-shot against text prompts like "blood", "injury" — flexible, no training, but the image encoder is ~25–40MB quantized; (b) fine-tune a small classifier on a gore dataset. |
| **Weapons (gun, rifle)** | YOLO fine-tuned on Roboflow/Kaggle weapon datasets | Good, but requires a training run — already flagged as deferred in `06-ai-detection.md` §2.3. |

**Honest expectation to set now:** nudity is well-served by existing models; violence is workable with effort; **blood/gore has the weakest off-the-shelf support and is the most likely to need training or a zero-shot approach**. A demo that covers nudity + weapons + a violence signal is realistic. One that reliably catches all gore is a research problem, and should be described that way rather than promised (SKILL.md rule 10).

### Phase B — Scene classifier wired in ✅ (2026-09-08)

**Built:**
- `packages/vision/scene_classifier.py` and `apps/extension/src/inference/scene-classifier.js` — the same model, same preprocessing rules, on both sides.
- `SceneRisk.sexual` added (`types.py`) and included in `max_score()`, mirrored in `risk.js`. Without this the NSFW output had nowhere to go and sexual content would have scored 0 — a silent safety failure, so it has its own test.
- Model vendored into the extension by `setup.js`; the offscreen document loads it as a second session.

**The pipeline now runs cheap-gate-first, as Phase A recommended:**

```text
image ──► scene classifier (~13ms)
             ├─ flagged  ──► BLUR_FRAME, object detection skipped entirely
             └─ clean    ──► object detection (~200ms) ──► region blur for knife/scissors
```

Skipping object detection on an already-flagged frame isn't just an optimisation: the verdict is whole-frame protection either way, so finding a knife inside an already-blurred frame changes nothing and costs ~200ms.

**Deliberate choice — `violence` stays 0.** The classifier folds violence into its NSFL score rather than reporting it separately. Putting a number in `SceneRisk.violence` that we never measured would misrepresent the model; the field is kept so a dedicated violence model can fill it later. Tested explicitly.

**Threshold left at 0.6, deliberately.** SKILL.md §27 argues missing harmful content is the worse failure, which points to lowering it. But tuning it without a harmful evaluation set is guessing, and an over-blurring extension gets switched off — which protects nobody. It stays at the documented default until Phase 9c provides data.

**Verified:** 159/159 tests, including 5 new cross-engine parity scenarios for scene risk (gore escalation, sexual escalation, below-threshold, scene-overrides-object, and the real measured bus.jpg scores staying ALLOW). Python and browser engines agree on all of them. The classifier scores 0.03–0.04 on every safe fixture against a 0.6 threshold, so false positives on ordinary content look unlikely — though that is a false-positive check only, not evidence it catches harm.

**Still unmeasured, and it matters:** detection accuracy on actual harmful content. No test here asserts it, because none can without an evaluation set. See Open Decisions.

---

### Phase D — On/off toggle ✅ (2026-09-08)

The extension is now the single control the product was described as: switch it on, harmful content gets blurred.

**"Off" means genuinely idle, not merely hidden.** Switching off tears down the IntersectionObserver, clears every overlay, empties the analysis queue and cancels video sampling timers. A tool that kept analysing every image while the user believed it was off would be dishonest about what it does with their CPU and their browsing — and switching it off is exactly what a suspicious user does to check.

**Design points:**
- **Defaults to on.** A safety tool that ships switched off protects nobody until someone remembers to enable it. Pinned by a test.
- **All tabs, not just the active one.** The toggle broadcasts to every open tab. Applying only to the tab in front would leave others unprotected while the popup reported "on".
- **Toolbar badge.** An `OFF` badge and hover text mean "is this protecting me right now?" is answerable without opening anything. Driven by a `chrome.storage.onChanged` listener rather than set alongside each write, so the badge can't disagree with the stored state.
- **Backstop in the service worker.** Analysis requests are refused outright while disabled. The content script already stops asking, so this should never fire — which is the point: "off means no images are analysed" shouldn't depend on one call site staying correct forever.
- **Advanced settings collapsed.** The debug label is a testing aid, not a product feature, so it moved into a collapsed `Advanced` section and is now described as *adding to* automatic gore/sexual detection rather than being the thing that makes protection work.
- **No stats while off.** The panel reports "Not being checked" instead of analysis counts, which would imply work that isn't happening.

**Verified:** 161/161, including new toggle-semantics tests (default-on, off survives the message path the debug label silently broke, off preserves other settings, partial overrides merge rather than replacing whole sections).

---

### Phase C — Weapon detection ⏸️ blocked, deliberately (2026-09-09)

Researched rather than built, because every available route needs a
decision that isn't mine to make.

**What exists:** several community weapon-detection models on Roboflow
Universe (rifle/handgun/knife sets, 3k–9k images each). All are
YOLOv8-derived.

**Why none were adopted:**
- **Unverifiable quality.** They are community-trained with no independent
  benchmark. Dropping an unvetted model into a child-safety tool while
  having no evaluation set to judge it against would mean claiming a
  capability nobody has measured — the specific thing SKILL.md rule 10
  prohibits, and the thing this project has already been bitten by.
- **Access is gated.** Roboflow downloads need an account and API key, so
  acquisition is yours.
- **Licensing compounds.** Each is YOLOv8-derived and therefore AGPL,
  pulling against the option (now live) of dropping AGPL entirely by
  disabling object detection.
- **Cost stacks.** Another ~10–25MB and another inference pass per image,
  on top of a budget where object detection is already ~4× the classifier.

**Recommendation:** leave firearms as a documented gap for the demo. The
scene classifier already covers gore and sexual content, which is the bulk
of the harm surface. Revisit once an evaluation set exists — with data, a
weapon model can be judged instead of hoped about.

---

### Phase G — Cross-language parity extended ✅ (2026-09-09)

Scene-classifier preprocessing is now pinned across both implementations
(`tests/integration/test_scene_parity.py`): tensor length, 0-255 scale
(catching a stray `/255` on either side), per-channel means (catching
channel swaps and NHWC/NCHW mix-ups), leading values, and label order.

The fixture deliberately gives each channel a distinct distribution — a
flat or grey image would pass even with red and blue swapped. That class of
bug produces three plausible probabilities from a wrong tensor and breaks
nothing visibly, which is exactly why it needs a test rather than a read.

Required splitting `packPixels()` out of `preprocess()` so it can be
exercised without a canvas. One bug found and fixed while writing it:
`Math.max(...tensor)` overflowed the call stack on 150k elements.

---

### Phase H — Hardening, privacy audit, limitations ✅ (2026-09-09)

**Privacy audit result: clean.** Two `fetch()` calls, both for the image
being analysed. No telemetry, no external hosts, no remote config, no model
CDN. Logs carry counts, actions and risk levels — never URLs, image data or
page text; error messages carry an HTTP status only. Storage holds one key:
the on/off flag and the debug label.

**Two things surfaced rather than buried** (`docs/12`): images are fetched a
*second* time by the offscreen document, so servers see duplicate requests
— that's the mechanism that defeats canvas tainting; and `<all_urls>` is
genuinely broad, which is why the audit is written down as evidence rather
than asserted.

**Also written:** `docs/09-testing-evaluation.md` (what is tested, and the
larger section on what deliberately isn't).

---

### Phase F — Performance & quantization ✅ (2026-09-09) — a negative result

**Hypothesis:** INT8 quantization would shrink the models and speed up inference, easing the ~230ms/frame in-browser budget.

**Measured (isolated processes, 25 runs after warmup, `intra_op_num_threads=2`):**

| Model | Size | Median latency | Verdict |
|---|---|---|---|
| safety-classifier fp32 | 12.53 MB | 17.4 ms | — |
| safety-classifier INT8 | 3.74 MB (−70%) | 17.3 ms | no gain |
| yolo11n fp32 | 10.14 MB | 46.5 ms | — |
| yolo11n INT8 | 2.74 MB (−73%) | 69.8 ms | **50% slower** |

**Rejected.** Large size savings, but no speedup for the classifier and a substantial regression for YOLO. Dynamic INT8 has no optimised kernel path for these architectures on this runtime, so dequantisation overhead dominates.

**And it moves decisions, not just numbers.** Quantized YOLO found the same five objects on `bus.jpg` at the same positions, but with confidences shifted enough to cross the 0.5 threshold — one person scored 0.399 fp32 and 0.575 INT8. It flipped *toward* detection there, but nothing guarantees the direction; a threshold-crossing shift is exactly what you cannot accept in the layer deciding whether a child sees something.

Size mattered less than assumed anyway: models are bundled and loaded once per session, not fetched per page. Trading latency for download size is the wrong way round here.

**Honest caveat:** measured with native `onnxruntime`. The browser's wasm backend may have different INT8 kernel coverage, so this doesn't prove quantization is useless in-browser — only that there's no evidence for adopting it, and the burden of proof sits with adoption.

**The real performance lever, found while measuring:** object detection is ~4× the classifier's cost and contributes least to the product's purpose.

| Configuration | Native | Est. in-browser (~5.5× wasm penalty) |
|---|---|---|
| classifier + YOLO | ~53 ms | ~300 ms |
| classifier only | ~13 ms | ~70 ms |

Turning YOLO off is a **~4× throughput improvement**, and it simultaneously removes the only AGPL-licensed model from the stack. The cost is losing knife/scissors *region* blur — while gore and sexual content, the actual targets, are unaffected because the classifier handles them. `detection.objectDetectionEnabled` already exists for this; the default is unchanged because dropping a detection capability is a product call, not mine.

---

**Size budget matters:** the extension already carries ~14MB wasm + ~10MB YOLO. Each added model compounds it, and every model runs per image at ~230ms. This is why Phase F (quantization) follows directly, and why "run every model on every image" is not the design — expect a cheap gate first, heavier models only on suspicious content.

---

## Chunk 0 — Fix the known bug + correct stale docs ✅ Done (2026-09-08)
- `packages/shared/config.py:56`: `inference_fps` default `30` → `8` (matches `configs/default.yaml`, README, and the test) — full suite now 142/142 passing
- Updated the stale "Not Started" status tables in `03-development-phases.md` and `README.md` to reflect Phases 1–8 actually being done, and marked Phase 9 "Parked" / Phase 10 "In Progress"

---

## Remaining phases

### Phase 9 — Video Quality & Performance
Full-pipeline benchmarking (decode → inference → tracker → blur → encode) on real clips, not just raw model speed. Sweep `inference_fps` / `temporal_persistence_frames`, profile memory, document recommended defaults with evidence.

### Phase 9b — Streamlit Dashboard *(closes a gap — this was a Phase 6 deliverable that never got built)*
`apps/dashboard/app.py`: before/after view for images and video, config sliders, doubles as the QA tool for every phase after this one.

### Phase 9c — Test Data Fixtures
Source or create actual harmful test images (and eventually short clips) — currently `test-data/images/harmful/` and both video folders are empty. Needs a provenance/licensing decision from you before anything goes in the repo (see Open Decisions below).

---

### Phase 10a — Browser Extension Skeleton ✅ Implemented (2026-09-08), pending your manual load-check
**Goal:** Manifest V3 scaffold only — `manifest.json`, background service worker, popup, a content script that just counts/logs discovered `<img>`/`<video>` elements. **No AI yet.**
**Built:** `apps/extension/` — see [`apps/extension/README.md`](../apps/extension/README.md) for what's there and how to load it.
**Verification:** manifest JSON and all JS validated for syntax. **Still needs you** to `chrome://extensions` → Developer mode → Load unpacked → confirm the popup shows correct image/video counts on a plain page and on youtube.com — loading an unpacked extension goes through a native OS file picker, which isn't something I can drive from here.

### Phase 10b — Real In-Browser Detection (static pages)
Split into three sub-chunks once the real engineering risk areas became clear (Worker/CSP plumbing, YOLO pre/postprocessing correctness, and cross-origin image access are three separable failure modes — better to prove each in isolation than debug all three at once):

#### 10b-i — Detection-engine plumbing proof ✅ Implemented (2026-09-08), pending your manual check
**Goal:** prove the whole chain works before touching a single real pixel: content script → spawn Web Worker → load vendored `onnxruntime-web` (wasm-only, single-threaded — see note below) → load `models/yolo11n.onnx` → run one inference call on a blank tensor.
**Built:** `apps/extension/src/inference/detector.worker.js`, `apps/extension/scripts/setup.js` (vendors the model + wasm runtime, kept out of git like `models/*.onnx` already was), manifest updated with `web_accessible_resources` and a `wasm-unsafe-eval` CSP. Popup now shows "Detection engine: ready" with load/inference timings and the model's output tensor shape.
**Design note:** forced `numThreads = 1` — onnxruntime-web's multi-threaded wasm path needs the *host page* to be cross-origin isolated (COOP/COEP headers), which this extension has no control over on arbitrary sites. Single-threaded is slower but actually loads everywhere, which matters more than speed at this stage given the "works on YouTube, works everywhere" goal — threading can be revisited in Phase 12 if profiling says it's worth the added fragility.
**Verification:** manifest/JS syntax validated; `setup.js` run end-to-end successfully. **Still needs you**: run `node apps/extension/scripts/setup.js`, reload the extension, reload a page, and confirm the popup shows "Detection engine: ready" with a plausible output shape (something like `[1, 84, 8400]` for a YOLO11 COCO model) rather than "failed".

**Two real bugs found during manual testing on youtube.com** (2026-09-08), both instructive for the "works everywhere" goal — this is exactly the kind of thing sub-chunking was meant to surface early:

1. `new Worker(chrome.runtime.getURL(...))` from the content script throws `SecurityError: Failed to construct 'Worker'` — the Worker constructor enforces same-origin between the page and the worker script, and `chrome-extension://...` is cross-origin from `https://www.youtube.com`. `web_accessible_resources` permits `fetch()`/`importScripts()` to reach a file but not `new Worker()` to load it directly.
2. The standard workaround for #1 (fetch the worker source, wrap it in a `Blob`, construct the worker from the resulting `blob:` URL, which inherits the page's origin) **still failed** — YouTube's own page CSP (`script-src`, used as the `worker-src` fallback) doesn't include `blob:` in its allowed sources, so the browser blocks worker creation outright. This is enforced by the *host page*, not the extension — no extension-side CSP change can fix it, and other strict sites will hit the same wall.

**Actual fix:** stop trying to run inference in the content script's page context at all. Moved it into the **background service worker** instead (`service-worker.js` now loads onnxruntime-web + the model and runs the self-test directly; `content-script.js` just does `chrome.runtime.sendMessage`). The service worker runs under the extension's own origin/CSP, completely decoupled from any host page's policy — this generalizes to every site, not just a YouTube-specific patch, and it was already off the page's main thread so no nested Worker is needed. Removed the now-unused `web_accessible_resources` block and the standalone `detector.worker.js` file.

**Third bug**, also found via manual testing: `importScripts()` inside an MV3 service worker is only valid during the worker's initial *synchronous* evaluation — calling it lazily inside an async message-handler callback (the first version of the service-worker fix did exactly this) throws `Failed to execute 'importScripts' ... script failed to load`, even though the file is present and the identical call works fine at the top level. **Fixed** by hoisting `importScripts(...)` (and the `ort.env.wasm.*` config that depends on it) to the top level of `service-worker.js`, executed unconditionally when the file first loads; only the genuinely-async `ort.InferenceSession.create(...)` stayed lazy.

**Fourth bug, and the deepest one:** with the load-order issue fixed, `ort.InferenceSession.create(...)` itself failed inside the service worker with `no available backend found. ERR: [wasm] TypeError: import() is disallowed on ServiceWorkerGlobalScope by the HTML specification`. onnxruntime-web's wasm backend uses a dynamic `import()` internally — and that's not a configuration problem, it's a hard platform restriction: the HTML spec disallows dynamic `import()` inside `ServiceWorkerGlobalScope` entirely, in every browser. **No service worker can run onnxruntime-web's wasm backend, full stop.**

**Actual fix:** moved inference into a **[chrome.offscreen](https://developer.chrome.com/docs/extensions/reference/api/offscreen) document** — a hidden, extension-owned page (`src/offscreen/offscreen.html` + `offscreen.js`) that's a normal DOM/Window context (dynamic `import()` works fine there) while still running under the extension's own origin/CSP (immune to host-page CSP, same as the service worker was). Final architecture:

```text
content script --(MYKID_SELF_TEST)--> background service worker
                                          |
                                          |-- ensures the offscreen document exists
                                          |     (chrome.offscreen.createDocument)
                                          |
                                          `--(MYKID_RUN_SELF_TEST)--> offscreen document
                                                                        (loads onnxruntime-web +
                                                                         the model, runs inference)
```

The service worker is now a thin router; all actual model work happens in the offscreen document. Added the `offscreen` permission to `manifest.json`.

**Fifth bug — this one self-inflicted, not a platform constraint:** with the offscreen document in place, the dynamic `import()` was finally *allowed*, and promptly failed with `Failed to fetch dynamically imported module: .../ort-wasm-simd-threaded.mjs`. Cause: `setup.js` vendored only two of the three files onnxruntime-web actually needs. The real load chain is:

```text
ort.wasm.min.js             -- API surface; dynamically imports ->
ort-wasm-simd-threaded.mjs  -- Emscripten glue module; fetches ->
ort-wasm-simd-threaded.wasm -- the compiled runtime
```

The `.wasm` filename is never referenced by `ort.wasm.min.js` at all — the `.mjs` glue is what loads it — so omitting the glue breaks the chain in the middle and surfaces as a misleading `no available backend found`. **Fixed** by vendoring all three files in `setup.js`, which now also fails loudly if any expected file is missing from the package's `dist/` rather than silently producing a broken extension. Verified by grepping the library for every referenced `.mjs`/`.wasm` filename and every dynamic `import()` call site — there is exactly one, and it now resolves.

**Lesson worth keeping:** vendoring a wasm library means shipping its whole load chain, not just the binary. Check what the entry file actually imports before assuming a file list — this cost four test cycles that one `grep` would have prevented.

#### 10b-ii — Real image pre/postprocessing ✅ Implemented (2026-09-08), pending your manual check
**Goal:** letterbox resize + normalize a real image, run it through the model, decode the raw output (NMS, class labelling, map boxes back to original image coordinates).

**Built:**
- `apps/extension/src/inference/detector.js` — letterbox (gray-114 padding, centered), RGBA→NCHW float32 normalization, `[1, 84, 8400]` output decoding, greedy per-class NMS at IoU 0.7, coordinate mapping back to original image space. Exposes a `MyKidDetector` global. Produces structured detections *only* — no risk logic, no blurring (SKILL.md §38).
- `scripts/parity_reference.py` — Python ground truth running the **same ONNX file** with the **same hand-written letterbox/decode maths**, so a mismatch points at a specific stage instead of just "the two disagree". Reusable for Phase 13.
- `scripts/fetch_detection_samples.py` — the project's own `test-data/images/safe/` fixtures are blank and produce **zero detections**, which cannot validate box decoding at all (a pipeline with broken coordinate maths passes them just as well as a correct one). This copies the standard sample images bundled with the installed `ultralytics` package — AGPL-3.0, so gitignored and regenerated, consistent with how the model itself is handled.
- Popup "Run detection test" button → routes through the service worker to the offscreen document, runs the full pipeline on the sample image, renders the detections.

**Reference output** (`python scripts/parity_reference.py test-data/images/detection-samples/bus.jpg --conf 0.25`), cross-checked against Ultralytics' own pipeline (same 5 objects, boxes within a few px; small confidence deltas expected between the `.pt` and the exported `.onnx`):

```text
letterbox: scale=0.592593 padX=80 padY=0
bus        0.9392   11.9, 228.4, 787.3, 506.8
person     0.9020   48.6, 398.0, 194.6, 506.6
person     0.8493   670.6, 392.6, 139.4, 487.0
person     0.8328   223.1, 405.6, 122.1, 454.1
person     0.3993   0.0, 550.2, 66.0, 321.6
```

**Verification:** all JS/manifest validated, Python suite still 142/142. **Still needs you**: click "Run detection test" in the popup and confirm the browser reproduces the table above. Sub-pixel differences are fine (canvas scaling vs. `cv2.INTER_LINEAR`); different labels, counts, or boxes off by more than a few px are not.

#### 10b-iii — Wire into the real page ✅ Implemented (2026-09-08), pending your manual check
**Goal:** for each real `<img>` on the page, run 10b-ii's pipeline, then a ported risk engine and protection engine, protecting flagged images in place.

**Built:**
- `src/shared/config.js` — ported thresholds, mirroring `configs/default.yaml` (SKILL.md §26: no scattered magic numbers).
- `src/inference/risk.js` — port of `risk_engine.py`: same ordering, thresholds, coverage escalation and box padding.
- `src/content/protection.js` — DOM protection via **positioned overlays, never mutating the image**. Rewriting `src` or canvas-replacing pixels would fight the page (SPAs reset `src` on navigation, lazy-loaders reassign it, `srcset` re-picks on resize) and is impossible for cross-origin images anyway.
- `src/content/content-script.js` — discovery, per-image state (`QUEUED → PROCESSING → PROCESSED | SKIPPED | FAILED`, SKILL.md §21), bounded-concurrency queue.

**Two design decisions worth recording:**

1. **Canvas tainting sidestepped entirely.** The plan assumed cross-origin images would be unreadable due to canvas tainting, forcing us to skip most real-world images. Instead the content script passes the image *URL* to the offscreen document, which fetches and decodes it with the **extension's own host permissions** — page-origin CORS rules simply don't apply. This is strictly better than the anticipated fallback. `blob:` URLs remain genuinely unfetchable (they're origin-scoped) and are skipped; that's the honest residual limitation, revisited in 10c.
2. **Performance is the real constraint, not correctness.** Measured inference is ~230ms/image, and a YouTube feed carries 90–125 images — naively that's 20–30s of solid compute. Mitigations: a 128px minimum dimension filter (icons/avatars/tracking pixels dominate real pages and can't meaningfully carry harmful content at that size), concurrency capped at 2, and a 40-image-per-page safety valve. All in config, all revisited in 10c/Phase 12.

**Verified:** browser and Python risk engines produce **identical verdicts** across 7 scenarios including padded box coordinates and coverage escalation — now pinned by `tests/integration/test_risk_parity.py` (+ `scripts/risk_parity_runner.js`), which fails if the port ever drifts. This is the first concrete piece of Phase 13, landed early because 10b-iii is what created the duplication. Suite: 144/144.

**Test affordance:** COCO only flags knife/scissors, which essentially never appear while browsing, and there are still no harmful fixtures (Phase 9c) — so the DETECT→DECIDE→PROTECT chain is impossible to *observe* working. The popup now has a "treat as harmful" selector (person/car/dog/cat/bus) that makes the risk engine flag a common object instead. It changes no production default; it's a debugging lens.

**Still needs you:** pick e.g. `person` in the popup, reload a page with people in it, and confirm blur overlays land on them and track the layout on scroll/resize.

### Phase 10c — Dynamic Content & Feed Compatibility ✅ Implemented (2026-09-08), pending your manual check
**Goal:** work without a page refresh — lazily-loaded images, infinite-scroll grids, SPA navigation.

**Driven by testing feedback:** protection "worked partially", which turned out to be self-inflicted rather than mysterious:
- A **40-image-per-page cap** meant most of a ~94-image feed was never analysed at all.
- A **128px minimum on both dimensions** rejected legitimate grid thumbnails, which are often wide but short.
- `backdrop-filter` overlays depend on the stacking context they land in and **silently fail** on some real layouts.

**Built:**
- **`MutationObserver`** — images added after load (infinite feeds) are picked up, plus `src`/`srcset` attribute changes, which is how most lazy-loaders work (they swap the attribute on an existing element rather than inserting a new one).
- **`IntersectionObserver`** — only images at/near the viewport are analysed, with a 200px margin so protection lands before the image is visible. This is what makes the per-page cap unnecessary: cost scales with what's on screen (~10-20 images), not with page size. Cap removed; size floor lowered 128 → 64px.
- **SPA navigation detection** — `popstate`, YouTube's `yt-navigate-finish`, plus a 500ms URL poll as the general fallback. (Patching `history.pushState` from a content script does *not* work: the isolated world has its own copy and the page's own calls bypass it.)
- **Real protected pixels** (`src/inference/protection-render.js`) — the offscreen document already holds the decoded bitmap, so it renders the protected image there and hands back a finished image the content script simply positions. Removes the stacking-context fragility entirely and enables **true pixelation**, which CSS couldn't do.
- Debug label changes now **apply immediately** via a `MYKID_RESCAN` message instead of requiring a reload.

**Two bugs caught while writing this, worth recording:**
1. `IntersectionObserver.observe()` on an already-observed target is a **no-op that delivers no new entry** — so a lazy-loader swapping `src` on a watched element would never be re-queued. Fixed by unobserving first.
2. On SPA navigation the obvious move is to clear every overlay — but that briefly *unprotects* images still on screen and still harmful. Now only overlays whose image left the DOM are pruned. For a child-safety tool, protection lingering a moment too long beats coming down a moment too early.

**Still needs you:** on YouTube, scroll the feed (new thumbnails should get protected as they come into view, no refresh) and click into a video (navigation should trigger a rescan without a reload).

---

### Phase 11a — Browser Video: Basic Overlay ✅ Implemented (2026-09-08), pending your manual check
**Goal:** sample frames from playing `<video>` elements, analyse them, and protect harmful regions.

**Built:** `src/content/video-protection.js` — frame sampling, analysis, region overlays, temporal persistence, fullscreen handling.

**Four design decisions that video forced, each different from the image path:**

1. **Frames are captured, not fetched.** The image path passes a URL to the offscreen document, which fetches it with extension privileges. That's impossible here: YouTube and most streaming sites feed video through MSE `blob:` URLs, which are origin-scoped and unfetchable. So frames are drawn from the element in page context and shipped as `data:` URLs — which the existing analysis path accepts unchanged, no new plumbing.
2. **Region crops, not a full-frame overlay.** Overlaying the whole rendered protected frame would visibly freeze the video to the 2fps sample rate. Instead each region gets its own overlay showing that region's *crop* of the protected frame (via `background-position`), so harmful areas are obscured while the rest keeps playing at full framerate.
3. **`position: fixed`, and the layer follows fullscreen.** The fullscreen renderer only paints the fullscreen element's subtree, so an overlay parented to `<body>` **disappears exactly when the video is largest**. That's a safety failure, not a cosmetic one, so the layer re-parents into the fullscreen element on `fullscreenchange` rather than deferring this to 11c.
4. **2fps, not the config's 8.** `configs/default.yaml` specifies `inference_fps: 8` for the Python pipeline, which processes offline. In-browser inference measures ~230ms/frame, so 8fps would need a 125ms budget — arithmetically impossible. Browser video config is separate and set to 2fps, with a documented reason rather than a rate we can't hit.

**Temporal persistence** (`temporalPersistenceMs: 2000`) carries protection across frames where detection misses, so a single dropped frame doesn't flash content into view — the browser equivalent of `temporal_persistence_frames` (SKILL.md §17), expressed in time because sampling here is time-based rather than frame-indexed.

**Known limitation, failing open:** cross-origin video without CORS headers taints the capture canvas, and `toDataURL` throws. Those elements are marked unreadable and skipped rather than retried every 500ms. MSE-fed players like YouTube are generally readable because the page supplied the bytes itself. The popup reports unreadable frame counts instead of hiding them.

**Still needs you:** play a YouTube video with the debug label set to `person`, confirm regions get blurred while the rest of the picture keeps playing, and check that protection survives entering fullscreen.

---

### Phase 11a — Browser Video: Basic Overlay
**Goal:** single `<video>` element on a plain test page, canvas overlay positioned on top, frame-sampling loop (`requestVideoFrameCallback`), basic region/full-frame protection.
**Verification:** overlay tracks the video correctly through play/pause/seek/resize.

### Phase 11b — SPA / Navigation Compatibility (YouTube core) ✅ Implemented (2026-09-08), pending your manual check
**Goal:** survive client-side navigation — the reused `<video>` element, stale overlays, and leaked timers/listeners.

**The central problem:** on YouTube, navigating between videos does **not** create a new `<video>` element. The same element is reused and its MSE source is swapped underneath it. Element identity never changes, so nothing in 11a would have noticed — the extension would have kept showing the *previous* video's protection over completely different content. Fixed by listening for `loadstart` / `emptied` / `loadedmetadata` on each watched element and resetting protection when they fire.

**Also built:**
- **Removal cleanup.** The MutationObserver previously only handled `addedNodes`. Videos hold a sampling timer, three event listeners and an IntersectionObserver registration — none of which were being released. Every SPA navigation leaked a set, accumulating for as long as the tab stayed open. `removedNodes` is now handled too.
- **Sampling gates.** Frames are only analysed when the video is playing, on screen (`IntersectionObserver`), and in a visible tab. YouTube keeps several `<video>` elements around; sampling paused or off-screen ones is exactly the waste that makes an extension feel heavy.
- **Self-rescheduling sample loop**, replacing `setInterval`. Analysis takes ~230ms+ against a 500ms interval at 2fps, so on a slower machine fixed intervals would queue work faster than it completes. Scheduling the next wait only *after* the previous analysis finishes makes the rate self-limiting.

**Opposite navigation policies for images and video, deliberately:**

| | On navigation | Why |
|---|---|---|
| **Images** | Keep overlays; prune only detached ones | A blanket clear briefly unprotects images still on screen and still harmful. Stale-but-protecting is the safer failure. |
| **Video** | Clear overlays immediately | The element persists while its content is replaced wholesale, so old protection would blur the *wrong video*. Sampling re-establishes it within ~500ms. |

**Still needs you:** with the debug label on `person`, navigate between YouTube videos via in-page links and confirm protection resets and re-establishes for each new video rather than carrying over — and that it still works after several navigations (the leak fix).

### Phase 11c — YouTube-Specific Player Quirks ✅ Implemented (2026-09-08), pending your manual check
**Goal:** player chrome, quality switches, ads, theater/fullscreen, Shorts, miniplayer.

**Two of these turned out to be defects in 11a/11b rather than polish:**

1. **The overlay covered the player's own controls.** The layer sat at `z-index: 2147483647` on a page-level container — above YouTube's entire chrome — so a full-frame blur obscured the scrubber and buttons. Clicks still passed through (`pointer-events: none`), but the player was visually unusable. **Fixed** by giving each video its own layer *inside the player container*, at `z-index: 5`: above the picture, below the site's controls (which sit in the tens).
2. **Quality switches dropped protection.** 11b cleared protection on `loadstart`/`emptied`, but those also fire for quality changes and rebuffering, where the content hasn't changed at all — briefly exposing content already judged harmful. **Fixed** by leaving protection standing on media-source events and re-verifying quickly (150ms) instead. An ad showing stale blur for a moment is cosmetic; unblurring harmful content is a safety failure. Genuine navigation is still handled by URL change, where clearing *is* correct.

**Also handled:**
- **Fullscreen simplified, not special-cased.** Since the layer now lives inside the player container — which is what actually goes fullscreen — the `fullscreenchange` re-parenting hack from 11a is gone. Only a reposition is needed.
- **Miniplayer / theater transitions.** YouTube moves the `<video>` between containers. The layer now re-attaches when the video's parent changes, instead of being left behind rendering protection in the wrong place.
- **Shorts.** The visibility test needed to satisfy two opposite shapes: a stacked Shorts feed where adjacent videos peek in at the edges (needs a *ratio* test to exclude them), and a player taller than the viewport that fills the screen at a low ratio (needs an *absolute area* test, or a maximised video would be judged invisible and never analysed). Both tests now apply.
- **Ads** need no special handling and deliberately get none: an ad is content like any other and should be analysed, so sampling simply continues through it.

**Still needs you:** on YouTube with the debug label on `person` — check the scrubber/controls stay visible over a blurred video, toggle fullscreen and theater mode, change quality mid-video (protection should *not* flicker off), and scroll a Shorts feed.

---

## Bug: the debug label never blurred anything (found 2026-09-08, fixed)

Reported as "person is never blurred". It was a real defect, and it invalidated the testing route for everything from 10c onward — which is why 11a–11c went unverified while being described as working.

**Cause:** the `MYKID_RESCAN` handler cleared per-image state and then called `observe()` again on the same elements. `IntersectionObserver.observe()` on an **already-observed target is a no-op that delivers no entry**, so nothing was ever re-queued and no image was re-analysed. Changing the dropdown appeared to do nothing because it genuinely did nothing.

Notably this is the *same* mistake already identified and fixed in `onImageChanged` during 10c — fixed in one call site, missed in the other.

**Fix:** the rescan now disconnects the observer entirely and rebuilds it, guaranteeing fresh entries for every image.

**Second cause, found once diagnostics existed (the real one):** with the rescan fixed, a page reported `Detected: person 58` and `Protected: 0`. Detection was fine; the risk engine never saw the setting. `MyKidConfig.load()` was being called **inside the offscreen document**, which gets only a restricted subset of the extension APIs — `chrome.storage` among the casualties. The failure was swallowed by a bare `catch {}` that returned defaults, so `treatLabelAsHarmful` was silently `null` and the harmful set stayed `[knife, scissors]`. Exactly the silent error-hiding SKILL.md rule 4 prohibits, and it cost two rounds of debugging.

**Fix:** the offscreen document no longer reads config at all. The **service worker** — which does have `chrome.storage` — reads the overrides and passes them in the message; the offscreen document builds its config from that via `MyKidConfig.fromOverrides()`. This removes the guesswork about which APIs are reachable where, rather than betting on it. `load()` now also warns loudly instead of failing silently.

**Diagnostics that made this findable:** the popup reports every label the model saw *and* the harmful set the engine actually applied. `Detected: person 58` next to `harmful: knife, scissors` states the bug outright, where a bare `Protected: 0` had been consistent with a dozen different faults.

**Guards added so this isn't re-asserted without evidence:**
- `TestDebugLabelOverride` in `tests/integration/test_risk_parity.py` — pins that `person` is ignored by default, *is* protected when the override is set, and that the override **adds to** rather than replaces the real safety labels (a regression there would silently disable knife/scissors detection while looking fine).
- **Popup diagnostics.** The offscreen document returns every label the model saw plus the harmful set actually applied, and the popup reports both with a plain-language diagnosis. "No images analysed yet", "analysed but the model recognised nothing", and "objects found but none match the harmful set (…)" are three completely different faults that previously all presented as a silent zero.
- `TestConfigDelivery` in the same file — pins that config reaches the risk engine **when `chrome.storage` is unavailable**, reproducing the offscreen document's actual environment, and that losing config access can never silently disable knife/scissors. Degraded settings are tolerable; degraded safety is not.

### Phase 11d — Cross-Site Hardening ⏸️ **Deferred (2026-09-08)**
Deprioritised for the demo: depth on YouTube + general browsing matters more than breadth right now. The design is already site-agnostic (no YouTube-specific selectors — `yt-navigate-finish` is an additive fast path alongside a generic URL poll), so this is validation work rather than new architecture, and can be picked up later without rework.

---

### Phase 12 — Browser AI Deployment Hardening
Quantization (INT8) and Worker-perf tuning on the model — most of the core wiring already lands in 10b; this is the size/latency optimization pass once real usage data exists.

### Phase 13 — Full Integration
Python-vs-browser parity test suite (headless browser, same fixtures), end-to-end extension test across the sites validated in 11d.

### Phase 14 — Testing & Hardening
Final Definition-of-Done pass against SKILL.md §33, privacy audit (no network calls, no leaked image data in logs), write the remaining docs (`09-testing-evaluation.md`, `10-implementation-rules.md`), document known limitations honestly.

---

## Open decisions needed from you (not engineering calls)

- **Harmful test-data sourcing — now blocking, not optional.** A gore or nudity detector cannot be validated against blank fixtures and a photo of a bus. Established academic/benchmark datasets exist for NSFW and violence detection, generally under access agreements requiring you to register as the responsible party. That acquisition is yours to make: I can wire up whatever you provide and build the evaluation harness around it, but I won't source, generate, or synthesise harmful imagery. Note also that some categories (particularly anything involving minors) are legally restricted regardless of research intent — those must never be collected, and detection there should rely on established third-party services rather than a local model.
- **AGPL-3.0 on YOLOv11n** — resolve before distributing publicly. Note this compounds as models are added: each new model brings its own licence, and NudeNet (MIT) mixes fine while other candidates may not.
- **Accuracy expectations for a child-safety tool.** False negatives (harmful content shown) and false positives (ordinary content blurred, making the extension annoying enough to switch off) both matter, and they trade against each other. SKILL.md §27 says missing genuinely harmful content is the more serious failure — worth confirming that's still the stance, since it implies deliberately over-blurring.
- **~~Scene-level classifier~~** — resolved by the amended queue: it's now Phase B, not deferred.

---

## How we'll proceed from here

For each remaining chunk: I write a short concrete plan (goal, deliverable, verification) → you confirm → I implement just that chunk → test → document → report → move to the next one. Nothing beyond Chunk 0 gets implemented without a confirm.
