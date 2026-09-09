# Fuzzy — Technical Report

> How this project works, why it is built the way it is, and what it
> genuinely does and does not do.
>
> **Part 1** explains it in plain language, no background assumed.
> **Part 2** is the technical account, from an empty repository to a
> working Chrome extension.
>
> Written 2026-09-09. Repository: `MyKidExtension`. Shipped extension name:
> **Fuzzy**.

---
---

# Part 1 — In plain terms

## What problem this solves

A child browsing the web will eventually land on a picture or video they
shouldn't see. Not because they went looking — a search result, a video
thumbnail, an autoplaying clip in a feed.

Existing answers are mostly blunt: block a whole site, or allow it. That
either shuts off things a child legitimately uses, or lets everything
through. And most of them work by sending what the child is looking at to a
company's servers to be checked.

Fuzzy takes a different approach. It looks at each picture and video **on
the device**, decides whether it is likely to be harmful, and if so blurs
it before the child really sees it. Nothing is uploaded. There is no
account, no server, no history.

## How it works, in one picture

```text
    a picture on a page
            │
            ▼
   ┌─────────────────┐     "What is in this?"
   │   1. LOOK       │     An AI model examines the picture and reports
   │                 │     what it sees: gore, nudity, a knife, a bus.
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐     "Does that matter?"
   │   2. DECIDE     │     Rules weigh what was found. A knife is not
   │                 │     automatically harmful — it depends.
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐     "Hide it."
   │   3. COVER      │     Blur the offending part, or the whole picture
   │                 │     if the entire scene is the problem.
   └────────┬────────┘
            │
            ▼
      child sees a blur
```

Those three steps are kept strictly separate, and that is the single most
important design decision in the project. The part that *looks* never
decides anything. The part that *decides* never draws anything. The reason
is that "what is in this picture" and "is this okay for a child" are
genuinely different questions, and mixing them makes both harder to change.

A knife makes the point. A knife in a cooking video is not the same as a
knife in a violent one. If the detector blurred everything it recognised as
a knife, the extension would ruin every recipe video on the internet and
get switched off within a day — protecting nobody. So the detector reports
"knife, 90% sure, in this rectangle", and a separate layer decides what to
do about it.

## What it actually catches

| Fuzzy hides | How |
|---|---|
| Gore, blood, injuries, violent scenes | Blurs the whole picture |
| Nudity and sexual content | Blurs the whole picture |
| Knives and scissors | Blurs just that part |

Whole-picture versus part-of-the-picture is not arbitrary. "There is a
knife at these coordinates" has a location, so only that area needs
covering. "This is a violent scene" has no location — the whole image is
the problem — so the whole thing is blurred.

## What it does not catch

This matters as much as the list above, and the extension says so in its
own interface rather than hiding it in documentation:

- **Guns and other weapons.** It knows knives and scissors, nothing else.
- **Drugs, self-harm, hate symbols.**
- **Anything written or spoken.** It looks at pictures. It cannot read text
  in an image or hear audio.
- **Very short moments in video.** It checks about twice a second, so
  something on screen for a quarter-second can slip through.

## The honest caveat

**Nobody has measured how well it detects harmful content.**

The model runs, and it correctly leaves ordinary pictures alone — that has
been tested. But testing whether it *catches* gore requires a collection of
genuinely harmful images to test against, and this project does not have
one. Assembling one is a decision for the project owner, with real legal
and ethical weight, so it has deliberately not been improvised.

The model's authors report 97.76% accuracy. That is their number, on their
own private data, and it has not been independently checked here. It should
be read as a claim, not a result.

So: Fuzzy demonstrably runs, demonstrably leaves safe content alone, and
plausibly catches harmful content. The third one is unverified, and this
report will not pretend otherwise.

## One thing it is not

Fuzzy is a **harm-reduction tool, not a lock**. It puts a blur over things.
A determined teenager can switch it off, or open the developer console and
remove the overlay. It reduces accidental exposure. It is not parental
control software and should not be sold as one.

---
---

# Part 2 — The technical account

## 2.1 Shape of the system

Two pipelines, one set of rules.

```text
                    ┌──────────────────────────┐
                    │   Shared safety rules    │
                    │  thresholds, risk logic  │
                    └────────────┬─────────────┘
                                 │  implemented twice,
                                 │  pinned together by tests
                ┌────────────────┴────────────────┐
                ▼                                 ▼
   ┌────────────────────────┐        ┌────────────────────────┐
   │  Python reference      │        │  Chrome extension      │
   │  packages/             │        │  apps/extension/       │
   │                        │        │                        │
   │  offline, for testing  │        │  live, on real pages   │
   │  and measurement       │        │                        │
   └────────────────────────┘        └────────────────────────┘
```

The Python side came first and remains the reference implementation: easy
to test, easy to measure, no browser in the way. The extension is the
product. They necessarily duplicate logic, because a browser cannot run
Python — and that duplication is the single largest source of risk in the
project, addressed in §2.8.

## 2.2 Layers

Both pipelines are built from the same five layers. Each has one job.

```text
 ┌───────────────────────────────────────────────────────────────┐
 │ CONTENT ADAPTER    gets pixels from a file, a page, a video    │
 ├───────────────────────────────────────────────────────────────┤
 │ VISION ENGINE      pixels ──► "knife, 0.91, at (320,180)"      │
 │                    pixels ──► "gore 0.02, sexual 0.01"         │
 │                    knows nothing about safety                  │
 ├───────────────────────────────────────────────────────────────┤
 │ RISK ENGINE        detections ──► LOW / MEDIUM / HIGH          │
 │                                ──► ALLOW / BLUR_REGION /       │
 │                                    BLUR_FRAME                  │
 │                    draws nothing                               │
 ├───────────────────────────────────────────────────────────────┤
 │ PROTECTION ENGINE  image + regions ──► protected image         │
 │                    decides nothing                             │
 ├───────────────────────────────────────────────────────────────┤
 │ DELIVERY           puts the result on screen                   │
 └───────────────────────────────────────────────────────────────┘
```

The separation earns its keep constantly. When the object detector turned
out to be the wrong model for the job (§2.5), the risk engine, the
protection engine and the entire browser delivery layer were untouched. A
model swap that would otherwise have rippled through everything changed one
layer.

## 2.3 The Python reference pipeline

Built first, bottom-up.

**`packages/shared/`** — types, configuration, logging, errors.

Every threshold lives in one config object loaded from `configs/default.yaml`
with environment-variable overrides. No magic numbers anywhere else. This
sounds like bookkeeping until you need to know why an image was blurred:
the answer is always a named value in one file.

Types are explicit: `BoundingBox`, `Detection`, `SceneRisk`, `RiskLevel`,
`ProtectionAction`, `AnalysisResult`. `BoundingBox` carries its own
`padded()` and `area()`, so box arithmetic exists once rather than being
re-derived at each call site — and gets ported once, correctly, to
JavaScript.

**`packages/vision/`** — the model abstraction.

`VisionModel` is an abstract base with `load()` and `predict()`.
`YoloModel` implements it. The abstraction exists so a model swap does not
leak outward, and it paid off exactly as intended in §2.5.

`SceneClassifier` deliberately does **not** implement `VisionModel`. That
interface returns `List[Detection]` — objects with boxes — and a scene
classifier has none. Forcing it to fit would mean inventing fake bounding
boxes for "this scene is violent", which is meaningless and would produce
blurs over arbitrary rectangles.

**`packages/risk/risk_engine.py`** — the decision layer.

```python
overall_risk = highest risk among detections
if scene score >= scene_risk_threshold:  overall_risk = HIGH

action = ALLOW
if overall_risk >= MEDIUM:               action = BLUR_REGION
if scene score >= scene_risk_threshold:  action = BLUR_FRAME
if harmful area > 60% of image:          action = BLUR_FRAME
```

The last rule is a usability judgement with a safety justification: once
harmful regions cover most of an image, a patchwork of blurred rectangles
looks worse and hides less than blurring the whole thing.

**`packages/protection/protection_engine.py`** — Gaussian blur, pixelation,
solid cover. Blur radius scales with region size, so a small region is not
obliterated and a large one is not under-blurred.

**`packages/pipeline/`** — wires the layers together for images and video.

**`packages/video_processing/temporal_tracker.py`** — video needs memory.
Running detection on every frame is too slow, and detection is jittery
frame to frame. An object missed on a single frame must not flash the
content into view. The tracker keeps protection alive for a configured
number of frames after the last sighting.

## 2.4 Getting into the browser: five walls

This is where most of the engineering went, and none of it was
anticipated. Each obstacle was found by running the thing and reading the
error.

**Wall 1 — Workers must be same-origin.**

The obvious design: content script spawns a Web Worker, worker runs the
model off the main thread.

```text
SecurityError: Failed to construct 'Worker': Script at
'chrome-extension://…/detector.worker.js' cannot be accessed from
origin 'https://www.youtube.com'.
```

`new Worker()` requires the script to be same-origin as the page. Declaring
the file web-accessible lets `fetch()` read it — it does not let `Worker()`
load it.

**Wall 2 — the page's CSP, not ours.**

The standard workaround is to fetch the worker source, wrap it in a `Blob`,
and construct from a `blob:` URL, which inherits the page's origin. On
YouTube:

```text
Creating a worker from 'blob:https://www.youtube.com/…' violates the
following Content Security Policy directive: "script-src …"
```

YouTube's own CSP forbids `blob:` workers. This is enforced by the *page*.
No extension-side configuration can override it, and any strict site will
do the same. The whole approach was a dead end.

**Wall 3 — `importScripts` timing in service workers.**

Next idea: run inference in the background service worker, which has the
extension's own origin and is immune to page CSP. It loaded the library —
then failed:

```text
Failed to execute 'importScripts' … the script failed to load.
```

`importScripts()` is only valid during a service worker's initial
*synchronous* evaluation. Called later, from inside an async message
handler, it fails — even though the file is present and the identical call
works at the top level. Hoisting it fixed this.

**Wall 4 — service workers cannot use dynamic `import()`.**

With loading fixed, the model itself refused to run:

```text
no available backend found. ERR: [wasm] TypeError: import() is disallowed
on ServiceWorkerGlobalScope by the HTML specification.
```

onnxruntime-web uses a dynamic `import()` internally. The HTML spec
disallows dynamic `import()` inside a service worker, in every browser.
This is not a bug to work around: **no service worker can run
onnxruntime-web's wasm backend.**

**The resolution — an offscreen document.**

Chrome provides exactly the missing thing: a hidden, extension-owned page.
It is a normal DOM/Window context, so dynamic `import()` works; it runs
under the extension's own origin and CSP, so no page can block it.

```text
content script ──(analyse this URL)──► service worker
                                            │  ensures the offscreen
                                            │  document exists
                                            ▼
                                     offscreen document
                                     (models live here)
```

**Wall 5 — an incomplete file list.** The first offscreen attempt failed
with `Failed to fetch dynamically imported module: …ort-wasm-simd-threaded.mjs`.
This one was self-inflicted: only two of the three required files had been
vendored. The real chain is

```text
ort.wasm.min.js  ──dynamically imports──►  ort-wasm-simd-threaded.mjs
                                     ──fetches──►  …simd-threaded.wasm
```

The `.wasm` filename never appears in `ort.wasm.min.js` at all — the glue
module loads it — so omitting the glue breaks the chain in the middle and
surfaces as a misleading "no available backend found". One `grep` of the
library would have caught it before four test cycles did.

**A problem that solved itself.** Reading pixels from a cross-origin image
in a page normally "taints" the canvas and is blocked — the expected
blocker for most real-world images. But the offscreen document does not
need the page's copy: it re-fetches the image URL with the **extension's**
host permissions, where page-origin CORS rules do not apply. The
anticipated limitation evaporated. Only `blob:` URLs remain genuinely
unreachable, since they are scoped to the page's origin.

## 2.5 Choosing models — and discovering the first one was wrong

The original model was YOLOv11n on COCO, chosen in an early phase for
being small, fast and browser-ready.

COCO has 80 classes. Exactly two are safety-relevant: **knife** and
**scissors**. It has no concept of blood, gore, violence or nudity.

So the pipeline was complete, correct, well-tested — and could not detect a
single thing the product exists for. No threshold change reaches categories
a model does not have. This was documented from the start as a known
limitation, but stayed a footnote while the browser plumbing was being
built.

A benchmark of candidates (`scripts/benchmark_safety_models.py`) produced:

| Model | Size | Median latency | Covers | Licence |
|---|---|---|---|---|
| **image-safety-classifier-xs** | 12.5 MB | **12.8 ms** | gore, violence, sexual | **MIT** |
| NudeNet 320n | 11.6 MB | 25.3 ms | nudity, with boxes | AGPL-derived |
| YOLOv11n *(incumbent)* | 10.1 MB | 202 ms | knife, scissors | AGPL-3.0 |

The chosen model is a SwiftFormer-XS finetune returning three
probabilities: **NSFL** (gore/violence), **NSFW** (sexual), **SFW**. Gore
was expected to be the hardest category to cover; it turned out to be
handled by the smallest, fastest, most permissively licensed option.

Two details were verified rather than assumed:

- **Output order is `[NSFL, NSFW, SFW]`** — read from the model's own
  config, not guessed. Guessing wrong would put gore scores in the sexual
  field and vice versa: wrong protection, for the wrong reason, with
  nothing visibly broken.
- **Input is raw 0-255, not normalised.** Preprocessing is baked into the
  ONNX graph. Pre-normalising pushed the NSFW score on a photo of a bus
  from 0.032 to 0.18 — the graph normalising already-normalised values.

**The architecture inverted as a result.** The old pipeline ran the
expensive model on everything. The new one runs the cheap model first:

```text
image ──► scene classifier (~13 ms)
             ├─ flagged ──► blur whole frame, skip object detection
             └─ clean   ──► object detection (~200 ms) ──► blur regions
```

Skipping object detection on a flagged frame is not merely an
optimisation: the verdict is whole-frame protection either way, so finding
a knife inside an already-blurred frame changes nothing and costs 200 ms.

The scene classifier plugs into a `SceneRisk` socket the risk engine had
carried since the beginning and never received. One field had to be added
— `sexual` — because the NSFW score had nowhere to go. Without it, sexual
content would have scored zero and never triggered protection: a silent
safety failure, so it has its own test.

## 2.6 The browser extension

```text
apps/extension/src/
  shared/config.js          thresholds, mirrors the Python config
  inference/
    detector.js             letterbox, YOLO decoding, NMS
    scene-classifier.js     the harm classifier
    risk.js                 port of risk_engine.py
    protection-render.js    draws the protected pixels
  content/
    content-script.js       discovery, queue, orchestration
    protection.js           positions image overlays
    video-protection.js     frame sampling and video overlays
  offscreen/offscreen.js    where models actually run
  background/service-worker.js  router; owns the offscreen document
  popup/                    the UI
```

**Discovery is demand-driven.** An `IntersectionObserver` analyses images
as they approach the viewport, so cost scales with what is on screen
(10–20 images) rather than page size (100+ on a YouTube feed). An earlier
version used a flat 40-image cap, which silently left most of a feed
unexamined.

**Nothing requires a page reload.** A `MutationObserver` catches images
added by infinite scroll and `src` swaps by lazy-loaders. SPA navigation is
detected via `popstate`, YouTube's `yt-navigate-finish`, and a URL poll as
the general fallback — patching `history.pushState` does not work, because
a content script's isolated world has its own copy and the page's calls
bypass it.

**Protection draws real pixels.** The first version used CSS
`backdrop-filter`, which depends on the stacking context it lands in and
silently fails in some real layouts — and cannot pixelate at all. Since
the offscreen document already holds the decoded bitmap, it renders the
protected image there and hands back a finished picture the content script
simply positions. Predictable everywhere, and true pixelation becomes
possible.

**Video is genuinely different**, in ways that each forced a design change:

1. *Frames cannot be fetched.* YouTube feeds video through MSE `blob:`
   URLs. Frames are captured from the element and shipped as `data:` URLs
   — which the existing analysis path accepted unchanged.
2. *A full-frame overlay would freeze the picture.* Each harmful region
   instead shows its own **crop** of the protected frame, so the rest of
   the video keeps playing at full framerate.
3. *The overlay lives inside the player.* A page-level overlay at maximum
   z-index covered YouTube's own scrubber and buttons. Inside the player
   container at a low z-index, it sits above the picture and below the
   controls — and fullscreen then works with no special handling, because
   the player container is what goes fullscreen.
4. *YouTube reuses the same `<video>` element* across navigations, swapping
   the MSE source underneath. Element identity never changes, so without
   watching for source changes the extension would happily show the
   previous video's protection over completely different content.

**Images and video have opposite navigation policies**, deliberately:

| | On navigation | Why |
|---|---|---|
| Images | Keep overlays, prune detached ones | A blanket clear briefly unprotects images still on screen. Stale-but-protecting is the safer failure. |
| Video | Clear immediately | The element persists while its content is replaced wholesale — keeping old protection would blur the *wrong video*. |

## 2.7 Bugs worth recording

Four bugs shaped the final design. All were silent — nothing crashed, and
in each case the system reported itself as working.

**Config never reached the risk engine.** A page reported `person 58`
detected and `0` protected. Detection was perfect; the risk engine never
saw the setting. `MyKidConfig.load()` was being called *inside the
offscreen document*, which gets only a restricted subset of `chrome.*`
APIs — `chrome.storage` not reliably among them. A bare `catch {}`
swallowed the failure and returned defaults.

The fix removed the guesswork rather than betting on API availability: the
service worker, which definitely has storage access, reads the config and
passes it in the message. `load()` now warns loudly instead of failing
silently.

**`IntersectionObserver.observe()` is a no-op on an already-observed
target.** It delivers no new entry. Code that cleared per-image state and
re-observed the same elements therefore analysed nothing — which is why
toggling a setting appeared to do nothing. Found in one call site, fixed,
then missed in a second one and rediscovered the same way.

**A failed analysis was permanent.** An image marked `FAILED` was never
retried, and the observer would not fire again for it. Most failures were
transient — the service worker sleeping, or the offscreen document still
building two model sessions when the first requests arrived. The result was
images left permanently unblurred with no signal that anything had gone
wrong. Now retried four times with backoff.

**The extension analysed its own output.** Protection overlays are `<img>`
elements, so discovery found them, analysed them, and protected *them* —
each protection spawning another overlay. With two inference slots, the
extension competed with itself and real images stopped being processed.
The tell was the image count: 8 images on a page reported as 16.

That last one was found by a purpose-built harness
(`apps/extension/devtest/live-rescan.html`) that runs the real content
script against stubbed extension APIs. It exists because on a real site,
"the setting did not apply" and "nothing here matched" look identical.

## 2.8 Testing

173 tests. The interesting ones are not the unit tests.

Because the browser cannot run Python, the risk engine, the label mapping
and the classifier preprocessing all exist **twice**. Duplicated logic
drifts, and drift here is invisible: both sides keep running, both return
plausible values, and they quietly disagree about whether a child sees
something.

Three cross-language checks pin them together, each guarding a silent
failure:

- **Risk verdicts** — 12 scenarios compared verdict-for-verdict between the
  Python engine and the JavaScript port, including padded box coordinates
  and coverage escalation. A Node runner feeds the extension's real code
  the same inputs.
- **Tensor layout** — a channel swap or NHWC/NCHW mix-up yields three
  plausible probabilities from a wrong tensor. The fixture deliberately
  gives each colour channel a distinct distribution, so a red/blue swap
  cannot pass.
- **Config delivery** — verified to work *without* `chrome.storage`,
  reproducing the offscreen document's real environment. This is a
  regression test for the bug above.

**What is deliberately not tested: detection accuracy on harmful content.**
No test asserts that gore is caught, because none can without an evaluation
set. Safe fixtures establish only that the model does not fire
indiscriminately — a false-positive check. Claiming more would be
overclaiming.

A related discovery: the project's own `test-data/images/safe/` fixtures
are blank and produce **zero detections**, which cannot validate bounding
box decoding at all — a pipeline with broken coordinate maths passes them
exactly as well as a correct one. Real sample images had to be introduced
before the browser detector could be checked against Python, which it now
matches to roughly one pixel.

## 2.9 Performance

| Stage | Native (Python) | In-browser (wasm) |
|---|---|---|
| Scene classifier | ~13 ms | ~70 ms (est.) |
| Object detection | ~42 ms | ~230 ms (measured) |
| Both | ~53 ms | ~300 ms |

WASM costs roughly 5× native. Absolute numbers move several-fold with
background load — an unpinned benchmark once reported 253 ms for an 11 ms
model — so measurements pin thread counts and report medians, and
comparisons rather than absolutes are the finding.

**INT8 quantization was tested and rejected.** It made models 70–73%
smaller but no faster for the classifier and ~50% *slower* for the
detector — dynamic INT8 has no optimised kernel path for these
architectures on this runtime. Worse, it moved decisions: quantized YOLO
found the same objects on the same image but with confidences shifted
across the 0.5 threshold (a person at 0.399 became 0.575). A
threshold-crossing shift is not acceptable in the layer deciding whether a
child sees something. Models are bundled and loaded once per session, so
trading latency for download size was the wrong way round.

**The real lever is elsewhere.** Object detection costs ~4× the classifier
and contributes least to the product's purpose. Disabling it is a ~4×
throughput gain that also removes the only AGPL-licensed component, at the
cost of knife/scissors region blur. The switch exists
(`detection.objectDetectionEnabled`); the default is unchanged, because
dropping a detection capability is a product decision.

## 2.10 Privacy

Audited, not asserted. Reproducible with four `grep`s, documented in
`docs/12-privacy-and-limitations.md`.

| Check | Result |
|---|---|
| Outbound network calls | 2, both fetching the image being analysed |
| External hosts / telemetry | none |
| Data leaving the device | none |
| Logged content | counts, actions, risk levels — never URLs, image data or page text |
| Persisted data | one key: the on/off flag and category settings |

Two things stated plainly rather than buried: images are fetched a
**second** time by the offscreen document, so servers see duplicate
requests — that is the mechanism that defeats canvas tainting; and
`<all_urls>` is a genuinely broad permission, which is why the audit is
written down as evidence rather than claimed.

## 2.11 What is left

| | Status |
|---|---|
| Detection accuracy | **Unmeasured.** Needs a harmful evaluation set — an owner decision with legal weight. |
| Firearms | Not detected. Available models are community-trained and unverifiable without the evaluation set above. |
| Video protection | Written and unit-tested, **not yet confirmed working in a browser**. |
| AGPL dependency | Removable by disabling object detection. |
| Cross-site breadth | Deferred. The design is site-agnostic; YouTube handling is an additive fast path, not a special case. |

## 2.12 What this project would do differently

- **Pick the model against the actual requirement, not the demo.** Months
  of correct engineering sat on a detector that could not see the target
  categories. The cost was not wasted code — the layering meant the swap
  touched one layer — but it was a wasted assumption.
- **Never write a bare `catch {}`.** Two of the four significant bugs were
  swallowed failures that presented as working systems. The one that hid a
  missing config read cost two full debugging cycles.
- **Build the observability before the feature.** "Detected: person 58,
  protected: 0" identified a bug in seconds that a bare "protected: 0" had
  obscured for two rounds.
- **Distinguish "written" from "working".** Several phases were marked
  complete on the strength of passing unit tests, then found broken in a
  browser. The status table in `docs/11-remaining-implementation-plan.md` now separates the two explicitly,
  because conflating them meant building three phases on an unverified
  foundation.

---

## Reference

| Document | Contents |
|---|---|
| `docs/06-ai-detection.md` | Model evaluation and selection, with benchmarks |
| `docs/09-testing-evaluation.md` | Testing strategy; what is and is not covered |
| `docs/11-remaining-implementation-plan.md` | Phase-by-phase history and status |
| `docs/12-privacy-and-limitations.md` | Privacy audit and the full limitations list |
| `apps/extension/README.md` | Setup and loading instructions |

```bash
python -m pytest tests/ -q                   # 173 tests
python scripts/benchmark_safety_models.py    # model size / latency / licence
python scripts/quantize_models.py            # the INT8 comparison
python scripts/parity_reference.py <image>   # ground-truth detections
node apps/extension/scripts/setup.js         # stage models into the extension
```
