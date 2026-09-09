/**
 * MyKid — Offscreen Document (Phase 10b-i)
 *
 * Hosts the actual onnxruntime-web inference. Exists because neither of
 * the two more obvious places works:
 *   - A Worker spawned from the content script (page context) is blocked
 *     by many real sites' own CSP — script-src/worker-src doesn't permit
 *     blob: worker construction (hit this on youtube.com).
 *   - The background service worker loads onnxruntime-web fine, but the
 *     library uses a dynamic `import()` internally, which the HTML spec
 *     disallows inside ServiceWorkerGlobalScope (hit this too).
 *
 * An offscreen document is a normal (hidden) DOM/Window context running
 * under the extension's own origin/CSP — immune to both problems. See
 * docs/11-remaining-implementation-plan.md for the full debugging trail.
 *
 * Phase 10b-i: self-test only. No real image handling yet (10b-ii/10b-iii).
 */

ort.env.wasm.numThreads = 1; // see service-worker.js / docs/11 for why
ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/onnxruntime-web/");

let sessionPromise = null;
let scenePromise = null;

function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(
      chrome.runtime.getURL("models/yolo11n.onnx"),
      { executionProviders: ["wasm"] }
    );
  }
  return sessionPromise;
}

/**
 * The scene classifier — the model that actually detects what this product
 * is for (gore, violence, sexual content). Loaded separately from the
 * object detector because it runs on a different schedule: it is the cheap
 * gate that runs on everything, while object detection is the expensive
 * step that runs only when there's reason to.
 */
function getSceneSession() {
  if (!scenePromise) {
    scenePromise = ort.InferenceSession.create(
      chrome.runtime.getURL("models/image-safety-classifier-xs.onnx"),
      { executionProviders: ["wasm"] }
    );
  }
  return scenePromise;
}

async function runSelfTest() {
  const loadStart = performance.now();
  const session = await getSession();
  const loadMs = performance.now() - loadStart;

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const dummyInput = new ort.Tensor(
    "float32",
    new Float32Array(1 * 3 * 640 * 640),
    [1, 3, 640, 640]
  );

  const inferStart = performance.now();
  const output = await session.run({ [inputName]: dummyInput });
  const inferMs = performance.now() - inferStart;

  return {
    ok: true,
    loadMs: Math.round(loadMs),
    inferMs: Math.round(inferMs),
    inputName,
    outputName,
    outputShape: output[outputName].dims,
  };
}

/**
 * Phase 10b-ii: run a real detection pass over an image, given a URL the
 * offscreen document can fetch.
 *
 * Kept deliberately dumb — it returns structured detections and nothing
 * else. Risk assessment and protection are separate layers (SKILL.md §38:
 * DETECT != DECIDE != PROTECT), wired up in 10b-iii.
 */
async function runDetection(imageUrl, confThreshold) {
  const session = await getSession();

  const response = await fetch(imageUrl);
  if (!response.ok) {
    throw new Error(`Could not fetch image (HTTP ${response.status})`);
  }
  const bitmap = await createImageBitmap(await response.blob());

  try {
    const result = await MyKidDetector.detect(session, bitmap, confThreshold);
    return { ok: true, ...result };
  } finally {
    bitmap.close();
  }
}

/**
 * Phase 10b-iii: full analysis for one page image — detect, then decide.
 *
 * Returns an AnalysisResult mirroring the Python pipeline's shape, so the
 * content script only has to render the verdict, never compute it.
 */
async function analyseImage(imageUrl, overrides) {
  // Config arrives in the message rather than being read from storage —
  // offscreen documents get only a restricted subset of the extension
  // APIs, and reading storage here failed silently, leaving the risk
  // engine on defaults. See service-worker.js for the full story.
  const config = MyKidConfig.fromOverrides(overrides);

  const response = await fetch(imageUrl);
  if (!response.ok) {
    throw new Error(`Could not fetch image (HTTP ${response.status})`);
  }
  const bitmap = await createImageBitmap(await response.blob());

  try {
    // --- Stage 1: scene classification (the cheap gate) -------------
    //
    // ~13ms native vs ~200ms for object detection, and it covers the
    // categories the product actually exists for. So it runs first, on
    // everything, and decides whether the expensive stage is worth it
    // (docs/06-ai-detection.md, "Recommended architecture").
    const scene = await MyKidSceneClassifier.classify(
      await getSceneSession(),
      bitmap
    );

    // Zero out categories the user has switched off, before the risk
    // engine sees them. Filtering here rather than inside the engine keeps
    // the decision logic byte-identical to the Python reference, so
    // cross-pipeline parity holds for every combination of settings and
    // not just the defaults.
    const sceneRisk = {
      violence: scene.sceneRisk.violence,
      graphic: config.categories.gore ? scene.sceneRisk.graphic : 0,
      sexual: config.categories.sexual ? scene.sceneRisk.sexual : 0,
    };

    const sceneFlagged =
      Math.max(sceneRisk.graphic, sceneRisk.sexual) >=
      config.detection.sceneRiskThreshold;

    // --- Stage 2: object detection (only when it can change anything) --
    //
    // Skipped when the scene is already flagged: the verdict is whole-frame
    // protection either way, so finding a knife inside an already-blurred
    // frame changes nothing and costs ~200ms. Also skipped when object
    // detection is turned off entirely.
    let detection = null;
    if (
      !sceneFlagged &&
      config.categories.weapons &&
      config.detection.objectDetectionEnabled
    ) {
      detection = await MyKidDetector.detect(
        await getSession(),
        bitmap,
        config.detection.objectConfidenceThreshold
      );
    }

    const analysis = MyKidRisk.evaluate(
      detection ? detection.detections : [],
      sceneRisk,
      bitmap.width,
      bitmap.height,
      config
    );

    // Render the protected pixels here, where the decoded bitmap already
    // lives — the content script receives a finished image to position,
    // not a filter to hope renders correctly in the page's layout.
    const renderStart = performance.now();
    const protectedImageUrl = await MyKidProtectionRender.render(
      bitmap,
      analysis,
      config
    );
    const renderMs = Math.round(performance.now() - renderStart);

    return {
      ok: true,
      analysis,
      protectedImageUrl,
      // Every label the model saw, not just the harmful ones. This is what
      // distinguishes "detection isn't running" from "detection ran and
      // found nothing we treat as harmful" — two very different faults
      // that look identical from the outside.
      detectedLabels: detection ? detection.detections.map((d) => d.label) : [],
      // What the risk engine actually considered harmful for this call.
      // Surfaced so a "detected but not protected" result can be checked
      // against the setting that produced it, instead of inferred.
      harmfulLabels: MyKidConfig.harmfulLabels(config),
      // Scene scores, so "why wasn't this blurred?" can be answered with
      // the actual numbers rather than guessed at.
      // Raw model scores, before category filtering — so the popup can
      // show what was actually seen even for a category that is switched
      // off, rather than a misleading zero.
      sceneScores: {
        gore: Number(scene.sceneRisk.graphic.toFixed(4)),
        sexual: Number(scene.sceneRisk.sexual.toFixed(4)),
        safe: Number((scene.scores.SFW ?? 0).toFixed(4)),
      },
      activeCategories: config.categories,
      sceneFlagged,
      objectDetectionRan: detection !== null,
      timings: {
        sceneMs: scene.inferMs,
        ...(detection ? detection.timings : {}),
        renderMs,
      },
    };
  } finally {
    bitmap.close();
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message && message.type === "MYKID_RUN_SELF_TEST") {
    runSelfTest()
      .then(sendResponse)
      .catch((err) =>
        sendResponse({ ok: false, error: err && err.message ? err.message : String(err) })
      );
    return true; // keep the message channel open for the async sendResponse
  }

  if (message && message.type === "MYKID_RUN_DETECTION") {
    runDetection(message.imageUrl, message.confThreshold ?? 0.25)
      .then(sendResponse)
      .catch((err) =>
        sendResponse({ ok: false, error: err && err.message ? err.message : String(err) })
      );
    return true;
  }

  if (message && message.type === "MYKID_RUN_ANALYSIS") {
    analyseImage(message.imageUrl, message.overrides)
      .then(sendResponse)
      .catch((err) =>
        sendResponse({ ok: false, error: err && err.message ? err.message : String(err) })
      );
    return true;
  }
});
