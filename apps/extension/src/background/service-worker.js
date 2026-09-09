/**
 * MyKid — Background Service Worker
 *
 * Routes self-test requests from content scripts to the offscreen
 * document, which does the actual inference (see src/offscreen/offscreen.js
 * for why: onnxruntime-web needs a normal DOM/Window context, which a
 * service worker isn't).
 *
 * Phase 10b-i: self-test relay only. No real image handling yet — that's
 * 10b-ii/10b-iii.
 */

const OFFSCREEN_DOCUMENT_PATH = "src/offscreen/offscreen.html";

async function ensureOffscreenDocument() {
  try {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT_PATH,
      reasons: ["WORKERS"],
      justification:
        "Runs onnxruntime-web model inference in a normal document context " +
        "(the library needs dynamic import(), which service workers can't do).",
    });
  } catch (err) {
    // Chrome throws if a document is already open at this URL — that's
    // the expected case on every call after the first.
    if (!String(err).includes("Only a single offscreen")) {
      throw err;
    }
  }
}

/**
 * Reflect the on/off state on the toolbar icon.
 *
 * A safety tool has to make its own state obvious. Without this, "is MyKid
 * protecting me right now?" needs a click to answer, and a user who
 * believes it's on when it isn't is worse off than one who knows it's off.
 */
async function updateBadge() {
  const { enabled } = await currentConfig();

  await chrome.action.setBadgeText({ text: enabled ? "" : "OFF" });
  await chrome.action.setBadgeBackgroundColor({ color: "#b3541e" });
  await chrome.action.setTitle({
    title: enabled
      ? "Fuzzy — on, watching out for you"
      : "Fuzzy — OFF, nothing is being checked",
  });
}

/** Full config, for decisions the service worker makes itself. */
async function currentConfig() {
  const overrides = await currentOverrides();
  return { enabled: overrides.enabled ?? true };
}

chrome.runtime.onInstalled.addListener((details) => {
  console.log(`[Fuzzy] service worker installed (reason: ${details.reason})`);
  updateBadge();
});

chrome.runtime.onStartup.addListener(updateBadge);

// The popup writes config; the badge follows it rather than being set in
// two places that could disagree.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.mykidConfig) updateBadge();
});

/**
 * Content scripts and the popup never talk to the offscreen document
 * directly — everything routes through here, so the offscreen document's
 * lifecycle has exactly one owner.
 */
/**
 * Read the user's config overrides and forward them with each request.
 *
 * The offscreen document can't read these itself: it gets only a
 * restricted subset of the extension APIs, and `chrome.storage` isn't
 * reliably among them. An earlier version had it call
 * `chrome.storage.local.get` directly, which failed silently and left the
 * risk engine running on defaults — so the "treat as harmful" setting was
 * detected, decided against, and never applied. Passing config explicitly
 * removes the guesswork about what's reachable where.
 */
async function currentOverrides() {
  try {
    const stored = await chrome.storage.local.get("mykidConfig");
    return stored.mykidConfig ?? {};
  } catch (err) {
    console.warn("[Fuzzy] could not read config overrides:", err);
    return {};
  }
}

const RELAYED_MESSAGES = {
  MYKID_SELF_TEST: async () => ({ type: "MYKID_RUN_SELF_TEST" }),
  MYKID_PARITY_TEST: async (message) => ({
    type: "MYKID_RUN_DETECTION",
    imageUrl: chrome.runtime.getURL("devtest/bus.jpg"),
    confThreshold: message.confThreshold ?? 0.25,
  }),
  MYKID_ANALYSE_IMAGE: async (message) => ({
    type: "MYKID_RUN_ANALYSIS",
    imageUrl: message.imageUrl,
    overrides: await currentOverrides(),
  }),
};

/**
 * Analysis requests are refused while disabled.
 *
 * The content script already stops asking when off, so this should never
 * fire — which is the point. It's the backstop that guarantees "off means
 * no images are analysed", rather than that resting on one call site
 * staying correct forever.
 */
async function isAnalysisAllowed(messageType) {
  if (messageType !== "MYKID_ANALYSE_IMAGE") return true;
  const { enabled } = await currentConfig();
  return enabled;
}

/**
 * Send to the offscreen document, tolerating it not being ready yet.
 *
 * `createDocument()` resolving does not guarantee the document's scripts
 * have run and registered their message listener, so an immediate send can
 * fail with "Receiving end does not exist". Chrome can also close an idle
 * offscreen document, leaving a stale assumption that one exists.
 *
 * Both are transient and both used to surface as a permanent analysis
 * failure for that image. Retrying costs a few hundred milliseconds once;
 * not retrying left images silently unprotected.
 */
async function sendToOffscreen(relayed, attempts = 3) {
  for (let attempt = 1; ; attempt++) {
    try {
      return await chrome.runtime.sendMessage(relayed);
    } catch (err) {
      const notListening = String(err).includes("Receiving end does not exist");
      if (!notListening || attempt >= attempts) throw err;

      await new Promise((resolve) => setTimeout(resolve, 200 * attempt));
      await ensureOffscreenDocument(); // it may have been closed entirely
    }
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const buildRelayed = message && RELAYED_MESSAGES[message.type];
  if (!buildRelayed) return;

  (async () => {
    try {
      if (!(await isAnalysisAllowed(message.type))) {
        sendResponse({ ok: false, error: "Fuzzy is switched off" });
        return;
      }
      await ensureOffscreenDocument();
      const relayed = await buildRelayed(message);
      sendResponse(await sendToOffscreen(relayed));
    } catch (err) {
      sendResponse({ ok: false, error: err && err.message ? err.message : String(err) });
    }
  })();

  return true; // keep the message channel open for the async sendResponse
});
