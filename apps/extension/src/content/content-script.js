/**
 * MyKid — Content Script (Phase 10c: dynamic content)
 *
 * The DOM layer. Discovers images, asks the offscreen document (via the
 * background service worker) to analyse them, and positions protection
 * over whatever comes back flagged. No inference, no risk logic here —
 * those live in src/inference/, per SKILL.md §38.
 *
 * Phase 10c makes this work *without a page refresh*, which is what real
 * sites demand:
 *   - MutationObserver picks up images added after load (infinite feeds).
 *   - Attribute watching catches lazy-loaders that swap `src` on an
 *     existing element rather than inserting a new one.
 *   - IntersectionObserver means we only spend inference on images the
 *     user can actually see, instead of racing through the whole DOM.
 *   - SPA navigation (YouTube et al.) is detected and triggers a re-scan,
 *     since those never fire a page load at all.
 *
 * Processing state per SKILL.md §21:
 *   UNPROCESSED -> OBSERVED -> QUEUED -> PROCESSING -> PROCESSED | SKIPPED | FAILED
 *
 * Per SKILL.md §29 this logs counts and states — never page content,
 * image data, or URLs.
 */
(function () {
  if (window.__mykidContentScriptLoaded) return;
  window.__mykidContentScriptLoaded = true;

  const State = {
    OBSERVED: "observed",
    QUEUED: "queued",
    PROCESSING: "processing",
    PROCESSED: "processed",
    SKIPPED: "skipped",
    FAILED: "failed",
  };

  // How many times to retry an image whose analysis failed.
  //
  // Failures here are usually transient: the service worker sleeps, or the
  // offscreen document is still building its model sessions when the first
  // requests arrive ("Receiving end does not exist"). Treating those as
  // final left images permanently unanalysed — and therefore permanently
  // unprotected — until something rebuilt the observers, which is why
  // toggling off and on appeared to "fix" the extension.
  const MAX_ANALYSIS_ATTEMPTS = 4;

  // Per-image record, keyed by element. Holds the state plus the src we
  // analysed, so a lazy-loader swapping in a new image re-triggers work.
  const imageRecords = new WeakMap();

  const stats = {
    discovered: 0,
    skipped: 0,
    processed: 0,
    failed: 0,
    retried: 0,
    // Why the last analysis failed. A bare failure count cannot
    // distinguish "one flaky image" from "the model file is missing and
    // nothing will ever work" — and the second needs a very different fix.
    lastError: null,
    protected: 0,
    queued: 0,
    // What the model actually saw, label -> count. Without this, a page
    // where nothing gets blurred is indistinguishable from a page where
    // nothing is being analysed at all — and those need opposite fixes.
    labelCounts: {},
    // What the risk engine treated as harmful, as reported by the engine
    // itself. Comparing this against labelCounts turns "why didn't it
    // blur?" into a directly readable answer.
    harmfulLabels: null,
    activeCategories: null,
    // Scene classifier (Phase B) — the model that detects the content this
    // product actually exists for. Tracked separately from object labels
    // because "the harm detector is running but scoring low" and "the harm
    // detector never ran" are different problems.
    sceneChecked: 0,
    sceneFlagged: 0,
    peakGore: 0,
    peakSexual: 0,
  };

  function resetStats() {
    stats.discovered = 0;
    stats.skipped = 0;
    stats.processed = 0;
    stats.failed = 0;
    stats.retried = 0;
    stats.lastError = null;
    stats.protected = 0;
    stats.queued = 0;
    stats.labelCounts = {};
    stats.harmfulLabels = null;
    stats.activeCategories = null;
    stats.sceneChecked = 0;
    stats.sceneFlagged = 0;
    stats.peakGore = 0;
    stats.peakSexual = 0;
  }

  let config = null;
  let intersectionObserver = null;
  let mutationObserver = null;

  // Bounded work queue — inference is ~230ms per image, so this paces the
  // work rather than letting a feed flood it.
  const queue = [];
  let activeWorkers = 0;

  function currentSrc(image) {
    return image.currentSrc || image.src || "";
  }

  /**
   * Our own protection overlays are <img> elements, so without excluding
   * them they get discovered, analysed, and protected in turn — each
   * protection spawning another overlay. The 1px stub used in the test
   * harness fell under the size floor and hid this, but a real protected
   * render (up to 800px) would feed straight back in.
   */
  function isOwnElement(element) {
    return Boolean(
      element.closest &&
        element.closest(
          `#mykid-protection-layer, .${MyKidVideoProtection.LAYER_CLASS}`
        )
    );
  }

  function countVisualElements() {
    return {
      images: Array.prototype.filter.call(document.images, (i) => !isOwnElement(i))
        .length,
      videos: document.querySelectorAll("video").length,
    };
  }

  /**
   * Is this image worth ~230ms of inference?
   *
   * Real pages are full of icons, avatars, spacers and tracking pixels.
   * The threshold is deliberately low enough to keep genuine content —
   * a YouTube grid thumbnail can be ~360x202, and an earlier, stricter
   * 128px-on-both-sides rule was rejecting legitimate thumbnails.
   */
  function isCandidate(image) {
    const src = currentSrc(image);
    if (!src) return false;

    // blob: URLs are scoped to the page's origin, so the offscreen
    // document can't re-fetch them. Skipped honestly rather than
    // mishandled — a documented limitation, see docs/11.
    if (src.startsWith("blob:")) return false;

    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    const minimum = config.browser.minImageDimension;

    // Not loaded yet — the load event will bring it back around.
    if (width === 0 || height === 0) return false;

    return width >= minimum && height >= minimum;
  }

  function enqueue(image) {
    const record = imageRecords.get(image);
    if (record && record.state === State.QUEUED) return;
    if (record && record.state === State.PROCESSING) return;
    if (record && record.state === State.PROCESSED && record.src === currentSrc(image)) {
      return; // already handled, and the image hasn't changed underneath us
    }

    imageRecords.set(image, { state: State.QUEUED, src: currentSrc(image) });
    queue.push({ image, attempt: 1 });
    stats.queued++;
    pumpQueue();
  }

  function pumpQueue() {
    while (
      activeWorkers < config.browser.maxConcurrentInferences &&
      queue.length > 0
    ) {
      const { image, attempt } = queue.shift();
      activeWorkers++;
      processImage(image, attempt)
        .catch((err) => console.error("[Fuzzy] image processing failed:", err))
        .finally(() => {
          activeWorkers--;
          pumpQueue();
        });
    }
  }

  /**
   * An analysis attempt failed. Retry it a few times before giving up.
   *
   * Failing open is the right behaviour for an image we genuinely cannot
   * inspect (SKILL.md §28), but "cannot inspect" and "did not manage to
   * inspect yet" are different things. Most failures here are transient —
   * the service worker sleeping, or the offscreen document still building
   * its two model sessions when the first requests land. Treating those as
   * final left images unanalysed for the life of the page, i.e. silently
   * unprotected, with no signal that anything had gone wrong.
   */
  function recordFailure(image, src, attempt, reason) {
    if (attempt < MAX_ANALYSIS_ATTEMPTS) {
      imageRecords.set(image, { state: State.FAILED, src, attempt });
      stats.retried++;

      // Backs off, because the usual cause is something still warming up.
      const delayMs = 350 * 2 ** (attempt - 1);
      setTimeout(() => {
        if (!image.isConnected || !config || !config.enabled) return;

        // Left alone if anything has moved on since: the image was
        // re-queued elsewhere, already analysed, or its src changed.
        const record = imageRecords.get(image);
        if (!record || record.state !== State.FAILED || record.src !== currentSrc(image)) {
          return;
        }

        queue.push({ image, attempt: attempt + 1 });
        pumpQueue();
      }, delayMs);
      return;
    }

    imageRecords.set(image, { state: State.FAILED, src, attempt });
    stats.failed++;
    if (reason) stats.lastError = String(reason).slice(0, 160);
    console.warn(
      `[Fuzzy] gave up on an image after ${attempt} attempts` +
        (reason ? ` (${reason})` : "")
    );
  }

  /**
   * Analyse one image and protect it if flagged.
   *
   * The image URL goes to the offscreen document, which fetches and
   * decodes it with the extension's own host permissions. That sidesteps
   * canvas tainting: reading a cross-origin image's pixels in page
   * context is blocked, the extension fetching that URL itself is not.
   */
  async function processImage(image, attempt = 1) {
    const src = currentSrc(image);
    if (!src || !image.isConnected) return;

    imageRecords.set(image, { state: State.PROCESSING, src, attempt });

    let response;
    try {
      response = await chrome.runtime.sendMessage({
        type: "MYKID_ANALYSE_IMAGE",
        imageUrl: src,
      });
    } catch (err) {
      // The service worker can be torn down mid-flight, or the offscreen
      // document may not be listening yet.
      recordFailure(image, src, attempt, String(err));
      return;
    }

    if (!response || !response.ok) {
      recordFailure(image, src, attempt, response && response.error);
      return;
    }

    imageRecords.set(image, { state: State.PROCESSED, src, attempt });
    stats.processed++;

    for (const label of response.detectedLabels ?? []) {
      stats.labelCounts[label] = (stats.labelCounts[label] ?? 0) + 1;
    }
    // Reported back by the engine that actually made the decision, so the
    // popup shows the setting in force rather than what we assume it is.
    if (response.harmfulLabels) stats.harmfulLabels = response.harmfulLabels;
    // Reported by the engine that applied them, so the popup shows the
    // categories actually in force rather than what it assumes.
    if (response.activeCategories) stats.activeCategories = response.activeCategories;

    if (response.sceneScores) {
      stats.sceneChecked++;
      if (response.sceneFlagged) stats.sceneFlagged++;
      stats.peakGore = Math.max(stats.peakGore, response.sceneScores.gore);
      stats.peakSexual = Math.max(stats.peakSexual, response.sceneScores.sexual);
    }

    // The image may have been swapped or removed while we were working.
    if (!image.isConnected || currentSrc(image) !== src) return;

    if (response.analysis.action !== MyKidRisk.ProtectionAction.ALLOW) {
      const applied = MyKidProtection.apply(
        image,
        response.analysis,
        response.protectedImageUrl
      );
      if (applied) {
        stats.protected++;
        console.log(
          `[Bubble] protection applied — action=${response.analysis.action}, ` +
            `risk=${response.analysis.overallRisk}, ` +
            `regions=${response.analysis.protectionRegions.length}`
        );
      }
    }
  }

  /**
   * Only images at or near the viewport get analysed. On a feed with
   * hundreds of images this is the difference between the extension
   * being usable and it burning minutes of CPU on content nobody scrolled
   * to — and it replaces the crude per-page cap the previous version used.
   */
  function getIntersectionObserver() {
    if (intersectionObserver) return intersectionObserver;

    intersectionObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const image = entry.target;
          if (isCandidate(image)) {
            enqueue(image);
          } else {
            imageRecords.set(image, { state: State.SKIPPED, src: currentSrc(image) });
            stats.skipped++;
          }
        }
      },
      {
        // Start work slightly before the image scrolls into view so
        // protection is in place by the time it's actually visible.
        rootMargin: "200px",
        threshold: 0.01,
      }
    );

    return intersectionObserver;
  }

  /** Begin watching one image (idempotent). */
  function observe(image) {
    const record = imageRecords.get(image);
    if (record && record.state !== State.SKIPPED) return;
    if (record && record.src === currentSrc(image)) return;

    imageRecords.set(image, { state: State.OBSERVED, src: currentSrc(image) });
    stats.discovered++;
    getIntersectionObserver().observe(image);

    // Images still loading report 0x0; re-evaluate once they're decoded.
    if (!image.complete) {
      image.addEventListener("load", () => onImageChanged(image), { once: true });
    }
  }

  /** An image's pixels changed — re-evaluate it and drop stale protection. */
  function onImageChanged(image) {
    const record = imageRecords.get(image);
    if (record && record.src === currentSrc(image) && record.state === State.PROCESSED) {
      return;
    }

    if (MyKidProtection.isProtected(image)) {
      MyKidProtection.remove(image);
    }

    imageRecords.delete(image);

    // IntersectionObserver.observe() on an already-observed target is a
    // no-op and will NOT re-deliver an entry — so a lazy-loader swapping
    // src on an element we're already watching would never be re-queued.
    // Unobserve first to force a fresh intersection callback.
    if (intersectionObserver) intersectionObserver.unobserve(image);

    observe(image);
  }

  function scan(root) {
    // Single choke point for the on/off state. The DOM and navigation
    // watchers stay installed while disabled and funnel through here, so
    // nothing is analysed until the toggle is back on — and re-enabling
    // needs no page reload.
    if (!config || !config.enabled) return;

    const images =
      root instanceof Element && root.tagName === "IMG"
        ? [root]
        : (root.querySelectorAll?.("img") ?? []);

    for (const image of images) {
      if (!isOwnElement(image)) observe(image);
    }

    scanVideos(root);
  }

  /**
   * A subtree left the DOM — release anything we were holding for it.
   * Images clean up lazily (their overlays are pruned on the next
   * reposition), but videos hold timers and listeners that must be
   * released explicitly or they leak across every SPA navigation.
   */
  function cleanupRemoved(root) {
    const videos =
      root.tagName === "VIDEO" ? [root] : (root.querySelectorAll?.("video") ?? []);
    for (const video of videos) MyKidVideoProtection.unwatch(video);

    const images =
      root.tagName === "IMG" ? [root] : (root.querySelectorAll?.("img") ?? []);
    for (const image of images) {
      MyKidProtection.remove(image);
      if (intersectionObserver) intersectionObserver.unobserve(image);
    }
  }

  /**
   * Videos are handled by a separate module — they need frame sampling
   * rather than one-shot analysis, and can't be fetched by URL the way
   * images can (see video-protection.js).
   */
  function scanVideos(root) {
    const videos =
      root instanceof Element && root.tagName === "VIDEO"
        ? [root]
        : (root.querySelectorAll?.("video") ?? []);

    for (const video of videos) {
      // Dimensions are only known once metadata has loaded.
      if (video.readyState >= 1) {
        MyKidVideoProtection.watch(video, config);
      } else {
        video.addEventListener(
          "loadedmetadata",
          () => MyKidVideoProtection.watch(video, config),
          { once: true }
        );
      }
    }
  }

  /**
   * Watch for content that appears after load: infinite scroll, lazy
   * loading, and any other DOM churn. Without this the extension only
   * ever protects whatever happened to be present at document_idle.
   */
  function watchDom() {
    mutationObserver = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === "childList") {
          for (const node of mutation.addedNodes) {
            if (node.nodeType === Node.ELEMENT_NODE) scan(node);
          }
          // Removals matter as much as additions on an SPA: without this,
          // every navigation would leave behind sampling timers and
          // listeners for <video> elements that no longer exist, and they
          // accumulate for as long as the tab is open.
          for (const node of mutation.removedNodes) {
            if (node.nodeType === Node.ELEMENT_NODE) cleanupRemoved(node);
          }
        } else if (mutation.type === "attributes") {
          // A lazy-loader swapping src/srcset on an existing element.
          onImageChanged(mutation.target);
        }
      }
    });

    mutationObserver.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src", "srcset"],
    });
  }

  /**
   * SPA navigation never fires a page load, so nothing would otherwise
   * re-trigger a scan — this is the "works without refreshing"
   * requirement. YouTube emits its own event; the URL poll is the
   * general fallback that works on any SPA (patching history.pushState
   * from a content script doesn't help, because the isolated world has
   * its own copy and the page's own calls bypass it).
   */
  function watchNavigation() {
    let lastUrl = location.href;

    const onNavigated = () => {
      if (location.href === lastUrl) return;
      lastUrl = location.href;

      // Images: deliberately NOT clearing every overlay. Pruning only the
      // ones whose image left the DOM removes genuinely stale overlays,
      // while a blanket clear would briefly unprotect images still on
      // screen and still harmful. For a child-safety tool, protection
      // lingering a moment too long beats coming down a moment too early.
      MyKidProtection.repositionAll();

      // Videos: the opposite call, for the opposite reason. The element
      // survives navigation while its content is replaced wholesale, so
      // keeping the old protection would blur the wrong video entirely.
      // Sampling re-establishes it within ~500ms if warranted.
      MyKidVideoProtection.handleNavigation();

      queue.length = 0;

      console.log("[Bubble] navigation detected — rescanning");
      scan(document);
    };

    window.addEventListener("popstate", onNavigated);
    window.addEventListener("yt-navigate-finish", onNavigated); // YouTube
    setInterval(onNavigated, 500);
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message && message.type === "MYKID_GET_COUNTS") {
      sendResponse({
        enabled: Boolean(config && config.enabled),
        ...countVisualElements(),
        stats: {
          ...stats,
          pending: queue.length,
          protectionsActive: MyKidProtection.count(),
        },
        video: MyKidVideoProtection.getStats(),
      });
    }
    if (message && message.type === "MYKID_RESCAN") {
      // Config changed (the on/off toggle, or the debug label) —
      // re-evaluate everything without making the user reload the page.
      teardown();
      resetStats();

      MyKidConfig.load().then((fresh) => {
        config = fresh;
        if (config.enabled) {
          scan(document);
        } else {
          console.log("[Bubble] disabled — protection removed, analysis stopped");
        }
      });
      sendResponse({ ok: true });
    }
  });

  /**
   * Stand down completely: drop overlays, stop watching, cancel queued and
   * in-flight video sampling.
   *
   * Used both when switching off and before a rescan. Switching off has to
   * actually stop work, not just hide it — an extension that keeps
   * analysing every image while the user believes it is off is dishonest
   * about what it's doing with their CPU and their browsing.
   *
   * The IntersectionObserver is disconnected rather than left in place,
   * because observe() on an already-observed target is a no-op that
   * delivers no entry — re-observing after clearing state would leave
   * every image silently un-analysed.
   */
  function teardown() {
    MyKidProtection.clearAll();
    MyKidVideoProtection.clearAll();

    for (const image of document.images) imageRecords.delete(image);
    queue.length = 0;

    if (intersectionObserver) {
      intersectionObserver.disconnect();
      intersectionObserver = null;
    }
  }

  async function start() {
    config = await MyKidConfig.load();

    // Listeners are installed regardless of state so the toggle can take
    // effect without a reload; they simply have nothing to act on while
    // disabled, because scan() is never called.
    MyKidProtection.watchLayout();
    MyKidVideoProtection.watchFullscreen();
    watchDom();
    watchNavigation();

    if (!config.enabled) {
      console.log("[Bubble] disabled — no analysis will run on this page");
      return;
    }

    scan(document);

    const counts = countVisualElements();
    console.log(
      `[Bubble] active — ${counts.images} image(s), ${counts.videos} video(s) on load; ` +
        `${stats.discovered} observed, analysing as they come into view`
    );
  }

  start().catch((err) => {
    // Never let an extension failure take the page down with it.
    console.error("[Bubble] startup failed:", err);
  });
})();
