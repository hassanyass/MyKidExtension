/**
 * MyKid — Video protection (Phases 11a–11c)
 *
 * Samples frames from playing <video> elements, sends them for analysis,
 * and overlays protection on the harmful regions.
 *
 * Three things make video meaningfully different from images:
 *
 * 1. **Frames can't be fetched by URL.** YouTube and most streaming sites
 *    feed video through MSE `blob:` URLs, which the offscreen document
 *    can't re-fetch. Frames are captured from the element here, in the
 *    page, and shipped as data: URLs — which the existing analysis path
 *    already accepts unchanged.
 *
 * 2. **Overlaying a whole protected frame would freeze the video.** For
 *    region protection we position one overlay per region, each showing
 *    the corresponding *crop* of the protected frame. Harmful areas are
 *    obscured; everything else keeps playing at full framerate.
 *
 * 3. **The overlay lives inside the player, not on top of the page.**
 *    See the layer notes below — this is what keeps player controls
 *    usable and makes fullscreen work without special-casing.
 *
 * Phase 11b: SPA survival. On YouTube, navigating between videos does not
 * create a new <video> element — the same element is reused with its MSE
 * source swapped underneath, so source changes are watched explicitly.
 *
 * Phase 11c: player quirks — control chrome, quality switches, ads,
 * theater/fullscreen, and Shorts.
 *
 * Exposes a global `MyKidVideoProtection`.
 */
var MyKidVideoProtection = (function () {
  const LAYER_CLASS = "mykid-video-layer";
  const OVERLAY_CLASS = "mykid-video-overlay";

  // Above the video, below the player's own chrome.
  //
  // The first version used the maximum z-index on a page-level layer,
  // which sat above YouTube's controls: a full-frame blur covered the
  // scrubber and buttons, making the player unusable. Sitting inside the
  // player container at a low z-index puts protection over the picture
  // while the site's controls (z-index in the tens) stay on top.
  const OVERLAY_Z_INDEX = 5;

  let stylesInjected = false;
  let rafHandle = null;
  let visibilityObserver = null;

  const videoStates = new Map(); // video element -> state record

  const stats = {
    discovered: 0,
    sampled: 0,
    failed: 0,
    unreadable: 0,
    sourceChanges: 0,
    navigations: 0,
  };

  function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;

    const style = document.createElement("style");
    style.textContent = `
      .${LAYER_CLASS} {
        position: absolute;
        top: 0; left: 0; width: 0; height: 0;
        pointer-events: none;
        z-index: ${OVERLAY_Z_INDEX};
      }
      .${OVERLAY_CLASS} {
        position: absolute;
        pointer-events: none;
        background-repeat: no-repeat;
        display: block;
      }
    `;
    (document.head || document.documentElement).appendChild(style);
  }

  /**
   * Each video gets its own overlay layer, inserted as a sibling of the
   * <video> inside the player container.
   *
   * Two problems this solves at once:
   *   - **Controls stay visible.** A page-level layer at max z-index
   *     covers the site's own player chrome; a layer inside the player at
   *     a low z-index sits above the picture but below the controls.
   *   - **Fullscreen just works.** The fullscreen renderer only paints the
   *     fullscreen element's subtree. Since the player container is what
   *     goes fullscreen, a layer inside it goes along — no re-parenting
   *     on `fullscreenchange`, which is what the previous version needed.
   */
  function getLayer(state) {
    const host = state.video.parentElement || document.body;

    if (!state.layerEl) {
      injectStyles();
      state.layerEl = document.createElement("div");
      state.layerEl.className = LAYER_CLASS;
    }

    // The layer must follow the video if the player moves it between
    // containers — YouTube does exactly that for the miniplayer, and for
    // theater/fullscreen transitions. Left behind, the layer would render
    // protection in the wrong place, or nowhere at all.
    if (state.layerEl.parentElement !== host) {
      host.appendChild(state.layerEl);
    }

    return state.layerEl;
  }

  function isCandidate(video, config) {
    const width = video.videoWidth || video.clientWidth;
    const height = video.videoHeight || video.clientHeight;
    return (
      width >= config.video.minVideoDimension &&
      height >= config.video.minVideoDimension
    );
  }

  /**
   * Capture the current frame as a downscaled JPEG data URL.
   *
   * Returns null when the frame can't be read — cross-origin video without
   * CORS taints the canvas and `toDataURL` throws. An honest limitation
   * (documented in docs/11) rather than something to paper over. MSE-fed
   * players like YouTube are generally readable, because the page supplied
   * the bytes itself.
   */
  function captureFrame(video, config) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;

    const scale = Math.min(1, config.video.captureMaxDimension / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));

    try {
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return {
        dataUrl: canvas.toDataURL("image/jpeg", 0.7),
        width: canvas.width,
        height: canvas.height,
      };
    } catch {
      return null; // tainted canvas
    }
  }

  function clearOverlays(state) {
    state.overlays.forEach((o) => o.remove());
    state.overlays = [];
    state.regions = [];
  }

  function renderProtection(state, analysis, protectedImageUrl, frameSize) {
    clearOverlays(state);

    if (analysis.action === MyKidRisk.ProtectionAction.ALLOW || !protectedImageUrl) {
      return;
    }

    const host = getLayer(state);
    const isFullFrame = analysis.action === MyKidRisk.ProtectionAction.BLUR_FRAME;
    const regions = isFullFrame
      ? [{ bbox: { x: 0, y: 0, width: frameSize.width, height: frameSize.height } }]
      : analysis.protectionRegions;

    for (let i = 0; i < regions.length; i++) {
      const overlay = document.createElement("div");
      overlay.className = OVERLAY_CLASS;
      overlay.style.backgroundImage = `url(${protectedImageUrl})`;
      host.appendChild(overlay);
      state.overlays.push(overlay);
    }

    state.regions = regions;
    state.frameSize = frameSize;
    state.protectedUntil =
      performance.now() + state.config.video.temporalPersistenceMs;

    reposition(state);
    ensureRepositionLoop();
  }

  /**
   * Where is the video, in the coordinate space of our layer?
   *
   * The layer is absolutely positioned, so its containing block is
   * `offsetParent` — the nearest positioned ancestor, whatever the player
   * happens to nest it in. Measuring against that keeps positioning
   * correct without assuming any particular DOM shape or having to mutate
   * the page's own CSS to create a containing block.
   */
  function videoBoxInLayerSpace(state) {
    const videoRect = state.video.getBoundingClientRect();
    const anchor = state.layerEl?.offsetParent;

    if (!anchor) {
      return {
        left: videoRect.left + window.scrollX,
        top: videoRect.top + window.scrollY,
        width: videoRect.width,
        height: videoRect.height,
      };
    }

    const anchorRect = anchor.getBoundingClientRect();
    return {
      left: videoRect.left - anchorRect.left + anchor.scrollLeft,
      top: videoRect.top - anchorRect.top + anchor.scrollTop,
      width: videoRect.width,
      height: videoRect.height,
    };
  }

  function reposition(state) {
    const { video, overlays, regions, frameSize } = state;
    if (!video.isConnected || overlays.length === 0) return;

    getLayer(state); // re-attach if the player moved the video elsewhere

    const box = videoBoxInLayerSpace(state);
    if (box.width === 0 || box.height === 0) {
      overlays.forEach((o) => (o.style.display = "none"));
      return;
    }

    const scaleX = box.width / frameSize.width;
    const scaleY = box.height / frameSize.height;

    regions.forEach((region, index) => {
      const overlay = overlays[index];
      if (!overlay) return;

      const offsetX = region.bbox.x * scaleX;
      const offsetY = region.bbox.y * scaleY;

      overlay.style.display = "block";
      overlay.style.left = `${box.left + offsetX}px`;
      overlay.style.top = `${box.top + offsetY}px`;
      overlay.style.width = `${region.bbox.width * scaleX}px`;
      overlay.style.height = `${region.bbox.height * scaleY}px`;
      overlay.style.backgroundSize = `${box.width}px ${box.height}px`;
      overlay.style.backgroundPosition = `-${offsetX}px -${offsetY}px`;
    });
  }

  function repositionAll() {
    for (const state of videoStates.values()) reposition(state);
  }

  /**
   * Drop protection once it has gone unconfirmed for long enough.
   *
   * Temporal persistence keeps blur up briefly after the last harmful
   * frame, so a single missed detection doesn't flash content into view
   * (SKILL.md §17). Expiry is a question about *elapsed time*, so it must
   * not be driven by the render loop: requestAnimationFrame stops
   * completely when a page isn't painting (background tab, occluded
   * window), which would strand protection indefinitely. setTimeout is
   * throttled in the background but still fires, so the sampling loop
   * checks this too.
   *
   * @returns {boolean} whether protection is still standing
   */
  function expireIfStale(state) {
    if (state.overlays.length === 0) return false;

    if (performance.now() > state.protectedUntil) {
      clearOverlays(state);
      return false;
    }
    return true;
  }

  /**
   * Overlays follow the video as the player resizes — theater mode,
   * fullscreen, window resize, or a Shorts feed scrolling. Runs only while
   * something is actually protected, and stops itself otherwise.
   *
   * Repositioning genuinely only matters while painting, so this loop is
   * the right home for it — and the wrong home for expiry.
   */
  function ensureRepositionLoop() {
    if (rafHandle !== null) return;

    const tick = () => {
      let anyActive = false;

      for (const state of videoStates.values()) {
        if (!expireIfStale(state)) continue;
        anyActive = true;
        reposition(state);
      }

      rafHandle = anyActive ? requestAnimationFrame(tick) : null;
    };

    rafHandle = requestAnimationFrame(tick);
  }

  function countProtected() {
    let total = 0;
    for (const state of videoStates.values()) {
      if (state.overlays.length > 0) total++;
    }
    return total;
  }

  /**
   * Should we spend inference on this video right now?
   *
   * Sampling a paused, off-screen, or background-tab video is pure waste.
   * On a Shorts feed — a column of stacked <video> elements — the
   * viewport check is what keeps us analysing only the one being watched.
   */
  function shouldSample(state) {
    const { video } = state;
    if (state.stopped || state.unreadable || state.busy) return false;
    if (document.visibilityState !== "visible") return false;
    if (!state.visible) return false;
    if (video.paused || video.ended) return false;
    if (video.readyState < 2) return false;
    return true;
  }

  async function sampleFrame(state) {
    if (!shouldSample(state)) return;

    state.busy = true;
    try {
      const frame = captureFrame(state.video, state.config);
      if (!frame) {
        state.unreadable = true;
        stats.unreadable++;
        console.log(
          "[Bubble] video frame unreadable (cross-origin without CORS) — skipping this element"
        );
        return;
      }

      const response = await chrome.runtime.sendMessage({
        type: "MYKID_ANALYSE_IMAGE",
        imageUrl: frame.dataUrl,
      });

      stats.sampled++;

      if (!response || !response.ok) {
        stats.failed++;
        return;
      }

      if (response.analysis.action !== MyKidRisk.ProtectionAction.ALLOW) {
        renderProtection(state, response.analysis, response.protectedImageUrl, {
          width: frame.width,
          height: frame.height,
        });
      }
      // ALLOW leaves existing protection to expire via temporal
      // persistence rather than tearing it down instantly.
    } catch {
      stats.failed++;
    } finally {
      state.busy = false;
    }
  }

  /**
   * Self-rescheduling sample loop.
   *
   * Deliberately not setInterval: analysis takes ~230ms+ and the interval
   * at 2fps is 500ms, so on a slower machine fixed intervals would queue
   * work faster than it completes. Scheduling the next wait only after the
   * previous analysis finishes makes the rate self-limiting.
   */
  function scheduleNextSample(state, delayMs) {
    if (state.stopped) return;

    clearTimeout(state.timer);
    const interval = delayMs ?? 1000 / state.config.video.inferenceFps;

    state.timer = setTimeout(async () => {
      if (!state.video.isConnected) {
        unwatch(state.video);
        return;
      }

      // Expire here as well as in the render loop: this timer keeps
      // running when the page isn't painting, and stale blur sitting over
      // changed content is exactly what persistence must not cause.
      expireIfStale(state);

      await sampleFrame(state);
      scheduleNextSample(state);
    }, interval);
  }

  function getVisibilityObserver() {
    if (visibilityObserver) return visibilityObserver;

    visibilityObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const state = videoStates.get(entry.target);
          if (!state) continue;

          // "Visible" needs to hold for two opposite shapes, so neither
          // test alone is enough:
          //   - A Shorts feed stacks videos; adjacent ones peek in at the
          //     edges and shouldn't be sampled, hence the ratio test.
          //   - A player taller than the viewport can fill the screen at a
          //     low ratio, so a big absolute intersection counts too —
          //     otherwise a maximised video would be judged invisible and
          //     never analysed at all.
          const rect = entry.intersectionRect;
          const intersectionArea = rect.width * rect.height;
          const viewportArea = window.innerWidth * window.innerHeight;

          state.visible =
            entry.isIntersecting &&
            (entry.intersectionRatio >= 0.25 ||
              intersectionArea >= viewportArea * 0.3);
        }
      },
      { threshold: [0, 0.25, 0.5] }
    );

    return visibilityObserver;
  }

  /**
   * The media source changed underneath a reused element.
   *
   * This fires for genuinely different content (a new video, an ad roll)
   * *and* for events where the content is unchanged — quality switches
   * and rebuffering both tear down and rebuild the MSE source.
   *
   * The previous version cleared protection unconditionally, which meant a
   * quality switch briefly exposed content that had already been judged
   * harmful. Since we can't reliably tell the cases apart from the media
   * events alone, protection is now left standing and re-verified quickly:
   * an ad showing stale blur for a moment is a cosmetic cost, whereas
   * unblurring harmful content is a safety failure. Real navigation is
   * handled separately, by URL change, where clearing *is* correct.
   */
  function onSourceChanged(state) {
    stats.sourceChanges++;
    state.unreadable = false; // a new source may well be readable

    // Re-verify promptly rather than waiting out the full sample interval.
    scheduleNextSample(state, 150);
  }

  /** Start sampling a video element. */
  function watch(video, config) {
    if (videoStates.has(video)) return;
    if (!isCandidate(video, config)) return;

    const state = {
      video,
      config,
      layerEl: null,
      overlays: [],
      regions: [],
      frameSize: { width: 1, height: 1 },
      protectedUntil: 0,
      busy: false,
      unreadable: false,
      stopped: false,
      visible: true,
      timer: null,
    };

    videoStates.set(video, state);
    stats.discovered++;

    const onChange = () => onSourceChanged(state);
    state.listeners = [
      ["loadstart", onChange],
      ["emptied", onChange],
      ["loadedmetadata", onChange],
    ];
    for (const [event, handler] of state.listeners) {
      video.addEventListener(event, handler);
    }

    getVisibilityObserver().observe(video);
    scheduleNextSample(state);

    console.log(`[Bubble] watching video (${video.videoWidth}x${video.videoHeight})`);
  }

  function unwatch(video) {
    const state = videoStates.get(video);
    if (!state) return;

    state.stopped = true;
    clearTimeout(state.timer);
    clearOverlays(state);
    state.layerEl?.remove();

    for (const [event, handler] of state.listeners ?? []) {
      video.removeEventListener(event, handler);
    }
    if (visibilityObserver) visibilityObserver.unobserve(video);

    videoStates.delete(video);
  }

  function clearAll() {
    for (const video of Array.from(videoStates.keys())) unwatch(video);
  }

  /**
   * SPA navigation — genuinely different content, unlike a quality switch.
   *
   * Overlays describe the *previous* video, so they go immediately. This
   * is the opposite of the image policy, deliberately: an <img> keeps its
   * content across navigation, so stale-but-protecting is the safer
   * failure there. A reused <video> has its content replaced wholesale, so
   * keeping old protection would blur the wrong video entirely. Sampling
   * re-establishes protection within ~500ms if warranted.
   */
  function handleNavigation() {
    stats.navigations++;

    for (const state of videoStates.values()) {
      clearOverlays(state);
      state.protectedUntil = 0;
      state.unreadable = false;
      scheduleNextSample(state, 150);
    }

    for (const video of Array.from(videoStates.keys())) {
      if (!video.isConnected) unwatch(video);
    }
  }

  /**
   * Fullscreen no longer needs the layer re-parented — it lives inside the
   * player container, which is what goes fullscreen. Positioning still
   * needs a nudge, since the player's box changes.
   */
  function watchFullscreen() {
    document.addEventListener("fullscreenchange", repositionAll);
  }

  function getStats() {
    return { ...stats, watching: videoStates.size, protectedNow: countProtected() };
  }

  return {
    LAYER_CLASS,
    watch,
    unwatch,
    clearAll,
    handleNavigation,
    watchFullscreen,
    repositionAll,
    getStats,
  };
})();
