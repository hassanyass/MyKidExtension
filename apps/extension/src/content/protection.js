/**
 * MyKid — Protection overlay placement (Phase 10c)
 *
 * Positions the protected image (rendered in the offscreen document by
 * protection-render.js) on top of the original.
 *
 * Two deliberate choices:
 *
 * 1. **Overlay, never mutate.** We don't rewrite `src` or replace pixels
 *    in place, because that fights the page: SPAs reset `src` on
 *    navigation, lazy-loaders reassign it, and `srcset` re-picks on
 *    resize. An overlay leaves the page's own state untouched.
 *
 * 2. **Real pixels, not a CSS filter.** An earlier version used
 *    `backdrop-filter`, which depends on the stacking context it lands in
 *    and silently fails on some real layouts. Showing an actual rendered
 *    image behaves the same everywhere.
 *
 * Overlays follow their image through scrolling, resizing and reflow via
 * ResizeObserver plus rAF-throttled scroll/resize handling.
 *
 * Exposes a global `MyKidProtection`.
 */
var MyKidProtection = (function () {
  const OVERLAY_CLASS = "mykid-protection-overlay";
  const LAYER_ID = "mykid-protection-layer";

  let stylesInjected = false;
  let layer = null;
  let resizeObserver = null;

  const activeProtections = new Map(); // image element -> protection record

  function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;

    const style = document.createElement("style");
    style.textContent = `
      #${LAYER_ID} {
        position: absolute;
        top: 0; left: 0; width: 0; height: 0;
        pointer-events: none;
        z-index: 2147483646;
      }
      .${OVERLAY_CLASS} {
        position: absolute;
        pointer-events: none;
        object-fit: fill;
        display: block;
        max-width: none;
        max-height: none;
      }
    `;
    (document.head || document.documentElement).appendChild(style);
  }

  function getLayer() {
    if (layer && layer.isConnected) return layer;

    layer = document.createElement("div");
    layer.id = LAYER_ID;
    (document.body || document.documentElement).appendChild(layer);
    return layer;
  }

  function getResizeObserver() {
    if (resizeObserver) return resizeObserver;

    resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const protection = activeProtections.get(entry.target);
        if (protection) reposition(protection);
      }
    });
    return resizeObserver;
  }

  /** Page-coordinate rect (viewport rect + scroll offset). */
  function pageRect(element) {
    const rect = element.getBoundingClientRect();
    return {
      left: rect.left + window.scrollX,
      top: rect.top + window.scrollY,
      width: rect.width,
      height: rect.height,
    };
  }

  function reposition(protection) {
    const { image, overlay } = protection;

    if (!image.isConnected) {
      remove(image);
      return;
    }

    const box = pageRect(image);

    // Hidden, collapsed, or scrolled far out of layout — don't draw.
    if (box.width === 0 || box.height === 0) {
      overlay.style.display = "none";
      return;
    }

    overlay.style.display = "block";
    overlay.style.left = `${box.left}px`;
    overlay.style.top = `${box.top}px`;
    overlay.style.width = `${box.width}px`;
    overlay.style.height = `${box.height}px`;
  }

  /**
   * Show the protected version of an image.
   * ALLOW is a no-op — safe content is left completely untouched
   * (SKILL.md §33: "Safe images remain unchanged").
   */
  function apply(image, analysis, protectedImageUrl) {
    if (
      analysis.action === MyKidRisk.ProtectionAction.ALLOW ||
      !protectedImageUrl
    ) {
      return null;
    }

    // Replacing an existing protection (e.g. the image's src changed).
    remove(image);

    injectStyles();

    const overlay = document.createElement("img");
    overlay.className = OVERLAY_CLASS;
    overlay.src = protectedImageUrl;
    overlay.alt = "";
    overlay.setAttribute("aria-hidden", "true");
    getLayer().appendChild(overlay);

    const protection = { image, overlay, action: analysis.action };
    activeProtections.set(image, protection);
    getResizeObserver().observe(image);
    reposition(protection);

    return protection;
  }

  function remove(image) {
    const protection = activeProtections.get(image);
    if (!protection) return;

    protection.overlay.remove();
    activeProtections.delete(image);
    if (resizeObserver) resizeObserver.unobserve(image);
  }

  /** Drop every overlay — used on SPA navigation. */
  function clearAll() {
    for (const image of Array.from(activeProtections.keys())) {
      remove(image);
    }
  }

  function repositionAll() {
    for (const protection of Array.from(activeProtections.values())) {
      reposition(protection);
    }
  }

  /**
   * Overlays live in page coordinates, so they must follow the image when
   * the page reflows — which on a lazy-loading feed happens constantly.
   */
  function watchLayout() {
    let frame = null;
    const schedule = () => {
      if (frame !== null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        repositionAll();
      });
    };

    window.addEventListener("resize", schedule, { passive: true });
    window.addEventListener("scroll", schedule, { passive: true, capture: true });
  }

  function count() {
    return activeProtections.size;
  }

  function isProtected(image) {
    return activeProtections.has(image);
  }

  return {
    apply,
    remove,
    clearAll,
    repositionAll,
    watchLayout,
    count,
    isProtected,
    OVERLAY_CLASS,
  };
})();
