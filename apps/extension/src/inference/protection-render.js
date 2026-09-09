/**
 * MyKid — Protection renderer (Phase 10c)
 *
 * Produces the actual protected pixels for an image. Port of
 * packages/protection/protection_engine.py, running in the offscreen
 * document where the decoded image is already available.
 *
 * Why render pixels here instead of applying a CSS filter in the page:
 * `backdrop-filter` depends on the stacking context it lands in, so it
 * silently fails on some real-world layouts, and CSS has no true
 * pixelation. Here we have the real bitmap, so gaussian / pixelate /
 * cover can all be applied faithfully — and the result is a plain image
 * the content script just positions, which behaves identically no matter
 * how the host page is structured.
 *
 * This layer protects; it does not detect and it does not decide
 * (SKILL.md §38).
 *
 * Exposes a global `MyKidProtectionRender`.
 */
var MyKidProtectionRender = (function () {
  // Blurred output carries no fine detail by definition, so there's no
  // reason to ship full-resolution pixels back to the content script.
  // Caps the data URL size on large images.
  const MAX_OUTPUT_DIMENSION = 800;

  // Gaussian blur radius as a fraction of the region's shorter side, so
  // small regions aren't obliterated and large ones aren't under-blurred.
  // Mirrors the kernel-size heuristic in protection_engine.py.
  const BLUR_RATIO = 0.15;
  const MIN_BLUR_PX = 8;
  const MAX_BLUR_PX = 40;

  function blurRadiusFor(width, height) {
    const radius = Math.min(width, height) * BLUR_RATIO;
    return Math.max(MIN_BLUR_PX, Math.min(MAX_BLUR_PX, radius));
  }

  /** Gaussian blur one rectangle of the canvas, in place. */
  function applyGaussian(ctx, source, region, scale) {
    const { x, y, width, height } = scaleRegion(region, scale);
    if (width <= 0 || height <= 0) return;

    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, width, height);
    ctx.clip();
    ctx.filter = `blur(${blurRadiusFor(width, height)}px)`;
    // Redraw the whole image through the clip so the blur samples
    // neighbouring pixels rather than smearing the region's own edges.
    ctx.drawImage(source, 0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
    ctx.filter = "none";
  }

  /** True pixelation: downscale hard, then upscale with smoothing off. */
  function applyPixelate(ctx, region, scale) {
    const { x, y, width, height } = scaleRegion(region, scale);
    if (width <= 0 || height <= 0) return;

    const blocks = 12;
    const smallW = Math.max(1, Math.floor(width / blocks));
    const smallH = Math.max(1, Math.floor(height / blocks));

    const scratch = new OffscreenCanvas(smallW, smallH);
    const scratchCtx = scratch.getContext("2d");
    scratchCtx.drawImage(ctx.canvas, x, y, width, height, 0, 0, smallW, smallH);

    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scratch, 0, 0, smallW, smallH, x, y, width, height);
    ctx.restore();
  }

  /** Solid cover — the strongest option. */
  function applyCover(ctx, region, scale) {
    const { x, y, width, height } = scaleRegion(region, scale);
    if (width <= 0 || height <= 0) return;

    ctx.save();
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(x, y, width, height);
    ctx.restore();
  }

  function scaleRegion(region, scale) {
    return {
      x: Math.round(region.x * scale),
      y: Math.round(region.y * scale),
      width: Math.round(region.width * scale),
      height: Math.round(region.height * scale),
    };
  }

  function applyMode(ctx, source, region, mode, scale) {
    switch (mode) {
      case "pixelate":
        return applyPixelate(ctx, region, scale);
      case "cover":
        return applyCover(ctx, region, scale);
      case "gaussian":
      default:
        return applyGaussian(ctx, source, region, scale);
    }
  }

  /**
   * Render the protected version of an image.
   *
   * @param {ImageBitmap} bitmap  the decoded original
   * @param {Object} analysis     result from MyKidRisk.evaluate
   * @param {Object} config
   * @returns {Promise<string|null>} data URL of the protected image, or
   *          null when no protection applies (ALLOW).
   */
  async function render(bitmap, analysis, config) {
    if (analysis.action === MyKidRisk.ProtectionAction.ALLOW) return null;

    const scale = Math.min(
      1,
      MAX_OUTPUT_DIMENSION / Math.max(bitmap.width, bitmap.height)
    );
    const outputWidth = Math.max(1, Math.round(bitmap.width * scale));
    const outputHeight = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = new OffscreenCanvas(outputWidth, outputHeight);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0, outputWidth, outputHeight);

    const mode = config.protection.mode;

    if (analysis.action === MyKidRisk.ProtectionAction.BLUR_FRAME) {
      applyMode(
        ctx,
        bitmap,
        { x: 0, y: 0, width: bitmap.width, height: bitmap.height },
        mode,
        scale
      );
    } else {
      for (const region of analysis.protectionRegions) {
        applyMode(ctx, bitmap, region.bbox, region.mode || mode, scale);
      }
    }

    const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.82 });
    return await blobToDataUrl(blob);
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
  }

  return { render, MAX_OUTPUT_DIMENSION };
})();
