# Privacy Audit & Known Limitations

> Audited 2026-09-09 against the extension as built. Re-run the checks in
> §1 after any change that touches networking, logging or storage.

This document exists because SKILL.md rule 10 says not to pretend the
system is perfect, and §23 requires local-first processing to be
demonstrated rather than asserted. Everything below is either a measured
result or a known gap — nothing here is aspirational.

---

## 1. Privacy audit

### Method

Static audit of `apps/extension/src/`, reproducible with:

```bash
grep -rn "fetch(\|XMLHttpRequest\|WebSocket\|sendBeacon" apps/extension/src/
grep -rnE "https?://[a-zA-Z0-9.-]+" apps/extension/src/ apps/extension/manifest.json
grep -rn "console\." apps/extension/src/
grep -rn "storage.local.set" apps/extension/src/
```

### Findings

| Check | Result |
|---|---|
| Outbound network calls | **2**, both `fetch(imageUrl)` in the offscreen document — the image being analysed. Nothing else. |
| Hardcoded external hosts | **None.** No analytics, no telemetry, no remote config, no model CDN. |
| Data sent off-device | **None.** No image data, URLs, page content or identifiers leave the browser. |
| Model execution | Entirely local (`onnxruntime-web`, wasm, bundled in the extension). |
| Logged content | Counts, actions, risk levels and video dimensions only. **No URLs, no image data, no page text.** Error messages carry an HTTP status, never the URL. |
| Persisted data | One key, `mykidConfig` — the on/off flag and the debug label. No history, no per-site records, no identifiers. |
| Permissions | `offscreen`, `storage`, and `host_permissions: <all_urls>`. |

### Two things worth stating plainly rather than burying

**Images are fetched a second time.** The content script passes an image
*URL* to the offscreen document, which re-fetches it with the extension's
own privileges. This is what sidesteps canvas tainting on cross-origin
images, and it's why protection works at all on real sites. The cost: each
analysed image is requested twice, so a server sees a duplicate hit. No new
destination is contacted — only URLs the page already loaded — but the
request pattern is observably different from browsing without the
extension.

**`<all_urls>` is broad, and necessarily so.** It's required to fetch
images from arbitrary CDNs. It's the permission that makes the extension
work everywhere, and also the one a cautious user should scrutinise. It
grants the ability to read page content and fetch any URL; the audit above
is the evidence that the extension does neither beyond what's described.

---

## 2. Known limitations

### Detection

1. **Accuracy on harmful content is unmeasured.** There is no harmful
   evaluation set in this project, so no test here asserts that gore or
   sexual content is actually caught. The model's authors report 97.76% on
   a proprietary dataset; that is self-reported, on their own data, and is
   not independent evidence. **This is the single largest gap.**
2. **Safe-content behaviour *is* measured.** All four safe fixtures score
   0.03–0.08 against a 0.6 threshold, so indiscriminate blurring looks
   unlikely. That is a false-positive check only.
3. **Covered categories are gore/violence (NSFL), sexual (NSFW), and
   knife/scissors.** Not covered: firearms, drugs, self-harm, hate
   symbols, or anything requiring text or audio understanding.
4. **`SceneRisk.violence` is always 0.** The classifier folds violence
   into its NSFL score; the field is reserved for a future dedicated model
   rather than filled with a number that was never measured.
5. **The 0.6 scene threshold is a default, not a tuned value.** Tuning it
   requires the evaluation set from (1). SKILL.md §27 argues for erring
   toward over-blurring; that argument can't be acted on responsibly
   without data.

### Coverage gaps in the browser

6. **`blob:` images are skipped.** They're scoped to the page's origin and
   the offscreen document cannot re-fetch them.
7. **Cross-origin video without CORS is skipped.** Reading such frames
   taints the canvas and `toDataURL` throws. MSE-fed players (YouTube) are
   generally readable because the page supplied the bytes itself.
8. **Only images near the viewport are analysed.** Fast scrolling can
   outrun analysis, so an image may be visible briefly before protection
   lands.
9. **Video is sampled at ~2 fps.** Harmful content shorter than roughly
   500ms can pass without being caught. Raising this is bounded by
   inference cost, not by choice.
10. **CSS background images are not handled** — only `<img>` and `<video>`
    elements.

### Behavioural

11. **Protection is an overlay, not enforcement.** It reduces exposure; it
    does not prevent a determined user from removing it via devtools or by
    switching the extension off. This is a harm-reduction tool, not a
    security control, and shouldn't be described as one.
12. **First analysis on a page pays model load.** Subsequent images reuse
    the loaded session.
13. **Temporal persistence keeps video protection for ~2s after the last
    harmful frame**, so protection can briefly linger over content that has
    already changed. Chosen deliberately: flicker that exposes content is
    worse than blur that outstays it.

### Licensing

14. **YOLOv11n is AGPL-3.0** and is the only encumbered component. The
    scene classifier is MIT over an Apache-2.0 base. Disabling object
    detection (`detection.objectDetectionEnabled`) leaves a fully
    permissive stack, costs ~4× less per image, and loses only
    knife/scissors region blur.

### Verification status

15. **Confirmed in-browser:** extension loads, engine loads, detection
    maths matches the Python reference to ~1px, and protection applies to
    real page images.
16. **Not yet confirmed in-browser:** video protection (11a–11c), infinite
    scroll, and SPA re-scan. The code is written and unit-tested; it has
    not been observed working. See `docs/11` for the current status table.

---

## 3. What would close the biggest gaps

| Gap | What it needs |
|---|---|
| Unmeasured detection accuracy | A licensed harmful-content evaluation set (owner decision — see `docs/11`, Open Decisions) |
| Firearms not detected | A weapon-dataset fine-tune (Phase C) |
| Video verification | A browser session with the toggle on (Phase E) |
| AGPL dependency | Disable object detection, or license YOLO commercially |
