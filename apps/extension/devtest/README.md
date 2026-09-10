# Dev test harnesses

## live-rescan.html

Runs the real content script against stubbed extension APIs, so config
changes can be checked for taking effect **live** — which is difficult to
judge on a real site, where you cannot tell "the setting did not apply"
apart from "nothing on this page matched".

The stub reports a `person` in every image, so toggling the practice label
should blur or unblur everything without a reload.

Because file previews are served from a `data:` URL, relative `<script src>`
paths do not resolve. Inline them first:

```bash
python - <<'PY'
import pathlib
base = pathlib.Path("apps/extension")
html = (base / "devtest" / "live-rescan.html").read_text(encoding="utf-8")
scripts = ["src/shared/config.js", "src/inference/risk.js",
           "src/content/protection.js", "src/content/video-protection.js",
           "src/content/content-script.js"]
inlined = "\n".join("<script>\n" + (base / s).read_text(encoding="utf-8") + "\n</script>"
                    for s in scripts)
html = html.replace("\n".join(f'    <script src="../{s}"></script>' for s in scripts), inlined)
(base / "devtest" / "live-rescan.inlined.html").write_text(html, encoding="utf-8")
PY
```

Then open `apps/extension/devtest/live-rescan.inlined.html`.

**Found by this harness:** protection overlays are themselves `<img>`
elements, so they were being discovered and analysed in turn — each
protection spawning another overlay. Harmless with the 1px stub used here,
but a real protected render (up to 800px) clears the size floor and feeds
straight back in, flooding the two inference slots with the extension's own
overlays and starving real images.

## video-check.html

Runs the real `video-protection.js` against a genuinely playing `<video>`.
The video is a canvas `captureStream()`, so frames are same-origin and
readable — the same situation as YouTube, where the page supplies the bytes
itself. The page also fakes a player container with controls at `z-index: 30`,
so overlay layering can be checked.

Inline the scripts first (as for `live-rescan.html`), then press "run checks".

Ten checks: discovery, frame capture, that the captured frame is a real
readable JPEG, canvas not tainted, overlay appears when flagged, overlay
positioned inside the video box, blur below the player controls, temporal
persistence holding, persistence expiring, and paused video not being
sampled.

**Found by this harness:** protection expiry was driven by
`requestAnimationFrame`. Expiry is a question about elapsed *time*, and rAF
stops entirely when a page isn't painting (background tab, occluded window),
which stranded blur indefinitely. Expiry now also runs from the sampling
timer, which throttles in the background but keeps firing.
