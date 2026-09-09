/**
 * MyKid — Browser-side configuration (Phase 10b-iii)
 *
 * Mirrors configs/default.yaml. Per SKILL.md §26 and agent rule 7, every
 * threshold lives here — no magic numbers scattered through the extension.
 * Keep the defaults in sync with the Python config; Phase 13 checks that
 * both pipelines reach the same decisions, which only holds if they're
 * working from the same numbers.
 *
 * Exposes a global `MyKidConfig` (classic script, no bundler).
 */
var MyKidConfig = (function () {
  const DEFAULTS = {
    // The product's main control. On by default: this is a safety tool,
    // and one that starts switched off protects nobody until someone
    // remembers to enable it.
    //
    // "Off" means genuinely idle — no analysis, no model loading, no
    // overlays — not merely hidden protection. A safety tool that keeps
    // burning CPU while the user believes it is off is both dishonest and
    // a reason to uninstall it.
    enabled: true,
    // Harm categories, each independently switchable. All on by default:
    // a safety tool should cover everything it can unless the user
    // deliberately narrows it, not require them to opt in category by
    // category and discover the gaps later.
    //
    // Filtering happens at the source (offscreen.js) rather than in the
    // risk engine — a disabled category's score is zeroed before the
    // engine ever sees it. That keeps the decision logic identical to the
    // Python reference, so cross-pipeline parity holds under every
    // combination of settings rather than only the default one.
    categories: {
      // NSFL — blood, gore, injury, violent scenes. Scene-level, so it
      // protects the whole image.
      gore: true,
      // NSFW — nudity and sexual content. Scene-level.
      //
      // Deliberately NOT exposed as a switch in the popup, and forced on
      // in fromOverrides(). Two reasons: it is not a protection a parent
      // would sensibly turn off, and the label does not belong in a panel
      // a child may open. The key stays here so the engine still reads it
      // through the normal path rather than special-casing the category.
      sexual: true,
      // knife / scissors, via object detection. Region-level, so only the
      // object is blurred. Costs ~4x the scene classifier.
      weapons: true,
    },
    detection: {
      // Minimum confidence for a detection to be considered at all.
      objectConfidenceThreshold: 0.5,
      // Scene-level risk score that forces whole-image protection.
      //
      // The classifier's three outputs are a softmax, so 0.6 on gore or
      // sexual means the model is fairly confident. SKILL.md §27 says
      // missing harmful content is the worse failure, which argues for
      // going lower — but lowering it without a harmful evaluation set
      // would be guessing, and an over-blurring extension gets switched
      // off, which protects nobody. Left at the documented default until
      // Phase 9c provides data to tune against.
      sceneRiskThreshold: 0.6,
      // Object detection (knife/scissors) is ~15x more expensive than the
      // scene classifier and covers far less of the harm surface. Kept on
      // by default for region-level blur, but switchable — turning it off
      // also removes the only AGPL-licensed model from the stack
      // (docs/06-ai-detection.md, "Licensing consequence").
      objectDetectionEnabled: true,
      // If harmful regions cover more than this fraction of the image,
      // blur the whole thing rather than a patchwork of boxes.
      maxAllowedObjectAreaRatio: 0.6,
    },
    protection: {
      // Fraction of each box's size added as padding before blurring.
      blurPadding: 0.1,
      // gaussian | pixelate | cover
      mode: "gaussian",
    },
    browser: {
      // Images smaller than this (px, either dimension) are skipped.
      // Icons, avatars, spacer GIFs and tracking pixels dominate real
      // pages and aren't worth ~230ms of inference each.
      //
      // Kept low deliberately: an earlier 128px floor rejected legitimate
      // content, because grid thumbnails are often wide but short
      // (~360x202, and smaller in dense layouts). 64 clears avatars and
      // icons while keeping real thumbnails.
      minImageDimension: 64,
      // Concurrent inferences. At ~230ms each this paces the work without
      // saturating the machine.
      maxConcurrentInferences: 2,
    },
    video: {
      // Frames per second sent to the model.
      //
      // configs/default.yaml uses 8 for the Python pipeline, which
      // processes offline and can afford it. In-browser inference measures
      // ~230ms per frame, so 8fps is arithmetically impossible — it would
      // need a 125ms budget. 2fps leaves headroom for the page itself to
      // stay responsive (SKILL.md §22), and is honest about the ceiling
      // rather than configuring a rate we can't hit.
      inferenceFps: 2,
      // Keep protection up this long after the last harmful frame.
      // Detection at 2fps is jittery — an object missed on a single frame
      // shouldn't flash the content into view. This is the browser
      // equivalent of temporal_persistence_frames in the Python pipeline
      // (SKILL.md §17), expressed in time because sampling here is
      // time-based rather than frame-indexed.
      temporalPersistenceMs: 2000,
      // Videos smaller than this (px, either dimension) are ignored —
      // background/decorative loops, autoplay previews in UI chrome.
      minVideoDimension: 128,
      // Captured frames are downscaled to this before being sent for
      // analysis. The model letterboxes to 640 anyway, so anything larger
      // is wasted encoding and message payload.
      captureMaxDimension: 640,
    },
    debug: {
      // TEST AFFORDANCE, OFF BY DEFAULT.
      //
      // The COCO model only gives us knife and scissors as safety-relevant
      // classes (docs/06-ai-detection.md §6), and the repo currently has no
      // harmful test fixtures (Phase 9c). That makes the DETECT -> DECIDE ->
      // PROTECT chain impossible to *see* working on ordinary pages, since
      // nothing ever triggers it.
      //
      // Setting this to a COCO label (e.g. "person") makes the risk engine
      // treat that label as harmful, so protection can be observed on real
      // content. It changes nothing about the production defaults — it is a
      // debugging lens, not a safety rule. Toggled from the popup.
      treatLabelAsHarmful: null,
    },
  };

  /** Safety-relevant labels. Ported from packages/shared/labels.py. */
  const SAFETY_RELEVANT_LABELS = ["knife", "scissors"];

  let overrides = {};

  /**
   * Build a config from a plain overrides object, with no storage access.
   *
   * This exists because contexts differ in what they can reach:
   * `chrome.storage` is available to content scripts and the service
   * worker, but offscreen documents get only a restricted subset of the
   * extension APIs. The offscreen document therefore receives its
   * overrides in the message rather than reading them itself.
   */
  function fromOverrides(sourceOverrides) {
    return {
      ...DEFAULTS,
      enabled: sourceOverrides?.enabled ?? DEFAULTS.enabled,
      categories: {
        ...DEFAULTS.categories,
        ...(sourceOverrides?.categories ?? {}),
        // Always on, whatever is stored — including a `false` left behind
        // by an earlier build that did expose the switch.
        sexual: true,
      },
      detection: { ...DEFAULTS.detection, ...(sourceOverrides?.detection ?? {}) },
      debug: { ...DEFAULTS.debug, ...(sourceOverrides?.debug ?? {}) },
    };
  }

  /** Load persisted overrides (currently just the debug toggle). */
  async function load() {
    if (!chrome?.storage?.local) {
      // Deliberately loud. An earlier version swallowed this in a bare
      // catch and silently returned defaults, which made the debug label
      // appear to do nothing — detection ran, but the risk engine never
      // saw the setting. Silent fallbacks hide exactly this class of bug
      // (SKILL.md rule 4).
      console.warn(
        "[Bubble] chrome.storage unavailable in this context — using defaults. " +
          "Config must be passed in explicitly here."
      );
      overrides = {};
      return get();
    }

    try {
      const stored = await chrome.storage.local.get("mykidConfig");
      overrides = stored.mykidConfig ?? {};
    } catch (err) {
      console.warn("[Bubble] could not read stored config:", err);
      overrides = {};
    }
    return get();
  }

  function get() {
    return fromOverrides(overrides);
  }

  async function setDebugLabel(label) {
    overrides = { ...overrides, debug: { treatLabelAsHarmful: label || null } };
    await chrome.storage.local.set({ mykidConfig: overrides });
  }

  async function setCategory(name, on) {
    overrides = {
      ...overrides,
      categories: { ...(overrides.categories ?? {}), [name]: Boolean(on) },
    };
    await chrome.storage.local.set({ mykidConfig: overrides });
  }

  async function setEnabled(enabled) {
    overrides = { ...overrides, enabled: Boolean(enabled) };
    await chrome.storage.local.set({ mykidConfig: overrides });
  }

  /**
   * Labels the risk engine should treat as harmful, given current config.
   */
  function harmfulLabels(config) {
    // Weapons off means the object-detection labels carry no weight, even
    // if detection somehow still ran.
    const labels = config.categories.weapons ? [...SAFETY_RELEVANT_LABELS] : [];

    // Accepts an array (multiple test labels) or a single string, so
    // settings saved by older builds keep working.
    const extra = config.debug.treatLabelAsHarmful;
    if (Array.isArray(extra)) {
      labels.push(...extra.filter(Boolean));
    } else if (extra) {
      labels.push(extra);
    }
    return labels;
  }

  return {
    DEFAULTS,
    SAFETY_RELEVANT_LABELS,
    load,
    get,
    fromOverrides,
    setDebugLabel,
    setEnabled,
    setCategory,
    harmfulLabels,
  };
})();
