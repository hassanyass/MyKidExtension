/**
 * MyKid — Popup script (Phase 10a counts + Phase 10b-i self-test status)
 *
 * Asks the content script of the active tab for its element counts and
 * the detection-engine self-test result, and renders both. This is
 * purely a verification aid — the real UI (per-detection status,
 * protection toggle, etc.) comes later.
 */
const statusEl = document.getElementById("status");

/**
 * The standalone engine self-test used to run on every page load and be
 * reported here. It's gone: the analysis path itself now proves the engine
 * works (and reports timings), so a separate self-test only added a model
 * load and a full inference every time the popup opened. "Run detection
 * test" below covers the same ground on demand.
 */

function renderLabelCounts(labelCounts) {
  const entries = Object.entries(labelCounts ?? {}).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return "";

  const top = entries
    .slice(0, 6)
    .map(([label, count]) => `${label}&nbsp;${count}`)
    .join(" · ");

  return `<p class="count-line detected">Detected: ${top}</p>`;
}

function renderSceneStats(stats) {
  if (!stats || stats.sceneChecked === 0) return "";

  // Peak scores make the harm detector's behaviour legible: near-zero
  // across a whole page means it ran and saw nothing alarming, which is a
  // different situation from it not running at all.
  return `
    <p class="count-line">Harm scan: <strong>${stats.sceneChecked}</strong> checked
      · <strong>${stats.sceneFlagged}</strong> flagged</p>
    <p class="count-line detected">Peak scores — gore ${stats.peakGore.toFixed(3)}
      · sexual ${stats.peakSexual.toFixed(3)}</p>
  `;
}

function renderStats(stats) {
  if (!stats) return "";

  // When nothing was protected, say why rather than showing a bare zero —
  // "analysed nothing" and "analysed plenty, matched nothing" need
  // completely different fixes.
  let diagnosis = "";
  if (stats.protected === 0) {
    if (stats.processed === 0 && stats.pending === 0) {
      diagnosis = `<p class="count-line warning">No images analysed yet
        (scroll, or they may all be below the size threshold).</p>`;
    } else if (Object.keys(stats.labelCounts ?? {}).length === 0) {
      diagnosis = `<p class="count-line warning">Analysed, but the model
        found no objects it recognises.</p>`;
    } else {
      const harmful = stats.harmfulLabels
        ? stats.harmfulLabels.join(", ")
        : "unknown";
      diagnosis = `<p class="count-line">Objects found, but none match the
        harmful set in force: <strong>${harmful}</strong></p>`;
    }
  }

  return `
    <p class="count-line">Analysed: <strong>${stats.processed}</strong>
      · skipped ${stats.skipped} · failed ${stats.failed}
      · pending ${stats.pending ?? 0}</p>
    <p class="count-line">Protected: <strong>${stats.protected}</strong>
      (${stats.protectionsActive} overlay(s) active)</p>
    ${renderSceneStats(stats)}
    ${renderLabelCounts(stats.labelCounts)}
    ${diagnosis}
  `;
}

function renderVideoStats(video) {
  if (!video || video.watching === 0) return "";
  return `
    <p class="count-line">Video: watching <strong>${video.watching}</strong>
      · ${video.sampled} frame(s) sampled
      · ${video.protectedNow} protected now</p>
    ${
      video.sourceChanges > 0
        ? `<p class="count-line">${video.sourceChanges} source change(s) handled</p>`
        : ""
    }
    ${
      video.unreadable > 0
        ? `<p class="count-line warning">${video.unreadable} video(s) unreadable
             (cross-origin without CORS)</p>`
        : ""
    }
  `;
}

function render(counts) {
  renderToggle(counts.enabled);
  if (counts.stats?.activeCategories) {
    renderCategories(counts.stats.activeCategories);
  }

  // Showing analysis stats while switched off would imply work is
  // happening that isn't.
  if (!counts.enabled) {
    statusEl.innerHTML = `
      <p class="count-line">Found on this page: ${counts.images} image(s),
        ${counts.videos} video(s).</p>
      <p class="count-line warning">Not being checked — protection is off.</p>
    `;
    return;
  }

  statusEl.innerHTML = `
    <p class="count-line">Images found: <strong>${counts.images}</strong></p>
    <p class="count-line">Videos found: <strong>${counts.videos}</strong></p>
    ${renderStats(counts.stats)}
    ${renderVideoStats(counts.video)}
  `;
}

// --- Power toggle: the product's main control -----------------------
const powerButton = document.getElementById("power");
const toggleState = document.getElementById("toggle-state");
const toggleHint = document.getElementById("toggle-hint");

/**
 * Push the config change to every open tab, not just the active one.
 *
 * A safety toggle that only applies to the tab you happened to be looking
 * at would leave other tabs unprotected while reporting itself as on.
 * Tabs without a content script (chrome:// pages, or ones not reloaded
 * since install) simply error and are skipped.
 */
function broadcastRescan() {
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      if (tab.id === undefined) continue;
      chrome.tabs.sendMessage(tab.id, { type: "MYKID_RESCAN" }, () => {
        void chrome.runtime.lastError; // no content script here; expected
      });
    }
  });
}

function renderToggle(enabled) {
  powerButton.setAttribute("aria-checked", String(enabled));
  powerButton.classList.toggle("on", enabled);
  toggleState.textContent = enabled ? "Protection on" : "Protection off";
  toggleState.classList.toggle("off", !enabled);
  toggleHint.textContent = enabled
    ? "Checking images and video as you browse."
    : "Nothing is being checked or blurred.";
}

function setEnabled(enabled) {
  chrome.storage.local.get("mykidConfig", (stored) => {
    const overrides = stored?.mykidConfig ?? {};
    overrides.enabled = enabled;
    chrome.storage.local.set({ mykidConfig: overrides }, () => {
      renderToggle(enabled);
      broadcastRescan();
      // Re-read counts so the panel reflects the new state right away.
      setTimeout(refreshStatus, 150);
    });
  });
}

powerButton.addEventListener("click", () => {
  setEnabled(powerButton.getAttribute("aria-checked") !== "true");
});

// --- Harm categories ------------------------------------------------
const categoryInputs = document.querySelectorAll("#categories input[data-category]");

function renderCategories(categories) {
  for (const input of categoryInputs) {
    // Default on: absent means "not yet configured", which for a safety
    // tool should mean covered, not skipped.
    input.checked = categories?.[input.dataset.category] ?? true;
  }
}

for (const input of categoryInputs) {
  input.addEventListener("change", () => {
    chrome.storage.local.get("mykidConfig", (stored) => {
      const overrides = stored?.mykidConfig ?? {};
      overrides.categories = {
        ...(overrides.categories ?? {}),
        [input.dataset.category]: input.checked,
      };
      chrome.storage.local.set({ mykidConfig: overrides }, () => {
        broadcastRescan();
        setTimeout(refreshStatus, 150);
      });
    });
  });
}

// --- Debug labels (multi-select) ------------------------------------
const harmfulLabelSelect = document.getElementById("harmful-label");

function selectedLabels() {
  return Array.from(harmfulLabelSelect.selectedOptions)
    .map((option) => option.value)
    .filter(Boolean);
}

chrome.storage.local.get("mykidConfig", (stored) => {
  const config = stored?.mykidConfig ?? {};

  // Accepts the older single-string form as well as the array, so a
  // setting saved by a previous build still applies.
  const saved = config.debug?.treatLabelAsHarmful;
  const labels = Array.isArray(saved) ? saved : saved ? [saved] : [];
  for (const option of harmfulLabelSelect.options) {
    option.selected = labels.includes(option.value);
  }

  renderToggle(config.enabled ?? true);
  renderCategories(config.categories);
});

harmfulLabelSelect.addEventListener("change", () => {
  const labels = selectedLabels();
  chrome.storage.local.get("mykidConfig", (stored) => {
    const overrides = stored?.mykidConfig ?? {};
    overrides.debug = { treatLabelAsHarmful: labels.length ? labels : null };
    chrome.storage.local.set({ mykidConfig: overrides }, () => {
      broadcastRescan();
      setTimeout(refreshStatus, 150);
    });
  });
});

function renderUnavailable() {
  statusEl.innerHTML = `
    <p class="warning">MyKid isn't active on this page.</p>
    <p class="count-line">(Restricted pages like chrome:// or the Web Store can't be inspected.)</p>
  `;
}

// --- Phase 10b-ii: detection parity test ---------------------------
//
// Runs the browser detection pipeline over a fixed sample image, so its
// output can be diffed against `python scripts/parity_reference.py` on the
// same file. Same model, same letterboxing, same decoding — the numbers
// should line up.
const parityButton = document.getElementById("run-parity");
const parityOutput = document.getElementById("parity-output");

parityButton.addEventListener("click", () => {
  parityButton.disabled = true;
  parityOutput.textContent = "Running…";

  chrome.runtime.sendMessage({ type: "MYKID_PARITY_TEST" }, (result) => {
    parityButton.disabled = false;

    if (chrome.runtime.lastError || !result) {
      parityOutput.innerHTML = `<p class="warning">${
        chrome.runtime.lastError?.message ?? "no response"
      }</p>`;
      return;
    }
    if (!result.ok) {
      parityOutput.innerHTML = `<p class="warning">${result.error}</p>`;
      return;
    }

    const rows = result.detections
      .map(
        (d) =>
          `<tr><td>${d.label}</td><td>${d.confidence.toFixed(4)}</td>` +
          `<td>${d.bbox.x.toFixed(1)}, ${d.bbox.y.toFixed(1)}, ` +
          `${d.bbox.width.toFixed(1)}, ${d.bbox.height.toFixed(1)}</td></tr>`
      )
      .join("");

    parityOutput.innerHTML = `
      <p class="count-line">${result.detections.length} detection(s) ·
        pre ${result.timings.preprocessMs}ms ·
        infer ${result.timings.inferMs}ms ·
        decode ${result.timings.decodeMs}ms</p>
      <table>
        <tr><th>label</th><th>conf</th><th>x, y, w, h</th></tr>
        ${rows}
      </table>
    `;
    console.log("[MyKid] parity test result:", result);
  });
});

function refreshStatus() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];
    if (!tab || tab.id === undefined) {
      renderUnavailable();
      return;
    }

    chrome.tabs.sendMessage(tab.id, { type: "MYKID_GET_COUNTS" }, (counts) => {
      if (chrome.runtime.lastError || !counts) {
        // No content script on this tab (restricted page, or the tab hasn't
        // reloaded since the extension was installed/updated).
        renderUnavailable();
        return;
      }
      render(counts);
    });
  });
}

refreshStatus();
