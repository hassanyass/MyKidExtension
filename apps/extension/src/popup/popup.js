/**
 * Fuzzy — popup.
 *
 * Two audiences share this panel: an adult setting it up, and a child who
 * may well open it out of curiosity. So the top half is plain language and
 * big controls, and everything technical (counts, scores, self-check) is
 * folded into "Grown-up settings".
 *
 * Sexual-content detection is deliberately not offered as a switch. It
 * stays permanently on — it is not a protection worth offering to disable,
 * and the label does not belong in a panel a child may read.
 */
const statusEl = document.getElementById("status");
const powerButton = document.getElementById("power");
const toggleRow = document.getElementById("toggle-row");
const toggleState = document.getElementById("toggle-state");
const toggleHint = document.getElementById("toggle-hint");
const techEl = document.getElementById("tech");

/**
 * Push config changes to every open tab, not just the active one — a
 * protection switch that applied only to the tab in front would leave the
 * others unguarded while reporting itself as on.
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

function saveConfig(mutate) {
  chrome.storage.local.get("mykidConfig", (stored) => {
    const overrides = stored?.mykidConfig ?? {};
    mutate(overrides);
    chrome.storage.local.set({ mykidConfig: overrides }, () => {
      broadcastRescan();
      setTimeout(refreshStatus, 200);
    });
  });
}

// --- Power ----------------------------------------------------------

function renderToggle(enabled) {
  powerButton.setAttribute("aria-checked", String(enabled));
  powerButton.classList.toggle("on", enabled);
  toggleRow.classList.toggle("off", !enabled);
  toggleState.textContent = enabled ? "Fuzzy is on" : "Fuzzy is off";
  toggleHint.textContent = enabled
    ? "Keeping an eye on pictures and videos."
    : "Nothing is being checked right now.";
}

powerButton.addEventListener("click", () => {
  const next = powerButton.getAttribute("aria-checked") !== "true";
  renderToggle(next); // respond instantly; storage catches up
  saveConfig((overrides) => {
    overrides.enabled = next;
  });
});

// --- Categories -----------------------------------------------------

const categoryInputs = document.querySelectorAll("#categories input[data-category]");

function renderCategories(categories) {
  for (const input of categoryInputs) {
    // Absent means "not configured yet", which for a safety tool should
    // mean covered rather than skipped.
    input.checked = categories?.[input.dataset.category] ?? true;
  }
}

for (const input of categoryInputs) {
  input.addEventListener("change", () => {
    saveConfig((overrides) => {
      overrides.categories = {
        ...(overrides.categories ?? {}),
        [input.dataset.category]: input.checked,
      };
    });
  });
}

// --- Practice mode (any number of labels at once) --------------------

const practiceInputs = document.querySelectorAll("#practice-labels input");

for (const input of practiceInputs) {
  input.addEventListener("change", () => {
    const labels = Array.from(practiceInputs)
      .filter((box) => box.checked)
      .map((box) => box.value);

    saveConfig((overrides) => {
      overrides.debug = { treatLabelAsHarmful: labels.length ? labels : null };
    });
  });
}

// --- Status ----------------------------------------------------------

function renderTech(stats, video) {
  if (!stats) {
    techEl.innerHTML = "";
    return;
  }

  const rows = [
    ["Checked", String(stats.processed)],
    ["Skipped (too small)", String(stats.skipped)],
    ["Waiting", String(stats.pending ?? 0)],
    ["Could not read", String(stats.failed)],
  ];

  if (stats.retried) rows.push(["Retried", String(stats.retried)]);
  if (stats.lastError) rows.push(["Last error", stats.lastError]);

  if (stats.sceneChecked > 0) {
    rows.push([
      "Highest scores seen",
      `gore ${stats.peakGore.toFixed(2)} · grown-up ${stats.peakSexual.toFixed(2)}`,
    ]);
  }
  if (video && video.watching) {
    rows.push(["Videos watched", `${video.watching} (${video.sampled} frames)`]);
  }
  if (stats.harmfulLabels && stats.harmfulLabels.length) {
    rows.push(["Hiding", stats.harmfulLabels.join(", ")]);
  }

  techEl.innerHTML = rows
    .map(([label, value]) => `<div class="row"><span>${label}</span><b>${value}</b></div>`)
    .join("");
}

/**
 * Plain-language status.
 *
 * The message explains a zero rather than leaving it ambiguous: "nothing
 * found" and "nothing checked yet" look identical as a bare 0 and mean
 * very different things.
 */
function renderStatus(counts) {
  const stats = counts.stats || {};
  const found = counts.images + counts.videos;
  const processed = stats.processed || 0;
  const pending = stats.pending || 0;
  const hidden = stats.protected || 0;

  let message;
  if (hidden > 0) {
    message = `<p class="message happy">Fuzzy hid ${hidden} thing${
      hidden === 1 ? "" : "s"
    } on this page.</p>`;
  } else if (processed === 0 && pending === 0) {
    message = found
      ? `<p class="message">Nothing checked here yet — try scrolling.</p>`
      : `<p class="message">No pictures or videos on this page.</p>`;
  } else if (pending > 0) {
    message = `<p class="message">Still looking…</p>`;
  } else if (processed === 0 && stats.failed > 0) {
    // Everything failed: this is a broken install, not a clean page, and
    // saying "all clear" here would be actively misleading.
    message = `<p class="message warn">Fuzzy could not check this page.
      Open Grown-up settings for the reason.</p>`;
  } else {
    message = `<p class="message happy">All clear — nothing needed hiding.</p>`;
  }

  statusEl.innerHTML = `
    <div class="tally">
      <div class="tally-card"><b>${found}</b><span>on page</span></div>
      <div class="tally-card"><b>${processed}</b><span>checked</span></div>
      <div class="tally-card hidden-count"><b>${hidden}</b><span>hidden</span></div>
    </div>
    ${message}
  `;
}

function renderOff(counts) {
  statusEl.innerHTML = `
    <div class="tally">
      <div class="tally-card"><b>${counts.images + counts.videos}</b><span>on page</span></div>
      <div class="tally-card"><b>—</b><span>checked</span></div>
      <div class="tally-card hidden-count"><b>—</b><span>hidden</span></div>
    </div>
    <p class="message warn">Fuzzy is off, so nothing is being checked.</p>
  `;
}

function renderUnavailable() {
  statusEl.innerHTML = `
    <p class="message">Fuzzy cannot look at this page.</p>
    <p class="message">Try a normal website — browser pages are off limits.</p>
  `;
  techEl.innerHTML = "";
}

function render(counts) {
  renderToggle(counts.enabled);
  if (counts.stats && counts.stats.activeCategories) {
    renderCategories(counts.stats.activeCategories);
  }

  if (!counts.enabled) {
    renderOff(counts);
    renderTech(null);
    return;
  }

  renderStatus(counts);
  renderTech(counts.stats, counts.video);
}

function refreshStatus() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];
    if (!tab || tab.id === undefined) {
      renderUnavailable();
      return;
    }
    chrome.tabs.sendMessage(tab.id, { type: "MYKID_GET_COUNTS" }, (counts) => {
      if (chrome.runtime.lastError || !counts) {
        renderUnavailable();
        return;
      }
      render(counts);
    });
  });
}

// --- Self-check -------------------------------------------------------

const parityButton = document.getElementById("run-parity");
const parityOutput = document.getElementById("parity-output");

parityButton.addEventListener("click", () => {
  parityButton.disabled = true;
  parityOutput.textContent = "Checking…";

  chrome.runtime.sendMessage({ type: "MYKID_PARITY_TEST" }, (result) => {
    parityButton.disabled = false;

    if (chrome.runtime.lastError || !result || !result.ok) {
      const reason =
        (result && result.error) ||
        (chrome.runtime.lastError && chrome.runtime.lastError.message) ||
        "no response";
      parityOutput.innerHTML = `<p class="message warn">${reason}</p>`;
      return;
    }

    const rows = result.detections
      .map((d) => `<tr><td>${d.label}</td><td>${d.confidence.toFixed(3)}</td></tr>`)
      .join("");

    parityOutput.innerHTML = `
      <div class="row"><span>Found</span><b>${result.detections.length} object(s)</b></div>
      <div class="row"><span>Speed</span><b>${result.timings.inferMs}ms</b></div>
      <table><tr><th>what</th><th>sure?</th></tr>${rows}</table>
    `;
  });
});

// --- Boot -------------------------------------------------------------

chrome.storage.local.get("mykidConfig", (stored) => {
  const config = stored?.mykidConfig ?? {};

  renderToggle(config.enabled ?? true);
  renderCategories(config.categories);

  // Accepts the older single-string form as well as a list, so settings
  // saved by a previous build still apply.
  const saved = config.debug && config.debug.treatLabelAsHarmful;
  const labels = Array.isArray(saved) ? saved : saved ? [saved] : [];
  for (const input of practiceInputs) input.checked = labels.includes(input.value);
});

refreshStatus();
