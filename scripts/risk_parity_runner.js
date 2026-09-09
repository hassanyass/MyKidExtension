#!/usr/bin/env node
/**
 * Runs the browser extension's risk engine over scenarios supplied as JSON
 * on stdin, and prints its verdicts as JSON on stdout.
 *
 * Exists so the Python test suite can compare the two risk engines
 * directly (tests/integration/test_risk_parity.py). The extension's risk
 * engine is a port of packages/risk/risk_engine.py, and this is what keeps
 * the two from silently drifting apart — a drift would mean the browser
 * and the reference pipeline disagree about whether a child sees something.
 *
 * Input:  [{ "detections": [{label, confidence, bbox:{x,y,width,height}}],
 *            "width": int, "height": int }]
 * Output: [{ "overall_risk": str, "action": str, "regions": [...] }]
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const EXTENSION_ROOT = path.resolve(__dirname, "..", "apps", "extension");

// The extension scripts are written for a browser; give them the minimum
// they touch. config.js reads chrome.storage, nothing else here does.
const input = JSON.parse(fs.readFileSync(0, "utf8"));

// Accept either a bare scenario array or {scenarios, debugLabel}, so the
// debug "treat this label as harmful" path can be exercised too.
const scenarios = Array.isArray(input) ? input : input.scenarios;
const debugLabel = Array.isArray(input) ? null : (input.debugLabel ?? null);

// "configOnly" asks for the resolved config instead of risk verdicts, so
// the Python suite can assert on the extension's config semantics (the
// on/off toggle, override merging) without duplicating them in a second
// language where they could drift.
const configOnly = !Array.isArray(input) && input.configOnly;

// Either a bare debug label (older call sites) or a full override object.
const explicitConfig = Array.isArray(input) ? null : input.storedConfig;
const storedConfig = debugLabel
  ? { mykidConfig: { debug: { treatLabelAsHarmful: debugLabel } } }
  : explicitConfig && Object.keys(explicitConfig).length
    ? { mykidConfig: explicitConfig }
    : {};

// `noStorage` reproduces the offscreen document's environment, where
// chrome.storage isn't available. Config must reach it through the
// message instead — see the ConfigDelivery tests.
const noStorage = !Array.isArray(input) && input.noStorage === true;

const sandbox = {
  chrome: noStorage
    ? { runtime: {} }
    : { storage: { local: { get: async () => storedConfig, set: async () => {} } } },
  console,
};
vm.createContext(sandbox);

for (const file of ["src/shared/config.js", "src/inference/risk.js"]) {
  const fullPath = path.join(EXTENSION_ROOT, file);
  vm.runInContext(fs.readFileSync(fullPath, "utf8"), sandbox, { filename: file });
}

const { MyKidConfig, MyKidRisk } = sandbox;

// With storage available, load() picks up the debug override. Without it
// (the offscreen document's situation), config must be supplied directly —
// which is exactly what the service worker now does.
const configPromise = noStorage
  ? Promise.resolve(MyKidConfig.fromOverrides(storedConfig.mykidConfig))
  : MyKidConfig.load();

configPromise.then((config) => {
if (configOnly) {
  process.stdout.write(
    JSON.stringify({
      enabled: config.enabled,
      categories: config.categories,
      detection: config.detection,
      debug: config.debug,
      harmfulLabels: MyKidConfig.harmfulLabels(config),
    })
  );
  return;
}

const results = scenarios.map((scenario) => {
  const analysis = MyKidRisk.evaluate(
    scenario.detections,
    scenario.sceneRisk ?? null,
    scenario.width,
    scenario.height,
    config
  );

  return {
    overall_risk: analysis.overallRisk,
    action: analysis.action,
    regions: analysis.protectionRegions.map((region) => ({
      x: Number(region.bbox.x.toFixed(4)),
      y: Number(region.bbox.y.toFixed(4)),
      width: Number(region.bbox.width.toFixed(4)),
      height: Number(region.bbox.height.toFixed(4)),
    })),
  };
});

process.stdout.write(JSON.stringify(results));
});
