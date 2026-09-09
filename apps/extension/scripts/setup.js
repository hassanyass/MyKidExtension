#!/usr/bin/env node
/**
 * MyKid extension setup script.
 *
 * The extension needs two things that are intentionally NOT committed to
 * git — one is a 10MB+ binary, the other a third-party build artifact,
 * both regenerable:
 *   1. models/yolo11n.onnx      <- copied from the project's top-level models/
 *   2. vendor/onnxruntime-web/  <- the wasm inference runtime, fetched via npm
 *
 * Run this once after cloning the repo, and again any time the model or
 * the pinned onnxruntime-web version changes, before loading the
 * extension unpacked in Chrome.
 *
 * Usage: node apps/extension/scripts/setup.js
 */
const { execSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");

const EXT_ROOT = path.resolve(__dirname, "..");
const PROJECT_ROOT = path.resolve(EXT_ROOT, "..", "..");
const ORT_VERSION = "1.29.0";

function step(label, fn) {
  process.stdout.write(`- ${label}... `);
  fn();
  console.log("done");
}

step("copying models/yolo11n.onnx into the extension", () => {
  const src = path.join(PROJECT_ROOT, "models", "yolo11n.onnx");
  const destDir = path.join(EXT_ROOT, "models");
  const dest = path.join(destDir, "yolo11n.onnx");
  if (!fs.existsSync(src)) {
    throw new Error(
      `Model not found at ${src}. Run \`python scripts/export_onnx.py\` first (see docs/06-ai-detection.md).`
    );
  }
  fs.mkdirSync(destDir, { recursive: true });
  fs.copyFileSync(src, dest);
});

step("copying the scene safety classifier into the extension", () => {
  // The model that actually detects gore / violence / sexual content
  // (Phase A). Downloaded by scripts/benchmark_safety_models.py rather
  // than committed — it's a 12.5MB binary we didn't author.
  const src = path.join(
    PROJECT_ROOT,
    "models",
    "candidates",
    "image-safety-classifier-xs.onnx"
  );
  if (!fs.existsSync(src)) {
    throw new Error(
      `Scene classifier not found at ${src}.\n` +
        "  Run `python scripts/benchmark_safety_models.py` first — it downloads the model.\n" +
        "  Without it the extension can only detect knives and scissors."
    );
  }
  fs.mkdirSync(path.join(EXT_ROOT, "models"), { recursive: true });
  fs.copyFileSync(
    src,
    path.join(EXT_ROOT, "models", "image-safety-classifier-xs.onnx")
  );
});

step("copying the parity-test sample image into the extension", () => {
  // Used by the popup's "Run detection test" button to verify the browser
  // pipeline against scripts/parity_reference.py on identical input.
  // Optional: skip quietly if the sample hasn't been generated yet.
  const src = path.join(
    PROJECT_ROOT,
    "test-data",
    "images",
    "detection-samples",
    "bus.jpg"
  );
  if (!fs.existsSync(src)) {
    console.log(
      "\n  (skipped — run `python scripts/fetch_detection_samples.py` to enable the parity test)"
    );
    return;
  }
  const destDir = path.join(EXT_ROOT, "devtest");
  fs.mkdirSync(destDir, { recursive: true });
  fs.copyFileSync(src, path.join(destDir, "bus.jpg"));
});

step(`fetching onnxruntime-web@${ORT_VERSION}`, () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "mykid-ort-"));
  try {
    execSync(`npm install onnxruntime-web@${ORT_VERSION} --no-save --silent`, {
      cwd: tmpDir,
      stdio: "ignore",
    });
    const distDir = path.join(tmpDir, "node_modules", "onnxruntime-web", "dist");
    const vendorDir = path.join(EXT_ROOT, "vendor", "onnxruntime-web");
    fs.mkdirSync(vendorDir, { recursive: true });
    // Wasm-only backend build (no webgl/webgpu) — smallest option that
    // covers plain CPU inference, which is all Phase 10b needs.
    //
    // All three files are required, and the load chain is why:
    //   ort.wasm.min.js            -- the API surface; dynamically imports ->
    //   ort-wasm-simd-threaded.mjs -- Emscripten glue module; fetches ->
    //   ort-wasm-simd-threaded.wasm -- the actual compiled runtime
    // Omitting the .mjs glue fails at runtime with a misleading
    // "no available backend found" error, because the wasm backend's
    // dynamic import() of the glue is what actually throws.
    for (const file of [
      "ort.wasm.min.js",
      "ort-wasm-simd-threaded.mjs",
      "ort-wasm-simd-threaded.wasm",
    ]) {
      const from = path.join(distDir, file);
      if (!fs.existsSync(from)) {
        throw new Error(
          `Expected ${file} in onnxruntime-web@${ORT_VERSION}'s dist/ but it wasn't there. ` +
            `The package layout may have changed — check dist/ and update this list.`
        );
      }
      fs.copyFileSync(from, path.join(vendorDir, file));
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

console.log(
  "\nSetup complete. Load apps/extension/ as an unpacked extension in Chrome (chrome://extensions -> Load unpacked)."
);
