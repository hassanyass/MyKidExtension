#!/usr/bin/env node
/**
 * Runs the extension's scene-classifier preprocessing over raw RGBA pixels
 * supplied on stdin, and prints the resulting tensor summary as JSON.
 *
 * Exists so the Python suite can verify both pipelines pack pixels
 * identically (tests/integration/test_scene_parity.py). A mismatch here —
 * wrong channel order, NHWC instead of NCHW, an accidental /255 — does not
 * crash anything: the model still runs and still returns three plausible
 * probabilities. It is just quietly wrong about whether a child sees
 * something, which is why it is pinned by a test rather than by reading.
 *
 * Input:  { "rgba": [...], "size": 224 }
 * Output: { "length": n, "first12": [...], "channelMeans": [r, g, b], "max": n }
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const EXTENSION_ROOT = path.resolve(__dirname, "..", "apps", "extension");

// The module expects browser globals it won't touch on this path.
const sandbox = { console, OffscreenCanvas: function () {}, ort: {} };
vm.createContext(sandbox);
vm.runInContext(
  fs.readFileSync(
    path.join(EXTENSION_ROOT, "src/inference/scene-classifier.js"),
    "utf8"
  ),
  sandbox,
  { filename: "scene-classifier.js" }
);

const { MyKidSceneClassifier } = sandbox;
const input = JSON.parse(fs.readFileSync(0, "utf8"));

const tensor = MyKidSceneClassifier.packPixels(Uint8ClampedArray.from(input.rgba));

const pixelCount = MyKidSceneClassifier.INPUT_SIZE * MyKidSceneClassifier.INPUT_SIZE;
const channelMeans = [0, 1, 2].map((channel) => {
  let total = 0;
  for (let i = 0; i < pixelCount; i++) total += tensor[channel * pixelCount + i];
  return Number((total / pixelCount).toFixed(6));
});

// Looped rather than Math.max(...tensor): the tensor holds 150k+ values
// and spreading it overflows the call stack.
let max = -Infinity;
for (let i = 0; i < tensor.length; i++) {
  if (tensor[i] > max) max = tensor[i];
}

process.stdout.write(
  JSON.stringify({
    length: tensor.length,
    inputSize: MyKidSceneClassifier.INPUT_SIZE,
    labels: MyKidSceneClassifier.LABELS,
    first12: Array.from(tensor.slice(0, 12)),
    channelMeans,
    max,
  })
);
