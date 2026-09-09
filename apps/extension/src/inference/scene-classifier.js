/**
 * MyKid — Scene-level safety classifier (Phase B)
 *
 * Answers "what is happening in this image" rather than "what objects are
 * present" (SKILL.md §9). This is what actually detects the content the
 * product exists for — gore, violence, sexual content — none of which the
 * COCO object detector has any concept of.
 *
 * Model: `image-safety-classifier-xs` (SwiftFormer-XS finetune, MIT),
 * selected in Phase A. See docs/06-ai-detection.md for the benchmark.
 *
 * Scene risk has no bounding boxes by nature, so it produces whole-frame
 * protection. Inventing boxes for it is explicitly forbidden by
 * SKILL.md §13.
 *
 * The browser mirror of packages/vision/scene_classifier.py — keep the two
 * in step; Phase 13 checks they agree.
 *
 * Exposes a global `MyKidSceneClassifier`.
 */
var MyKidSceneClassifier = (function () {
  const INPUT_SIZE = 224;

  // Fixed by the model's own config (pretrained_cfg.label_names), read from
  // the model rather than assumed.
  const LABELS = ["NSFL", "NSFW", "SFW"];

  /**
   * Resize to 224x224 as raw 0-255 RGB in NCHW order.
   *
   * Normalisation is baked into the ONNX graph, so the input must NOT be
   * pre-normalised. Verified in Phase A: normalising first pushes the NSFW
   * score on a photo of a bus from 0.032 to 0.18, because the graph then
   * normalises already-normalised values.
   *
   * Note this is a plain resize, not the letterbox used for detection —
   * a classifier has no boxes to map back, so aspect distortion costs
   * nothing and matches how the model was trained.
   */
  function preprocess(source) {
    const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);

    return packPixels(ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data);
  }

  /**
   * RGBA bytes -> NCHW float32, alpha dropped, values left in 0-255.
   *
   * Split out from preprocess() so it can be tested without a canvas: this
   * is where a silent divergence from the Python pipeline would live. Get
   * the channel order or the layout wrong and the model still runs, still
   * returns three plausible probabilities, and is quietly wrong — which is
   * exactly the failure a test has to catch. See
   * tests/integration/test_scene_parity.py.
   */
  function packPixels(rgba) {
    const pixelCount = INPUT_SIZE * INPUT_SIZE;
    const tensor = new Float32Array(pixelCount * 3);

    const greenOffset = pixelCount;
    const blueOffset = pixelCount * 2;

    for (let i = 0; i < pixelCount; i++) {
      const offset = i * 4;
      tensor[i] = rgba[offset];
      tensor[greenOffset + i] = rgba[offset + 1];
      tensor[blueOffset + i] = rgba[offset + 2];
    }

    return tensor;
  }

  /**
   * Classify one image's scene-level risk.
   *
   * @param {ort.InferenceSession} session
   * @param {ImageBitmap} source
   * @returns {Promise<{sceneRisk: Object, scores: Object, inferMs: number}>}
   */
  async function classify(session, source) {
    const tensorData = preprocess(source);
    const inputTensor = new ort.Tensor("float32", tensorData, [
      1, 3, INPUT_SIZE, INPUT_SIZE,
    ]);

    const start = performance.now();
    const outputMap = await session.run({ [session.inputNames[0]]: inputTensor });
    const probabilities = outputMap[session.outputNames[0]].data;
    const inferMs = performance.now() - start;

    const scores = {};
    LABELS.forEach((label, index) => {
      scores[label] = probabilities[index];
    });

    return {
      // `violence` stays 0: this model folds violence into NSFL rather
      // than scoring it separately, and putting a number there we didn't
      // measure would misrepresent the model. The field is kept so a
      // dedicated violence model can fill it later.
      sceneRisk: {
        violence: 0,
        graphic: scores.NSFL,
        sexual: scores.NSFW,
      },
      scores,
      inferMs: Math.round(inferMs),
    };
  }

  return { INPUT_SIZE, LABELS, preprocess, packPixels, classify };
})();
