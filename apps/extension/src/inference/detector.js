/**
 * MyKid — Detection pre/postprocessing (Phase 10b-ii)
 *
 * The browser-side counterpart to the Python reference pipeline. This is a
 * deliberate, step-for-step mirror of scripts/parity_reference.py — if you
 * change the maths here, change it there too, and re-run the parity check.
 * The two must agree, because Phase 13 validates that the browser and
 * Python pipelines reach the same safety decisions for the same input.
 *
 * Responsibilities (SKILL.md §6.3): produce structured detections only.
 * No risk assessment, no blurring — those stay in their own layers.
 *
 * Exposes a global `MyKidDetector` (classic script, no bundler).
 */
var MyKidDetector = (function () {
  const MODEL_INPUT_SIZE = 640;
  const LETTERBOX_PAD_VALUE = 114; // flat gray, matching Ultralytics' default
  const DEFAULT_IOU_THRESHOLD = 0.7; // Ultralytics' default NMS IoU

  // COCO-80 class names, indexed by class id.
  // Ported from packages/shared/labels.py — keep the two in sync.
  const COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
  ];

  // Safety-relevant labels. Ported from SAFETY_CATEGORIES in
  // packages/shared/labels.py — currently weapons only, because COCO gives
  // us nothing else usable (see docs/06-ai-detection.md §6).
  const SAFETY_RELEVANT_LABELS = new Set(["knife", "scissors"]);

  /**
   * Draw an image into a square letterboxed canvas: scaled to fit while
   * preserving aspect ratio, remainder padded with flat gray.
   *
   * @param {ImageBitmap|HTMLImageElement} source
   * @returns {{imageData: ImageData, scale: number, padX: number, padY: number,
   *            originalWidth: number, originalHeight: number}}
   */
  function letterbox(source) {
    const originalWidth = source.width;
    const originalHeight = source.height;

    const scale = Math.min(
      MODEL_INPUT_SIZE / originalWidth,
      MODEL_INPUT_SIZE / originalHeight
    );
    const newWidth = Math.round(originalWidth * scale);
    const newHeight = Math.round(originalHeight * scale);
    const padX = Math.floor((MODEL_INPUT_SIZE - newWidth) / 2);
    const padY = Math.floor((MODEL_INPUT_SIZE - newHeight) / 2);

    const canvas = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    ctx.fillStyle = `rgb(${LETTERBOX_PAD_VALUE}, ${LETTERBOX_PAD_VALUE}, ${LETTERBOX_PAD_VALUE})`;
    ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    ctx.drawImage(source, padX, padY, newWidth, newHeight);

    return {
      imageData: ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
      scale,
      padX,
      padY,
      originalWidth,
      originalHeight,
    };
  }

  /**
   * RGBA ImageData -> normalized NCHW Float32Array, as the model expects.
   * Alpha is dropped; channels are scaled to 0..1.
   */
  function toInputTensorData(imageData) {
    const { data } = imageData;
    const pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
    const tensor = new Float32Array(pixelCount * 3);

    const greenOffset = pixelCount;
    const blueOffset = pixelCount * 2;

    for (let i = 0; i < pixelCount; i++) {
      const rgba = i * 4;
      tensor[i] = data[rgba] / 255;
      tensor[greenOffset + i] = data[rgba + 1] / 255;
      tensor[blueOffset + i] = data[rgba + 2] / 255;
    }

    return tensor;
  }

  /** Intersection-over-union for two [x1, y1, x2, y2] boxes. */
  function iou(boxA, boxB) {
    const interX1 = Math.max(boxA[0], boxB[0]);
    const interY1 = Math.max(boxA[1], boxB[1]);
    const interX2 = Math.min(boxA[2], boxB[2]);
    const interY2 = Math.min(boxA[3], boxB[3]);

    const interW = Math.max(0, interX2 - interX1);
    const interH = Math.max(0, interY2 - interY1);
    const intersection = interW * interH;
    if (intersection === 0) return 0;

    const areaA = Math.max(0, boxA[2] - boxA[0]) * Math.max(0, boxA[3] - boxA[1]);
    const areaB = Math.max(0, boxB[2] - boxB[0]) * Math.max(0, boxB[3] - boxB[1]);
    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
  }

  /** Greedy per-class non-max suppression, highest confidence first. */
  function nonMaxSuppression(detections, iouThreshold) {
    const threshold = iouThreshold ?? DEFAULT_IOU_THRESHOLD;
    const remaining = detections.slice().sort((a, b) => b.confidence - a.confidence);
    const kept = [];

    while (remaining.length > 0) {
      const best = remaining.shift();
      kept.push(best);

      for (let i = remaining.length - 1; i >= 0; i--) {
        const candidate = remaining[i];
        if (
          candidate.classId === best.classId &&
          iou(candidate._xyxy, best._xyxy) >= threshold
        ) {
          remaining.splice(i, 1);
        }
      }
    }

    return kept;
  }

  /**
   * Decode a YOLOv8/v11 [1, 84, 8400] output tensor into detections in
   * original-image coordinates.
   *
   * Layout: 84 channels = 4 box coords (cx, cy, w, h) + 80 class scores.
   * There is no separate objectness score — confidence is the top class score.
   */
  function decodeOutput(output, geometry, confThreshold) {
    const [, channelCount, anchorCount] = output.dims;
    const data = output.data;
    const classCount = channelCount - 4;
    const { scale, padX, padY, originalWidth, originalHeight } = geometry;

    const detections = [];

    for (let anchor = 0; anchor < anchorCount; anchor++) {
      // Find the highest-scoring class for this anchor.
      let bestScore = 0;
      let bestClassId = -1;
      for (let c = 0; c < classCount; c++) {
        const score = data[(4 + c) * anchorCount + anchor];
        if (score > bestScore) {
          bestScore = score;
          bestClassId = c;
        }
      }

      if (bestClassId < 0 || bestScore < confThreshold) continue;

      const cx = data[0 * anchorCount + anchor];
      const cy = data[1 * anchorCount + anchor];
      const w = data[2 * anchorCount + anchor];
      const h = data[3 * anchorCount + anchor];

      // Model space (640x640, letterboxed) -> original image space
      const x1 = Math.max(0, (cx - w / 2 - padX) / scale);
      const y1 = Math.max(0, (cy - h / 2 - padY) / scale);
      const x2 = Math.min(originalWidth, (cx + w / 2 - padX) / scale);
      const y2 = Math.min(originalHeight, (cy + h / 2 - padY) / scale);

      const label = COCO_LABELS[bestClassId] ?? "unknown";

      detections.push({
        label,
        classId: bestClassId,
        confidence: bestScore,
        bbox: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 },
        safetyRelevant: SAFETY_RELEVANT_LABELS.has(label),
        _xyxy: [x1, y1, x2, y2],
      });
    }

    const kept = nonMaxSuppression(detections);
    kept.forEach((d) => delete d._xyxy);

    return kept.sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Full detection pass over one image.
   *
   * @param {ort.InferenceSession} session
   * @param {ImageBitmap} source
   * @param {number} confThreshold
   * @returns {Promise<{detections: Array, geometry: Object, timings: Object}>}
   */
  async function detect(session, source, confThreshold) {
    const preprocessStart = performance.now();
    const geometry = letterbox(source);
    const tensorData = toInputTensorData(geometry.imageData);
    const inputTensor = new ort.Tensor("float32", tensorData, [
      1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
    ]);
    const preprocessMs = performance.now() - preprocessStart;

    const inferStart = performance.now();
    const outputMap = await session.run({ [session.inputNames[0]]: inputTensor });
    const output = outputMap[session.outputNames[0]];
    const inferMs = performance.now() - inferStart;

    const decodeStart = performance.now();
    const detections = decodeOutput(output, geometry, confThreshold);
    const decodeMs = performance.now() - decodeStart;

    return {
      detections,
      geometry: {
        scale: geometry.scale,
        padX: geometry.padX,
        padY: geometry.padY,
        originalWidth: geometry.originalWidth,
        originalHeight: geometry.originalHeight,
      },
      outputShape: output.dims,
      timings: {
        preprocessMs: Math.round(preprocessMs),
        inferMs: Math.round(inferMs),
        decodeMs: Math.round(decodeMs),
      },
    };
  }

  return {
    MODEL_INPUT_SIZE,
    COCO_LABELS,
    SAFETY_RELEVANT_LABELS,
    letterbox,
    toInputTensorData,
    decodeOutput,
    nonMaxSuppression,
    iou,
    detect,
  };
})();
