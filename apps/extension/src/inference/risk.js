/**
 * MyKid — Risk Engine (Phase 10b-iii)
 *
 * Converts raw detections into a risk level and a protection action.
 * A faithful port of packages/risk/risk_engine.py — the decision logic,
 * ordering and thresholds are deliberately identical, because Phase 13
 * validates that both pipelines reach the same verdict on the same input.
 *
 * This layer decides; it does not detect and it does not blur
 * (SKILL.md §38: DETECT != DECIDE != PROTECT).
 *
 * Exposes a global `MyKidRisk` (classic script, no bundler).
 */
var MyKidRisk = (function () {
  const RiskLevel = { LOW: "low", MEDIUM: "medium", HIGH: "high" };

  const ProtectionAction = {
    ALLOW: "allow",
    BLUR_REGION: "blur_region",
    BLUR_FRAME: "blur_frame",
  };

  const RISK_WEIGHTS = {
    [RiskLevel.LOW]: 0,
    [RiskLevel.MEDIUM]: 1,
    [RiskLevel.HIGH]: 2,
  };

  /**
   * Expand a box by a padding ratio, clamped to the image bounds.
   * Ported from BoundingBox.padded() in packages/shared/types.py.
   */
  function padBox(bbox, paddingRatio, imageWidth, imageHeight) {
    const padW = bbox.width * paddingRatio;
    const padH = bbox.height * paddingRatio;

    const x = Math.max(0, bbox.x - padW);
    const y = Math.max(0, bbox.y - padH);

    return {
      x,
      y,
      width: Math.min(imageWidth - x, bbox.width + 2 * padW),
      height: Math.min(imageHeight - y, bbox.height + 2 * padH),
    };
  }

  function boxArea(bbox) {
    return Math.max(0, bbox.width) * Math.max(0, bbox.height);
  }

  /**
   * Assign a risk level to each detection based on its label.
   *
   * The Python vision layer filters to safety-relevant labels before the
   * risk engine ever sees them; here the detector returns everything, so
   * the filtering happens at this boundary instead. Same outcome.
   */
  function classifyDetections(detections, config) {
    const harmful = new Set(MyKidConfig.harmfulLabels(config));

    return detections
      .filter(
        (d) =>
          harmful.has(d.label) &&
          d.confidence >= config.detection.objectConfidenceThreshold
      )
      .map((d) => ({ ...d, risk: RiskLevel.MEDIUM }));
  }

  function calculateOverallRisk(harmfulDetections, sceneRisk, config) {
    let maxWeight = RISK_WEIGHTS[RiskLevel.LOW];

    for (const detection of harmfulDetections) {
      maxWeight = Math.max(maxWeight, RISK_WEIGHTS[detection.risk]);
    }

    if (maxScore(sceneRisk) >= config.detection.sceneRiskThreshold) {
      maxWeight = Math.max(maxWeight, RISK_WEIGHTS[RiskLevel.HIGH]);
    }

    return (
      Object.keys(RISK_WEIGHTS).find((level) => RISK_WEIGHTS[level] === maxWeight) ??
      RiskLevel.LOW
    );
  }

  /**
   * Highest scene-level score. Mirrors SceneRisk.max_score() in
   * packages/shared/types.py — `sexual` added in Phase B alongside the
   * scene classifier, whose NSFW output had nowhere to go.
   */
  function maxScore(sceneRisk) {
    if (!sceneRisk) return 0;
    return Math.max(
      sceneRisk.violence ?? 0,
      sceneRisk.graphic ?? 0,
      sceneRisk.sexual ?? 0
    );
  }

  function exceedsCoverageThreshold(harmfulDetections, imageWidth, imageHeight, config) {
    const imageArea = imageWidth * imageHeight;
    if (imageArea === 0) return false;

    const harmfulArea = harmfulDetections.reduce(
      (total, d) => total + (d.bbox ? boxArea(d.bbox) : 0),
      0
    );

    return harmfulArea / imageArea > config.detection.maxAllowedObjectAreaRatio;
  }

  /**
   * Evaluate detections into a full analysis result.
   *
   * @returns {{detections: Array, overallRisk: string, action: string,
   *            protectionRegions: Array}}
   */
  function evaluate(detections, sceneRisk, imageWidth, imageHeight, config) {
    const harmfulDetections = classifyDetections(detections, config);
    const overallRisk = calculateOverallRisk(harmfulDetections, sceneRisk, config);

    let action = ProtectionAction.ALLOW;
    if (overallRisk === RiskLevel.MEDIUM || overallRisk === RiskLevel.HIGH) {
      action = ProtectionAction.BLUR_REGION;
    }

    if (maxScore(sceneRisk) >= config.detection.sceneRiskThreshold) {
      action = ProtectionAction.BLUR_FRAME;
    }

    if (
      action === ProtectionAction.BLUR_REGION &&
      exceedsCoverageThreshold(harmfulDetections, imageWidth, imageHeight, config)
    ) {
      action = ProtectionAction.BLUR_FRAME;
    }

    const protectionRegions = [];
    if (action === ProtectionAction.BLUR_REGION) {
      for (const detection of harmfulDetections) {
        if (!detection.bbox) continue;
        protectionRegions.push({
          bbox: padBox(
            detection.bbox,
            config.protection.blurPadding,
            imageWidth,
            imageHeight
          ),
          mode: config.protection.mode,
          sourceLabel: detection.label,
        });
      }
    }

    return {
      detections: harmfulDetections,
      overallRisk,
      action,
      protectionRegions,
    };
  }

  return {
    RiskLevel,
    ProtectionAction,
    evaluate,
    padBox,
    classifyDetections,
  };
})();
