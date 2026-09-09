# Testing & Evaluation

> What is tested, what is deliberately not tested, and why the difference
> matters. Written for Phase 14; kept current as the suite changes.

## Suite at a glance

```bash
python -m pytest tests/ -q          # 167 tests
```

| Area | Location | Covers |
|---|---|---|
| Shared types, config, errors, logging | `tests/unit/` | Foundations |
| Image processing, risk, protection | `tests/unit/` | Reference pipeline |
| End-to-end image & video pipelines | `tests/integration/` | Python pipeline |
| **Browser↔Python risk parity** | `tests/integration/test_risk_parity.py` | 12 scenarios, both engines |
| **Scene classifier** | `tests/integration/test_scene_classifier.py` | Contract + safe-content behaviour |
| **Scene preprocessing parity** | `tests/integration/test_scene_parity.py` | Tensor layout across languages |

Node-dependent tests skip automatically when Node isn't installed, so the
suite stays runnable on a Python-only machine.

## Why cross-language parity is tested at all

The browser cannot run the Python pipeline, so the risk engine, the label
mapping and the classifier preprocessing all exist twice. Duplicated logic
drifts, and drift here is invisible: both sides keep running, both return
plausible values, and they quietly disagree about whether a child sees
something.

Three specific failures are pinned because each is silent:

1. **Risk verdicts** — 12 scenarios compared verdict-for-verdict, including
   padded box coordinates and coverage escalation.
2. **Tensor layout** — a channel swap or NHWC/NCHW mix-up produces three
   plausible probabilities from a wrong tensor. The fixture deliberately
   gives each channel a distinct distribution so a swap cannot pass.
3. **Config delivery** — verified to work *without* `chrome.storage`,
   reproducing the offscreen document's real environment. This is a
   regression test for a bug that shipped: config read from a context that
   couldn't reach storage, failed into a bare `catch`, and left the risk
   engine on defaults while detection ran perfectly.

## What is NOT tested, and cannot currently be

**Detection accuracy on harmful content.** There is no harmful evaluation
set in this project. No test asserts that gore, violence or sexual content
is detected, because none can. Safe fixtures establish only that the model
runs and does not fire indiscriminately — a false-positive check, not a
detection check.

This is the largest gap in the project and no amount of additional testing
closes it without data. See `docs/12-privacy-and-limitations.md` §2 and
`docs/11` Open Decisions.

**Browser behaviour.** The suite tests logic, not the extension running in
Chrome. Confirmed by hand: extension loads, engine loads, detection maths
matches Python to ~1px, protection applies to real page images. Not yet
confirmed: video protection, infinite scroll, SPA re-scan.

## Fixtures

| Set | Contents | Notes |
|---|---|---|
| `test-data/images/safe/` | 2 blank images | Produce **zero** detections — useless for validating box decoding, which is why the set below exists |
| `test-data/images/detection-samples/` | `bus.jpg`, `zidane.jpg` | Known detections. AGPL, gitignored; regenerate with `scripts/fetch_detection_samples.py` |
| `test-data/images/harmful/` | **empty** | The blocking gap |

## Reproducible measurements

```bash
python scripts/benchmark_safety_models.py   # model size / latency / licence
python scripts/quantize_models.py           # INT8 comparison (rejected — see docs/11 Phase F)
python scripts/parity_reference.py <image>  # ground-truth detections
```

Latency figures move several-fold with background load and thread
contention — an unpinned run once reported 253ms for an 11ms model.
Measurements pin `intra_op_num_threads` and report medians and minimums;
treat absolute numbers as machine-specific and comparisons as the finding.
