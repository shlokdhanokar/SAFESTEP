# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SafeStep detects obstacles from a camera feed and guides a visually impaired user with directional audio. It is a `src/`-layout Python package (`src/safestep/`) with a thin [main.py](main.py) shim at the root for `python main.py`.

Everything here is shaped by one fact: **this is an assistive device, so the failure modes are asymmetric.** Missing an obstacle can injure someone; a false alarm merely annoys. But over-warning gets the device switched off, which then misses everything. Changes that alter what the user is or is not told deserve more care than their diff size suggests.

## Commands

```bash
pip install -e .                  # install the package + console script
pip install -r requirements-dev.txt

pytest                            # 372 tests; needs no camera, display or audio
pytest tests/test_alerts.py       # one module
pytest -k cooldown                # one topic

flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics   # blocking gate
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics   # advisory

python scripts/fetch_models.py    # download detector weights into models/ (gitignored)

# browser demo (docs/, published to GitHub Pages)
python scripts/generate_tables.py           # regenerate docs/js/tables.mjs from Python
python scripts/generate_parity_fixtures.py  # re-record Python's answers
node scripts/check_parity.mjs               # assert the JS port still agrees
npm run serve                               # http://127.0.0.1:8123
```

CI ([.github/workflows/python-app.yml](.github/workflows/python-app.yml)) runs flake8 plus pytest on Python 3.9/3.11/3.12, and additionally against OpenCV 4.x. Both flake8 gates are currently clean, including complexity — keep them that way.

## Verifying changes without hardware

`--source` accepts a **video file** as well as a camera index. This is the main way to exercise the real pipeline here:

```bash
python main.py --source clip.mp4 --detector color --headless --no-speech --no-earcons --log-level DEBUG
```

Generate a clip with `cv2.VideoWriter` if you need one. Only live capture, actual speech output and earcon playback genuinely require hardware.

## Architecture

Pipeline: `FrameSource → Detector → enrich → IoUTracker → AlertPolicy → Announcer`, wired in [app.py](src/safestep/app.py). Every stage is injected into `SafeStepApp`, which is why the loop itself is unit-testable.

**Detection is pluggable and lazily imported.** [detection/base.py](src/safestep/detection/base.py) defines the `Detector` protocol; backends are `color`, `ssd`, `yolo`, chosen by `build_detector()`. An explicit backend that cannot load raises with remediation text; `auto` falls back through the list. Never make an explicit request silently downgrade.

**Neural tensor decoding is separated from network I/O.** `decode_yolov8_output()` and `decode_ssd_output()` are pure functions over numpy arrays, tested against hand-built tensors in [tests/test_detection_neural.py](tests/test_detection_neural.py). Put new model-output parsing there, not inside a `detect()` method — it is the only way to test it without weights. [tests/test_integration_yolo.py](tests/test_integration_yolo.py) covers the remaining `cv2.dnn` plumbing with a real forward pass and skips itself when `models/yolov8n.onnx` is absent.

**PyTorch is a build-time dependency, never a runtime one.** `ultralytics`/`torch` are only needed once to export `yolov8n.onnx`; the resulting file is portable. `import safestep.app` must never pull in `torch` — there is a check for this alongside the `pyttsx3` one.

**Audio never blocks the capture loop.** This was the original design's central defect. `Pyttsx3Speaker` runs a worker thread that *creates its own engine* (pyttsx3 drivers have thread affinity) and holds a **single-slot mailbox**: a new message replaces an unspoken one rather than queueing, because a stale obstacle warning is worse than none.

**Importing the package is side-effect free.** No camera, no model, no TTS engine. `pyttsx3` is imported inside the speech worker thread; `cv2.dnn` model loading happens only in a detector constructor. There is a subprocess test asserting `pyttsx3` is absent from `sys.modules` after importing `safestep.app` — do not "tidy" that import to the top of the module.

**`AlertPolicy` is pure logic over an injectable clock** ([alerts.py](src/safestep/alerts.py)), so cooldown and rate-limit behaviour is tested deterministically with a fake clock rather than `sleep`. Four rules in order: relevance floor, confirmation (`min_hits`), priority (one announcement, most urgent, centre-preferring), then rate limiting (per-track cooldown + global gap).

**The web dashboard reuses the pipeline, it does not reimplement it.** [web/pipeline.py](src/safestep/web/pipeline.py) drives the same detector, tracker, `AlertPolicy` and haptics on a worker thread and publishes a `Snapshot`; [web/server.py](src/safestep/web/server.py) broadcasts it. FastAPI/uvicorn are an optional extra (`pip install 'safestep[web]'`) and imported lazily, so a headless Pi without them still runs.

**Frame and telemetry ship in one WebSocket message.** Sending them on separate channels would let boxes drift out of sync with the picture they describe — unacceptable in a tool about spatial awareness. `Snapshot.seq` lets a slow client skip frames instead of accumulating stale ones.

**Browser audio is synthesised client-side**, not streamed. The Web Audio API gives sample-accurate stereo-panned earcons and a real analyser trace, with no cross-process audio plumbing. `haptics.py` is pure arithmetic with no hardware dependency: it drives the on-screen meters today and would drive GPIO PWM unchanged.

**The browser demo in [docs/](docs/) is a second implementation, not a wrapper.** It re-does spatial reasoning, tracking, the alert policy and haptics in JavaScript so the page can be static and run inference on-device. Two implementations of safety-relevant logic is a liability, so it is held together by generation and testing rather than discipline: the lookup tables are **generated** by `scripts/generate_tables.py`, and `scripts/check_parity.mjs` replays ~2,500 recorded Python results through the JS modules. **Change Python behaviour and you must re-run both scripts**, or CI fails — deliberately. Never hand-edit `docs/js/tables.mjs`.

## Things that will bite you

- **OpenCV 5.0 removed the Caffe importer.** `cv2.dnn.readNetFromCaffe` does not exist there, so the `ssd` backend raises `BackendUnavailable` on 5.x. Guard any new Caffe/Darknet usage with a capability check like `caffe_support_available()`. ONNX works on both lines and is the recommended path.
- **`cv2.dnn.NMSBoxes` filters with strict `>`, not `>=`.** The YOLO decoder passes it a score threshold of `0.0` deliberately and applies confidence in its own vectorised prefilter, so a detection scoring exactly at the threshold is not silently dropped. Do not "restore" the threshold argument.
- **Distance is a pinhole estimate, not a measurement.** `distance = real_height × focal_px ÷ pixel_height`, with per-class heights in `KNOWN_HEIGHTS_M`. It needs a correct `--hfov` and returns `None` for unknown classes, which then fall back to a frame-fraction heuristic. A monocular camera fundamentally cannot see drop-offs or kerbs.
- **Distance picks height or width, and the choice matters more than the formula.** Height is the better cue in general (it does not change as a person turns), but it is useless when the object is vertically incomplete. `estimate_distance()` falls back to width when the box is cut off by the top/bottom edge, *or* when its aspect ratio is far from the class's canonical shape -- the latter catches partial detections that touch no edge, such as a hand picked up as "person". Measured on real hardware: a subject at 0.5 m read ~2.0 m before this and reads 0.5 m after. `KNOWN_WIDTHS_M` must stay in step with `KNOWN_HEIGHTS_M`; a class missing from both gets no distance at all.
- **Truncated boxes under-report distance, and the correction is deliberate.** When a box touches the top or bottom frame edge only part of the object's height is visible, so the pinhole figure becomes an *upper bound* — a person at 0.5 m reads as 2.5 m. This was measured on real hardware, and the error always points the same way: under-warning about the nearest obstacles. `enrich()` therefore flags `Detection.truncated` and takes `max(pinhole_band, frame_fraction_band)`. Do not "simplify" that back to trusting the pinhole value alone. Horizontal clipping does not trigger this — only height feeds the estimate.
- **`RunStats.elapsed_s` excludes camera startup.** Opening a camera took 43 s on this Windows box; folding it into processing time made the reported fps wrong by ~9x. `RunClock` splits `startup_s` from `elapsed_s`, and `fps` uses the latter.
- **The colour backend is not a safety mechanism.** It detects coloured blobs. It exists so the app runs before weights are fetched, and its defaults were widened from the original (which required saturation ≥ 150 and so ignored walls, concrete and most real hazards). Say so in any user-facing text.
- **Detection may run every Nth frame** (`--detect-every-n`, 3 under `--profile pi`). The tracker carries identities across the gaps, so boxes go slightly stale between passes. Anything reading `detections` outside a detection frame must tolerate that.
- **Radar bearing convention: angle 0 is straight ahead (up), positive to the right, plotted as `(cx + sin a * r, cy - cos a * r)`.** An earlier version added a spurious `-pi/2`, which rotated the whole plot 90 degrees and drew a centre obstacle hard left on the baseline. Every radar element -- blips, sector dividers, sweep -- must go through the same `polar()` helper.
- **Haptic strength is continuous, not banded.** `intensity_for_distance()` ramps linearly from full power at 0.5 m to nothing at 5 m, with `_INTENSITY_FLOOR` per band so anything worth announcing stays perceptible. The band table is only the fallback for detections with no metric distance. Do not collapse this back to `_INTENSITY[proximity]`: stepping between three levels means a wearer cannot feel an obstacle approaching.
- **`ort.env.wasm.wasmPaths` resolves relative to the ORT bundle, not the page.** A relative path there yields `/vendor/vendor/...`; a root-relative one breaks when Pages serves from a `/SAFESTEP/` subpath. Use `new URL('../vendor/ort/', import.meta.url).href`.
- **Probe `navigator.gpu.requestAdapter()`, not just `navigator.gpu`.** The object exists in environments with no usable adapter (headless Chromium, for one). Asking ORT for `webgpu` anyway makes it fall back internally *without telling you*, so the UI reports a backend it is not running.
- **Never rebuild dashboard DOM every frame.** Telemetry arrives at 40–90 fps. Re-assigning `innerHTML` at that rate restarts CSS entry animations continuously, which pinned every transcript message at `opacity: 0` and left the panel looking empty while the DOM was full. The transcript and detections table are guarded by content keys — keep it that way; removing the guard also roughly halves the UI frame rate.
