<p align="center">
  <h1 align="center">🦯 SafeStep</h1>
  <p align="center">
    <strong>Real-time obstacle detection and audio guidance for visually impaired users</strong>
  </p>
  <p align="center">
    <a href="https://github.com/shlokdhanokar/SAFESTEP/actions"><img src="https://github.com/shlokdhanokar/SAFESTEP/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  </p>
</p>

---

## ⚠️ Safety notice — read this first

**SafeStep is an experimental aid, not a certified mobility device.**

- It **does not replace** a white cane, a guide dog, or orientation and mobility training.
- It **will miss obstacles.** Glass, drop-offs, kerbs, potholes, overhanging branches and head-height hazards are all poorly detected or not detected at all. A monocular camera cannot see downward steps.
- Distance readings are **estimates** derived from apparent object size, not measurements. They assume typical object dimensions and degrade with unusual sizes, partial occlusion and tilt.
- When an obstacle is cut off by the top or bottom of the frame — which happens constantly at close range — only a lower bound on its size is visible, so the reported distance is an **upper bound**, shown as `<2.5m`. SafeStep escalates the urgency of such obstacles rather than trusting the number.
- Silence does **not** mean the path is clear.

Treat it as a supplementary hint channel alongside your primary mobility aid, and validate it in a safe, controlled space before relying on it anywhere else.

---

## What it does

```
camera ─► detector ─► distance + direction ─► tracker ─► alert policy ─► earcon + speech
```

1. **Detect.** A neural detector (YOLOv8n or MobileNet-SSD) finds objects and labels them. A colour-threshold fallback runs with no model files at all.
2. **Locate.** Each box is converted to a direction (left / ahead / right) and a distance estimated from the pinhole camera model, `distance = real_height × focal_length ÷ pixel_height`.
3. **Track.** An IoU tracker gives obstacles stable identities across frames, so the same chair is not announced twice.
4. **Prioritise.** At most one obstacle is announced at a time: the most urgent, preferring what is directly ahead. A list is useless to someone walking.
5. **Alert.** A stereo-panned tone fires immediately (pitch and pulse rate encode urgency, panning encodes direction), and speech follows with the detail — `"person, very close, ahead"`.

Speech is the *secondary* channel by design. A spoken sentence takes over a second; at walking pace the user has covered two metres before it finishes. The tone arrives in tens of milliseconds.

---

## Try it in your browser

**[shlokdhanokar.github.io/SAFESTEP](https://shlokdhanokar.github.io/SAFESTEP/)** — no install, works on a phone.

Point your camera at a room and SafeStep names what it sees, estimates range and
bearing, and warns you with a tone, a spoken alert and a haptic pattern.

Everything runs **on your device**. YOLOv8n executes in your browser via
onnxruntime-web (WebGPU where available, WASM everywhere else); there is no
backend, and no frame ever leaves your machine. First load pulls ~15 MB of model
and runtime, then it is cached and works offline.

The browser build shares no code with the Python app — it is a second
implementation of the same logic — so a
[parity harness](scripts/check_parity.mjs) replays ~2,500 recorded Python
results through the JavaScript and fails CI on any disagreement. Shipping a demo
that behaves differently from the device it advertises would be worse than
shipping no demo.

```bash
npm install && npm run vendor     # fetch the onnxruntime files
cp models/yolov8n.onnx docs/models/
npm run serve                     # http://127.0.0.1:8123
```

---

## Install

```bash
git clone https://github.com/shlokdhanokar/SAFESTEP.git
cd SAFESTEP
pip install -e .
```

Then fetch detector weights (optional — the app runs without them):

```bash
python scripts/fetch_models.py
```

Weights are never committed. If a download fails, the script prints the exact file and destination so you can place it manually.

---

## Run

```bash
safestep                                  # default camera, auto-select detector
safestep --detector yolo                  # force the YOLOv8 backend
safestep --source clip.mp4 --headless     # run against a video file, no window
safestep --profile pi                     # Raspberry Pi defaults
safestep --help                           # every option
```

`python main.py …` works identically without installing.

Press **Q** or **Esc** in the preview window to quit, or Ctrl-C in the terminal.

### Browser dashboard

```bash
pip install 'safestep[web]'
safestep --web --detector yolo          # then open http://127.0.0.1:8000
```

A live operator view: camera feed with detection overlay, per-object range and
bearing, a field radar, haptic motor meters, and a running transcript of every
spoken alert.

| Panel | Shows |
|---|---|
| **Camera Vision** | Live feed with corner-bracket boxes coloured by urgency, zone guides, range labels (`<1.8m` marks an upper bound from a truncated box) |
| **Nearest Obstacle** | The one obstacle the alert policy would announce, with band, bearing and confidence |
| **Haptic Feedback** | Left/right vibration-motor duty cycles, overall strength and pulse rate. Strength scales **continuously** with distance — full power inside 0.5 m, fading to nothing at 5 m — so an approaching obstacle is felt building rather than snapping between levels |
| **Field** | Top-down radar placing each obstacle by bearing and range |
| **Audio · Live Transcript** | Scrolling level history of the earcons plus every alert as it is spoken |

Audio is synthesised **in the browser** with the Web Audio API rather than
streamed from the server: the earcons are sample-accurate, stereo-panned by
bearing, and the waveform is a real analyser trace rather than an animation.
Click **enable audio** first — browsers block autoplay until you interact.

The dashboard is fully self-contained (no CDN, no fonts, no external requests),
because a wearable device cannot assume an internet connection. `--web` implies
headless: the browser replaces the desktop preview rather than running beside it.

### Options worth knowing

| Flag | Why you would use it |
|---|---|
| `--source` | Camera index (`0`) **or a video file path**. File sources make the whole pipeline runnable with no hardware. |
| `--hfov` | Your camera's horizontal field of view in degrees. Distance estimates are only as good as this number — measure it. |
| `--profile pi` | Headless, 480×360, detection every 3rd frame. |
| `--detect-every-n` | Throughput lever. Detection runs every Nth frame; the tracker covers the gaps. |
| `--classes` | Restrict announcements, e.g. `--classes person chair`. |
| `--cooldown` / `--min-gap` | How talkative it is. Per-object silence window, and a floor between any two alerts. |
| `--min-proximity` | Quietest band worth announcing. `near` is a good choice in busy places. |
| `--no-speech` / `--no-earcons` | Disable either audio channel. |
| `--mirror` | Flip the image. Use with a **front-facing** laptop or phone camera, where left and right are otherwise reported reversed. A worn camera faces away from you and needs no flip. |

---

## Detector backends

| Backend | Model | Size | Notes |
|---|---|---|---|
| `yolo` | YOLOv8n (ONNX) | ~12 MB | **Recommended.** Best accuracy, no PyTorch at runtime, works on OpenCV 4.x *and* 5.x. |
| `ssd` | MobileNet-SSD (Caffe) | ~22 MB | **Requires OpenCV 4.x.** OpenCV 5.0 removed the Caffe importer, so this backend refuses to start there. |
| `color` | none | 0 | HSV threshold fallback. Detects coloured blobs, *not* obstacles. Demo and development only. |
| `auto` | — | — | Default. Tries yolo → ssd → color and logs what it picked. |

Asking for a specific backend that cannot load fails loudly with remediation steps rather than silently downgrading — a device that quietly stops detecting is worse than one that refuses to start.

### Getting the YOLO ONNX file

OpenCV needs ONNX, but upstream publishes only the PyTorch checkpoint. `fetch_models.py` downloads it and exports it if `ultralytics` is installed. The export is a one-off and the result is portable, so you can export on a laptop and copy `yolov8n.onnx` to a Pi that has neither PyTorch nor ultralytics:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12
```

---

## Hardware

- Webcam or Pi Camera module
- Headphones — **stereo matters**, the directional cue is carried by panning. Bone-conduction headphones are strongly preferable so ambient hearing stays unobstructed.
- Raspberry Pi 4 or any laptop

---

## Development

```bash
pip install -r requirements-dev.txt
pytest                      # 372 tests, no camera / display / audio needed
pytest tests/test_alerts.py # one module
pytest -k cooldown          # one topic

flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics   # CI blocking gate
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

The whole suite runs with nothing plugged in. That is a hard design constraint, not a coincidence: detectors are pure functions of a frame, the alert policy takes an injectable clock, audio sits behind protocols with fakes, and neural tensor decoding is tested against hand-built tensors so no weights are needed.

One module is the deliberate exception. [tests/test_integration_yolo.py](tests/test_integration_yolo.py) loads the real network and runs a real forward pass, covering the `cv2.dnn` plumbing that synthetic tensors cannot reach. It **skips itself** when `models/yolov8n.onnx` is absent, so it never blocks a fresh checkout.

To exercise the real pipeline without a camera, point it at a video file:

```bash
safestep --source clip.mp4 --headless --detector color --log-level DEBUG
```

### Layout

```
src/safestep/
  geometry.py     BBox: area, centroid, IoU, clipping
  spatial.py      zones, proximity bands, pinhole distance, Detection
  tracking.py     IoU tracker -> stable track ids
  alerts.py       priority, cooldown, dedupe, phrasing   (pure, injectable clock)
  detection/      Detector protocol + color / ssd / yolo backends + factory
  feedback/       Announcer protocol + earcons / speech / composite
  camera.py       camera or video file, buffer pinning, reconnect
  app.py          the loop
  cli.py          argument parsing
```

---

## Roadmap

- [ ] **Depth sensing** — stereo pair or ToF module for true distance rather than an apparent-size estimate
- [ ] **Downward hazards** — kerbs, steps and drop-offs, the biggest current blind spot
- [ ] **Ground-plane estimation** — distinguish obstacles in the walking path from objects merely in frame
- [ ] **Multilingual speech**
- [ ] **Wearable enclosure** and battery profiling
- [ ] **User testing** with visually impaired participants and O&M instructors

---

## Contributing

Contributions are welcome. Please keep the test suite hardware-free, and be conservative about anything that changes what the user is or is not told — under-warning is a safety issue, and over-warning gets the device switched off.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

[OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) · [pyttsx3](https://pyttsx3.readthedocs.io/) · [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
