/**
 * YOLOv8 inference in the browser, via onnxruntime-web.
 *
 * The desktop app runs the same ONNX file through OpenCV's DNN module; here it
 * runs on WASM in the visitor's own browser. Nothing leaves their machine,
 * which is the whole point of the public demo: a camera-based aid that uploads
 * your camera would be a poor advertisement for it.
 *
 * `decodeYolov8Output` is a port of `decode_yolov8_output` in
 * `src/safestep/detection/yolo_onnx.py`, including the deliberate choice to
 * apply confidence in the prefilter and give NMS a threshold of zero.
 */

import { BBox } from './geometry.mjs';
import { COCO_CLASSES, YOLO_INPUT_SIZE } from './tables.mjs';

export { COCO_CLASSES };

/**
 * Greedy non-maximum suppression.
 *
 * Stands in for `cv2.dnn.NMSBoxes`. Note that OpenCV compares scores with a
 * strict `>`; the Python side sidesteps that by passing a threshold of 0 and
 * filtering beforehand, and this does the same, so a detection sitting exactly
 * on the confidence threshold survives in both.
 */
export function nms(boxes, scores, iouThreshold) {
  const order = scores
    .map((s, i) => [s, i])
    .sort((a, b) => b[0] - a[0])
    .map(([, i]) => i);

  const keep = [];
  const suppressed = new Uint8Array(boxes.length);

  for (const i of order) {
    if (suppressed[i]) continue;
    keep.push(i);
    for (const j of order) {
      if (j === i || suppressed[j]) continue;
      if (boxes[i].iou(boxes[j]) > iouThreshold) suppressed[j] = 1;
    }
  }
  return keep;
}

/**
 * Convert a raw YOLOv8 forward pass into detections.
 *
 * `data` is the flat output tensor with dims `[1, 4 + numClasses, numAnchors]`:
 * rows 0-3 are the box in centre form in input-space pixels, the rest are
 * per-class scores. YOLOv8 has no separate objectness score, so the class
 * score is the confidence.
 */
export function decodeYolov8Output(data, dims, frameWidth, frameHeight, {
  confidence = 0.45,
  nmsThreshold = 0.45,
  classesOfInterest = null,
  inputSize = YOLO_INPUT_SIZE,
} = {}) {
  const channels = dims[1];
  const anchors = dims[2];
  const numClasses = channels - 4;

  const xFactor = frameWidth / inputSize;
  const yFactor = frameHeight / inputSize;

  // Row-major [channel][anchor], so channel c of anchor a is at c*anchors + a.
  const at = (c, a) => data[c * anchors + a];

  const boxes = [];
  const scores = [];
  const classIds = [];

  for (let a = 0; a < anchors; a += 1) {
    let bestClass = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c += 1) {
      const s = at(4 + c, a);
      if (s > bestScore) { bestScore = s; bestClass = c; }
    }
    if (bestScore < confidence) continue;

    const cx = at(0, a); const cy = at(1, a);
    const bw = at(2, a); const bh = at(3, a);

    boxes.push(new BBox(
      (cx - bw / 2) * xFactor, (cy - bh / 2) * yFactor,
      Math.max(0, bw * xFactor), Math.max(0, bh * yFactor),
    ));
    scores.push(bestScore);
    classIds.push(bestClass);
  }

  if (!boxes.length) return [];

  const wanted = classesOfInterest
    ? new Set([...classesOfInterest].map((c) => c.toLowerCase()))
    : null;

  const results = [];
  for (const idx of nms(boxes, scores, nmsThreshold)) {
    const classId = classIds[idx];
    if (classId < 0 || classId >= COCO_CLASSES.length) continue;
    const label = COCO_CLASSES[classId];
    if (wanted && !wanted.has(label.toLowerCase())) continue;

    const b = boxes[idx];
    const bbox = BBox.fromXYXY(b.x, b.y, b.x + b.w, b.y + b.h)
      .clipped(frameWidth, frameHeight);
    if (bbox.area <= 0) continue;

    results.push({ bbox, label, confidence: scores[idx] });
  }
  return results;
}

/** Loads the model and runs frames through it. */
export class YoloDetector {
  constructor(ort, session, { confidence = 0.45, nmsThreshold = 0.45, backend = 'wasm' } = {}) {
    this.name = 'yolo';
    this.backend = backend;
    this.ort = ort;
    this.session = session;
    this.confidence = confidence;
    this.nmsThreshold = nmsThreshold;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];

    // Reused between frames: allocating a 640x640x3 float array per frame
    // would keep the garbage collector busy enough to cost real frame rate.
    this._input = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
    this._canvas = new OffscreenCanvas(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
  }

  static async create(ort, modelUrl, options = {}, onProgress = null) {
    const providers = options.providers ?? ['wasm'];
    if (onProgress) onProgress('fetching model');
    const response = await fetch(modelUrl);
    if (!response.ok) throw new Error(`model fetch failed: HTTP ${response.status}`);
    const buffer = new Uint8Array(await response.arrayBuffer());

    if (onProgress) onProgress(`starting runtime (${providers[0]})`);
    const session = await ort.InferenceSession.create(buffer, {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
    return new YoloDetector(ort, session, { ...options, backend: providers[0] });
  }

  /**
   * Detect obstacles in a video frame.
   * `source` is anything drawImage accepts (a <video> element here).
   */
  async detect(source, frameWidth, frameHeight) {
    const S = YOLO_INPUT_SIZE;

    // Plain resize, not letterboxed, matching the Python preprocessing so the
    // inverse is a single independent scale factor per axis.
    this._ctx.drawImage(source, 0, 0, S, S);
    const { data } = this._ctx.getImageData(0, 0, S, S);

    // RGBA bytes -> planar RGB floats in 0..1, which is what the model wants.
    const plane = S * S;
    const input = this._input;
    for (let i = 0, px = 0; px < plane; px += 1, i += 4) {
      input[px] = data[i] / 255;
      input[plane + px] = data[i + 1] / 255;
      input[2 * plane + px] = data[i + 2] / 255;
    }

    const tensor = new this.ort.Tensor('float32', input, [1, 3, S, S]);
    const output = await this.session.run({ [this.inputName]: tensor });
    const result = output[this.outputName];

    return decodeYolov8Output(result.data, result.dims, frameWidth, frameHeight, {
      confidence: this.confidence,
      nmsThreshold: this.nmsThreshold,
    });
  }
}
