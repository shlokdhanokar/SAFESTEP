/**
 * Axis-aligned bounding boxes.
 *
 * A direct port of `src/safestep/geometry.py`. The Python and JavaScript
 * implementations are held in lockstep by a parity harness
 * (`scripts/check_parity.mjs`), so behaviour must not drift between them.
 */

export class BBox {
  constructor(x, y, w, h) {
    if (w < 0 || h < 0) throw new RangeError(`BBox extent must be non-negative, got w=${w} h=${h}`);
    this.x = x; this.y = y; this.w = w; this.h = h;
  }

  get x2() { return this.x + this.w; }
  get y2() { return this.y + this.h; }
  get area() { return this.w * this.h; }
  get cx() { return this.x + this.w / 2; }
  get cy() { return this.y + this.h / 2; }

  clipped(width, height) {
    const x1 = Math.max(0, Math.min(this.x, width));
    const y1 = Math.max(0, Math.min(this.y, height));
    const x2 = Math.max(x1, Math.min(this.x2, width));
    const y2 = Math.max(y1, Math.min(this.y2, height));
    return new BBox(x1, y1, x2 - x1, y2 - y1);
  }

  iou(other) {
    const ix1 = Math.max(this.x, other.x);
    const iy1 = Math.max(this.y, other.y);
    const ix2 = Math.min(this.x2, other.x2);
    const iy2 = Math.min(this.y2, other.y2);

    const iw = ix2 - ix1;
    const ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0;

    const intersection = iw * ih;
    const union = this.area + other.area - intersection;
    return union <= 0 ? 0 : intersection / union;
  }

  /**
   * Build from corner coordinates, rounding to whole pixels.
   *
   * Uses banker's rounding to match Python's `round()`; plain `Math.round`
   * differs on exact .5 values and would break parity.
   */
  static fromXYXY(x1, y1, x2, y2) {
    const a = bankersRound(x1), b = bankersRound(x2);
    const c = bankersRound(y1), d = bankersRound(y2);
    const left = Math.min(a, b), right = Math.max(a, b);
    const top = Math.min(c, d), bottom = Math.max(c, d);
    return new BBox(left, top, right - left, bottom - top);
  }
}

/** Round half to even, matching Python's built-in `round()`. */
export function bankersRound(value) {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}
