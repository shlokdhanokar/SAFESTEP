/**
 * Assert the JavaScript port agrees with Python, case for case.
 *
 * Two implementations of safety-relevant logic is a liability unless something
 * holds them together. `scripts/generate_parity_fixtures.py` records Python's
 * answers; this replays them through the browser modules and fails on any
 * disagreement, so a change in either language is caught immediately.
 *
 *   python scripts/generate_parity_fixtures.py
 *   node scripts/check_parity.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { BBox } from '../docs/js/geometry.mjs';
import {
  enrich, focalLengthPx, Proximity, zoneFor,
} from '../docs/js/spatial.mjs';
import { KNOWN_HEIGHTS_M, KNOWN_WIDTHS_M } from '../docs/js/tables.mjs';
import { IoUTracker } from '../docs/js/tracking.mjs';
import { AlertPolicy, describe } from '../docs/js/alerts.mjs';
import { cueFor, strongestCue } from '../docs/js/haptics.mjs';
import { decodeYolov8Output } from '../docs/js/detector.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const fixture = JSON.parse(readFileSync(join(ROOT, 'tests/fixtures/parity.json'), 'utf8'));
const { frameWidth: W, frameHeight: H, focalPx: FOCAL } = fixture.meta;

const EPS = 1e-6;
let checks = 0;
const failures = [];

function fail(section, detail) { failures.push(`[${section}] ${detail}`); }

function eq(section, label, actual, expected) {
  checks += 1;
  const near = typeof actual === 'number' && typeof expected === 'number'
    ? Math.abs(actual - expected) <= EPS * Math.max(1, Math.abs(expected))
    : actual === expected;
  if (!near) fail(section, `${label}: js=${JSON.stringify(actual)} py=${JSON.stringify(expected)}`);
}

const box = (o) => new BBox(o.x, o.y, o.w, o.h);
const boxEq = (section, label, a, e) => {
  eq(section, `${label}.x`, a.x, e.x); eq(section, `${label}.y`, a.y, e.y);
  eq(section, `${label}.w`, a.w, e.w); eq(section, `${label}.h`, a.h, e.h);
};

// ── generated tables must match their Python source ──────────────────────
for (const [label, value] of Object.entries(fixture.tables.heights)) {
  eq('tables', `height:${label}`, KNOWN_HEIGHTS_M[label], value);
}
for (const [label, value] of Object.entries(fixture.tables.widths)) {
  eq('tables', `width:${label}`, KNOWN_WIDTHS_M[label], value);
}
eq('tables', 'heightCount', Object.keys(KNOWN_HEIGHTS_M).length,
   Object.keys(fixture.tables.heights).length);
eq('tables', 'widthCount', Object.keys(KNOWN_WIDTHS_M).length,
   Object.keys(fixture.tables.widths).length);

// ── focal length ─────────────────────────────────────────────────────────
eq('meta', 'focalPx', focalLengthPx(W, 65.0), FOCAL);

// ── geometry ─────────────────────────────────────────────────────────────
for (const c of fixture.geometry) {
  if (c.fromXYXY) {
    boxEq('geometry', 'fromXYXY', BBox.fromXYXY(...c.fromXYXY), c.result);
    continue;
  }
  const a = box(c.a);
  eq('geometry', 'iou', a.iou(box(c.b)), c.iou);
  boxEq('geometry', 'clipped', a.clipped(W, H), c.clipped);
  eq('geometry', 'area', a.area, c.area);
  eq('geometry', 'cx', a.cx, c.cx);
  eq('geometry', 'cy', a.cy, c.cy);
}

// ── spatial enrichment ───────────────────────────────────────────────────
for (const c of fixture.spatial) {
  const got = enrich(box(c.box), c.label, 0.8, W, H, FOCAL);
  const want = c.expected;
  const tag = `${c.label}@${c.box.x},${c.box.y},${c.box.w},${c.box.h}`;
  boxEq('spatial', `${tag}.bbox`, got.bbox, want.bbox);
  eq('spatial', `${tag}.zone`, got.zone, want.zone);
  eq('spatial', `${tag}.proximity`, got.proximity, want.proximity);
  eq('spatial', `${tag}.distanceM`, got.distanceM === null ? null : +got.distanceM.toFixed(6),
     want.distanceM);
  eq('spatial', `${tag}.truncated`, got.truncated, want.truncated);
  eq('spatial', `${tag}.distanceMethod`, got.distanceMethod, want.distanceMethod);
}

// ── zones ────────────────────────────────────────────────────────────────
for (const c of fixture.zones) {
  eq('zones', `cx=${c.cx} cf=${c.centerFraction}`,
     zoneFor(new BBox(c.cx - 5, 100, 10, 100), W, c.centerFraction), c.expected);
}

// ── tracking ─────────────────────────────────────────────────────────────
{
  const tracker = new IoUTracker();
  fixture.tracking.forEach((step, i) => {
    const dets = step.boxes.map((b) => enrich(box(b), 'person', 0.9, W, H, FOCAL));
    const tracked = tracker.update(dets);
    eq('tracking', `frame${i}.count`, tracked.length, step.trackIds.length);
    tracked.forEach((d, j) => {
      eq('tracking', `frame${i}.id[${j}]`, d.trackId, step.trackIds[j]);
      eq('tracking', `frame${i}.hits[${j}]`, tracker.hitsFor(d.trackId), step.hits[j]);
    });
  });
}

// ── describe ─────────────────────────────────────────────────────────────
for (const c of fixture.describe) {
  eq('describe', `${c.proximity}/${c.zone}`,
     describe({ label: 'person', proximity: c.proximity, zone: c.zone }), c.text);
}

// ── alert policy timelines ───────────────────────────────────────────────
for (const scenario of fixture.alerts) {
  const clock = { t: 1000.0 };
  const policy = new AlertPolicy({
    cooldownS: scenario.cooldownS, minGapS: scenario.minGapS, minHits: 1,
    minProximity: Proximity.MODERATE, clock: () => clock.t,
  });
  for (const step of scenario.timeline) {
    clock.t = 1000.0 + step.t;
    const dets = step.detections.map((d) => ({
      bbox: new BBox(300, 200, 60, 200),
      label: d.label, confidence: 0.9, zone: d.zone,
      proximity: d.proximity, distanceM: 2.0, trackId: d.trackId,
      truncated: false, distanceMethod: 'height',
    }));
    const result = policy.select(dets);
    eq('alerts', `${scenario.name}@t=${step.t}`,
       result === null ? null : result.text, step.announced);
  }
}

// ── haptics ──────────────────────────────────────────────────────────────
for (const c of fixture.haptics) {
  const cue = cueFor(c.proximity, c.zone, c.distanceM);
  const tag = `${c.proximity}/${c.zone}/${c.distanceM}`;
  for (const field of ['left', 'right', 'pulseHz', 'intensity', 'strength']) {
    eq('haptics', `${tag}.${field}`, +cue[field].toFixed(6), c.expected[field]);
  }
  eq('haptics', `${tag}.active`, cue.active, c.expected.active);
}

for (const c of fixture.strongest) {
  const dets = c.group.map((g) => ({
    bbox: new BBox(100, 100, g.w, g.h), label: 'person', confidence: 0.9,
    zone: g.zone, proximity: g.proximity, distanceM: 2.0, trackId: null,
    truncated: false, distanceMethod: 'height',
  }));
  const cue = strongestCue(dets);
  eq('strongest', 'intensity', +cue.intensity.toFixed(6), c.expected.intensity);
  eq('strongest', 'zone', cue.zone, c.expected.zone);
  eq('strongest', 'proximity', cue.proximity, c.expected.proximity);
}

// ── YOLOv8 decode ────────────────────────────────────────────────────────
for (const c of fixture.decode) {
  const channels = 4 + c.numClasses;
  const data = new Float32Array(channels * c.anchors);
  const set = (ch, a, v) => { data[ch * c.anchors + a] = v; };
  c.rows.forEach(([cx, cy, w, h, cls, score], a) => {
    set(0, a, cx); set(1, a, cy); set(2, a, w); set(3, a, h); set(4 + cls, a, score);
  });

  const got = decodeYolov8Output(data, [1, channels, c.anchors],
                                 c.frameWidth, c.frameHeight, { confidence: 0.45 });
  const tag = `${c.name}@${c.frameWidth}x${c.frameHeight}`;
  eq('decode', `${tag}.count`, got.length, c.expected.length);
  got.forEach((r, i) => {
    const want = c.expected[i];
    if (!want) return;
    boxEq('decode', `${tag}[${i}]`, r.bbox, want.bbox);
    eq('decode', `${tag}[${i}].label`, r.label, want.label);
    eq('decode', `${tag}[${i}].confidence`, +r.confidence.toFixed(5), want.confidence);
  });
}

// ── report ───────────────────────────────────────────────────────────────
if (failures.length) {
  console.error(`\nPARITY FAILED: ${failures.length} of ${checks} assertions\n`);
  for (const f of failures.slice(0, 40)) console.error('  ' + f);
  if (failures.length > 40) console.error(`  ... and ${failures.length - 40} more`);
  process.exit(1);
}
console.log(`parity OK: ${checks} assertions across ${Object.keys(fixture).length - 2} sections`);
