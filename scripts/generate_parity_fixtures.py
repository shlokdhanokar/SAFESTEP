#!/usr/bin/env python3
"""Record Python's answers for a spread of cases, for the JS port to match.

The browser demo re-implements SafeStep's spatial reasoning, tracking, alert
policy and haptics in JavaScript. Two implementations of safety-relevant logic
is a liability unless something holds them together, so this writes a fixture
of inputs and Python's outputs, and `scripts/check_parity.mjs` asserts the
JavaScript agrees. A behaviour change in either language fails the harness.

    python scripts/generate_parity_fixtures.py
    node scripts/check_parity.mjs
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from safestep.alerts import AlertPolicy, describe  # noqa: E402
from safestep.geometry import BBox  # noqa: E402
from safestep.haptics import cue_for, strongest_cue  # noqa: E402
from safestep.spatial import (  # noqa: E402
    KNOWN_HEIGHTS_M,
    KNOWN_WIDTHS_M,
    Proximity,
    Zone,
    enrich,
    focal_length_px,
    zone_for,
)
from safestep.tracking import IoUTracker  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "parity.json"

FRAME_W, FRAME_H = 640, 480
FOCAL = focal_length_px(FRAME_W, 65.0)

# Boxes chosen to hit every branch of the distance selector: fully visible,
# vertically cut, horizontally cut, both, squat-aspect, tiny, oversized.
BOXES = [
    (280, 140, 60, 205), (150, 0, 340, 480), (300, 330, 130, 150),
    (300, 190, 60, 120), (0, 0, 640, 480), (0, 100, 80, 200),
    (560, 100, 80, 200), (300, 420, 40, 60), (10, 10, 30, 30),
    (620, 460, 40, 40), (100, 50, 200, 300), (250, 200, 140, 60),
]
LABELS = ["person", "chair", "car", "cell phone", "bottle", "dog", "obstacle", "bicycle"]


def box_payload(b: BBox) -> dict:
    return {"x": b.x, "y": b.y, "w": b.w, "h": b.h}


def detection_payload(d) -> dict:
    return {
        "bbox": box_payload(d.bbox),
        "label": d.label,
        "confidence": round(d.confidence, 6),
        "zone": d.zone.value,
        "proximity": int(d.proximity),
        "distanceM": None if d.distance_m is None else round(d.distance_m, 6),
        "truncated": d.truncated,
        "distanceMethod": d.distance_method,
        "trackId": d.track_id,
    }


def geometry_cases() -> list:
    cases = []
    for i, a in enumerate(BOXES):
        for b in BOXES[i:]:
            ba, bb = BBox(*a), BBox(*b)
            cases.append({
                "a": box_payload(ba), "b": box_payload(bb),
                "iou": round(ba.iou(bb), 9),
                "clipped": box_payload(ba.clipped(FRAME_W, FRAME_H)),
                "area": ba.area, "cx": ba.cx, "cy": ba.cy,
            })
    for x1, y1, x2, y2 in [(10.4, 20.6, 40.5, 60.4), (40, 60, 10, 20), (0.5, 1.5, 2.5, 3.5)]:
        cases.append({
            "fromXYXY": [x1, y1, x2, y2],
            "result": box_payload(BBox.from_xyxy(x1, y1, x2, y2)),
        })
    return cases


def spatial_cases() -> list:
    cases = []
    for box in BOXES:
        for label in LABELS:
            d = enrich(BBox(*box), label, 0.8, (FRAME_H, FRAME_W, 3), FOCAL)
            cases.append({
                "box": box_payload(BBox(*box)), "label": label,
                "expected": detection_payload(d),
            })
    return cases


def zone_cases() -> list:
    return [
        {"cx": cx, "centerFraction": cf, "expected": zone_for(BBox(cx - 5, 100, 10, 100), FRAME_W, cf).value}
        for cx in (5, 100, 211, 212, 320, 428, 429, 500, 635)
        for cf in (0.10, 0.34, 0.60)
    ]


def tracking_cases() -> list:
    """Drive a tracker through a scripted sequence and record the ids."""
    frames = [
        [(100, 100, 50, 100), (400, 100, 60, 60)],
        [(105, 100, 50, 100), (405, 100, 60, 60)],
        [(105, 100, 50, 100)],
        [],
        [],
        [(105, 100, 50, 100), (500, 300, 40, 40)],
        [(500, 300, 40, 40)],
    ]
    tracker = IoUTracker()
    steps = []
    for frame in frames:
        dets = [enrich(BBox(*b), "person", 0.9, (FRAME_H, FRAME_W, 3), FOCAL) for b in frame]
        tracked = tracker.update(dets)
        steps.append({
            "boxes": [box_payload(BBox(*b)) for b in frame],
            "trackIds": [d.track_id for d in tracked],
            "hits": [tracker.hits_for(d.track_id) for d in tracked],
        })
    return steps


def alert_cases() -> list:
    """Replay a timeline through the policy with a deterministic clock."""
    scenarios = []
    for name, min_gap, cooldown in [("tight", 0.0, 4.0), ("gapped", 1.2, 4.0)]:
        now = {"t": 1000.0}
        policy = AlertPolicy(cooldown_s=cooldown, min_gap_s=min_gap, min_hits=1,
                             min_proximity=Proximity.MODERATE, clock=lambda: now["t"])
        timeline = []
        script = [
            (0.0, [("person", Proximity.IMMEDIATE, Zone.CENTER, 1)]),
            (0.5, [("person", Proximity.IMMEDIATE, Zone.CENTER, 1)]),
            (1.5, [("person", Proximity.IMMEDIATE, Zone.CENTER, 1),
                   ("chair", Proximity.NEAR, Zone.LEFT, 2)]),
            (3.0, [("chair", Proximity.NEAR, Zone.LEFT, 2)]),
            (5.0, [("person", Proximity.IMMEDIATE, Zone.CENTER, 1)]),
            (5.2, [("bottle", Proximity.FAR, Zone.RIGHT, 3)]),
        ]
        for dt, spec in script:
            now["t"] = 1000.0 + dt
            dets = []
            for label, prox, zone, tid in spec:
                dets.append(enrich(BBox(300, 200, 60, 200), label, 0.9,
                                   (FRAME_H, FRAME_W, 3), FOCAL))
                dets[-1] = dets[-1].with_track_id(tid)
                dets[-1] = type(dets[-1])(
                    bbox=dets[-1].bbox, label=label, confidence=0.9, zone=zone,
                    proximity=prox, distance_m=dets[-1].distance_m,
                    track_id=tid, truncated=dets[-1].truncated,
                    distance_method=dets[-1].distance_method,
                )
            result = policy.select(dets)
            timeline.append({
                "t": dt,
                "detections": [
                    {"label": lb, "proximity": int(px), "zone": zn.value, "trackId": tid}
                    for lb, px, zn, tid in spec
                ],
                "announced": None if result is None else result.text,
            })
        scenarios.append({"name": name, "minGapS": min_gap, "cooldownS": cooldown,
                          "timeline": timeline})
    return scenarios


def describe_cases() -> list:
    cases = []
    for prox in Proximity:
        for zone in Zone:
            d = enrich(BBox(300, 200, 60, 200), "person", 0.9, (FRAME_H, FRAME_W, 3), FOCAL)
            d = type(d)(bbox=d.bbox, label="person", confidence=0.9, zone=zone,
                        proximity=prox, distance_m=d.distance_m, track_id=None,
                        truncated=d.truncated, distance_method=d.distance_method)
            cases.append({"proximity": int(prox), "zone": zone.value, "text": describe(d)})
    return cases


def haptic_cases() -> list:
    cases = []
    for prox in Proximity:
        for zone in Zone:
            for distance in (None, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 4.9, 6.0):
                cue = cue_for(prox, zone, distance)
                cases.append({
                    "proximity": int(prox), "zone": zone.value, "distanceM": distance,
                    "expected": {
                        "left": round(cue.left, 6), "right": round(cue.right, 6),
                        "pulseHz": round(cue.pulse_hz, 6),
                        "intensity": round(cue.intensity, 6),
                        "strength": round(cue.strongest, 6), "active": cue.active,
                    },
                })
    return cases


def strongest_cases() -> list:
    groups = [
        [(Proximity.MODERATE, Zone.LEFT, 60, 60), (Proximity.IMMEDIATE, Zone.RIGHT, 60, 60)],
        [(Proximity.NEAR, Zone.LEFT, 60, 60), (Proximity.NEAR, Zone.CENTER, 60, 60)],
        [(Proximity.NEAR, Zone.LEFT, 10, 10), (Proximity.NEAR, Zone.RIGHT, 200, 200)],
        [],
    ]
    out = []
    for group in groups:
        dets = []
        for prox, zone, w, h in group:
            d = enrich(BBox(100, 100, w, h), "person", 0.9, (FRAME_H, FRAME_W, 3), FOCAL)
            dets.append(type(d)(bbox=BBox(100, 100, w, h), label="person", confidence=0.9,
                                zone=zone, proximity=prox, distance_m=2.0, track_id=None,
                                truncated=False, distance_method="height"))
        cue = strongest_cue(dets)
        out.append({
            "group": [{"proximity": int(p), "zone": z.value, "w": w, "h": h}
                      for p, z, w, h in group],
            "expected": {"intensity": round(cue.intensity, 6), "zone": cue.zone.value,
                         "proximity": cue.proximity.name},
        })
    return out


def decode_cases() -> list:
    """Synthetic YOLOv8 tensors through the Python decoder.

    The riskiest part of the port: the transpose, centre-form conversion,
    per-axis rescale and NMS all have to agree exactly.
    """
    import numpy as np

    from safestep.detection.yolo_onnx import decode_yolov8_output

    num_classes, anchors = 80, 40
    scenarios = [
        ("empty", []),
        ("single", [(320, 320, 100, 200, 0, 0.90)]),
        ("threshold", [(320, 320, 100, 200, 0, 0.50)]),
        ("overlapping", [(320, 320, 100, 200, 0, 0.90),
                         (322, 321, 102, 198, 0, 0.85),
                         (318, 319, 98, 202, 0, 0.80)]),
        ("distinct", [(150, 320, 80, 160, 0, 0.90), (500, 320, 80, 160, 0, 0.90)]),
        ("mixed", [(150, 320, 80, 160, 0, 0.90), (500, 320, 80, 160, 56, 0.88)]),
        ("corner", [(0, 0, 200, 200, 0, 0.90)]),
        ("low", [(320, 320, 100, 200, 0, 0.20)]),
    ]

    cases = []
    for name, rows in scenarios:
        tensor = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
        for a, (cx, cy, w, h, cls, score) in enumerate(rows):
            tensor[0, 0, a], tensor[0, 1, a] = cx, cy
            tensor[0, 2, a], tensor[0, 3, a] = w, h
            tensor[0, 4 + cls, a] = score

        for frame_w, frame_h in [(640, 640), (1280, 720)]:
            out = decode_yolov8_output(tensor, frame_w, frame_h, confidence=0.45)
            cases.append({
                "name": name, "rows": rows, "anchors": anchors,
                "numClasses": num_classes, "frameWidth": frame_w, "frameHeight": frame_h,
                "expected": [
                    {"bbox": box_payload(r.bbox), "label": r.label,
                     "confidence": round(float(r.confidence), 5)}
                    for r in out
                ],
            })
    return cases


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"frameWidth": FRAME_W, "frameHeight": FRAME_H, "focalPx": FOCAL},
        "tables": {"heights": KNOWN_HEIGHTS_M, "widths": KNOWN_WIDTHS_M},
        "geometry": geometry_cases(),
        "spatial": spatial_cases(),
        "zones": zone_cases(),
        "tracking": tracking_cases(),
        "alerts": alert_cases(),
        "describe": describe_cases(),
        "haptics": haptic_cases(),
        "strongest": strongest_cases(),
        "decode": decode_cases(),
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    total = sum(len(v) for k, v in payload.items() if isinstance(v, list))
    print(f"wrote {OUT} ({total} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
