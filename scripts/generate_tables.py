#!/usr/bin/env python3
"""Emit the class dimension tables as a JavaScript module.

The browser demo re-implements SafeStep's spatial reasoning in JavaScript. The
*logic* is hand-ported and held in step by the parity harness, but the lookup
tables are generated so they can never silently diverge: add a class in Python,
re-run this, and the demo picks it up.

    python scripts/generate_tables.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from safestep.detection.yolo_onnx import COCO_CLASSES, INPUT_SIZE  # noqa: E402
from safestep.spatial import (  # noqa: E402
    IMMEDIATE_M,
    KNOWN_HEIGHTS_M,
    KNOWN_WIDTHS_M,
    MODERATE_M,
    NEAR_M,
    UNKNOWN_LABEL,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "js" / "tables.mjs"

HEADER = """/**
 * Class dimension tables and band edges.
 *
 * GENERATED from src/safestep/spatial.py by scripts/generate_tables.py --
 * do not edit by hand. Regenerate after changing the Python tables; the
 * parity harness fails if these drift.
 */

"""


def js_table(name: str, table: dict) -> str:
    body = ",\n".join(f"  {json.dumps(k)}: {v}" for k, v in sorted(table.items()))
    return f"export const {name} = {{\n{body}\n}};\n"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        HEADER
        + f"export const UNKNOWN_LABEL = {json.dumps(UNKNOWN_LABEL)};\n"
        + f"export const IMMEDIATE_M = {IMMEDIATE_M};\n"
        + f"export const NEAR_M = {NEAR_M};\n"
        + f"export const MODERATE_M = {MODERATE_M};\n"
        + f"export const YOLO_INPUT_SIZE = {INPUT_SIZE};\n\n"
        + "// Index order is fixed by the trained network; do not reorder.\n"
        + "export const COCO_CLASSES = [\n"
        + "".join(f"  {json.dumps(c)},\n" for c in COCO_CLASSES)
        + "];\n\n"
        + js_table("KNOWN_HEIGHTS_M", KNOWN_HEIGHTS_M)
        + "\n"
        + js_table("KNOWN_WIDTHS_M", KNOWN_WIDTHS_M),
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(KNOWN_HEIGHTS_M)} heights, {len(KNOWN_WIDTHS_M)} widths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
