#!/usr/bin/env python3
"""Download the neural detector weights into ``models/``.

Weights are never committed -- they are large binaries with their own licences.
This script fetches them on demand.

Usage::

    python scripts/fetch_models.py                 # everything available
    python scripts/fetch_models.py --model yolo    # just YOLOv8n
    python scripts/fetch_models.py --list          # show what is needed

Backend notes
-------------

**yolo (recommended).** OpenCV needs an ONNX file, but upstream publishes only
the PyTorch ``.pt`` checkpoint, so this script downloads that and exports it.
The export needs ``ultralytics`` (which pulls in PyTorch) *once*, on any
machine -- the resulting ``yolov8n.onnx`` is portable, so you can export on a
laptop and copy the file to a Raspberry Pi that has neither. If ``ultralytics``
is absent the script says exactly what to run.

**ssd.** Requires an OpenCV 4.x build. OpenCV 5.0 removed the Caffe importer,
so on 5.x these files download fine but the backend refuses to start. Use
``--detector yolo`` there.

SafeStep runs with no weights at all via ``--detector color``, though that
fallback detects coloured regions rather than obstacles.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

YOLO_ONNX_NAME = "yolov8n.onnx"
YOLO_PT_NAME = "yolov8n.pt"
YOLO_PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"


class Asset(NamedTuple):
    filename: str
    url: str
    sha256: Optional[str]  # None where upstream publishes no stable digest
    note: str


MODELS: Dict[str, List[Asset]] = {
    "ssd": [
        Asset(
            filename="MobileNetSSD_deploy.prototxt",
            url=(
                "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
                "master/deploy.prototxt"
            ),
            sha256=None,
            note="MobileNet-SSD network definition (~44 KB, needs OpenCV 4.x)",
        ),
        Asset(
            filename="MobileNetSSD_deploy.caffemodel",
            url=(
                "https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/"
                "master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel"
            ),
            sha256=None,
            note="MobileNet-SSD weights (~22 MB, needs OpenCV 4.x)",
        ),
    ],
    "yolo": [
        Asset(
            filename=YOLO_PT_NAME,
            url=YOLO_PT_URL,
            sha256=None,
            note="YOLOv8 nano checkpoint (~6 MB); exported to ONNX after download",
        ),
    ],
}

EXPORT_HINT = f"""
  {YOLO_ONNX_NAME} could not be produced automatically because 'ultralytics'
  is not installed. Either:

    1. Install it and re-run (one-off; pulls in PyTorch):
         pip install ultralytics
         python scripts/fetch_models.py --model yolo

    2. Or export on any machine that has it, then copy the file across:
         yolo export model={YOLO_PT_NAME} format=onnx opset=12
         # then place yolov8n.onnx in: {MODEL_DIR}

    3. Or skip YOLO entirely and use the colour fallback:
         safestep --detector color
"""


def human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: Asset, dest_dir: Path, force: bool = False) -> bool:
    dest = dest_dir / asset.filename

    if dest.is_file() and not force:
        print(f"  [skip] {asset.filename} already present ({human(dest.stat().st_size)})")
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {asset.filename} - {asset.note}")
    print(f"         {asset.url}")

    try:
        request = urllib.request.Request(asset.url, headers={"User-Agent": "safestep-fetch/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  [FAIL] {asset.filename}: {exc}", file=sys.stderr)
        print(f"         Place the file manually at: {dest}", file=sys.stderr)
        return False

    if asset.sha256:
        actual = sha256_of(tmp)
        if actual != asset.sha256:
            tmp.unlink(missing_ok=True)
            print(
                f"  [FAIL] {asset.filename}: checksum mismatch\n"
                f"         expected {asset.sha256}\n"
                f"         actual   {actual}",
                file=sys.stderr,
            )
            return False

    tmp.replace(dest)
    print(f"  [ ok ] {asset.filename} ({human(dest.stat().st_size)})")
    return True


def export_yolo_onnx(dest_dir: Path, force: bool = False) -> bool:
    """Convert the downloaded ``.pt`` checkpoint into the ONNX OpenCV needs."""
    onnx_path = dest_dir / YOLO_ONNX_NAME
    pt_path = dest_dir / YOLO_PT_NAME

    if onnx_path.is_file() and not force:
        print(f"  [skip] {YOLO_ONNX_NAME} already present ({human(onnx_path.stat().st_size)})")
        return True

    if not pt_path.is_file():
        print(f"  [FAIL] {YOLO_PT_NAME} not available to export from", file=sys.stderr)
        return False

    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"  [WARN] ultralytics not installed - cannot export {YOLO_ONNX_NAME}")
        print(EXPORT_HINT)
        return False

    print(f"  [conv] exporting {YOLO_PT_NAME} -> {YOLO_ONNX_NAME} (this takes a minute)")
    try:
        produced = YOLO(str(pt_path)).export(format="onnx", opset=12)
    except Exception as exc:  # noqa: BLE001 - surface the real reason, keep going
        print(f"  [FAIL] ONNX export failed: {exc}", file=sys.stderr)
        print(EXPORT_HINT, file=sys.stderr)
        return False

    produced_path = Path(produced)
    if produced_path.resolve() != onnx_path.resolve():
        shutil.move(str(produced_path), str(onnx_path))

    print(f"  [ ok ] {YOLO_ONNX_NAME} ({human(onnx_path.stat().st_size)})")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download SafeStep detector weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", choices=sorted(MODELS) + ["all"], default="all", help="Which model to fetch",
    )
    parser.add_argument("--dest", type=Path, default=MODEL_DIR, help="Destination directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--list", action="store_true", help="List required files and exit")
    args = parser.parse_args(argv)

    selected = sorted(MODELS) if args.model == "all" else [args.model]

    if args.list:
        for name in selected:
            print(f"{name}:")
            for asset in MODELS[name]:
                print(f"  {asset.filename:38s} {asset.note}")
            if name == "yolo":
                print(f"  {YOLO_ONNX_NAME:38s} exported locally from the checkpoint above")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Fetching model weights into {args.dest}\n")

    failures = 0
    for name in selected:
        print(f"{name}:")
        for asset in MODELS[name]:
            if not download(asset, args.dest, force=args.force):
                failures += 1
        if name == "yolo" and not export_yolo_onnx(args.dest, force=args.force):
            failures += 1
        print()

    if failures:
        print(
            f"{failures} item(s) could not be prepared. See the messages above.\n"
            f"SafeStep still runs with:  safestep --detector color\n"
            f"but that fallback detects coloured regions, not obstacles.",
            file=sys.stderr,
        )
        return 1

    print("All requested model files are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
