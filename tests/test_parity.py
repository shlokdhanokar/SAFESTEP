"""Guard the Python/JavaScript parity harness from the pytest side.

The browser demo re-implements SafeStep's spatial reasoning, tracking, alert
policy and haptics in JavaScript. `scripts/check_parity.mjs` proves the two
agree; these tests make sure that proof stays runnable and current, so a change
to the Python source cannot quietly leave the published demo behaving
differently from the device it advertises.

The Node-dependent tests skip when Node is absent, so the suite still runs on a
machine with only Python installed. CI has Node and runs them for real.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "parity.json"
TABLES = ROOT / "docs" / "js" / "tables.mjs"
CHECKER = ROOT / "scripts" / "check_parity.mjs"

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="Node.js not installed")


class TestHarnessIsPresent:
    def test_fixture_exists(self):
        assert FIXTURE.is_file(), "run scripts/generate_parity_fixtures.py"

    def test_generated_tables_exist(self):
        assert TABLES.is_file(), "run scripts/generate_tables.py"

    def test_checker_exists(self):
        assert CHECKER.is_file()

    def test_fixture_covers_every_section(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for section in ("geometry", "spatial", "zones", "tracking", "alerts",
                        "describe", "haptics", "strongest", "decode"):
            assert payload[section], f"{section} fixture is empty"


class TestGeneratedFilesAreCurrent:
    """A stale generated file means the demo ships different numbers."""

    def test_tables_match_the_python_source(self):
        from safestep.detection.yolo_onnx import COCO_CLASSES
        from safestep.spatial import KNOWN_HEIGHTS_M, KNOWN_WIDTHS_M

        text = TABLES.read_text(encoding="utf-8")
        for label in KNOWN_HEIGHTS_M:
            assert json.dumps(label) in text, f"{label} missing from generated tables"
        for label in KNOWN_WIDTHS_M:
            assert json.dumps(label) in text, f"{label} missing from generated tables"
        for name in COCO_CLASSES:
            assert json.dumps(name) in text, f"{name} missing from generated classes"

    def test_fixture_tables_match_the_python_source(self):
        from safestep.spatial import KNOWN_HEIGHTS_M, KNOWN_WIDTHS_M

        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        assert payload["tables"]["heights"] == KNOWN_HEIGHTS_M
        assert payload["tables"]["widths"] == KNOWN_WIDTHS_M


@needs_node
class TestParity:
    def test_javascript_agrees_with_python(self):
        result = subprocess.run(
            [node, str(CHECKER)], cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "parity OK" in result.stdout

    def test_harness_reports_a_meaningful_number_of_assertions(self):
        # Guards against the checker silently degrading into a no-op.
        result = subprocess.run(
            [node, str(CHECKER)], cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        count = int(result.stdout.split("parity OK: ")[1].split()[0])
        assert count > 1000, f"only {count} assertions ran"


@needs_node
class TestDemoAssets:
    def test_every_demo_module_parses(self):
        # Catches a syntax error before it reaches the published page.
        for module in sorted((ROOT / "docs" / "js").glob("*.mjs")):
            result = subprocess.run(
                [node, "--check", str(module)], capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, f"{module.name}: {result.stderr}"

    def test_demo_page_references_only_local_assets(self):
        # The demo must work offline after first load, and must not leak the
        # visitor's presence to a third party while they use their camera.
        html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        body = html.split("<body>", 1)[1]
        for marker in ("//cdn", "src=\"http", "href=\"http://", "@import url(http"):
            assert marker not in body, f"external reference {marker!r} in demo page"


class TestDemoAssetsPresent:
    def test_model_is_committed_for_pages(self):
        model = ROOT / "docs" / "models" / "yolov8n.onnx"
        assert model.is_file(), "docs/models/yolov8n.onnx is required for the demo"
        assert model.stat().st_size > 1_000_000

    def test_runtime_is_vendored(self):
        vendor = ROOT / "docs" / "vendor" / "ort"
        assert (vendor / "ort.webgpu.min.mjs").is_file()
        assert (vendor / "ort-wasm-simd-threaded.jsep.wasm").is_file()


@pytest.mark.skipif(sys.platform == "emscripten", reason="not applicable")
class TestNoDuplicatedConstants:
    """The tables are generated, not hand-copied, so they cannot drift."""

    def test_tables_module_is_marked_generated(self):
        text = TABLES.read_text(encoding="utf-8")
        assert "GENERATED" in text
        assert "generate_tables.py" in text
