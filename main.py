#!/usr/bin/env python3
"""SafeStep entry point.

Kept at the repository root so the documented ``python main.py`` still works.
All logic lives in the ``safestep`` package under ``src/``; this shim only makes
that package importable when running from a source checkout without installing.

Prefer the installed console script once you have run ``pip install -e .``::

    safestep --help
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safestep.cli import main  # noqa: E402  (path setup must precede the import)

if __name__ == "__main__":
    sys.exit(main())
