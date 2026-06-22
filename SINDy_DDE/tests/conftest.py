"""Pytest configuration: put the local ``src/`` package directory on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
# src/ enables ``import pri2m...``; repo root enables ``from tests._phase3_synth import ...``.
for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
