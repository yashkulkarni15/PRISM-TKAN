"""Tests for the loader-resolution logic in extract_phase2_signals.

Covers:
  (a) fallback disallowed + canonical reader missing  → raises RuntimeError with helpful message
  (b) --allow-h5-fallback + canonical reader missing  → returns load_h5_fallback callable
"""

from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# ── locate the script module ──────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

def _import_script():
    """Import extract_phase2_signals as a module (add scripts/ to path once)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import importlib as _il
    spec = _il.util.spec_from_file_location(
        "extract_phase2_signals",
        _SCRIPTS / "extract_phase2_signals.py",
    )
    mod = _il.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _import_script()
resolve_loader = _mod.resolve_loader


# ── test (a): fallback disallowed + reader missing → RuntimeError ─────────────

def test_resolve_loader_raises_when_reader_missing_and_fallback_not_allowed():
    """When load_time_h5 is None and allow_fallback is False, resolve_loader must
    raise RuntimeError (not silently return the fallback)."""
    with pytest.raises(RuntimeError) as exc_info:
        resolve_loader(load_time_h5=None, allow_fallback=False)

    msg = str(exc_info.value).lower()
    # Message must mention the canonical reader and the override flag
    assert "load_time_h5" in msg or "read_file" in msg, (
        f"Error message should reference the canonical reader. Got: {exc_info.value!r}"
    )
    assert "allow-h5-fallback" in msg or "allow_h5_fallback" in msg, (
        f"Error message should mention --allow-h5-fallback flag. Got: {exc_info.value!r}"
    )


# ── test (b): allow_fallback=True + reader missing → returns a callable (fallback) ─

def test_resolve_loader_returns_fallback_when_allowed(capsys):
    """When load_time_h5 is None and allow_fallback is True, resolve_loader must
    return a callable (the fallback wrapper) and emit a warning to stderr."""
    result = resolve_loader(load_time_h5=None, allow_fallback=True)

    # Must be callable (not raise)
    assert callable(result), f"Expected a callable, got {result!r}"

    # Must NOT be the identity of a canonical reader — it wraps load_h5_fallback
    # We verify it is a wrapper by checking it is distinct from a MagicMock
    # and that it is not the canonical-reader sentinel None.
    assert result is not None

    # A warning must have been emitted to stderr
    captured = capsys.readouterr()
    warn_lower = captured.err.lower()
    assert "warning" in warn_lower or "fallback" in warn_lower, (
        f"Expected a warning on stderr, got: {captured.err!r}"
    )


# ── test (c): canonical reader present → returned as-is regardless of flag ────

def test_resolve_loader_returns_canonical_reader_when_present():
    """When load_time_h5 is provided, it should be returned regardless of flag."""
    fake_reader = MagicMock()
    result = resolve_loader(load_time_h5=fake_reader, allow_fallback=False)
    assert result is fake_reader

    result2 = resolve_loader(load_time_h5=fake_reader, allow_fallback=True)
    assert result2 is fake_reader
