"""Tests for scripts.assign_signal_modes helpers."""

from __future__ import annotations

import pytest

from scripts.assign_signal_modes import _safe_float


class TestSafeFloat:
    """Unit tests for _safe_float."""

    def test_pos_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert _safe_float(float("-inf")) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_valid_string_float(self):
        assert _safe_float("1.5") == pytest.approx(1.5)

    def test_valid_int(self):
        assert _safe_float(2) == pytest.approx(2.0)

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_non_numeric_string_returns_none(self):
        assert _safe_float("abc") is None

    def test_empty_string_returns_none(self):
        assert _safe_float("") is None
