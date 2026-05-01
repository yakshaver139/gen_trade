"""Tests for the UI's NaN-safe formatter."""
from __future__ import annotations

from gentrade.ui.format import fmt


def test_fmt_normal_float():
    assert fmt(0.123, "+.2f") == "+0.12"


def test_fmt_negative_float():
    assert fmt(-0.5, "+.4f") == "-0.5000"


def test_fmt_int():
    assert fmt(7, "d") == "7"


def test_fmt_none_returns_default():
    assert fmt(None) == "—"
    assert fmt(None, default="N/A") == "N/A"


def test_fmt_nan_returns_default():
    assert fmt(float("nan")) == "—"


def test_fmt_inf_returns_default():
    assert fmt(float("inf")) == "—"
    assert fmt(float("-inf")) == "—"


def test_fmt_invalid_format_falls_back():
    """A format spec that doesn't match the value type → default."""
    assert fmt("not a number", "+.2f") == "—"


def test_fmt_zero_is_finite_not_default():
    assert fmt(0.0, "+.2f") == "+0.00"


def test_fmt_percent_spec():
    assert fmt(0.5, ".0%") == "50%"


def test_fmt_does_not_swallow_normal_strings_by_default():
    # When the format spec is empty / 's', plain strings format fine.
    assert fmt("hello", "s") == "hello"
