"""Tests for the UI's NaN-safe formatter + small render helpers."""
from __future__ import annotations

from gentrade.ui.format import fmt, link, render_html_table, safe_text


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


# ---------------------------------------------------------------------------
# safe_text + link + render_html_table
# ---------------------------------------------------------------------------

def test_safe_text_escapes_html():
    assert safe_text("<script>alert('x')</script>") == (
        "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;"
    )


def test_safe_text_handles_none():
    assert safe_text(None) == ""


def test_link_default_target_is_self():
    cell = link("/Run_detail?run_id=abc", "→ open")
    assert 'target="_self"' in cell["html"]
    assert "/Run_detail?run_id=abc" in cell["html"]
    assert "→ open" in cell["html"]


def test_link_external_target_blank_when_requested():
    cell = link("https://x", "x", same_tab=False)
    assert 'target="_blank"' in cell["html"]


def test_link_escapes_text_and_url():
    cell = link("/x?q=<script>", "<b>bad</b>")
    assert "<script>" not in cell["html"]
    assert "&lt;script&gt;" in cell["html"]


def test_render_html_table_basic_structure():
    columns = [
        {"key": "id", "label": "ID"},
        {"key": "open", "label": "Open"},
    ]
    rows = [{"id": "abc", "open": link("/x", "→")}]
    html = render_html_table(columns, rows)
    assert "<table" in html
    assert "<thead>" in html
    assert "<tbody>" in html
    assert "abc" in html
    assert 'target="_self"' in html


def test_render_html_table_supports_per_row_style():
    columns = [{"key": "outcome", "label": "Outcome"}]
    rows = [
        {"outcome": "WIN", "_row_style": "background-color: rgba(46,160,67,0.18);"},
        {"outcome": "LOSS", "_row_style": "background-color: rgba(214,39,40,0.18);"},
    ]
    html = render_html_table(columns, rows)
    assert "rgba(46,160,67,0.18)" in html
    assert "rgba(214,39,40,0.18)" in html
