"""Formatting + small rendering helpers shared across Streamlit pages.

The API surface returns JSON over the wire; Pydantic v2 maps non-finite
floats (NaN, +Inf, -Inf) to ``null`` by default. That means a ``sharpe``
field can come back as ``None`` even though the schema declares ``float``,
which crashes naive ``f"{x:+.2f}"`` formatting in the page code.

These helpers handle None / NaN / Inf uniformly. The HTML helpers exist
because Streamlit's ``LinkColumn`` opens links in a new tab with no API
to override the target; rendering link-heavy tables as HTML lets us
control the anchor target directly.
"""
from __future__ import annotations

import html as _html
import math
from collections.abc import Iterable, Mapping
from typing import Any


def fmt(value: Any, spec: str = "+.4f", default: str = "—") -> str:
    """Format ``value`` per ``spec``; return ``default`` for None / NaN / Inf."""
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return default


def safe_text(value: Any) -> str:
    """HTML-escape a value for inclusion in markup."""
    if value is None:
        return ""
    return _html.escape(str(value))


def render_html_table(
    columns: Iterable[Mapping[str, str]],
    rows: Iterable[Mapping[str, Any]],
    *,
    table_class: str = "gentrade-table",
) -> str:
    """Render a sortless HTML table with ``target='_self'`` on every link.

    ``columns`` is an ordered iterable of ``{"key": ..., "label": ...,
    "align": "left|right|center"}``. ``rows`` is an iterable of dicts
    keyed by column key; values may be raw strings (HTML-escaped here)
    or ``{"html": "..."}`` dicts to inject pre-rendered markup (e.g.
    anchors). Each row may also include a ``"_row_style"`` key with a
    CSS string applied to the ``<tr>``.

    Used by the runs / population tables so we can force same-tab
    navigation. ``st.dataframe``'s ``LinkColumn`` opens in ``_blank``
    with no override.
    """
    columns = list(columns)
    rows = list(rows)

    head_cells = "".join(
        f'<th style="text-align:{c.get("align", "left")};'
        'padding:0.4rem 0.75rem;border-bottom:1px solid #444;'
        'font-weight:600;">{label}</th>'.format(label=safe_text(c.get("label", c["key"])))
        for c in columns
    )

    body = []
    for r in rows:
        row_style = r.get("_row_style", "")
        cells = []
        for c in columns:
            v = r.get(c["key"])
            inner = v["html"] if isinstance(v, dict) and "html" in v else safe_text(v)
            align = c.get("align", "left")
            cells.append(
                f'<td style="text-align:{align};padding:0.35rem 0.75rem;'
                f'border-bottom:1px solid #2a2a2a;">{inner}</td>'
            )
        body.append(
            f'<tr style="{row_style}">{"".join(cells)}</tr>'
        )

    return (
        f'<div style="overflow-x:auto;">'
        f'<table class="{table_class}" '
        f'style="border-collapse:collapse;width:100%;font-size:0.92rem;">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        f"</table></div>"
    )


def link(href: str, text: str, *, same_tab: bool = True) -> dict[str, str]:
    """Build an anchor cell for ``render_html_table`` with same-tab default."""
    target = "_self" if same_tab else "_blank"
    return {
        "html": (
            f'<a href="{safe_text(href)}" target="{target}" '
            f'style="color:#4ea3ff;text-decoration:none;">'
            f"{safe_text(text)}</a>"
        )
    }
