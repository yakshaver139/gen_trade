"""Formatting helpers shared across Streamlit pages.

The API surface returns JSON over the wire; Pydantic v2 maps non-finite
floats (NaN, +Inf, -Inf) to ``null`` by default. That means a ``sharpe``
field can come back as ``None`` even though the schema declares ``float``,
which crashes naive ``f"{x:+.2f}"`` formatting in the page code.

These helpers handle None / NaN / Inf uniformly so every metric renders
as either a formatted number or the placeholder string.
"""
from __future__ import annotations

import math
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
