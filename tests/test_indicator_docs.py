"""Tests for the indicator docs lookup.

Confirms every catalogue indicator has a doc (no silent fallbacks for
known indicators), the fallback fires only for genuinely unknown names,
and the doc shape stays consistent.
"""
from __future__ import annotations

from gentrade.ui.indicator_docs import INDICATOR_DOCS, get_doc


def test_known_indicator_has_full_doc():
    doc = get_doc("momentum_rsi")
    assert "description" in doc and len(doc["description"]) > 0
    assert "formula" in doc and len(doc["formula"]) > 0
    assert "example" in doc and len(doc["example"]) > 0


def test_unknown_indicator_falls_back_to_generic():
    doc = get_doc("definitely_not_real")
    assert "description" in doc
    assert "definitely_not_real" in doc["description"]
    assert "ta" in doc["description"].lower()


def test_every_catalogue_indicator_has_a_doc():
    """If the catalogue gains an indicator, the docs file must keep up."""
    from gentrade.generate_strategy import LOADED_INDICATORS

    catalogue = {s["indicator"] for s in LOADED_INDICATORS}
    missing = [i for i in catalogue if i not in INDICATOR_DOCS]
    assert not missing, (
        f"INDICATOR_DOCS missing entries for {len(missing)} catalogue "
        f"indicators: {sorted(missing)}"
    )


def test_every_doc_has_description_formula_and_example():
    for indicator, doc in INDICATOR_DOCS.items():
        assert doc.get("description"), f"{indicator} has empty description"
        assert doc.get("formula"), f"{indicator} has empty formula"
        assert doc.get("example"), f"{indicator} has empty example"
