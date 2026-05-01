"""Tests for the run manifest (Phase 1).

The manifest is the reproducibility record: the fields persisted alongside
a Run so a historical run can be replayed byte-for-byte. This module pins
which fields are captured automatically and which the caller must supply.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from gentrade.manifest import Manifest, capture_manifest


def _windows():
    base = pd.Timestamp("2022-01-01", tz="UTC")
    return (
        (base, base + pd.Timedelta(days=30)),
        (base + pd.Timedelta(days=30), base + pd.Timedelta(days=60)),
        (base + pd.Timedelta(days=60), base + pd.Timedelta(days=90)),
    )


def test_capture_manifest_records_required_fields():
    train, val, test = _windows()
    config = {"target_pct": 0.015, "stop_loss_pct": 0.0075}

    m = capture_manifest(
        seed=42,
        train_window=train,
        validation_window=val,
        test_window=test,
        config_snapshot=config,
    )

    assert isinstance(m, Manifest)
    assert m.seed == 42
    assert m.train_window == train
    assert m.validation_window == val
    assert m.test_window == test
    assert m.config_snapshot == config
    assert m.started_at is not None


def test_capture_manifest_records_started_at_within_bounds():
    train, val, test = _windows()
    before = pd.Timestamp.utcnow()
    m = capture_manifest(seed=0, train_window=train, validation_window=val, test_window=test)
    after = pd.Timestamp.utcnow()

    assert before <= m.started_at <= after


def test_manifest_to_json_round_trips():
    train, val, test = _windows()
    config = {"target_pct": 0.015, "fee_bps": 10.0}

    m = capture_manifest(
        seed=7,
        train_window=train,
        validation_window=val,
        test_window=test,
        config_snapshot=config,
    )

    payload = m.to_json()
    parsed = json.loads(payload)

    assert parsed["seed"] == 7
    assert parsed["config_snapshot"] == config
    # windows survive as ISO strings
    assert parsed["train_window"][0] == train[0].isoformat()
    assert parsed["train_window"][1] == train[1].isoformat()


def test_manifest_rejects_overlapping_windows():
    base = pd.Timestamp("2022-01-01", tz="UTC")
    train = (base, base + pd.Timedelta(days=30))
    # validation starts BEFORE train ends — illegal per the spec invariant
    val = (base + pd.Timedelta(days=15), base + pd.Timedelta(days=45))
    test = (base + pd.Timedelta(days=45), base + pd.Timedelta(days=60))

    with pytest.raises(ValueError, match="windows must be chronologically ordered"):
        capture_manifest(
            seed=0, train_window=train, validation_window=val, test_window=test
        )


def test_manifest_default_config_snapshot_is_empty_dict():
    """Caller may omit the config snapshot if the run has no extra knobs."""
    train, val, test = _windows()
    m = capture_manifest(
        seed=0, train_window=train, validation_window=val, test_window=test
    )
    assert m.config_snapshot == {}
