"""Tests for the run manifest (Phase 1).

The manifest is the reproducibility record: the fields persisted alongside
a Run so a historical run can be replayed byte-for-byte. This module pins
which fields are captured automatically and which the caller must supply.
"""
from __future__ import annotations

import json
import subprocess

import pandas as pd
import pytest

from gentrade.manifest import (
    Manifest,
    capture_manifest,
    compute_data_hash,
    current_git_sha,
)


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


# ---------------------------------------------------------------------------
# code_sha
# ---------------------------------------------------------------------------

def _init_clean_repo(tmp_path):
    """Init a temporary git repo with one committed file."""
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    (tmp_path / "README").write_text("hello")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path, check=True, capture_output=True,
    )


def test_current_git_sha_returns_head_on_clean_tree(tmp_path):
    _init_clean_repo(tmp_path)
    sha = current_git_sha(repo_root=tmp_path)
    assert len(sha) == 40
    # idempotent
    assert current_git_sha(repo_root=tmp_path) == sha


def test_current_git_sha_refuses_dirty_tree(tmp_path):
    _init_clean_repo(tmp_path)
    (tmp_path / "README").write_text("dirty")  # uncommitted change

    with pytest.raises(RuntimeError, match="dirty"):
        current_git_sha(repo_root=tmp_path)


def test_current_git_sha_allows_dirty_when_explicit(tmp_path):
    _init_clean_repo(tmp_path)
    (tmp_path / "README").write_text("dirty")

    sha = current_git_sha(repo_root=tmp_path, allow_dirty=True)
    assert len(sha) == 40


def test_current_git_sha_raises_outside_a_repo(tmp_path):
    with pytest.raises(RuntimeError):
        current_git_sha(repo_root=tmp_path)


# ---------------------------------------------------------------------------
# data_hash
# ---------------------------------------------------------------------------

def _bars():
    base = pd.Timestamp("2022-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "open_ts": [base + pd.Timedelta(minutes=15 * i) for i in range(5)],
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [10, 11, 12, 13, 14],
        }
    )


def test_compute_data_hash_is_deterministic():
    a = compute_data_hash(_bars())
    b = compute_data_hash(_bars())
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_compute_data_hash_changes_when_data_changes():
    bars = _bars()
    a = compute_data_hash(bars)
    bars.iloc[0, bars.columns.get_loc("close")] = 999.0
    b = compute_data_hash(bars)
    assert a != b


def test_compute_data_hash_ignores_indicator_columns():
    """Adding a derived indicator column must not invalidate the data hash —
    the data is the OHLCV; indicators are a function of it."""
    bars = _bars()
    a = compute_data_hash(bars)
    bars["momentum_rsi"] = [50, 55, 60, 65, 70]
    b = compute_data_hash(bars)
    assert a == b
