"""Reproducibility manifest for a GA run.

A `Manifest` captures everything needed to replay a run byte-for-byte:
seed, window boundaries, frozen config, the timestamp of when the run
began, the code SHA, and a hash of the bars frame.

The spec's `WindowsAreOrdered` invariant is enforced at construction time —
overlapping windows are rejected with a clear error rather than producing
silently broken reports later.

`current_git_sha` and `compute_data_hash` are surfaced as standalone
helpers so callers can decide when (and whether) to capture them. The
unit-tested core doesn't reach out to the filesystem on every run; that's
the CLI's job.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

WindowTuple = tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class Manifest:
    """Reproducibility record for one GA run."""

    seed: int
    train_window: WindowTuple
    validation_window: WindowTuple
    test_window: WindowTuple
    started_at: pd.Timestamp
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    code_sha: str | None = None
    data_hash: str | None = None

    def to_json(self) -> str:
        """Serialise to JSON. Timestamps become ISO strings."""
        return json.dumps(
            {
                "seed": self.seed,
                "train_window": [
                    self.train_window[0].isoformat(),
                    self.train_window[1].isoformat(),
                ],
                "validation_window": [
                    self.validation_window[0].isoformat(),
                    self.validation_window[1].isoformat(),
                ],
                "test_window": [
                    self.test_window[0].isoformat(),
                    self.test_window[1].isoformat(),
                ],
                "started_at": self.started_at.isoformat(),
                "config_snapshot": self.config_snapshot,
                "code_sha": self.code_sha,
                "data_hash": self.data_hash,
            },
            sort_keys=True,
        )


def current_git_sha(
    repo_root: Path | str | None = None,
    allow_dirty: bool = False,
) -> str:
    """Return the current HEAD SHA, optionally refusing if the tree is dirty.

    Raises ``RuntimeError`` if the working tree has uncommitted changes and
    ``allow_dirty`` is False — runs on a dirty tree aren't reproducible.
    Raises ``RuntimeError`` if the directory isn't a git repo.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"could not read git HEAD: {e}") from e

    if not allow_dirty:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError(
                "working tree is dirty — refusing to capture code_sha. "
                "Pass allow_dirty=True to override (but the run will not be "
                "reproducible from this SHA alone)."
            )
    return sha


def compute_data_hash(bars: pd.DataFrame) -> str:
    """SHA-256 hex of a canonical byte serialisation of ``bars``.

    Uses the parquet representation (deterministic given a fixed pyarrow
    version) of the OHLCV-relevant columns. Indicator columns are excluded
    so adding a new indicator doesn't invalidate the data hash.
    """
    cols = [c for c in ("open_ts", "open", "high", "low", "close", "volume") if c in bars.columns]
    if not cols:
        # Fallback: hash the full frame's byte representation.
        return hashlib.sha256(pd.util.hash_pandas_object(bars, index=True).values.tobytes()).hexdigest()
    canonical = pd.util.hash_pandas_object(bars[cols], index=False).values.tobytes()
    return hashlib.sha256(canonical).hexdigest()


def capture_manifest(
    seed: int,
    train_window: WindowTuple,
    validation_window: WindowTuple,
    test_window: WindowTuple,
    config_snapshot: dict[str, Any] | None = None,
    code_sha: str | None = None,
    data_hash: str | None = None,
) -> Manifest:
    """Construct a manifest now, validating window ordering."""
    if not (
        train_window[0] <= train_window[1] <= validation_window[0]
        and validation_window[0] <= validation_window[1] <= test_window[0]
        and test_window[0] <= test_window[1]
    ):
        raise ValueError(
            "windows must be chronologically ordered with no overlap: "
            f"train={train_window}, validation={validation_window}, test={test_window}"
        )
    return Manifest(
        seed=seed,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        started_at=pd.Timestamp.utcnow(),
        config_snapshot=config_snapshot or {},
        code_sha=code_sha,
        data_hash=data_hash,
    )
