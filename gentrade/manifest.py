"""Reproducibility manifest for a GA run.

A `Manifest` captures everything needed to replay a run byte-for-byte:
seed, window boundaries, frozen config, and the timestamp of when the run
began. Code SHA and data hash are deferred to Phase 2 (persistence layer)
because they require git/file-system access that the unit-tested core
shouldn't depend on.

The spec's `WindowsAreOrdered` invariant is enforced at construction time —
overlapping windows are rejected with a clear error rather than producing
silently broken reports later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
