"""Asset registry for the API.

The API never accepts user-supplied file paths — clients refer to data
sources by an opaque ``asset`` name (e.g. ``BTCUSDC-15m``) and the server
resolves that to a CSV path via this registry. Asset → path mappings are
loaded from a JSON file at the path given in ``GENTRADE_ASSETS_PATH``;
when unset, the registry is empty and POST /runs returns 400.

This is the boundary that defends against path traversal: the only way
to point the GA at a file is to register it here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetEntry:
    """A registered data source the API is willing to load."""

    asset: str
    exchange: str
    interval: str
    path: str


_REGISTRY: dict[str, AssetEntry] | None = None


def configure_registry(entries: list[AssetEntry] | None) -> None:
    """Override the registry in-process (test hook). Pass ``None`` to reset."""
    global _REGISTRY
    _REGISTRY = None if entries is None else {e.asset: e for e in entries}


def _data_root() -> Path | None:
    """Return the configured data-root, if any. ``None`` disables the boundary check."""
    raw = os.getenv("GENTRADE_DATA_ROOT")
    return Path(raw).resolve() if raw else None


def _load_from_path(path: str) -> dict[str, AssetEntry]:
    """Load + validate the asset registry JSON.

    Each asset's ``path`` is resolved to an absolute path; if
    ``GENTRADE_DATA_ROOT`` is set, the absolute path must lie under it. This
    is the trust boundary that defends against the registry being pointed at
    arbitrary files via a malicious assets JSON (e.g. ``/etc/passwd``).
    """
    raw = json.loads(Path(path).read_text())
    root = _data_root()
    out: dict[str, AssetEntry] = {}
    for e in raw:
        resolved = Path(e["path"]).resolve()
        if root is not None and root not in resolved.parents and resolved != root:
            raise ValueError(
                f"asset {e['asset']!r} path {e['path']!r} escapes "
                f"GENTRADE_DATA_ROOT={root}"
            )
        out[e["asset"]] = AssetEntry(
            asset=e["asset"],
            exchange=e["exchange"],
            interval=e["interval"],
            path=str(resolved),
        )
    return out


def _registry() -> dict[str, AssetEntry]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    path = os.getenv("GENTRADE_ASSETS_PATH")
    _REGISTRY = _load_from_path(path) if path and Path(path).exists() else {}
    return _REGISTRY


def list_assets() -> list[AssetEntry]:
    return list(_registry().values())


def resolve(asset: str) -> AssetEntry | None:
    return _registry().get(asset)
