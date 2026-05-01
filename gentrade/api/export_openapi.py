"""Print the API's OpenAPI schema as JSON.

Run with: ``uv run python -m gentrade.api.export_openapi > openapi.json``.

The committed openapi.json is what the test in tests/test_api.py asserts
against. Regenerate it whenever you add or change an endpoint, and review
the diff before merging — that's the spec contract for downstream UIs.
"""
from __future__ import annotations

import json
import sys
import tempfile

from gentrade.api.app import create_app
from gentrade.persistence import init_db


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        engine = init_db(f"sqlite:///{tmp}/x.db")
        app = create_app(engine=engine)
        json.dump(app.openapi(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
