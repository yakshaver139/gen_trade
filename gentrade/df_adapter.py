"""Adapter that lets the GA work on either pandas or dask DataFrames at runtime.

Use ``PARALLEL=1`` to opt in to dask. Default is pandas. Dask is an optional
dependency (``pip install gentrade[parallel]``) so importing this module never
requires dask to be installed.
"""

import os
from typing import Any

import pandas as pd


def _use_dask() -> bool:
    return os.getenv("PARALLEL", "").lower() in ("1", "true", "yes")


class DFAdapter:
    def __init__(self, parallel: bool | None = None) -> None:
        if parallel is None:
            parallel = _use_dask()
        if parallel:
            import dask.dataframe as dd

            self.accessor: Any = dd
        else:
            self.accessor = pd

    def __getattribute__(self, attr: str) -> Any:
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            return getattr(self.__dict__["accessor"], attr)


dfa = DFAdapter()
