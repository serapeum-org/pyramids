"""Public container/variable NetCDF types (API-1, issue #614).

The concrete classes live in :mod:`pyramids.netcdf.netcdf` (where they are constructed,
avoiding an import cycle); this module re-exports them under the stable, documented import
path. Use :class:`Container` for a multidimensional store and :class:`Variable`
for a single extracted variable.
"""

from __future__ import annotations

from pyramids.netcdf.netcdf import Container, Variable

__all__ = ["Container", "Variable"]
