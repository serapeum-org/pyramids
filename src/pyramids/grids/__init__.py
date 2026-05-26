"""Adapters that turn exotic model grids into regular-grid :class:`~pyramids.dataset.Dataset`.

Weather/ocean models emit fields on grids that are not row-major rasters: ORCA
curvilinear ocean grids, octahedral reduced-Gaussian grids, HEALPix sphere
pixelizations, and others. pyramids already has two regridding bridges — a mesh path
(:func:`pyramids.netcdf.ugrid.interpolation.mesh_to_grid`) and a scattered-point path
(:func:`pyramids.dataset.ops.interpolate.grid_points`). The adapters here turn each
exotic grid into a mesh or point set and reuse those bridges; they do not implement a
new regridder.

- :func:`from_orca` — curvilinear ``(ny, nx)`` lon/lat → UGRID quad mesh → raster.
- :func:`from_octahedral` — ragged per-point lat/lon → scattered points → raster.
- :func:`from_healpix` — HEALPix pixels → raster (deferred pending the ``healpy``
  optional dependency; currently raises :class:`NotImplementedError`).
"""

from __future__ import annotations

from pyramids.grids.healpix import from_healpix
from pyramids.grids.octahedral import from_octahedral
from pyramids.grids.orca import from_orca

__all__ = [
    "from_healpix",
    "from_octahedral",
    "from_orca",
]
