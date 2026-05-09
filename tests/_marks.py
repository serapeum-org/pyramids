"""Optional-dependency probes, skipif decorators, and marker mapping.

One source of truth for every ``[project.optional-dependencies]`` group.

Two consumers:

* Inline ``@requires_<extra>`` decorators — kept for test files that
  already guard individual methods with ``pytest.mark.skipif(...)``.
* ``pytest_collection_modifyitems`` in :mod:`tests.conftest` — reads
  :data:`EXTRA_MARKERS` to auto-apply a skip when a test is tagged with
  ``@pytest.mark.<extra>`` but the extra is not installed.

Marker names use underscores (valid Python identifiers required by
``pytest.mark.<name>``), while the PyPI extra names use hyphens. The
mapping below spells both out explicitly.
"""

from __future__ import annotations

import importlib.util

import pytest


def _has(*module_names: str) -> bool:
    """Return True iff every named top-level module is importable.

    Uses :func:`importlib.util.find_spec` so the probe never triggers a
    heavy import — cheap to run during collection. A dotted name whose
    parent package is missing raises :class:`ModuleNotFoundError`
    (rather than returning ``None``), so the guard catches that and
    reports the module as absent.
    """
    found = True
    for name in module_names:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            found = False
            break
    return found


HAS_DASK = _has("dask")
HAS_DASK_ARRAY = _has("dask.array")
HAS_DASK_GEOPANDAS = _has("dask_geopandas")
HAS_ZARR = _has("zarr")
HAS_XARRAY = _has("xarray")
HAS_NETCDF4 = _has("netCDF4")
HAS_H5NETCDF = _has("h5netcdf")
HAS_CFTIME = _has("cftime")
HAS_KERCHUNK = _has("kerchunk")
HAS_H5PY = _has("h5py")
HAS_PYARROW = _has("pyarrow")
HAS_CLEOPATRA = _has("cleopatra")


# Per-extra installed? flag. Values are True when EVERY underlying
# module of that extra is importable — so ``netcdf_lazy`` requires
# kerchunk AND h5py AND the xarray stack AND the lazy stack, matching
# what ``pip install pyramids-gis[netcdf-lazy]`` pulls in.
_HAS_VIZ = HAS_CLEOPATRA
_HAS_LAZY = HAS_DASK_ARRAY and HAS_ZARR
_HAS_XARRAY = HAS_XARRAY
_HAS_NETCDF_LAZY = _HAS_LAZY and HAS_KERCHUNK and HAS_H5PY
_HAS_PARQUET = HAS_PYARROW
_HAS_PARQUET_LAZY = _HAS_LAZY and _HAS_PARQUET and HAS_DASK_GEOPANDAS


requires_plot = pytest.mark.skipif(
    not _HAS_VIZ, reason="pyramids-gis[viz] not installed"
)
requires_lazy = pytest.mark.skipif(
    not _HAS_LAZY, reason="pyramids-gis[lazy] not installed"
)
requires_xarray = pytest.mark.skipif(
    not _HAS_XARRAY, reason="pyramids-gis[xarray] not installed"
)
requires_netcdf_lazy = pytest.mark.skipif(
    not _HAS_NETCDF_LAZY, reason="pyramids-gis[netcdf-lazy] not installed"
)
requires_parquet = pytest.mark.skipif(
    not _HAS_PARQUET, reason="pyramids-gis[parquet] not installed"
)
requires_parquet_lazy = pytest.mark.skipif(
    not _HAS_PARQUET_LAZY, reason="pyramids-gis[parquet-lazy] not installed"
)


# Legacy aliases for existing callsites that import the older names.
# Remove in a follow-up once the open call-sites migrate.
requires_dask = requires_lazy
requires_dask_geopandas = requires_parquet_lazy
requires_kerchunk = pytest.mark.skipif(
    not HAS_KERCHUNK, reason="kerchunk not installed"
)
requires_pyarrow = requires_parquet


# Marker-name → skipif-decorator mapping consumed by the collection hook
# in ``tests/conftest.py``. A test annotated with ``@pytest.mark.<name>``
# auto-gains the matching skip when its extra is not installed.
EXTRA_MARKERS: dict[str, pytest.MarkDecorator] = {
    # ``plot`` is the canonical gate for the [viz] extra; no separate
    # ``viz`` marker is registered to avoid duplicate surface.
    "plot": requires_plot,
    "lazy": requires_lazy,
    "xarray": requires_xarray,
    "netcdf_lazy": requires_netcdf_lazy,
    "parquet": requires_parquet,
    "parquet_lazy": requires_parquet_lazy,
}
