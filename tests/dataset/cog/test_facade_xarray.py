"""xarray-input tests for pyramids.dataset.cog.facade (the write_cog facade).

These exercise the ``xr.DataArray`` input contract of ``write_cog`` /
``_dataarray_to_dataset`` / ``_normalize_to_dataset``. They live in their own
``xarray``-marked module (rather than alongside the ``core`` facade tests) so the
extras-free "pure wheel" core suite never collects them — they run only in the
xarray CI job.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyramids.dataset.cog.facade import (
    _dataarray_to_dataset,
    _normalize_to_dataset,
)

pytestmark = pytest.mark.xarray


class _Coord:
    """Minimal stand-in for an xarray coordinate (exposes `.values`)."""

    def __init__(self, values):
        self.values = np.asarray(values)


class _FakeDataArray:
    """Duck-typed DataArray for exercising _dataarray_to_dataset branches."""

    def __init__(self, values, x, y, attrs=None, rio=None):
        self.values = np.asarray(values)
        self._coords = {"longitude": _Coord(x), "latitude": _Coord(y)}
        self.coords = self._coords
        self.attrs = attrs or {}
        if rio is not None:
            self.rio = rio

    def __getitem__(self, key):
        return self._coords[key]


class TestDataArrayToDataset:
    """Tests for _dataarray_to_dataset (xarray paths)."""

    def _make_dataarray(self, *, attrs=None):
        """Build a small real xarray.DataArray with lon/lat coords.

        Args:
            attrs: Optional attribute dict to attach (e.g. a `crs` key).

        Returns:
            A `(4, 5)` float32 DataArray on a regular lon/lat grid.
        """
        xr = pytest.importorskip("xarray")
        data = np.arange(20, dtype="float32").reshape(4, 5)
        lat = np.array([12.0, 11.0, 10.0, 9.0])
        lon = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
        return xr.DataArray(
            data,
            coords={"latitude": lat, "longitude": lon},
            dims=("latitude", "longitude"),
            attrs=attrs or {},
        )

    def test_explicit_crs(self):
        """An explicit `crs` builds a Dataset from coordinate geometry.

        Test scenario:
            A lon/lat DataArray with `crs=4326` yields a 4x5 Dataset.
        """
        da = self._make_dataarray()
        ds = _dataarray_to_dataset(da, 4326, None)
        assert (ds.rows, ds.columns) == (4, 5), f"Unexpected shape {ds.shape}"
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_crs_from_attrs(self):
        """CRS is read from `da.attrs['crs']` when not passed explicitly.

        Test scenario:
            `attrs={'crs': 4326}` is honored.
        """
        da = self._make_dataarray(attrs={"crs": 4326})
        ds = _dataarray_to_dataset(da, None, None)
        assert ds.epsg == 4326, f"CRS from attrs not honored, got {ds.epsg}"

    def test_crs_from_rio_accessor(self):
        """CRS falls back to a `.rio.crs` accessor when present.

        Test scenario:
            A duck-typed DataArray exposing `rio.crs` is honored even
            though rioxarray is not a dependency.
        """
        fake = _FakeDataArray(
            np.zeros((2, 2), dtype="float32"),
            x=[30.0, 31.0],
            y=[11.0, 10.0],
            rio=SimpleNamespace(crs=4326),
        )
        ds = _dataarray_to_dataset(fake, None, None)
        assert ds.epsg == 4326, f"CRS from rio accessor not honored, got {ds.epsg}"

    def test_missing_coords_raises(self):
        """A DataArray lacking spatial coords raises ValueError.

        Test scenario:
            Coordinates named neither x/y nor lon/lat are rejected.
        """
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            np.zeros((2, 2), dtype="float32"),
            coords={"a": [0, 1], "b": [0, 1]},
            dims=("a", "b"),
        )
        with pytest.raises(ValueError, match="coordinates"):
            _dataarray_to_dataset(da, 4326, None)

    def test_single_cell_axis_raises(self):
        """Spatial coords with fewer than 2 cells raise ValueError.

        Test scenario:
            A 1x1 grid cannot yield a cell size and is rejected.
        """
        fake = _FakeDataArray(
            np.zeros((1, 1), dtype="float32"), x=[30.0], y=[10.0], rio=None
        )
        with pytest.raises(ValueError, match="at least 2 cells"):
            _dataarray_to_dataset(fake, 4326, None)

    def test_missing_crs_raises(self):
        """No explicit/embedded CRS raises ValueError.

        Test scenario:
            A DataArray with no crs attr and no rio accessor is rejected.
        """
        da = self._make_dataarray()
        with pytest.raises(ValueError, match="CRS"):
            _dataarray_to_dataset(da, None, None)


class TestNormalizeToDatasetDataArray:
    """The DataArray dispatch branch of _normalize_to_dataset."""

    def test_dataarray_dispatch(self):
        """A real xarray.DataArray is dispatched by its type name.

        Test scenario:
            An `xr.DataArray` routes to `_dataarray_to_dataset` and
            yields a Dataset of matching shape.
        """
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            np.arange(20, dtype="float32").reshape(4, 5),
            coords={
                "latitude": np.array([12.0, 11.0, 10.0, 9.0]),
                "longitude": np.array([30.0, 31.0, 32.0, 33.0, 34.0]),
            },
            dims=("latitude", "longitude"),
        )
        result = _normalize_to_dataset(da, 4326, None, None)
        assert (result.rows, result.columns) == (
            4,
            5,
        ), f"Unexpected shape {result.shape}"
