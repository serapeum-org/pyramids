"""Tests for the exotic-grid adapters in :mod:`pyramids.grids`.

Covers the three adapters that turn non-raster model grids into a regular-grid
:class:`~pyramids.dataset.Dataset`:

* :func:`pyramids.grids.from_orca` — curvilinear ``(ny, nx)`` lon/lat → UGRID quad
  mesh → raster.
* :func:`pyramids.grids.from_octahedral` — ragged per-point lat/lon → scattered points
  → raster.
* :func:`pyramids.grids.from_healpix` — deferred; raises :class:`NotImplementedError`
  until the ``healpy`` dependency is approved.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset.dataset import Dataset
from pyramids.grids import from_healpix, from_octahedral, from_orca


@pytest.fixture(scope="function")
def orca_grid():
    """Build a small synthetic ORCA-style curvilinear grid.

    Returns:
        tuple: ``(lon2d, lat2d, data2d)`` each a ``(4, 5)`` float array. ``lon2d`` and
        ``lat2d`` are a regular lon/lat mesh; ``data2d`` is a ramp of distinct values.
    """
    ny, nx = 4, 5
    lon2d = np.tile(np.arange(nx, dtype=float), (ny, 1))
    lat2d = np.tile(np.arange(ny, dtype=float)[:, None], (1, nx))
    data2d = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    return lon2d, lat2d, data2d


@pytest.fixture(scope="function")
def octahedral_points():
    """Build a small synthetic octahedral reduced-Gaussian point set.

    Returns:
        tuple: ``(lats, lons, values)`` 1-D float arrays of equal length describing
        four corner points of a 5x5 degree box with distinct values.
    """
    lats = np.array([0.0, 0.0, 5.0, 5.0])
    lons = np.array([0.0, 5.0, 0.0, 5.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    return lats, lons, values


class TestFromOrca:
    """Tests for :func:`pyramids.grids.from_orca`."""

    def test_returns_single_band_dataset(self, orca_grid):
        """from_orca returns a single-band regular-grid Dataset.

        Test scenario:
            A 4x5 curvilinear grid regridded at cell_size=0.5 yields a
            :class:`Dataset` with one band and a positive row/column count.
        """
        lon2d, lat2d, data2d = orca_grid
        ds = from_orca(lon2d, lat2d, data2d, cell_size=0.5)
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"
        assert ds.rows > 0 and ds.columns > 0, f"Empty grid: {ds.rows}x{ds.columns}"

    def test_crs_is_propagated(self, orca_grid):
        """from_orca stamps the requested EPSG on the output.

        Test scenario:
            Passing epsg=4326 (default) produces a Dataset reporting EPSG 4326.
        """
        lon2d, lat2d, data2d = orca_grid
        ds = from_orca(lon2d, lat2d, data2d, cell_size=0.5, epsg=4326)
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_extent_covers_node_bounds(self, orca_grid):
        """from_orca output extent spans the input coordinate range.

        Test scenario:
            The output bounding box should fall within the node coordinate
            min/max envelope (the mesh cannot extend past its own nodes).
        """
        lon2d, lat2d, data2d = orca_grid
        ds = from_orca(lon2d, lat2d, data2d, cell_size=0.5)
        xmin, ymin, xmax, ymax = (float(v) for v in ds.bbox)
        assert xmin >= lon2d.min() - 1e-6, f"xmin {xmin} below node min {lon2d.min()}"
        assert xmax <= lon2d.max() + 1e-6, f"xmax {xmax} above node max {lon2d.max()}"
        assert ymin >= lat2d.min() - 1e-6, f"ymin {ymin} below node min {lat2d.min()}"
        assert ymax <= lat2d.max() + 1e-6, f"ymax {ymax} above node max {lat2d.max()}"

    def test_values_within_source_range(self, orca_grid):
        """from_orca interpolated values stay within the source value range.

        Test scenario:
            Nearest-neighbour regridding cannot invent values outside the input
            field; valid (non-nodata) cells lie within ``[min, max]`` of the faces.
        """
        lon2d, lat2d, data2d = orca_grid
        nodata = -9999.0
        ds = from_orca(lon2d, lat2d, data2d, cell_size=0.5, nodata=nodata)
        arr = ds.read_array()
        valid = arr[arr != nodata]
        face_values = data2d[:-1, :-1]
        assert valid.size > 0, "No valid cells produced by from_orca."
        assert valid.min() >= face_values.min() - 1e-6, "Value below source min."
        assert valid.max() <= face_values.max() + 1e-6, "Value above source max."

    @pytest.mark.parametrize(
        "lon_shape, lat_shape, data_shape",
        [
            ((4, 5), (4, 4), (4, 5)),
            ((4, 5), (4, 5), (3, 5)),
            ((4, 5), (3, 5), (3, 5)),
        ],
    )
    def test_mismatched_shapes_raise(self, lon_shape, lat_shape, data_shape):
        """from_orca rejects coordinate/data arrays of differing shapes.

        Args:
            lon_shape: Shape of the longitude array.
            lat_shape: Shape of the latitude array.
            data_shape: Shape of the data array.

        Test scenario:
            Any shape disagreement between lon2d/lat2d/data2d raises ValueError
            mentioning "same shape".
        """
        lon2d = np.zeros(lon_shape)
        lat2d = np.zeros(lat_shape)
        data2d = np.zeros(data_shape)
        with pytest.raises(ValueError, match="same shape") as exc:
            from_orca(lon2d, lat2d, data2d, cell_size=0.5)
        assert "same shape" in str(exc.value), f"Unexpected message: {exc.value}"

    def test_non_2d_input_raises(self):
        """from_orca rejects 1-D coordinate arrays.

        Test scenario:
            Passing 1-D arrays (all the same shape) raises ValueError mentioning
            "2-D".
        """
        flat = np.arange(6, dtype=float)
        with pytest.raises(ValueError, match="2-D") as exc:
            from_orca(flat, flat, flat, cell_size=0.5)
        assert "2-D" in str(exc.value), f"Unexpected message: {exc.value}"

    @pytest.mark.parametrize("shape", [(1, 5), (4, 1), (1, 1)])
    def test_degenerate_grid_raises(self, shape):
        """from_orca rejects grids too small to form any quad cell.

        Args:
            shape: A 2-D shape with at least one dimension < 2.

        Test scenario:
            A grid with fewer than 2 rows or 2 columns raises ValueError
            mentioning "2 x 2".
        """
        arr = np.zeros(shape)
        with pytest.raises(ValueError, match=r"2 x 2") as exc:
            from_orca(arr, arr, arr, cell_size=0.5)
        assert "2 x 2" in str(exc.value), f"Unexpected message: {exc.value}"


class TestFromOctahedral:
    """Tests for :func:`pyramids.grids.from_octahedral`."""

    def test_returns_single_band_dataset(self, octahedral_points):
        """from_octahedral returns a single-band regular-grid Dataset.

        Test scenario:
            Four corner points regridded at cell_size=1.0 with nearest-neighbour
            yield a Dataset with one band and positive dimensions.
        """
        lats, lons, values = octahedral_points
        ds = from_octahedral(lats, lons, values, cell_size=1.0, algorithm="nearest")
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"
        assert ds.rows > 0 and ds.columns > 0, f"Empty grid: {ds.rows}x{ds.columns}"

    def test_crs_is_propagated(self, octahedral_points):
        """from_octahedral stamps the requested EPSG on the output.

        Test scenario:
            Passing epsg=4326 produces a Dataset reporting EPSG 4326.
        """
        lats, lons, values = octahedral_points
        ds = from_octahedral(lats, lons, values, cell_size=1.0, epsg=4326)
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_grid_resolution_matches_cell_size(self, octahedral_points):
        """from_octahedral honours cell_size for the output resolution.

        Test scenario:
            A 5x5 degree extent at cell_size=1.0 yields a 5x5 grid (rounded extent
            / cell_size).
        """
        lats, lons, values = octahedral_points
        ds = from_octahedral(lats, lons, values, cell_size=1.0, algorithm="nearest")
        assert ds.rows == 5, f"Expected 5 rows, got {ds.rows}"
        assert ds.columns == 5, f"Expected 5 columns, got {ds.columns}"

    def test_nearest_preserves_source_values(self, octahedral_points):
        """from_octahedral nearest-neighbour output equals one of the inputs per cell.

        Test scenario:
            With algorithm="nearest", every output cell value must be one of the
            four source values (no interpolation between them).
        """
        lats, lons, values = octahedral_points
        ds = from_octahedral(lats, lons, values, cell_size=1.0, algorithm="nearest")
        arr = ds.read_array()
        produced = set(np.unique(arr).tolist())
        allowed = set(values.tolist())
        assert produced.issubset(allowed), f"Unexpected values {produced - allowed}"

    def test_accepts_2d_input_by_raveling(self):
        """from_octahedral flattens multi-dimensional inputs of equal size.

        Test scenario:
            2-D lat/lon/value arrays of equal size are ravelled and gridded without
            error.
        """
        lats = np.array([[0.0, 0.0], [5.0, 5.0]])
        lons = np.array([[0.0, 5.0], [0.0, 5.0]])
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = from_octahedral(lats, lons, values, cell_size=1.0, algorithm="nearest")
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"

    @pytest.mark.parametrize(
        "n_lat, n_lon, n_val",
        [(4, 3, 4), (4, 4, 3), (3, 4, 4)],
    )
    def test_unequal_lengths_raise(self, n_lat, n_lon, n_val):
        """from_octahedral rejects lat/lon/value arrays of unequal length.

        Args:
            n_lat: Number of latitude values.
            n_lon: Number of longitude values.
            n_val: Number of data values.

        Test scenario:
            Any length disagreement raises ValueError mentioning "equal length".
        """
        with pytest.raises(ValueError, match="equal length") as exc:
            from_octahedral(
                np.zeros(n_lat), np.zeros(n_lon), np.zeros(n_val), cell_size=1.0
            )
        assert "equal length" in str(exc.value), f"Unexpected message: {exc.value}"


class TestFromHealpix:
    """Tests for the deferred :func:`pyramids.grids.from_healpix`."""

    def test_raises_not_implemented(self):
        """from_healpix raises NotImplementedError while deferred.

        Test scenario:
            Any call raises NotImplementedError whose message points at the
            supported adapters and the pending 'healpy' dependency.
        """
        with pytest.raises(NotImplementedError) as exc:
            from_healpix(np.zeros(12), cell_size=1.0)
        msg = str(exc.value)
        assert "healpy" in msg, f"Message should mention healpy: {msg}"
        assert "from_orca" in msg, f"Message should point at from_orca: {msg}"

    def test_raises_regardless_of_arguments(self):
        """from_healpix raises even when given a full set of valid-looking args.

        Test scenario:
            Supplying nside/nest/method does not change the deferred behaviour.
        """
        with pytest.raises(NotImplementedError):
            from_healpix(
                np.zeros(48), nside=2, nest=True, cell_size=2.0, method="nearest"
            )
