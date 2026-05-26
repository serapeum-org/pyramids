"""Tests for the exotic-grid adapters in :mod:`pyramids.grids`.

Covers the three adapters that turn non-raster model grids into a regular-grid
:class:`~pyramids.dataset.Dataset`:

* :func:`pyramids.grids.from_orca` — curvilinear ``(ny, nx)`` lon/lat → UGRID quad
  mesh → raster.
* :func:`pyramids.grids.from_octahedral` — ragged per-point lat/lon → scattered points
  → raster.
* :func:`pyramids.grids.from_healpix` — HEALPix pixels (RING or NESTED) → scattered
  points → raster, with the pixel→lon/lat math implemented in plain NumPy (no
  ``healpy`` dependency).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset.dataset import Dataset
from pyramids.grids import from_healpix, from_octahedral, from_orca
from pyramids.grids.healpix import _nest2ring, _ring_pix2lonlat


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
            Face values are corner-node means and the regrid cannot invent values
            outside the input field; valid (non-nodata) cells lie within
            ``[data2d.min, data2d.max]``.
        """
        lon2d, lat2d, data2d = orca_grid
        nodata = -9999.0
        ds = from_orca(lon2d, lat2d, data2d, cell_size=0.5, nodata=nodata)
        arr = ds.read_array()
        valid = arr[arr != nodata]
        assert valid.size > 0, "No valid cells produced by from_orca."
        assert valid.min() >= data2d.min() - 1e-6, "Value below source min."
        assert valid.max() <= data2d.max() + 1e-6, "Value above source max."

    def test_face_value_is_corner_node_mean(self):
        """from_orca uses the mean of a face's four corner nodes (no data dropped).

        Test scenario:
            On a 2x2 grid (a single quad face) the only cell's value equals the mean of
            all four node values, proving the last row/column contribute (the old
            implementation used only the upper-left node).
        """
        lon2d = np.array([[0.0, 1.0], [0.0, 1.0]])
        lat2d = np.array([[1.0, 1.0], [0.0, 0.0]])
        data2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = from_orca(lon2d, lat2d, data2d, cell_size=0.5, method="nearest")
        arr = ds.read_array()
        valid = arr[arr != ds.no_data_value[0]]
        assert valid.size > 0, "single quad face produced no cells"
        np.testing.assert_allclose(
            np.unique(valid),
            [2.5],
            atol=1e-9,
            err_msg="face value should be mean of 1..4",
        )

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

    def test_bbox_pins_output_extent(self, octahedral_points):
        """from_octahedral honours an explicit bbox for the output extent.

        Test scenario:
            Passing bbox=(-180, -90, 180, 90) with cell_size=5 yields a 72x36 grid whose
            bounding box equals the requested global extent, rather than the points'
            (much smaller) bounding box.
        """
        lats, lons, values = octahedral_points
        ds = from_octahedral(
            lats,
            lons,
            values,
            cell_size=5.0,
            algorithm="nearest",
            bbox=(-180.0, -90.0, 180.0, 90.0),
        )
        assert (ds.rows, ds.columns) == (36, 72), f"Got {ds.rows}x{ds.columns}"
        np.testing.assert_allclose(
            list(ds.bbox),
            [-180.0, -90.0, 180.0, 90.0],
            atol=1e-6,
            err_msg="bbox not pinned",
        )


class TestRingPix2Lonlat:
    """Tests for :func:`pyramids.grids.healpix._ring_pix2lonlat`."""

    def test_nside1_matches_canonical_centres(self):
        """RING pix2ang reproduces the canonical HEALPix nside=1 pixel centres.

        Test scenario:
            For nside=1 (12 pixels) the four north-cap pixels sit at lat=arcsin(2/3)
            (≈41.81°) and lon 45/135/225/315; the four equatorial pixels at lat=0 and
            lon 0/90/180/270; the four south-cap pixels at lat=-41.81°.
        """
        lon, lat = _ring_pix2lonlat(1, np.arange(12))
        cap = np.degrees(np.arcsin(2.0 / 3.0))
        np.testing.assert_allclose(
            lon, [45, 135, 225, 315, 90, 180, 270, 0, 45, 135, 225, 315], atol=1e-9
        )
        np.testing.assert_allclose(
            lat,
            [cap, cap, cap, cap, 0, 0, 0, 0, -cap, -cap, -cap, -cap],
            atol=1e-9,
        )

    @pytest.mark.parametrize("nside", [1, 2, 4, 8])
    def test_latitudes_symmetric_and_lon_in_range(self, nside):
        """RING centres are north/south symmetric with longitudes in [0, 360).

        Args:
            nside: HEALPix resolution parameter.

        Test scenario:
            The sorted latitude set is symmetric about the equator and every longitude
            falls in [0, 360).
        """
        npix = 12 * nside * nside
        lon, lat = _ring_pix2lonlat(nside, np.arange(npix))
        assert lon.min() >= 0.0 and lon.max() < 360.0, f"lon out of range for {nside}"
        np.testing.assert_allclose(
            np.sort(lat), -np.sort(-lat)[::-1], atol=1e-9, err_msg="lat not symmetric"
        )


class TestNest2Ring:
    """Tests for :func:`pyramids.grids.healpix._nest2ring`."""

    @pytest.mark.parametrize("nside", [1, 2, 4, 8])
    def test_is_bijection(self, nside):
        """nest2ring is a bijection over the full pixel index range.

        Args:
            nside: HEALPix resolution parameter.

        Test scenario:
            Mapping every nested index 0..npix-1 yields a permutation of the same range
            (no collisions, no out-of-range values).
        """
        npix = 12 * nside * nside
        ring = _nest2ring(nside, np.arange(npix))
        np.testing.assert_array_equal(
            np.sort(ring), np.arange(npix), err_msg=f"not a bijection for nside={nside}"
        )

    def test_known_reference_nside2(self):
        """nest2ring matches the published HEALPix reference for nside=2.

        Test scenario:
            nest2ring(2, 0..11) equals the canonical [13,5,4,0,15,7,6,1,17,9,8,2].
        """
        result = _nest2ring(2, np.arange(12)).tolist()
        assert result == [13, 5, 4, 0, 15, 7, 6, 1, 17, 9, 8, 2], f"Got {result}"

    @pytest.mark.parametrize("nside", [1, 2, 4, 8])
    def test_nested_centre_set_equals_ring(self, nside):
        """NESTED pixel centres are the same set as RING centres for a given nside.

        Args:
            nside: HEALPix resolution parameter.

        Test scenario:
            RING and NESTED index the same physical pixels, so the unordered set of
            centre coordinates must be identical.
        """
        npix = 12 * nside * nside
        idx = np.arange(npix)
        lon_r, lat_r = _ring_pix2lonlat(nside, idx)
        lon_n, lat_n = _ring_pix2lonlat(nside, _nest2ring(nside, idx))
        set_ring = set(zip(np.round(lon_r, 9), np.round(lat_r, 9)))
        set_nest = set(zip(np.round(lon_n, 9), np.round(lat_n, 9)))
        assert set_ring == set_nest, f"centre sets differ for nside={nside}"


class TestFromHealpix:
    """Tests for :func:`pyramids.grids.from_healpix`."""

    def test_ring_returns_single_band_dataset(self):
        """from_healpix regrids a RING field into a single-band Dataset.

        Test scenario:
            An nside=1 RING field (12 pixels) yields a Dataset with one band, positive
            dimensions, and the requested CRS.
        """
        ds = from_healpix(np.arange(12.0), cell_size=30.0)
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"
        assert ds.rows > 0 and ds.columns > 0, f"Empty grid: {ds.rows}x{ds.columns}"
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_nested_returns_single_band_dataset(self):
        """from_healpix regrids a NESTED field into a single-band Dataset.

        Test scenario:
            An nside=2 NESTED field (48 pixels) yields a single-band Dataset.
        """
        ds = from_healpix(np.arange(48.0), nside=2, nest=True, cell_size=20.0)
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"

    def test_nside_derived_from_length(self):
        """from_healpix derives nside from the value count when omitted.

        Test scenario:
            A 192-element field implies nside=4 (12*16) and regrids without an explicit
            nside.
        """
        ds = from_healpix(np.arange(192.0), cell_size=15.0)
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"

    def test_nearest_values_subset_of_source(self):
        """from_healpix nearest-neighbour output values come from the source field.

        Test scenario:
            With method="nearest", every output cell value must be one of the input
            pixel values (no interpolation between them).
        """
        values = np.arange(48.0)
        ds = from_healpix(values, nside=2, nest=True, cell_size=20.0, method="nearest")
        produced = set(np.unique(ds.read_array()).tolist())
        assert produced.issubset(set(values.tolist())), f"Unexpected: {produced}"

    def test_invalid_length_raises(self):
        """from_healpix rejects a value count that is not 12*nside**2.

        Test scenario:
            A length of 10 is not a valid HEALPix pixel count and raises ValueError.
        """
        with pytest.raises(ValueError, match="valid HEALPix pixel count") as exc:
            from_healpix(np.zeros(10), cell_size=30.0)
        assert "valid HEALPix" in str(exc.value), f"Unexpected: {exc.value}"

    def test_explicit_nside_mismatch_raises(self):
        """from_healpix rejects an explicit nside that disagrees with the length.

        Test scenario:
            nside=4 implies 192 pixels; passing 12 values raises ValueError.
        """
        with pytest.raises(ValueError, match="valid HEALPix pair") as exc:
            from_healpix(np.zeros(12), nside=4, cell_size=30.0)
        assert "192" in str(exc.value), f"Expected pixel count in message: {exc.value}"

    def test_nested_non_power_of_two_raises(self):
        """from_healpix rejects NESTED ordering with a non-power-of-two nside.

        Test scenario:
            nside=3 is a valid RING resolution (108 pixels) but NESTED ordering requires
            a power-of-two nside, so nest=True raises ValueError.
        """
        with pytest.raises(ValueError, match="power of two") as exc:
            from_healpix(np.zeros(108), nside=3, nest=True, cell_size=30.0)
        assert "power of two" in str(exc.value), f"Unexpected: {exc.value}"

    def test_bbox_pins_output_extent(self):
        """from_healpix honours an explicit bbox for the output extent.

        Test scenario:
            An nside=2 field gridded with bbox=(-180, -90, 180, 90) and cell_size=10
            yields a 36x18 grid whose bounding box equals the requested global extent.
        """
        ds = from_healpix(
            np.arange(48.0), nside=2, cell_size=10.0, bbox=(-180.0, -90.0, 180.0, 90.0)
        )
        assert (ds.rows, ds.columns) == (18, 36), f"Got {ds.rows}x{ds.columns}"
        np.testing.assert_allclose(
            list(ds.bbox),
            [-180.0, -90.0, 180.0, 90.0],
            atol=1e-6,
            err_msg="bbox not pinned",
        )
