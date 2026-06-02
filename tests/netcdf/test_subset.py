"""Tests for ``NetCDF.subset`` and its windowing helpers (issue #460).

``subset`` reads a windowed ``(variable, time, bbox)`` slice of a gridded
multidimensional cube into a georeferenced :class:`~pyramids.dataset.Dataset`
without materialising the whole variable. The pure helpers below carry the
tricky logic — ascending/descending axis ranges, non-spatial index selection,
time-axis detection, and reprojecting a lon/lat bbox onto a projected grid — so
they are tested in isolation. A live, opt-in test against the public NWM
retrospective store exercises the full remote path end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import osr

from pyramids.netcdf.netcdf import (
    NetCDF,
    _clamp_bound,
    _contiguous_range,
    _resolve_index_selector,
)

pytestmark = pytest.mark.core

NWM_LDASOUT = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr"


class TestContiguousRange:
    """``_contiguous_range`` works for either axis direction (NWM ``y`` ascends)."""

    def test_ascending_axis(self):
        coords = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert _contiguous_range(coords, 1.5, 3.5, "x", (0, 0, 0, 0)) == (2, 4)

    def test_descending_axis(self):
        coords = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        assert _contiguous_range(coords, 1.5, 3.5, "y", (0, 0, 0, 0)) == (1, 3)

    def test_inclusive_bounds(self):
        coords = np.array([0.0, 1.0, 2.0, 3.0])
        assert _contiguous_range(coords, 1.0, 2.0, "x", (0, 0, 0, 0)) == (1, 3)

    def test_empty_window_raises(self):
        coords = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="selects no cells"):
            _contiguous_range(coords, 10.0, 20.0, "x", (10, 10, 20, 20))


class TestResolveIndexSelector:
    """Non-spatial dimension selectors resolve to half-open ``(start, stop)``."""

    def test_none_on_length_one_ok(self):
        assert _resolve_index_selector(None, 1, "vis_nir") == (0, 1)

    def test_none_on_longer_dim_raises(self):
        with pytest.raises(ValueError, match="must be selected"):
            _resolve_index_selector(None, 4, "soil_layers_stag")

    def test_int_selects_one(self):
        assert _resolve_index_selector(2, 10, "time") == (2, 3)

    def test_negative_int_wraps(self):
        assert _resolve_index_selector(-1, 5, "time") == (4, 5)

    def test_tuple_range(self):
        assert _resolve_index_selector((0, 4), 10, "time") == (0, 4)

    def test_slice_range(self):
        assert _resolve_index_selector(slice(None, 3), 10, "time") == (0, 3)

    def test_bad_tuple_length_raises(self):
        with pytest.raises(ValueError, match="must be"):
            _resolve_index_selector((0, 1, 2), 10, "time")

    def test_slice_negative_start_wraps(self):
        assert _resolve_index_selector(slice(-2, None), 5, "time") == (3, 5)

    def test_slice_stop_beyond_size_clamps(self):
        assert _resolve_index_selector(slice(0, 999), 5, "time") == (0, 5)

    def test_tuple_negative_and_out_of_range_clamp(self):
        assert _resolve_index_selector((-3, 999), 5, "time") == (2, 5)

    def test_out_of_range_int_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_index_selector(99, 5, "time")

    def test_negative_int_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_index_selector(-99, 5, "time")

    def test_reversed_range_raises_empty(self):
        with pytest.raises(ValueError, match="empty index range"):
            _resolve_index_selector((3, 1), 10, "time")

    def test_equal_bounds_raises_empty(self):
        with pytest.raises(ValueError, match="empty index range"):
            _resolve_index_selector(slice(2, 2), 10, "time")


class TestClampBound:
    """``_clamp_bound`` wraps negatives and clamps a slice bound into [0, size]."""

    @pytest.mark.parametrize(
        "index, size, expected",
        [
            (2, 5, 2),
            (-1, 5, 4),
            (-99, 5, 0),
            (99, 5, 5),
            (0, 5, 0),
            (5, 5, 5),
        ],
    )
    def test_clamp(self, index, size, expected):
        """Bound is wrapped (negatives) then clamped to the inclusive [0, size] range.

        Args:
            index: Raw bound to normalise.
            size: Dimension length.
            expected: Normalised bound.
        """
        assert _clamp_bound(index, size) == expected, f"{index}/{size} -> {expected}"


class TestDetectTimeAxis:
    """The time selector targets the time-like axis, else the first non-spatial."""

    def test_named_time_dim(self):
        assert NetCDF._detect_time_axis(["time", "y", "x"], 1, 2) == 0

    def test_no_time_falls_back_to_first_non_spatial(self):
        # (level, member, y, x) with no time-like name -> first non-spatial axis.
        assert NetCDF._detect_time_axis(["level", "member", "y", "x"], 2, 3) == 0

    def test_only_spatial_returns_none(self):
        assert NetCDF._detect_time_axis(["y", "x"], 0, 1) is None


class TestReprojectBboxEnvelope:
    """Densified bbox reprojection onto a projected grid (G-D)."""

    def test_no_dst_crs_is_identity(self):
        bbox = (-78.0, 38.0, -75.0, 40.0)
        assert NetCDF._reproject_bbox_envelope(bbox, 4326, None, 25) == bbox

    def test_same_crs_is_identity(self):
        bbox = (-78.0, 38.0, -75.0, 40.0)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
        out = NetCDF._reproject_bbox_envelope(bbox, 4326, dst, 25)
        assert out == pytest.approx(bbox)

    def test_lonlat_into_lambert_conformal_conic(self):
        # NWM-style LCC sphere: a lon/lat box maps to projected metres east of
        # the -97 central meridian, and the densified envelope conservatively
        # over-covers the requested latitude span.
        lcc = osr.SpatialReference()
        lcc.ImportFromWkt(
            'PROJCS["Lambert_Conformal_Conic",'
            'GEOGCS["GCS_Sphere",DATUM["D_Sphere",'
            'SPHEROID["Sphere",6370000.0,0.0]],'
            'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
            'PROJECTION["Lambert_Conformal_Conic_2SP"],'
            'PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],'
            'PARAMETER["central_meridian",-97.0],'
            'PARAMETER["standard_parallel_1",30.0],'
            'PARAMETER["standard_parallel_2",60.0],'
            'PARAMETER["latitude_of_origin",40.0],UNIT["Meter",1.0]]'
        )
        min_x, min_y, max_x, max_y = NetCDF._reproject_bbox_envelope(
            (-78.0, 38.0, -75.0, 40.0), 4326, lcc, 25
        )
        # East of -97 -> positive easting, ~1.5-1.9 million metres.
        assert 1.4e6 < min_x < 1.9e6
        assert max_x > min_x
        assert max_y > min_y

    def test_wkt_string_source_crs(self):
        """A WKT/PROJ string source CRS is accepted via ``SetFromUserInput``.

        Test scenario:
            Passing ``crs`` as a WKT string (not an EPSG int) and a matching
            destination yields an identity envelope.
        """
        wgs84_wkt = osr.SpatialReference()
        wgs84_wkt.ImportFromEPSG(4326)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
        bbox = (-10.0, -5.0, 10.0, 5.0)
        out = NetCDF._reproject_bbox_envelope(bbox, wgs84_wkt.ExportToWkt(), dst, 10)
        assert out == pytest.approx(bbox)


class _FakeAttr:
    """Minimal stand-in for a GDAL attribute exposing ``ReadAsDoubleArray``."""

    def __init__(self, value: float) -> None:
        self._value = value

    def ReadAsDoubleArray(self):  # noqa: N802 (GDAL SWIG name)
        return [self._value]


class _FakeMDArray:
    """Minimal MDArray for ``_md_array_no_data`` — driver value + CF attrs."""

    def __init__(self, ndv=None, attrs=None) -> None:
        self._ndv = ndv
        self._attrs = attrs or {}

    def GetNoDataValueAsDouble(self):  # noqa: N802
        return self._ndv

    def GetAttribute(self, name):  # noqa: N802
        if name in self._attrs:
            return _FakeAttr(self._attrs[name])
        raise RuntimeError(f"Attribute {name} does not exist")


class TestMdArrayNoData:
    """``_md_array_no_data`` prefers the driver value, then CF attrs, else None."""

    def test_driver_value_wins(self):
        assert NetCDF._md_array_no_data(_FakeMDArray(ndv=-1.0)) == pytest.approx(-1.0)

    def test_missing_value_fallback(self):
        md = _FakeMDArray(ndv=None, attrs={"missing_value": -999900.0})
        assert NetCDF._md_array_no_data(md) == pytest.approx(-999900.0)

    def test_fill_value_fallback(self):
        md = _FakeMDArray(ndv=None, attrs={"_FillValue": -9999.0})
        assert NetCDF._md_array_no_data(md) == pytest.approx(-9999.0)

    def test_missing_value_preferred_over_fill_value(self):
        md = _FakeMDArray(ndv=None, attrs={"missing_value": 1.0, "_FillValue": 2.0})
        assert NetCDF._md_array_no_data(md) == pytest.approx(1.0)

    def test_none_when_no_value_or_attrs(self):
        assert NetCDF._md_array_no_data(_FakeMDArray()) is None


class _FakeCoord:
    """Coordinate MDArray stub returning a fixed array (or ``None``)."""

    def __init__(self, values) -> None:
        self._values = values

    def ReadAsArray(self):  # noqa: N802
        return self._values


class _FakeGroup:
    """Root-group stub: returns a coord, returns ``None``, or raises like GDAL."""

    def __init__(self, arrays) -> None:
        self._arrays = arrays

    def OpenMDArray(self, name):  # noqa: N802
        if name not in self._arrays:
            raise RuntimeError(f"Array {name} does not exist")
        return self._arrays[name]


class TestReadAxisCoords:
    """``_read_axis_coords`` returns float64 values or a clear error."""

    def test_success_returns_float64(self):
        rg = _FakeGroup({"x": _FakeCoord(np.array([0, 1, 2]))})
        out = NetCDF._read_axis_coords(rg, "x", "x")
        assert out.dtype == np.float64
        assert list(out) == [0.0, 1.0, 2.0]

    def test_missing_array_raises_clear_error(self):
        rg = _FakeGroup({})
        with pytest.raises(ValueError, match="no 1-D coordinate variable"):
            NetCDF._read_axis_coords(rg, "y", "y")

    def test_none_values_raises_clear_error(self):
        rg = _FakeGroup({"y": _FakeCoord(None)})
        with pytest.raises(ValueError, match="no 1-D coordinate variable"):
            NetCDF._read_axis_coords(rg, "y", "y")


def _synthetic_cube(
    tmp_path,
    *,
    with_coords=True,
    with_extra=False,
    with_no_time=False,
    x_descending=False,
    step=1.0,
):
    """Build a tiny local multidimensional NetCDF and return an opened ``NetCDF``.

    The ``y`` axis ascends (south->north) so the north-up normalisation in
    ``subset`` is exercised. ``temp`` holds ``np.arange`` values so band
    orientation can be verified exactly; with ``with_extra`` a 4-D ``flux`` adds
    a non-spatial ``level`` axis for ``**dims`` coverage; with ``with_no_time`` a
    3-D ``ens(member, y, x)`` adds a non-time leading axis. ``with_coords=False``
    drops the ``y`` / ``x`` coordinate variables (the missing-coordinate case).
    ``step`` scales the ``x`` / ``y`` cell spacing (to test cell-size handling).
    """
    xr = pytest.importorskip("xarray")
    n_t, n_y, n_x = 3, 4, 5
    temp = np.arange(n_t * n_y * n_x, dtype="float64").reshape(n_t, n_y, n_x)
    data_vars = {"temp": (("time", "y", "x"), temp)}
    if with_extra:
        n_lev = 2
        flux = np.arange(n_t * n_lev * n_y * n_x, dtype="float64").reshape(
            n_t, n_lev, n_y, n_x
        )
        data_vars["flux"] = (("time", "level", "y", "x"), flux)
    if with_no_time:
        n_member = 2
        ens = np.arange(n_member * n_y * n_x, dtype="float64").reshape(
            n_member, n_y, n_x
        )
        data_vars["ens"] = (("member", "y", "x"), ens)
    coords = {}
    if with_coords:
        x_asc = np.arange(n_x, dtype="float64") * step
        x_vals = x_asc[::-1] if x_descending else x_asc
        coords = {
            "time": np.arange(n_t),
            "y": 10.0 + np.arange(n_y, dtype="float64") * step,  # ascending
            "x": x_vals,
        }
        if with_extra:
            coords["level"] = np.arange(2)
        if with_no_time:
            coords["member"] = np.arange(2)
    ds = xr.Dataset(data_vars, coords=coords)
    return NetCDF.from_xarray(ds, path=str(tmp_path / "cube.nc"))


class TestSubsetOffline:
    """Offline integration tests for the ``subset`` body (no network).

    Cover the windowed read, north-up flip of an ascending ``y`` axis, band
    construction for a time range, bbox cropping in native coordinates, extra-dim
    selection, and the error paths (missing coordinate, unknown ``**dims`` key,
    out-of-range index).
    """

    def test_single_timestep_shape_and_north_up(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        ds = nc.subset("temp", time=0)
        assert (ds.rows, ds.columns) == (4, 5)
        assert ds.band_count == 1
        # y ascends [10..13]; north-up output row 0 is the northernmost (y=13),
        # i.e. xarray y-index 3 -> values [15, 16, 17, 18, 19].
        row0 = np.asarray(ds.read_array())[0]
        assert list(row0) == [15.0, 16.0, 17.0, 18.0, 19.0]
        # North-up geotransform: negative dy, top-left y at the upper edge.
        gt = ds.geotransform
        assert gt[5] < 0
        assert gt[3] == pytest.approx(13.5)

    def test_time_range_is_multiband_with_labels(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        ds = nc.subset("temp", time=(0, 3))
        assert ds.band_count == 3
        assert ds.band_names == ["time=0", "time=1", "time=2"]

    def test_full_grid_when_bbox_none(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        ds = nc.subset("temp", time=0)
        assert (ds.rows, ds.columns) == (4, 5)

    def test_descending_x_axis_normalised_west_to_east(self, tmp_path):
        nc = _synthetic_cube(tmp_path, x_descending=True)
        ds = nc.subset("temp", time=0)
        # Stored x descends [4..0]; west-to-east output row 0 (north, y=13) must
        # run x=0..4 -> the stored values reversed: data[0, 3] = [15..19] -> [19..15].
        row0 = np.asarray(ds.read_array())[0]
        assert list(row0) == [19.0, 18.0, 17.0, 16.0, 15.0]
        assert ds.geotransform[1] > 0, "x cell size must be positive (west-to-east)"

    def test_not_multidimensional_raises(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        classic = NetCDF.read_file(nc.file_name, open_as_multi_dimensional=False)
        with pytest.raises(ValueError, match="requires a multidimensional store"):
            classic.subset("temp", time=0)

    def test_missing_variable_raises(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        with pytest.raises(ValueError, match="is not a variable"):
            nc.subset("does_not_exist", time=0)

    def test_bbox_crops_in_native_coords(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        # Keep x in [1, 3] (3 cols) and y in [11, 12] (2 rows).
        ds = nc.subset("temp", time=0, bbox=(1.0, 11.0, 3.0, 12.0))
        assert (ds.rows, ds.columns) == (2, 3)

    def test_extra_dim_selection(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_extra=True)
        ds = nc.subset("flux", time=0, level=1)
        assert (ds.rows, ds.columns) == (4, 5)
        assert ds.band_count == 1

    def test_unselected_extra_dim_raises(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_extra=True)
        with pytest.raises(ValueError, match="must be selected"):
            nc.subset("flux", time=0)

    def test_unknown_dim_key_raises(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_extra=True)
        with pytest.raises(ValueError, match="unknown dimension selector"):
            nc.subset("flux", time=0, levl=1)

    def test_out_of_range_time_index_raises(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        with pytest.raises(ValueError, match="out of range"):
            nc.subset("temp", time=99)

    def test_missing_coordinate_variable_raises(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_coords=False)
        with pytest.raises(ValueError, match="no 1-D coordinate variable"):
            nc.subset("temp", time=0, bbox=(0.0, 0.0, 1.0, 1.0))

    def test_reversed_time_range_raises(self, tmp_path):
        # L1: an empty/reversed range is rejected with a clear message.
        nc = _synthetic_cube(tmp_path)
        with pytest.raises(ValueError, match="empty index range"):
            nc.subset("temp", time=(2, 1))

    def test_non_time_axis_selectable_by_name(self, tmp_path):
        # L2: a non-time leading axis (ens member) is addressable via **dims.
        nc = _synthetic_cube(tmp_path, with_no_time=True)
        ds = nc.subset("ens", member=1)
        assert (ds.rows, ds.columns) == (4, 5)
        assert ds.band_count == 1

    def test_non_time_axis_also_accepts_time_keyword(self, tmp_path):
        # L2: the dedicated time= keyword still drives that fallback axis.
        nc = _synthetic_cube(tmp_path, with_no_time=True)
        ds = nc.subset("ens", time=0)
        assert ds.band_count == 1

    def test_multi_range_band_labels_are_cartesian_product(self, tmp_path):
        # L3: ranging two non-spatial axes (time x level, each length 2) labels
        # every band as the Cartesian product, in C-order (level varies fastest).
        nc = _synthetic_cube(tmp_path, with_extra=True)
        ds = nc.subset("flux", time=(0, 2), level=(0, 2))
        assert ds.band_count == 4
        assert ds.band_names == [
            "time=0,level=0",
            "time=0,level=1",
            "time=1,level=0",
            "time=1,level=1",
        ]

    def test_single_cell_window_keeps_native_resolution(self, tmp_path):
        # N1: a 1x1 window carries the store's true cell size, not a 1.0 fallback.
        nc = _synthetic_cube(tmp_path, step=1000.0)
        ds = nc.subset("temp", time=0, bbox=(2000.0, 2010.0, 2000.0, 2010.0))
        assert (ds.rows, ds.columns) == (1, 1)
        assert ds.geotransform[1] == pytest.approx(1000.0)
        assert abs(ds.geotransform[5]) == pytest.approx(1000.0)


@pytest.mark.slow
@pytest.mark.vfs
class TestSubsetLiveNWM:
    """Opt-in live test against the public NWM retrospective gridded Zarr.

    Run with ``pytest -m "slow and vfs" tests/netcdf/test_subset.py``. Skipped by
    default and skipped gracefully when the bucket is unreachable. Verifies the
    three fixes together: anonymous remote multidim open (region pinned), CRS
    preserved on the windowed slice, and a bounded ``(time, bbox)`` read.
    """

    def test_subset_one_timestep_bbox(self):
        from pyramids.base.remote import CloudConfig

        try:
            with CloudConfig(aws_no_sign_request=True, aws_region="us-east-1"):
                nc = NetCDF.read_file(NWM_LDASOUT)
                ds = nc.subset("ACCET", time=0, bbox=(-78.0, 38.0, -75.0, 40.0))
        except (RuntimeError, OSError) as exc:  # network / bucket unreachable
            pytest.skip(f"NWM store unreachable: {exc}")

        assert ds.rows > 0 and ds.columns > 0
        # Far smaller than the full 3840 x 4608 grid -> the read was windowed.
        assert ds.rows < 3840 and ds.columns < 4608
        assert ds.band_count == 1
        assert "Lambert_Conformal_Conic" in (ds.crs or "")

    def test_subset_time_range_is_multiband(self):
        from pyramids.base.remote import CloudConfig

        try:
            with CloudConfig(aws_no_sign_request=True, aws_region="us-east-1"):
                nc = NetCDF.read_file(NWM_LDASOUT)
                ds = nc.subset("ACCET", time=(0, 3), bbox=(-78.0, 38.0, -75.0, 40.0))
        except (RuntimeError, OSError) as exc:
            pytest.skip(f"NWM store unreachable: {exc}")

        assert ds.band_count == 3
