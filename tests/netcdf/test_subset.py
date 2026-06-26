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

import gc
import os
from unittest.mock import Mock

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.base.remote import CloudConfig
from pyramids.netcdf.netcdf import (
    NetCDF,
    _clamp_bound,
    _contiguous_range,
    _resolve_index_selector,
)

pytestmark = pytest.mark.core


def _write_multidim(path, data_vars, coords):
    """Write a multidimensional NetCDF from plain dicts — pyramids' own GDAL writer.

    Mirrors ``NetCDF.from_xarray``'s internal builder (MEM multidim ->
    ``CreateCopy`` to netCDF -> reopen) without any xarray. Dimensions are
    inferred from the data variables' shapes; each coordinate becomes a 1-D
    indexing MDArray named after its dimension, so ``subset``'s name-based axis
    detection works exactly as it does for a real CF cube.

    Args:
        path: Output ``.nc`` path.
        data_vars: ``{name: (dim_names_tuple, ndarray)}``.
        coords: ``{dim_name: 1-D values}``; omit a dim to leave it
            coordinate-less (the missing-coordinate case).

    Returns:
        NetCDF: The reopened on-disk container.
    """
    sizes: dict[str, int] = {}
    for dim_names, arr in data_vars.values():
        for dim_name, size in zip(dim_names, np.asarray(arr).shape):
            sizes[dim_name] = int(size)
    src = gdal.GetDriverByName("MEM").CreateMultiDimensional("synthetic")
    root = src.GetRootGroup()
    gdal_dims = {d: root.CreateDimension(d, "", "", n) for d, n in sizes.items()}
    for name, values in coords.items():
        if name not in gdal_dims:
            continue
        values = np.asarray(values)
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        md = root.CreateMDArray(name, [gdal_dims[name]], ext)
        md.Write(np.ascontiguousarray(values))
    for name, (dim_names, arr) in data_vars.items():
        arr = np.asarray(arr)
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        md = root.CreateMDArray(name, [gdal_dims[d] for d in dim_names], ext)
        md.Write(np.ascontiguousarray(arr))
    dst = gdal.GetDriverByName("netCDF").CreateCopy(str(path), src, 0)
    if dst is None:
        raise RuntimeError(f"Failed to write NetCDF to {path}")
    dst.FlushCache()
    # Release the write handles before reopening — an open netCDF handle leaves the
    # on-disk file unrecognised by the reader (mirrors NetCDF.from_xarray).
    dst = None
    src = None
    return NetCDF.read_file(str(path), read_only=True)

NWM_LDASOUT = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr"

# The live NWM tests hit a public S3 bucket; opt in explicitly so the default
# suite never depends on network reachability of a third-party store.
_RUN_LIVE_NWM = os.environ.get("PYRAMIDS_RUN_NWM_SUBSET_TEST") == "1"

# Transport-level failures that justify a graceful skip of a live read. Anything
# NOT matching these (a real read_file / subset defect) must propagate and fail
# the test rather than be masked as "store unreachable".
_NETWORK_ERROR_MARKERS = (
    "curl error",
    "could not connect",
    "could not resolve",
    "connection timed out",
    "connection refused",
    "connection reset",
    "operation timed out",
    "timed out",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "no route to host",
    "http response code: 0",
)


def _skip_if_network_else_raise(exc: BaseException) -> None:
    """Skip on a genuine transport error; re-raise a real failure.

    A live S3 read should tolerate the bucket being unreachable, but must never
    mask a code bug. Only connection-shaped errors skip; anything else (a real
    ``read_file`` / ``subset`` defect) propagates and fails the test.

    Args:
        exc: The exception caught around the remote read.

    Raises:
        BaseException: ``exc`` itself, when it is not a transport error.
    """
    if any(marker in str(exc).lower() for marker in _NETWORK_ERROR_MARKERS):
        pytest.skip(f"NWM store unreachable: {exc}")
    raise exc


class TestNwmReachabilityGuard:
    """The live-NWM guard skips on transport errors but re-raises real failures."""

    def test_transport_error_skips(self):
        """A connection-shaped error skips gracefully, not fails."""
        with pytest.raises(pytest.skip.Exception):
            _skip_if_network_else_raise(
                RuntimeError("CURL error: Could not resolve host for the bucket")
            )

    def test_real_failure_propagates(self):
        """A non-transport error (a real bug) is re-raised, never masked as 'unreachable'."""
        with pytest.raises(ValueError, match="not found"):
            _skip_if_network_else_raise(ValueError("variable 'ACCET' not found"))


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


def _fake_md_array(ndv=None, attrs=None):
    """A Mock GDAL MDArray: ``GetNoDataValueAsDouble()`` + ``GetAttribute(name)``.

    Returns a :class:`unittest.mock.Mock` so the GDAL PascalCase API surface is
    faked without defining non-snake-case methods. ``attrs`` maps attribute name
    to its scalar value; an absent attribute raises ``RuntimeError`` like GDAL.
    """
    attrs = attrs or {}

    def get_attribute(name):
        if name not in attrs:
            raise RuntimeError(f"Attribute {name} does not exist")
        attr = Mock()
        attr.ReadAsDoubleArray.return_value = [attrs[name]]
        return attr

    md = Mock()
    md.GetNoDataValueAsDouble.return_value = ndv
    md.GetAttribute.side_effect = get_attribute
    return md


def _fake_group(arrays):
    """A Mock GDAL root group whose ``OpenMDArray(name)`` returns a coord stub.

    ``arrays`` maps a coordinate name to the array its ``ReadAsArray()`` yields
    (use ``None`` to simulate a dimension with no readable coordinate); a name
    absent from the mapping raises ``RuntimeError`` like GDAL.
    """

    def open_mdarray(name):
        if name not in arrays:
            raise RuntimeError(f"Array {name} does not exist")
        coord = Mock()
        coord.ReadAsArray.return_value = arrays[name]
        return coord

    rg = Mock()
    rg.OpenMDArray.side_effect = open_mdarray
    return rg


def _fake_group_with_attrs(coords):
    """A Mock root group whose coords expose CF attributes via ``GetAttributes``.

    ``coords`` maps a dimension name to either a ``{attr_name: value}`` dict (the
    coordinate's string attributes) or ``None`` to make ``OpenMDArray`` raise as
    GDAL does for a dimension with no coordinate variable.
    """

    def open_mdarray(name):
        if name not in coords or coords[name] is None:
            raise RuntimeError(f"Array {name} does not exist")
        attr_objs = []
        for attr_name, value in coords[name].items():
            attr = Mock()
            attr.GetName.return_value = attr_name
            # GDAL attributes are read via ``Attribute.Read()``; stub ``ReadAsString`` too
            # so the mock stays faithful regardless of which reader the code uses.
            attr.Read.return_value = value
            attr.ReadAsString.return_value = value
            attr_objs.append(attr)
        coord = Mock()
        coord.GetAttributes.return_value = attr_objs
        return coord

    rg = Mock()
    rg.OpenMDArray.side_effect = open_mdarray
    return rg


class TestAxisRole:
    """``_axis_role`` maps a coordinate's CF attributes to ``"Y"``/``"X"``/None."""

    @pytest.mark.parametrize(
        "attrs, expected",
        [
            ({"axis": "Y"}, "Y"),
            ({"axis": "X"}, "X"),
            ({"axis": "T"}, None),
            ({"standard_name": "latitude"}, "Y"),
            ({"standard_name": "longitude"}, "X"),
            ({"standard_name": "projection_y_coordinate"}, "Y"),
            ({"standard_name": "projection_x_coordinate"}, "X"),
            ({"standard_name": "grid_latitude"}, "Y"),
            ({"standard_name": "grid_longitude"}, "X"),
            ({"units": "degrees_north"}, "Y"),
            ({"units": "degree_n"}, "Y"),
            ({"units": "degrees_east"}, "X"),
            ({"units": "degree_e"}, "X"),
            ({"units": "metre"}, None),
            ({"long_name": "something"}, None),
        ],
    )
    def test_attribute_roles(self, attrs, expected):
        """Each recognised CF attribute maps to the right axis role (else None).

        Args:
            attrs: The coordinate variable's string attributes.
            expected: The expected ``"Y"`` / ``"X"`` / ``None`` role.
        """
        rg = _fake_group_with_attrs({"c": attrs})
        assert NetCDF._axis_role(rg, "c") == expected, f"attrs={attrs}"

    def test_missing_coordinate_is_none(self):
        """A dimension with no coordinate variable yields ``None`` (not an error).

        Test scenario:
            ``OpenMDArray`` raises ``RuntimeError`` -> role is ``None``.
        """
        rg = _fake_group_with_attrs({"c": None})
        assert NetCDF._axis_role(rg, "c") is None, "missing coord should be None"

    def test_unreadable_attribute_is_skipped(self):
        """An attribute whose value can't be read is tolerated; others still read.

        Test scenario:
            An unreadable ``valid_min`` (``Read`` raises) alongside ``axis=Y`` ->
            still detected as ``"Y"``.
        """
        bad = Mock()
        bad.GetName.return_value = "valid_min"
        bad.Read.side_effect = RuntimeError("not readable")
        good = Mock()
        good.GetName.return_value = "axis"
        good.Read.return_value = "Y"
        coord = Mock()
        coord.GetAttributes.return_value = [bad, good]
        rg = Mock()
        rg.OpenMDArray.side_effect = lambda name: coord
        assert NetCDF._axis_role(rg, "y") == "Y", "unreadable attr should be skipped"


class TestAssertFullRank:
    """``_assert_full_rank`` guards that a windowed read kept one axis per dim."""

    def test_matching_rank_is_accepted(self):
        """An array with one axis per dimension passes (returns ``None``).

        Test scenario:
            A (1, 4, 5) read for a 3-D variable -> no error.
        """
        assert NetCDF._assert_full_rank(np.zeros((1, 4, 5)), 3, "soil") is None

    def test_squeezed_read_raises(self):
        """A read missing an axis raises a clear ``RuntimeError``.

        Test scenario:
            A (4, 5) read for a 3-D variable -> RuntimeError naming the variable
            and the expected axis count.
        """
        with pytest.raises(RuntimeError, match="returned 2 axes, expected 3") as exc:
            NetCDF._assert_full_rank(np.zeros((4, 5)), 3, "soil")
        assert "soil" in str(exc.value), f"variable name missing: {exc.value}"


class TestCfSpatialAxes:
    """``_cf_spatial_axes`` finds the spatial axes from coordinate attributes."""

    def test_none_when_no_root_group(self):
        """``rg is None`` -> ``None`` (detection falls through to names).

        Test scenario:
            No multidimensional root group available.
        """
        assert NetCDF._cf_spatial_axes(None, ["time", "y", "x"]) is None

    def test_detects_interleaved_axes_by_attrs(self):
        """CF attrs locate y/x even when a layer dim is interleaved between them.

        Test scenario:
            ``(time, y, level, x)`` with axis attrs on y/x -> ``(1, 3)``.
        """
        rg = _fake_group_with_attrs(
            {
                "time": {"axis": "T"},
                "y": {"axis": "Y"},
                "level": {"long_name": "soil layer"},
                "x": {"axis": "X"},
            }
        )
        out = NetCDF._cf_spatial_axes(rg, ["time", "y", "level", "x"])
        assert out == (1, 3), f"expected (1, 3), got {out}"

    def test_none_when_only_one_axis_has_attrs(self):
        """Only one recognisable spatial axis -> ``None`` (incomplete).

        Test scenario:
            y carries an axis attr but x does not -> cannot resolve a pair.
        """
        rg = _fake_group_with_attrs(
            {"time": None, "y": {"axis": "Y"}, "x": {"long_name": "col"}}
        )
        assert NetCDF._cf_spatial_axes(rg, ["time", "y", "x"]) is None


class TestMdArrayNoData:
    """``_md_array_no_data`` prefers the driver value, then CF attrs, else None."""

    def test_driver_value_wins(self):
        assert NetCDF._md_array_no_data(_fake_md_array(ndv=-1.0)) == pytest.approx(-1.0)

    def test_missing_value_fallback(self):
        md = _fake_md_array(ndv=None, attrs={"missing_value": -999900.0})
        assert NetCDF._md_array_no_data(md) == pytest.approx(-999900.0)

    def test_fill_value_fallback(self):
        md = _fake_md_array(ndv=None, attrs={"_FillValue": -9999.0})
        assert NetCDF._md_array_no_data(md) == pytest.approx(-9999.0)

    def test_missing_value_preferred_over_fill_value(self):
        md = _fake_md_array(ndv=None, attrs={"missing_value": 1.0, "_FillValue": 2.0})
        assert NetCDF._md_array_no_data(md) == pytest.approx(1.0)

    def test_none_when_no_value_or_attrs(self):
        assert NetCDF._md_array_no_data(_fake_md_array()) is None


class TestReadAxisCoords:
    """``_read_axis_coords`` returns float64 values or a clear error."""

    def test_success_returns_float64(self):
        rg = _fake_group({"x": np.array([0, 1, 2])})
        out = NetCDF._read_axis_coords(rg, "x", "x")
        assert out.dtype == np.float64
        assert list(out) == [0.0, 1.0, 2.0]

    def test_missing_array_raises_clear_error(self):
        rg = _fake_group({})
        with pytest.raises(ValueError, match="no 1-D coordinate variable"):
            NetCDF._read_axis_coords(rg, "y", "y")

    def test_none_values_raises_clear_error(self):
        rg = _fake_group({"y": None})
        with pytest.raises(ValueError, match="no 1-D coordinate variable"):
            NetCDF._read_axis_coords(rg, "y", "y")


def _synthetic_cube(
    tmp_path,
    *,
    with_coords=True,
    with_extra=False,
    with_no_time=False,
    with_interleaved=False,
    x_descending=False,
    step=1.0,
):
    """Build a tiny local multidimensional NetCDF and return an opened ``NetCDF``.

    The ``y`` axis ascends (south->north) so the north-up normalisation in
    ``subset`` is exercised. ``temp`` holds ``np.arange`` values so band
    orientation can be verified exactly; with ``with_extra`` a 4-D ``flux`` adds
    a non-spatial ``level`` axis (``time, level, y, x``) for ``**dims`` coverage;
    with ``with_interleaved`` a 4-D ``soil`` puts the layer dim BETWEEN the
    spatial axes (``time, y, level, x``); with ``with_no_time`` a 3-D
    ``ens(member, y, x)`` adds a non-time leading axis. ``with_coords=False``
    drops the ``y`` / ``x`` coordinate variables (the missing-coordinate case).
    ``step`` scales the ``x`` / ``y`` cell spacing (to test cell-size handling).
    """
    n_t, n_y, n_x, n_lev = 3, 4, 5, 2
    temp = np.arange(n_t * n_y * n_x, dtype="float64").reshape(n_t, n_y, n_x)
    data_vars = {"temp": (("time", "y", "x"), temp)}
    if with_extra:
        flux = np.arange(n_t * n_lev * n_y * n_x, dtype="float64").reshape(
            n_t, n_lev, n_y, n_x
        )
        data_vars["flux"] = (("time", "level", "y", "x"), flux)
    if with_interleaved:
        soil = np.arange(n_t * n_y * n_lev * n_x, dtype="float64").reshape(
            n_t, n_y, n_lev, n_x
        )
        data_vars["soil"] = (("time", "y", "level", "x"), soil)
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
        if with_extra or with_interleaved:
            coords["level"] = np.arange(n_lev)
        if with_no_time:
            coords["member"] = np.arange(2)
    return _write_multidim(tmp_path / "cube.nc", data_vars, coords)


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

    def test_subset_result_survives_source_dataset_gc(self, tmp_path):
        """The NetCDF returned by subset() owns its GDAL handle and stays usable after GC (M2).

        Test scenario:
            subset() builds a temporary classic ``Dataset``, hands its GDAL raster to the
            returned ``NetCDF``, and clears the ``Dataset._raster`` so the discarded
            ``Dataset`` cannot close the handle the ``NetCDF`` now owns (API-2 ownership
            transfer). Force a ``gc.collect()`` to reclaim that temporary ``Dataset``,
            then read the array and write the result to a GeoTIFF — both must succeed,
            proving the handle survived the transfer (no dangling pointer / double-close).
        """
        nc = _synthetic_cube(tmp_path)
        result = nc.subset("temp", time=(0, 3))
        gc.collect()

        arr = np.asarray(result.read_array())
        assert arr.shape[0] == 3, f"expected 3 bands after GC, got shape {arr.shape}"

        out = tmp_path / "subset_after_gc.tif"
        result.to_file(out)
        assert out.exists() and out.stat().st_size > 0, "result not writable after source GC"

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


class TestDimensionSizesAndTimeValues:
    """G-1: surface true dimension sizes + raw time values (CF-unparseable case)."""

    def test_dimension_sizes_reports_true_lengths(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_extra=True)
        assert nc.dimension_sizes == {"time": 3, "y": 4, "x": 5, "level": 2}

    def test_get_time_values_returns_raw_coordinate(self, tmp_path):
        # The synthetic store has no CF time units (like the NWM Zarr), so
        # get_time_variable() is None but the raw coordinate is still readable.
        nc = _synthetic_cube(tmp_path)
        assert nc.get_time_variable() is None
        assert list(nc.get_time_values("time")) == [0, 1, 2]

    def test_get_time_values_missing_dim_returns_none(self, tmp_path):
        nc = _synthetic_cube(tmp_path)
        assert nc.get_time_values("nope") is None

    def test_dimension_sizes_empty_for_variable_subset(self, tmp_path):
        # A variable subset is a classic-mode dataset with no root group, so
        # dimension_sizes is empty (documented behaviour).
        nc = _synthetic_cube(tmp_path)
        var = nc.get_variable("temp")
        assert var.dimension_sizes == {}


class TestDetectSpatialAxes:
    """G-2: locate the spatial axes by override / CF attrs / name / trailing."""

    def test_explicit_override(self):
        assert NetCDF._detect_spatial_axes(
            None, ["time", "y", "lev", "x"], "y", "x"
        ) == (1, 3)

    def test_only_one_override_raises(self):
        with pytest.raises(ValueError, match="both y_dim and x_dim"):
            NetCDF._detect_spatial_axes(None, ["y", "x"], "y", None)

    def test_unknown_override_raises(self):
        with pytest.raises(ValueError, match="not a dimension"):
            NetCDF._detect_spatial_axes(None, ["y", "x"], "lat", "lon")

    def test_equal_override_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            NetCDF._detect_spatial_axes(None, ["time", "y", "x"], "x", "x")

    def test_well_known_names_when_no_attrs(self):
        # rg=None -> CF-attr detection skipped; fall back to known names. y/x are
        # interleaved around a layer dim, so this is NOT the trailing-two case.
        assert NetCDF._detect_spatial_axes(
            None, ["time", "y", "soil", "x"], None, None
        ) == (1, 3)

    def test_trailing_two_fallback(self):
        assert NetCDF._detect_spatial_axes(None, ["time", "a", "b"], None, None) == (
            1,
            2,
        )


class TestSubsetInterleavedLayer:
    """G-2: window a variable whose layer dim sits between y and x."""

    def test_interleaved_layer_selected_by_name(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_interleaved=True)
        ds = nc.subset("soil", time=0, level=1)
        assert (ds.rows, ds.columns) == (4, 5)
        assert ds.band_count == 1

    def test_interleaved_layer_correct_orientation_and_values(self, tmp_path):
        # soil is (time, y, level, x); y ascends -> output row 0 is the
        # northernmost (y index 3). Values are arange over (t, y, level, x).
        nc = _synthetic_cube(tmp_path, with_interleaved=True)
        ds = nc.subset("soil", time=0, level=0)
        n_y, n_x, n_lev = 4, 5, 2
        full = np.arange(3 * n_y * n_lev * n_x, dtype="float64").reshape(
            3, n_y, n_lev, n_x
        )
        expected_row0 = full[0, 3, 0, :]  # time 0, y index 3 (north), level 0
        assert list(np.asarray(ds.read_array())[0]) == list(expected_row0)

    def test_interleaved_unselected_layer_raises(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_interleaved=True)
        with pytest.raises(ValueError, match="must be selected"):
            nc.subset("soil", time=0)

    def test_explicit_y_x_dim_override(self, tmp_path):
        nc = _synthetic_cube(tmp_path, with_interleaved=True)
        ds = nc.subset("soil", time=0, level=0, y_dim="y", x_dim="x")
        assert (ds.rows, ds.columns) == (4, 5)

    def test_interleaved_layer_range_is_multiband_in_c_order(self, tmp_path):
        # L3: a ranged interleaved layer -> one band per layer, labelled and
        # ordered to match the C-order band flatten after moveaxis.
        nc = _synthetic_cube(tmp_path, with_interleaved=True)
        ds = nc.subset("soil", time=0, level=(0, 2))
        assert ds.band_count == 2
        assert ds.band_names == ["level=0", "level=1"]
        n_y, n_x, n_lev = 4, 5, 2
        full = np.arange(3 * n_y * n_lev * n_x, dtype="float64").reshape(
            3, n_y, n_lev, n_x
        )
        bands = np.asarray(ds.read_array())  # (2, rows, cols); row 0 = north (y=3)
        assert list(bands[0][0]) == list(full[0, 3, 0, :])
        assert list(bands[1][0]) == list(full[0, 3, 1, :])


@pytest.mark.slow
@pytest.mark.vfs
@pytest.mark.skipif(
    not _RUN_LIVE_NWM,
    reason="set PYRAMIDS_RUN_NWM_SUBSET_TEST=1 to run the live NWM S3 subset test",
)
class TestSubsetLiveNWM:
    """Opt-in live test against the public NWM retrospective gridded Zarr.

    Off by default — it hits a public S3 bucket, so it runs only when
    ``PYRAMIDS_RUN_NWM_SUBSET_TEST=1`` is set (and is also marked ``slow``/``vfs``).
    When it runs, a transport-level failure skips gracefully via
    :func:`_skip_if_network_else_raise`, but a real ``read_file`` / ``subset``
    defect propagates and fails. Verifies the three fixes together: anonymous
    remote multidim open (region pinned), CRS preserved on the windowed slice,
    and a bounded ``(time, bbox)`` read.
    """

    def test_subset_one_timestep_bbox(self):
        try:
            with CloudConfig(aws_no_sign_request=True, aws_region="us-east-1"):
                nc = NetCDF.read_file(NWM_LDASOUT)
                ds = nc.subset("ACCET", time=0, bbox=(-78.0, 38.0, -75.0, 40.0))
        except (RuntimeError, OSError) as exc:  # transport vs. real failure
            _skip_if_network_else_raise(exc)

        assert ds.rows > 0 and ds.columns > 0
        # Far smaller than the full 3840 x 4608 grid -> the read was windowed.
        assert ds.rows < 3840 and ds.columns < 4608
        assert ds.band_count == 1
        assert "Lambert_Conformal_Conic" in (ds.crs or "")

    def test_subset_time_range_is_multiband(self):
        try:
            with CloudConfig(aws_no_sign_request=True, aws_region="us-east-1"):
                nc = NetCDF.read_file(NWM_LDASOUT)
                ds = nc.subset("ACCET", time=(0, 3), bbox=(-78.0, 38.0, -75.0, 40.0))
        except (RuntimeError, OSError) as exc:
            _skip_if_network_else_raise(exc)

        assert ds.band_count == 3

    def test_dimension_sizes_and_time_values(self):
        # G-1: true dim sizes + raw time coordinate are surfaced even though GDAL
        # can't parse the CF time units (so get_time_variable() is None).
        try:
            with CloudConfig(aws_no_sign_request=True, aws_region="us-east-1"):
                nc = NetCDF.read_file(NWM_LDASOUT)
                sizes = nc.dimension_sizes
                time_vals = nc.get_time_values("time")
        except (RuntimeError, OSError) as exc:
            _skip_if_network_else_raise(exc)

        assert sizes.get("time") == 128568
        assert sizes.get("y") == 3840 and sizes.get("x") == 4608
        assert time_vals is not None and time_vals.size == 128568

    def test_subset_interleaved_layer_variable(self):
        # G-2: SOIL_M is (time, y, soil_layers_stag, x) — layer between y and x.
        try:
            with CloudConfig(aws_no_sign_request=True, aws_region="us-east-1"):
                nc = NetCDF.read_file(NWM_LDASOUT)
                ds = nc.subset(
                    "SOIL_M",
                    time=0,
                    soil_layers_stag=0,
                    bbox=(-78.0, 38.0, -75.0, 40.0),
                )
        except (RuntimeError, OSError) as exc:
            _skip_if_network_else_raise(exc)

        assert ds.rows > 0 and ds.columns > 0
        assert ds.rows < 3840 and ds.columns < 4608
        assert ds.band_count == 1
        assert "Lambert_Conformal_Conic" in (ds.crs or "")
