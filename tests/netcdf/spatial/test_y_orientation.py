"""Tests for Y-axis orientation consistency in the NetCDF class.

NetCDF files from external tools (WRF, ERA5, NOAH) store latitude
south-to-north (row 0 = southernmost). GDAL's raster convention is
row 0 = northernmost (negative Y pixel size). Both ``get_variable()``
and ``_read_variable()`` must flip such data so row 0 = north.

Files created by pyramids ``create_from_array`` already follow GDAL
convention and should NOT be flipped.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal, osr

from pyramids.netcdf.netcdf import Container, NetCDF

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def noah_nc():
    """Noah precipitation file — external file with south-to-north lat."""
    return NetCDF.read_file(
        "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc",
        open_as_multi_dimensional=True,
    )


class TestExternalFileOrientation:
    """External NetCDF files (south-to-north) must be flipped on read."""

    def test_get_variable_negative_y_pixel_size(self, noah_nc):
        """Extracted variable must have negative Y pixel size.

        Test scenario:
            Negative Y = GDAL convention (origin at north, going south).
        """
        var = noah_nc.get_variable("Band1")
        gt = var.geotransform
        assert gt[5] < 0, f"Y pixel size should be negative, got {gt[5]}"

    def test_get_variable_origin_at_north(self, noah_nc):
        """Geotransform Y origin should be at the north edge (~90).

        Test scenario:
            Noah file covers the globe, so origin Y should be near 90.
        """
        var = noah_nc.get_variable("Band1")
        gt = var.geotransform
        assert gt[3] > 0, f"Y origin should be positive (north), got {gt[3]}"


class TestReadVariableConsistency:
    """_read_variable and get_variable must return the same data."""

    def test_noah_2d_consistency(self, noah_nc):
        """Both read paths should produce identical arrays.

        Test scenario:
            Read Band1 via _read_variable and get_variable().read_array(),
            compare element-by-element.
        """
        from_read = noah_nc._read_variable("Band1")
        var = noah_nc.get_variable("Band1")
        from_get = var.read_array(band=0)
        assert_allclose(
            from_read,
            from_get,
            rtol=1e-5,
            err_msg="_read_variable and get_variable data mismatch",
        )

    def test_pyramids_created_2d_consistency(self):
        """Files created by pyramids should also be consistent.

        Test scenario:
            create_from_array → both read paths should agree.
        """
        arr = np.arange(50, dtype=np.float64).reshape(10, 5)
        geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="test",
        )
        from_read = nc._read_variable("test")
        var = nc.get_variable("test")
        from_get = var.read_array(band=0)
        assert_allclose(
            from_read,
            from_get,
            err_msg="Pyramids-created file: _read_variable != get_variable",
        )

    def test_pyramids_created_3d_consistency(self):
        """3D files created by pyramids should also be consistent.

        Test scenario:
            create_from_array with 3D → both read paths should agree.
        """
        arr = np.arange(150, dtype=np.float64).reshape(3, 10, 5)
        geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="test3d",
            extra_dim_name="time",
        )
        from_read = nc._read_variable("test3d")
        var = nc.get_variable("test3d")
        from_get = var.read_array()
        assert_allclose(
            from_read,
            from_get,
            err_msg="Pyramids-created 3D: _read_variable != get_variable",
        )


class TestPyramidsCreatedNotFlipped:
    """Files created by create_from_array are already in GDAL order."""

    def test_2d_data_preserved_as_is(self):
        """create_from_array data should round-trip without flipping.

        Test scenario:
            Create with known values, read back, verify unchanged.
        """
        arr = np.arange(50, dtype=np.float64).reshape(10, 5)
        geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="seq",
        )
        var = nc.get_variable("seq")
        read_back = var.read_array(band=0)
        assert_allclose(
            read_back,
            arr,
            err_msg="create_from_array data should not be altered",
        )

    def test_negative_y_pixel_size(self):
        """Pyramids-created files should have negative Y pixel size.

        Test scenario:
            The geotransform from create_from_array should already be
            in GDAL convention.
        """
        arr = np.ones((5, 5), dtype=np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="v",
        )
        var = nc.get_variable("v")
        gt = var.geotransform
        assert gt[5] < 0, f"Y pixel size should be negative, got {gt[5]}"


class TestOneDimNotFlipped:
    """1D arrays (dimension coordinates) should never be flipped."""

    def test_x_coordinate_not_flipped(self):
        """The x dimension coordinate should be returned as-is.

        Test scenario:
            Read the x coordinate, verify it's 1D and not altered.
        """
        arr = np.ones((5, 8), dtype=np.float64)
        geo = (10.0, 0.5, 0, 15.0, 0, -0.5)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="v",
        )
        x_vals = nc._read_variable("x")
        assert x_vals is not None, "x coordinate should be readable"
        assert x_vals.ndim == 1, f"Expected 1D, got {x_vals.ndim}D"
        assert x_vals[0] < x_vals[-1], "x should be ascending (west to east)"


ORIENTATION_CASES = [
    ("cf__9v__1d7-2d2__geos__y-desc.nc", "CMI", False, "geostationary radian scan angle, descending -> keep"),
    ("cf__6v__1d2-2d4__geog__y-asc.nc", "Band1", True, "geographic ascending (NOAH) -> flip"),
    ("cf__5v__1d4-3d1__geog__y-desc.nc", "t2m", False, "geographic descending (ERA5) -> keep"),
    ("coards__4v__1d3-3d1__y-desc.nc", "air", False, "geographic descending (COARDS) -> keep"),
]


def _scaled_axis(array, axis_from_end):
    """The axis' coordinate values with `scale_factor` / `add_offset` applied."""
    dim = array.GetDimensions()[array.GetDimensionCount() - axis_from_end]
    return np.asarray(dim.GetIndexingVariable().GetUnscaled().ReadAsArray())


def _north_up_reference(path, variable):
    """First-principles ground truth: the raw array ordered north-up (row 0 = north, col 0 = west).

    Independent of both read paths under test, and of GDAL's classic driver — which cannot serve as a
    reference here: it returns pure fill for some 4-D packed variables, and it mis-orients a
    GDAL-written ``WRITE_BOTTOMUP=NO`` file (flipping the pixels while keeping a north-up
    geotransform).

    Both coordinates are read through ``GetUnscaled()`` so a packed axis — a geostationary granule's
    radian scan angle, stored with a **negative** ``scale_factor`` — is interpreted physically rather
    than by its raw storage order (#705).
    """
    root = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER).GetRootGroup()
    array = root.OpenMDArray(variable)
    y_values, x_values = _scaled_axis(array, 2), _scaled_axis(array, 1)
    raw = np.asarray(array.ReadAsArray())
    # Row 0 already sits at the north when Y descends; col 0 at the west when X ascends.
    rows = slice(None) if y_values[0] > y_values[-1] else slice(None, None, -1)
    cols = slice(None) if x_values[0] < x_values[-1] else slice(None, None, -1)
    return raw[..., rows, cols]


def _assert_orientation(var, expect_flip, label, path, variable):
    """Assert the flip decision, a north-up geotransform, and agreement with the coordinate reference.

    Checking against an *independent* reference is the whole point: the previous version of this
    helper compared pyramids' two internal read paths against each other, and both were mirrored the
    same way, so a geostationary raster shipped upside down (#705). It also checks that materializing
    does not change the pixels.
    """
    assert (
        var._md_y_flipped is expect_flip
    ), f"{label}: expected _md_y_flipped={expect_flip}, got {var._md_y_flipped}"
    assert (
        var.geotransform[5] < 0
    ), f"{label}: geotransform must be north-up, got gt[5]={var.geotransform[5]}"
    before = np.asarray(var.read_array())
    np.testing.assert_array_equal(
        before,
        _north_up_reference(path, variable),
        err_msg=f"{label}: row 0 of read_array() is not at the northernmost Y coordinate",
    )
    var._materialize_md_view()
    np.testing.assert_array_equal(
        np.asarray(var.read_array()),
        before,
        err_msg=f"{label}: materializing the view changed the pixels",
    )


def _utm_netcdf(path, bottom_up):
    """Write an 8x6 UTM32N raster to netCDF, bottom-up or top-down, and return the path.

    GDAL's netCDF writer defaults to ``WRITE_BOTTOMUP=YES``, which stores the rows south→north and
    emits an **ascending** ``y`` (``projection_y_coordinate``) axis; ``WRITE_BOTTOMUP=NO`` keeps the
    rows top-down and emits a descending one. Between them they cover both projected cells of the
    Y-orientation 2x2, neither of which any on-disk fixture provides.
    """
    src = gdal.GetDriverByName("MEM").Create("", 8, 6, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32636)
    src.SetProjection(srs.ExportToWkt())
    src.SetGeoTransform((500000.0, 100.0, 0.0, 4000000.0, 0.0, -100.0))
    src.GetRasterBand(1).WriteArray(np.arange(48, dtype=np.float32).reshape(6, 8))
    src.GetRasterBand(1).SetNoDataValue(-9999.0)
    options = [f"WRITE_BOTTOMUP={'YES' if bottom_up else 'NO'}"]
    gdal.Translate(path, src, format="netCDF", creationOptions=options)
    return path


@pytest.fixture(scope="module")
def projected_descending_nc(tmp_path_factory):
    """A UTM (projected) netCDF written top-down, so its Y axis descends (row 0 = north)."""
    return _utm_netcdf(
        str(tmp_path_factory.mktemp("proj_desc") / "utm_projected_descending.nc"),
        bottom_up=False,
    )


@pytest.fixture(scope="module")
def projected_ascending_nc(tmp_path_factory):
    """A UTM (projected) netCDF written bottom-up, so its Y axis ascends (row 0 = south)."""
    return _utm_netcdf(
        str(tmp_path_factory.mktemp("proj_asc") / "utm_projected_ascending.nc"),
        bottom_up=True,
    )


@pytest.fixture(scope="module")
def x_descending_nc(tmp_path_factory):
    """A geographic netCDF whose longitude runs east→west (negative pixel width).

    Legal CF — the convention only asks that a coordinate be monotonic — but no on-disk fixture has
    one, and no known producer writes one. GDAL's classic netCDF driver never flips X; it reports a
    negative `gt[1]` instead, which pyramids' `abs()`-based cell size and bbox arithmetic cannot
    represent. Written by hand so the east-to-west branch of `_read_md_array` has a case.
    """
    path = str(tmp_path_factory.mktemp("x_desc") / "lon_descending.nc")
    src = gdal.GetDriverByName("MEM").Create("", 5, 4, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    # Origin at the east edge, walking west: lon = 29, 27, 25, 23, 21.
    src.SetGeoTransform((30.0, -2.0, 0.0, 10.0, 0.0, -1.0))
    src.GetRasterBand(1).WriteArray(np.arange(20, dtype=np.float32).reshape(4, 5))
    gdal.Translate(path, src, format="netCDF", creationOptions=["WRITE_BOTTOMUP=NO"])
    return path


class TestOrientationAllCases:
    """The multidim read must come back north-up for every Y-axis case in the 2x2 (CRS x direction).

    The flip is decided from the **scale/offset-applied** Y coordinate, the way GDAL's own classic
    netCDF driver decides `bBottomUp`. That covers every axis kind with one rule: geographic degrees,
    projected metres, and the radian scan angles of a geostationary granule — the last of which is
    packed with a *negative* `scale_factor`, so its raw values ascend while the physical axis descends
    (#705). Each case asserts the flip decision, a north-up geotransform, and equality with an
    independently-derived north-up reference (`_north_up_reference`).
    """

    @pytest.mark.parametrize(
        "filename, variable, expect_flip, label",
        ORIENTATION_CASES,
        ids=[c[0].split(".")[0] for c in ORIENTATION_CASES],
    )
    def test_orientation_matches_coordinate_reference(self, filename, variable, expect_flip, label):
        """The multidim read is north-up and byte-identical to the coordinate-ordered reference.

        Test scenario:
            Read the variable through the public MDIM path and compare it against the raw array
            reordered so row 0 sits at the largest *scaled* Y coordinate.
        """
        path = f"tests/data/netcdf/{filename}"
        var = NetCDF.read_file(path).get_variable(variable)
        _assert_orientation(var, expect_flip, label, path, variable)

    def test_projected_descending_is_kept(self, projected_descending_nc):
        """Projected + descending Y is kept, not flipped.

        Test scenario:
            A UTM-projected netCDF whose `y` axis descends (row 0 = north) is already north-up, so
            `_read_md_array` must leave it alone.
        """
        var = NetCDF.read_file(projected_descending_nc).get_variable("Band1")
        srs = var.raster.GetSpatialRef()
        assert srs is not None and srs.IsProjected(), "fixture should carry a projected CRS"
        _assert_orientation(
            var, False, "projected descending (UTM) -> keep", projected_descending_nc, "Band1"
        )

    def test_projected_ascending_is_flipped(self, projected_ascending_nc):
        """Projected + ascending Y is flipped, exactly like an ascending geographic latitude.

        Test scenario:
            A UTM-projected netCDF whose `y` axis ascends (row 0 = south) must be reversed on read.
            This fills the 2x2 cell that had no fixture — the geostationary granule was miscatalogued
            as the projected-ascending case, and it is neither. Note this case does **not** by itself
            discriminate the #705 fix: the old `gt[5] > 0` rule flipped a projected-ascending axis
            too. Only the geostationary case (raw ascends, scaled descends) separates the two rules.
        """
        var = NetCDF.read_file(projected_ascending_nc).get_variable("Band1")
        srs = var.raster.GetSpatialRef()
        assert srs is not None and srs.IsProjected(), "fixture should carry a projected CRS"
        _assert_orientation(
            var, True, "projected ascending (UTM) -> flip", projected_ascending_nc, "Band1"
        )


def _mdim_cube(lat_values, lon_values, bands=1):
    """An in-memory multidim store holding `v(time?, lat, lon)`, plus the array as written.

    GDAL's netCDF writer emits one 2-D variable per band, so a multi-band variable with a chosen
    coordinate direction cannot be produced by `gdal.Translate`; build the MDArray directly.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("m")
    rg = store.GetRootGroup()
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float32)
    band_dims = [rg.CreateDimension("time", None, None, bands)] if bands > 1 else []
    y_dim = rg.CreateDimension("lat", None, None, len(lat_values))
    x_dim = rg.CreateDimension("lon", None, None, len(lon_values))
    lat = rg.CreateMDArray("lat", [y_dim], dtype)
    lat.WriteArray(np.asarray(lat_values, "f4"))
    lon = rg.CreateMDArray("lon", [x_dim], dtype)
    lon.WriteArray(np.asarray(lon_values, "f4"))
    y_dim.SetIndexingVariable(lat)
    x_dim.SetIndexingVariable(lon)
    shape = ((bands,) if bands > 1 else ()) + (len(lat_values), len(lon_values))
    data = np.arange(int(np.prod(shape)), dtype="f4").reshape(shape)
    rg.CreateMDArray("v", band_dims + [y_dim, x_dim], dtype).WriteArray(data)
    return store, data


class TestMultiBandMaterialize:
    """A flipped multi-band cube round-trips through the NumPy re-flip in the materialize path."""

    def test_multiband_y_flip_survives_materialize(self):
        """A 3-D bottom-up variable reads the same before and after materializing.

        Test scenario:
            `_materialize_from_raw_view` flips a non-contiguous `[..., ::-1, :]` view of the whole
            band stack in one `WriteArray`. A band-ordering or stride bug there would corrupt every
            band but the first, which a 2-D fixture cannot catch.
        """
        path = "tests/data/netcdf/cf__7v__1d3-2d3-3d1__y-asc.nc"
        var = NetCDF.read_file(path).get_variable("tos")
        assert var._md_y_flipped is True, "the fixture's latitude ascends"
        before = np.asarray(var.read_array())
        assert before.ndim == 3 and before.shape[0] > 1, f"expected a band stack, got {before.shape}"
        np.testing.assert_array_equal(before, _north_up_reference(path, "tos"))
        var._materialize_md_view()
        np.testing.assert_array_equal(
            np.asarray(var.read_array()), before, err_msg="materialize changed a band"
        )
        assert var.raster.ReadAsArray(1, 1, 3, 2).shape[-2:] == (2, 3), "window read must work"

    def test_multiband_x_flip_matches_numpy_reference(self):
        """A 3-D east-to-west variable is reversed along columns, on every band."""
        store, data = _mdim_cube([4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0], bands=3)
        var = Container(store).get_variable("v")
        assert var._md_x_flipped is True and var._md_y_flipped is False
        expected = data[..., ::-1]
        np.testing.assert_array_equal(np.asarray(var.read_array()), expected)
        var._materialize_md_view()
        np.testing.assert_array_equal(
            np.asarray(var.read_array()), expected, err_msg="materialize changed a band"
        )

    def test_both_axes_flipped_materializes_correctly(self):
        """A cube stored south-to-north *and* east-to-west is reversed on both axes."""
        store, data = _mdim_cube([1.0, 2.0, 3.0, 4.0], [5.0, 4.0, 3.0, 2.0, 1.0], bands=2)
        var = Container(store).get_variable("v")
        assert var._md_y_flipped is True and var._md_x_flipped is True
        expected = data[..., ::-1, ::-1]
        np.testing.assert_array_equal(np.asarray(var.read_array()), expected)
        var._materialize_md_view()
        np.testing.assert_array_equal(np.asarray(var.read_array()), expected)
        assert var.geotransform[1] > 0 and var.geotransform[5] < 0, "must be north-up, west-east"


class _FakeDim:
    """A dimension whose indexing variable yields the given 1-D coordinate values."""

    def __init__(self, values):
        self._values = None if values is None else np.asarray(values)

    def GetIndexingVariable(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """Return a stand-in MDArray, or `None` when the dimension has no coordinate."""
        return None if self._values is None else _FakeIndexingVariable(self._values)


class _RaisingDim:
    """A dimension whose indexing variable cannot be read."""

    def GetIndexingVariable(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """Fail the way a GDAL SWIG call fails."""
        raise RuntimeError("no indexing variable")


class _UnscaledlessDim:
    """A dimension whose `GetUnscaled()` declines, leaving only raw values and a scale factor."""

    def __init__(self, raw_values, scale):
        self._raw = np.asarray(raw_values)
        self._scale = scale

    def GetIndexingVariable(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """Return the packed coordinate variable."""
        return self

    def GetDimensionCount(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """Coordinate variables are 1-D."""
        return 1

    def GetUnscaled(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """Decline, the way GDAL does when it cannot build the unscaled view."""
        return None

    def GetScale(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """The coordinate's `scale_factor`."""
        return self._scale

    def ReadAsArray(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """The raw, packed storage values."""
        return self._raw


class _GeostationaryView:
    """A classic-dataset stand-in carrying a geostationary CRS and a raw-order geotransform."""

    def __init__(self, geotransform):
        self._gt = geotransform

    def GetGeoTransform(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """The view's raw-derived geotransform."""
        return self._gt

    def GetSpatialRef(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """A geostationary spatial reference."""
        srs = osr.SpatialReference()
        srs.SetGEOS(-75.0, 35786023.0, 0.0, 0.0)
        return srs


class _FakeIndexingVariable:
    """A 1-D MDArray stand-in whose `GetUnscaled()` is the identity."""

    def __init__(self, values):
        self._values = values

    def GetDimensionCount(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """Coordinate variables are 1-D."""
        return 1

    def GetUnscaled(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """The values are already scaled."""
        return self

    def ReadAsArray(self):  # noqa: N802 - mirrors the GDAL SWIG API
        """Return the coordinate values."""
        return self._values


class TestScaledAxisAscends:
    """The orientation predicate reports `None` whenever the coordinate cannot settle the direction."""

    @pytest.mark.parametrize(
        "values, expected, label",
        [
            ([1.0, 2.0, 3.0], True, "ascending"),
            ([3.0, 2.0, 1.0], False, "descending"),
            ([5.0, 5.0, 5.0], None, "constant"),
            ([7.0], None, "size-1"),
            (None, None, "no indexing variable"),
            ([np.nan, 2.0, 3.0], None, "NaN first endpoint"),
            ([1.0, 2.0, np.nan], None, "NaN last endpoint"),
            ([np.inf, 2.0, 3.0], None, "infinite endpoint"),
        ],
        ids=["asc", "desc", "constant", "size-1", "no-coord", "nan-first", "nan-last", "inf"],
    )
    def test_direction_or_unknown(self, values, expected, label):
        """A non-finite endpoint must report `None`, not a direction.

        Test scenario:
            `NaN != NaN` is True and `NaN < x` is False, so an unguarded `first < last` classifies a
            NaN-tipped axis as descending — keeping a bottom-up Y (or reversing a west-to-east X) and
            silently mirroring the raster, the exact failure mode of #705.
        """
        result = NetCDF._scaled_axis_ascends([_FakeDim(values)], 0)
        assert result is expected, f"{label}: expected {expected}, got {result}"

    def test_unreadable_indexing_variable_reports_unknown(self):
        """A GDAL failure reading the indexing variable reports `None`, not a direction."""
        assert NetCDF._scaled_axis_ascends([_RaisingDim()], 0) is None

    @pytest.mark.parametrize(
        "raw, scale, expected",
        [
            ([0, 1, 2, 3], -5.6e-05, False),
            ([0, 1, 2, 3], 5.6e-05, True),
            ([3, 2, 1, 0], -5.6e-05, True),
            ([0, 1, 2, 3], None, True),
        ],
        ids=["negative-scale", "positive-scale", "negative-scale-desc-raw", "no-scale"],
    )
    def test_declined_unscaled_view_accounts_for_the_scale_sign(self, raw, scale, expected):
        """When `GetUnscaled()` declines, the raw order is corrected by the scale factor's sign.

        Test scenario:
            Reading the raw values as if they were scaled inverts the direction of a negatively-packed
            axis — precisely how a GOES scan angle, whose raw values ascend while the physical angle
            descends, got mirrored in #705.
        """
        result = NetCDF._scaled_axis_ascends([_UnscaledlessDim(raw, scale)], 0)
        assert result is expected, f"raw={raw} scale={scale}: expected {expected}, got {result}"


class TestGeostationaryFallbackNeverFlips:
    """With an unreadable coordinate, a geostationary axis must not be flipped on the raw gt sign."""

    def test_y_fallback_refuses_to_flip_a_geostationary_axis(self):
        """A raw-order `gt[5] > 0` would re-mirror a geostationary raster (#705); it must be ignored.

        Test scenario:
            GDAL derives the view's geotransform from the *raw* scan angles, which ascend under the
            negative `scale_factor`. If the scaled coordinate cannot be read, that positive `gt[5]`
            is the only signal left — and it is the wrong one, because the cube then adopts the
            classic driver's north-up metre geotransform.
        """
        view = _GeostationaryView((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert NetCDF._y_axis_is_bottom_up([_RaisingDim()], 0, view) is False

    def test_x_fallback_refuses_to_flip_a_geostationary_axis(self):
        """The mirror carve-out for columns."""
        view = _GeostationaryView((0.0, -1.0, 0.0, 0.0, 0.0, -1.0))
        assert NetCDF._x_axis_is_right_to_left([_RaisingDim()], 0, view) is False

    def test_non_geostationary_fallback_still_uses_the_geotransform_sign(self):
        """A plain raster with no usable coordinate still falls back to the geotransform sign."""
        view = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Float32)
        view.SetGeoTransform((0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        assert NetCDF._y_axis_is_bottom_up([_RaisingDim()], 0, view) is True
        assert NetCDF._x_axis_is_right_to_left([_RaisingDim()], 0, view) is True


class TestCorrectFlippedGeotransform:
    """The wrapper-side geotransform correction re-anchors whichever axis GDAL left describing the
    pre-flip order."""

    @staticmethod
    def _cube(gt, y_flipped, x_flipped):
        """A minimal stand-in carrying only what `_correct_flipped_geotransform` reads."""
        return SimpleNamespace(
            _geotransform=gt,
            _md_y_flipped=y_flipped,
            _md_x_flipped=x_flipped,
            _rows=4,
            _columns=5,
            _cell_size=None,
        )

    def test_y_flip_reanchors_origin_to_the_north(self):
        """A positive `gt[5]` after a Y flip is re-anchored to the north edge."""
        cube = self._cube((10.0, 2.0, 0.0, 0.0, 0.0, 1.0), y_flipped=True, x_flipped=False)
        NetCDF._correct_flipped_geotransform(cube)
        assert cube._geotransform == (10.0, 2.0, 0.0, 4.0, 0.0, -1.0)
        assert cube._cell_size == 2.0

    def test_x_flip_reanchors_origin_to_the_west(self):
        """A negative `gt[1]` after an X flip is re-anchored to the west edge.

        Test scenario:
            GDAL normally corrects a reversed view's geotransform itself, so this branch only fires
            when the reversed dimension had no indexing variable. Exercised directly because no
            fixture can reach it.
        """
        cube = self._cube((30.0, -2.0, 0.0, 10.0, 0.0, -1.0), y_flipped=False, x_flipped=True)
        NetCDF._correct_flipped_geotransform(cube)
        assert cube._geotransform == (20.0, 2.0, 0.0, 10.0, 0.0, -1.0)
        assert cube._cell_size == 2.0

    def test_both_axes_reanchored_together(self):
        """Both corrections compose into one north-up, west-to-east geotransform."""
        cube = self._cube((30.0, -2.0, 0.0, 0.0, 0.0, 1.0), y_flipped=True, x_flipped=True)
        NetCDF._correct_flipped_geotransform(cube)
        assert cube._geotransform == (20.0, 2.0, 0.0, 4.0, 0.0, -1.0)

    def test_already_north_up_is_untouched(self):
        """An already-correct geotransform is left alone, and `_cell_size` is not rewritten."""
        gt = (10.0, 2.0, 0.0, 4.0, 0.0, -1.0)
        cube = self._cube(gt, y_flipped=True, x_flipped=True)
        NetCDF._correct_flipped_geotransform(cube)
        assert cube._geotransform == gt
        assert cube._cell_size is None, "no change means no _cell_size write"

    def test_unflipped_cube_keeps_a_positive_y_pixel_size(self):
        """Without a recorded flip the geotransform is trusted as-is — the #705 guard."""
        gt = (10.0, 2.0, 0.0, 0.0, 0.0, 1.0)
        cube = self._cube(gt, y_flipped=False, x_flipped=False)
        NetCDF._correct_flipped_geotransform(cube)
        assert cube._geotransform == gt, "a geostationary cube must not be re-anchored"


class TestXAxisOrientation:
    """The X axis is normalized to `col 0 = west`, the mirror of `row 0 = north`."""

    def test_x_descending_is_flipped(self, x_descending_nc):
        """An east→west longitude is reversed, and the geotransform describes the real extent.

        Test scenario:
            The file spans lon 20→30 with its columns stored east-first. Reading it must return the
            columns west-first under a positive pixel width. Before the X rule, the coordinate-derived
            geotransform took `lon[0]` (29) for the west edge, so the raster came back mirrored
            west-east under a bbox shifted a full grid width east — silently wrong pixels, the same
            failure mode as #705.
        """
        var = NetCDF.read_file(x_descending_nc).get_variable("Band1")
        assert var._md_x_flipped is True, "east-to-west longitude must be reversed"
        assert var._md_y_flipped is False, "the latitude already descends; it must not be reversed"
        gt = var.geotransform
        assert gt[1] > 0, f"pixel width must be positive after the X flip, got {gt[1]}"
        assert var.bbox == pytest.approx([20.0, 6.0, 30.0, 10.0]), f"wrong extent: {var.bbox}"
        np.testing.assert_array_equal(
            np.asarray(var.read_array()),
            _north_up_reference(x_descending_nc, "Band1"),
            err_msg="col 0 of read_array() is not at the westernmost X coordinate",
        )

    def test_x_flip_survives_materialize(self, x_descending_nc):
        """The NumPy re-flip in `_materialize_from_raw_view` reproduces the lazy `GetView` flip."""
        var = NetCDF.read_file(x_descending_nc).get_variable("Band1")
        before = np.asarray(var.read_array())
        var._materialize_md_view()
        np.testing.assert_array_equal(
            np.asarray(var.read_array()), before, err_msg="materializing changed the pixels"
        )
        assert var.raster.ReadAsArray(1, 1, 3, 2).shape[-2:] == (2, 3), "window read must work"

    @pytest.mark.parametrize(
        "filename, variable",
        [(c[0], c[1]) for c in ORIENTATION_CASES],
        ids=[c[0].split(".")[0] for c in ORIENTATION_CASES],
    )
    def test_repo_fixtures_all_ascend_in_x(self, filename, variable):
        """Document the invariant: every on-disk fixture stores X west→east, so none is X-flipped."""
        var = NetCDF.read_file(f"tests/data/netcdf/{filename}").get_variable(variable)
        assert var._md_x_flipped is False, f"{filename} unexpectedly stores X east-to-west"


class TestDiskRoundTripOrientation:
    """Save to disk, reload, verify orientation preserved."""

    def test_orientation_after_disk_roundtrip(self, noah_nc, tmp_path):
        """Data orientation should be preserved after save → reload.

        Test scenario:
            Read Band1, save the container, reload, compare arrays.
        """
        var_orig = noah_nc.get_variable("Band1")
        arr_orig = var_orig.read_array(band=0)
        out = str(tmp_path / "orientation_test.nc")
        noah_nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        var_reloaded = reloaded.get_variable("Band1")
        arr_reloaded = var_reloaded.read_array(band=0)
        assert_allclose(
            arr_orig,
            arr_reloaded,
            rtol=1e-5,
            err_msg="Orientation changed after disk round-trip",
        )
