"""Regression tests for geostationary CRS reporting on read (issue #706).

A geostationary (GOES/Himawari/MTG) fixed-grid projection is a custom CRS with
no EPSG authority code, so ``NetCDF.get_variable(...).epsg`` must report
``None`` rather than the misleading ``4326`` that the generic ``epsg_from_wkt``
fallback would otherwise produce, while ``.crs`` keeps the full geostationary
WKT and ``to_crs`` still reprojects without a manual ``set_crs``.

The fixture is built on the fly through GDAL's multidimensional (MDIM) API — a
minimal GOES-style granule (radian ``x`` / ``y`` scan angles + a
``goes_imager_projection`` grid mapping) — so the test needs no committed binary
and no network.
"""

import os

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF
from tests._marks import requires_lazy

pytestmark = pytest.mark.core

_GEOS_LON_0 = -75.0


def _attr_str(arr, name, value):
    """Write a scalar string attribute onto an MDIM array."""
    arr.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString()).Write(value)


def _attr_f64(arr, name, value):
    """Write a scalar float64 attribute onto an MDIM array."""
    dt = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    arr.CreateAttribute(name, [], dt).Write(float(value))


def _write_geostationary_mdim(path: str, ny: int = 120, nx: int = 150, n_time: int | None = None) -> None:
    """Write a minimal GOES-style geostationary NetCDF via the GDAL MDIM API.

    The ``x`` / ``y`` coordinates are packed ``int16`` scan angles in radians
    (``scale_factor`` / ``add_offset``); ``goes_imager_projection`` carries the
    CF ``grid_mapping_name = "geostationary"`` parameters; ``CMI_C02`` is the
    data variable that references them. When ``n_time`` is set, a leading
    ``time`` dimension is added so the variable has a non-spatial axis to reduce.
    """
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    rg = ds.GetRootGroup()
    dy = rg.CreateDimension("y", "", "", ny)
    dx = rg.CreateDimension("x", "", "", nx)
    i16 = gdal.ExtendedDataType.Create(gdal.GDT_Int16)
    if n_time is not None:
        dt = rg.CreateDimension("time", "", "", n_time)
        tv = rg.CreateMDArray("time", [dt], gdal.ExtendedDataType.Create(gdal.GDT_Int32))
        tv.Write(np.arange(n_time, dtype=np.int32))
        _attr_str(tv, "standard_name", "time")
        _attr_str(tv, "units", "days since 2024-01-01")

    x = rg.CreateMDArray("x", [dx], i16)
    x.Write(np.arange(nx, dtype=np.int16))
    _attr_str(x, "units", "rad")
    _attr_str(x, "axis", "X")
    _attr_str(x, "standard_name", "projection_x_coordinate")
    _attr_f64(x, "scale_factor", 2.8e-05)
    _attr_f64(x, "add_offset", -0.045)

    y = rg.CreateMDArray("y", [dy], i16)
    y.Write(np.arange(ny, dtype=np.int16))
    _attr_str(y, "units", "rad")
    _attr_str(y, "axis", "Y")
    _attr_str(y, "standard_name", "projection_y_coordinate")
    _attr_f64(y, "scale_factor", -2.8e-05)
    _attr_f64(y, "add_offset", 0.065)

    gp = rg.CreateMDArray("goes_imager_projection", [], gdal.ExtendedDataType.Create(gdal.GDT_Int32))
    gp.Write(np.array(0, dtype=np.int32))
    _attr_str(gp, "grid_mapping_name", "geostationary")
    _attr_f64(gp, "perspective_point_height", 35786023.0)
    _attr_f64(gp, "semi_major_axis", 6378137.0)
    _attr_f64(gp, "semi_minor_axis", 6356752.31414)
    _attr_f64(gp, "inverse_flattening", 298.2572221)
    _attr_f64(gp, "latitude_of_projection_origin", 0.0)
    _attr_f64(gp, "longitude_of_projection_origin", _GEOS_LON_0)
    _attr_str(gp, "sweep_angle_axis", "x")

    cmi_dims = [dt, dy, dx] if n_time is not None else [dy, dx]
    cmi_shape = (n_time, ny, nx) if n_time is not None else (ny, nx)
    cmi = rg.CreateMDArray("CMI_C02", cmi_dims, gdal.ExtendedDataType.Create(gdal.GDT_UInt16))
    cmi.Write(np.zeros(cmi_shape, dtype=np.uint16))
    _attr_str(cmi, "grid_mapping", "goes_imager_projection")
    _attr_str(cmi, "coordinates", "y x")
    _attr_f64(cmi, "scale_factor", 0.00031746)
    _attr_f64(cmi, "add_offset", 0.0)


@pytest.fixture
def geos_cube(tmp_path) -> NetCDF:
    """Read ``CMI_C02`` from a freshly MDIM-written geostationary granule."""
    path = str(tmp_path / "synthetic_geos.nc")
    _write_geostationary_mdim(path)
    return NetCDF.read_file(path).get_variable("CMI_C02")


class TestGeostationaryCRS:
    """A geostationary variable reports no EPSG code but a usable CRS (#706)."""

    def test_epsg_is_none_not_4326(self, geos_cube: NetCDF):
        """`.epsg` is `None` — a geostationary CRS has no EPSG code, so the old
        misleading `4326` must not be reported."""
        assert geos_cube.epsg is None

    def test_crs_is_geostationary(self, geos_cube: NetCDF):
        """`.crs` keeps the full geostationary WKT."""
        assert "Geostationary_Satellite" in geos_cube.crs

    def test_is_geostationary_flag(self, geos_cube: NetCDF):
        """The dataset is detected as geostationary on read."""
        assert geos_cube._is_geostationary() is True

    def test_central_meridian_is_sub_satellite_longitude(self, geos_cube: NetCDF):
        """The reconstructed CRS carries the sub-satellite longitude."""
        srs = geos_cube.raster.GetSpatialRef()
        assert srs.GetProjParm("central_meridian", 999.0) == pytest.approx(_GEOS_LON_0)

    def test_to_crs_works_without_manual_set_crs(self, geos_cube: NetCDF):
        """`to_crs(4326)` reprojects straight from the read CRS — no hand-built
        WKT, no `set_crs` — and yields a non-degenerate extent."""
        warped = geos_cube.to_crs(4326)
        minx, miny, maxx, maxy = warped.bbox
        assert maxx - minx > 0.1, f"degenerate width: {warped.bbox}"
        assert maxy - miny > 0.1, f"degenerate height: {warped.bbox}"


class TestGeostationaryContainerOps:
    """Container operations preserve the geostationary CRS via WKT (#706).

    A geostationary variable has ``.epsg is None``; the container fan-out that
    rebuilds each variable through ``create_from_array(epsg=...)`` must carry the
    CRS through the WKT instead of crashing on the missing EPSG code.
    """

    def test_container_resample_preserves_geostationary_crs(self, tmp_path):
        """`container.resample` keeps the geostationary CRS (no `epsg cannot be None`)."""
        path = str(tmp_path / "geos.nc")
        _write_geostationary_mdim(path)
        out = NetCDF.read_file(path).resample(cell_size=4000.0).get_variable("CMI_C02")
        assert out.epsg is None
        assert "Geostationary_Satellite" in out.crs

    def test_container_reduce_preserves_geostationary_crs(self, tmp_path):
        """`container.reduce` over a non-spatial dim keeps the geostationary CRS."""
        path = str(tmp_path / "geos.nc")
        _write_geostationary_mdim(path, n_time=3)
        reduced = NetCDF.read_file(path).reduce(dim="time", how="mean")
        out = reduced.get_variable("CMI_C02")
        assert out.epsg is None
        assert "Geostationary_Satellite" in out.crs

    def test_polygonize_preserves_geostationary_crs(self, tmp_path):
        """`to_polygons` builds its scratch raster without an epsg-None crash."""
        path = str(tmp_path / "geos.nc")
        _write_geostationary_mdim(path)
        var = NetCDF.read_file(path).get_variable("CMI_C02")
        gdf = var.to_polygons()
        assert gdf is not None and "geometry" in gdf.columns

    @requires_lazy
    def test_to_zarr_writes_geostationary_wkt(self, tmp_path):
        """`to_zarr` writes without an `int(None)` crash and stores the geostationary WKT."""
        import glob

        path = str(tmp_path / "geos.nc")
        _write_geostationary_mdim(path)
        var = NetCDF.read_file(path).get_variable("CMI_C02")
        store = str(tmp_path / "geos.zarr")
        var.to_zarr(store)
        assert os.path.exists(store)
        # the geostationary CRS is preserved in the store's `spatial_ref` metadata
        meta = glob.glob(os.path.join(store, "**", "*.json"), recursive=True) + glob.glob(
            os.path.join(store, "**", ".z*"), recursive=True
        )
        assert any("Geostationary_Satellite" in open(f, encoding="utf-8").read() for f in meta)


    def test_set_variable_preserves_geostationary_crs(self, tmp_path):
        """`set_variable` carries the geostationary WKT instead of erasing it."""
        path = str(tmp_path / "geos.nc")
        _write_geostationary_mdim(path)
        container = NetCDF.read_file(path)
        container.set_variable("CMI_copy", container.get_variable("CMI_C02"))
        copy = container.get_variable("CMI_copy")
        assert copy.epsg is None
        assert "Geostationary_Satellite" in copy.crs

    def test_bounds_carries_geostationary_crs(self, tmp_path):
        """`.bounds` attaches the geostationary CRS rather than a CRS-less frame."""
        path = str(tmp_path / "geos.nc")
        _write_geostationary_mdim(path)
        var = NetCDF.read_file(path).get_variable("CMI_C02")
        assert var.bounds.crs is not None

    def test_aligner_rejects_geostationary_reference(self, tmp_path):
        """Constructing an `Aligner` on a no-EPSG reference raises a clear error."""
        from pyramids.dataset.ops.reproject import Aligner

        path = str(tmp_path / "geos.nc")
        _write_geostationary_mdim(path)
        var = NetCDF.read_file(path).get_variable("CMI_C02")
        with pytest.raises(ValueError, match=r"no EPSG code"):
            Aligner(var)


class TestNonGeostationaryEpsgUnaffected:
    """The `None`-for-geostationary rule must not touch ordinary CRSs."""

    def test_latlon_epsg_stays_4326(self):
        """A plain lat/lon NetCDF still reports its EPSG code."""
        arr = np.zeros((5, 6), "f4")
        nc = NetCDF.create_from_array(arr, geo=(0, 1, 0, 5, 0, -1), epsg=4326, variable_name="t")
        assert nc.get_variable("t").epsg == 4326

    def test_latlon_container_resample_unchanged(self):
        """A plain lat/lon container still resamples and keeps its EPSG code."""
        arr = np.zeros((6, 7), "f4")
        nc = NetCDF.create_from_array(arr, geo=(0, 1, 0, 6, 0, -1), epsg=4326, variable_name="t")
        assert nc.resample(cell_size=2.0).get_variable("t").epsg == 4326
