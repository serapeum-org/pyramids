"""Geolocation-array warp: `geolocate` + `geolocation` accessor (#1033)."""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import osr

from pyramids.base._errors import GeolocationArrayError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

CURV = "tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc"


def _make_geoloc_dataset(tmp_path, *, drop_x=False) -> Dataset:
    """Build a small raster carrying a GDAL GEOLOCATION domain of lon/lat arrays."""
    rows, cols = 4, 4
    lon = np.tile(np.linspace(-10.0, -7.0, cols), (rows, 1)).astype("float64")
    lat = np.tile(np.linspace(40.0, 37.0, rows).reshape(rows, 1), (1, cols))
    lon_path = tmp_path / "lon.tif"
    lat_path = tmp_path / "lat.tif"
    for arr, path in [(lon, lon_path), (lat.astype("float64"), lat_path)]:
        Dataset.from_array(
            arr,
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        ).to_file(str(path))
    data = np.arange(rows * cols).reshape(rows, cols).astype("float32")
    ds = Dataset.from_array(
        data,
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    domain = {
        "X_DATASET": str(lon_path),
        "Y_DATASET": str(lat_path),
        "X_BAND": "1",
        "Y_BAND": "1",
        "PIXEL_OFFSET": "0",
        "PIXEL_STEP": "1",
        "LINE_OFFSET": "0",
        "LINE_STEP": "1",
        "SRS": srs.ExportToWkt(),
    }
    if drop_x:
        del domain["X_DATASET"]
    ds.set_meta_data(domain, domain="GEOLOCATION")
    return ds


class TestGeolocationAccessor:
    """`geolocation` / `has_geolocation` on a base Dataset."""

    def test_plain_raster_has_none(self):
        """A plain raster has no geolocation arrays."""
        ds = Dataset.from_array(
            np.zeros((2, 2)),
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        )
        assert ds.geolocation is None
        assert ds.has_geolocation is False

    def test_accessor_returns_domain(self, tmp_path):
        """The accessor returns the GEOLOCATION domain dict when present."""
        ds = _make_geoloc_dataset(tmp_path)
        assert ds.has_geolocation is True
        domain = ds.geolocation
        assert domain["X_DATASET"].endswith("lon.tif")
        assert domain["Y_DATASET"].endswith("lat.tif")


class TestGeolocate:
    """`geolocate` on a base Dataset with a synthetic GEOLOCATION domain."""

    def test_geolocate_grids_to_requested_crs(self, tmp_path):
        """geolocate produces a north-up affine grid in the requested CRS."""
        ds = _make_geoloc_dataset(tmp_path)
        out = ds.geolocate(to_epsg=4326)
        assert type(out) is Dataset
        assert out.epsg == 4326
        gt = out.geotransform
        assert gt[5] < 0, f"not north-up: {gt}"
        assert gt[2] == 0, f"not axis-aligned: {gt}"
        assert gt[4] == 0, f"not axis-aligned: {gt}"

    def test_geolocate_to_epsg_none_uses_domain_srs(self, tmp_path):
        """With to_epsg=None the result adopts the domain's own SRS."""
        out = _make_geoloc_dataset(tmp_path).geolocate()
        assert out.epsg == 4326

    def test_geolocate_cell_size_honored(self, tmp_path):
        """A requested cell_size sets the output resolution."""
        out = _make_geoloc_dataset(tmp_path).geolocate(to_epsg=4326, cell_size=0.5)
        assert out.geotransform[1] == pytest.approx(0.5)

    def test_facade_parity(self, tmp_path):
        """`ds.geolocate` mirrors `ds.georef.geolocate`."""
        ds = _make_geoloc_dataset(tmp_path)
        assert ds.geolocate(to_epsg=4326).epsg == ds.georef.geolocate(to_epsg=4326).epsg

    def test_no_domain_raises(self):
        """geolocate on a plain raster raises GeolocationArrayError."""
        ds = Dataset.from_array(
            np.zeros((2, 2)),
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        )
        with pytest.raises(GeolocationArrayError, match="no geolocation arrays"):
            ds.geolocate()

    def test_missing_x_dataset_raises(self, tmp_path):
        """A domain missing X_DATASET raises GeolocationArrayError naming it."""
        ds = _make_geoloc_dataset(tmp_path, drop_x=True)
        with pytest.raises(GeolocationArrayError, match="X_DATASET"):
            ds.geolocate(to_epsg=4326)

    def test_bad_crs_raises(self, tmp_path):
        """An unrecognisable to_epsg raises (a ValueError subclass)."""
        ds = _make_geoloc_dataset(tmp_path)
        with pytest.raises(ValueError):
            ds.geolocate(to_epsg="not-a-crs")

    def test_bad_method_raises(self, tmp_path):
        """An unknown resampling method raises ValueError."""
        ds = _make_geoloc_dataset(tmp_path)
        with pytest.raises(ValueError):
            ds.geolocate(to_epsg=4326, method="bogus")

    def test_lazy_pins_source_on_base_dataset(self, tmp_path):
        """A lazy geolocate on a base Dataset pins its source and reads through."""
        out = _make_geoloc_dataset(tmp_path).geolocate(to_epsg=4326, lazy=True)
        assert out._warp_source is not None
        assert out.read_array().size > 0


class TestGeolocateNetCDF:
    """The classic-handle nuance: a NetCDF swath variable geolocates."""

    def test_variable_geolocates(self):
        """A NetCDF swath variable warps to the requested grid as a base Dataset."""
        from pyramids.netcdf import NetCDF

        var = NetCDF.read_file(CURV).get_variable("Tair")
        out = var.geolocate(to_epsg=4326)
        assert type(out) is Dataset
        assert out.epsg == 4326
        assert out.band_count == var.band_count

    def test_variable_geolocation_accessor(self):
        """The accessor reads the domain via the classic handle (multidim drops it)."""
        from pyramids.netcdf import NetCDF

        var = NetCDF.read_file(CURV).get_variable("Tair")
        domain = var.geolocation
        assert domain is not None
        assert "X_DATASET" in domain
        assert "Y_DATASET" in domain

    def test_variable_geolocate_lazy_pins_source(self):
        """A lazy geolocate pins its source and reads after the wrapper is dropped."""
        from pyramids.netcdf import NetCDF

        out = (
            NetCDF.read_file(CURV)
            .get_variable("Tair")
            .geolocate(to_epsg=4326, lazy=True)
        )
        assert out._warp_source is not None
        assert out.read_array().shape[0] == out.band_count

    def test_container_raises(self):
        """A NetCDF container (no variable extracted) has no geolocation arrays."""
        from pyramids.netcdf import NetCDF

        container = NetCDF.read_file(CURV)
        with pytest.raises(GeolocationArrayError):
            container.geolocate()

    def test_close_drops_geolocation_source_memo(self):
        """`close()` releases the reopened geolocation handle it memoised (#564 handle contract).

        `_geolocation_source` memoises a base `Dataset` over the classic `NETCDF:"<file>":<var>`
        handle. `close()` must drop and close it, otherwise that handle keeps the source file open
        past `close()`.
        """
        from pyramids.netcdf import NetCDF

        var = NetCDF.read_file(CURV).get_variable("Tair")
        assert var.geolocation is not None, (
            "precondition: fixture must carry geolocation arrays"
        )
        assert "_geolocation_source_memo" in var.__dict__, (
            "precondition: the memo must be populated"
        )
        memo = var.__dict__["_geolocation_source_memo"]
        var.close()
        assert "_geolocation_source_memo" not in var.__dict__, (
            "close() must drop the reopened geolocation-source handle so the source file is released"
        )
        assert memo._raster is None, (
            "close() must also close the memo's GDAL handle, not merely drop the reference"
        )


class TestGeolocateFromBytes:
    """An in-memory (`from_bytes` / `/vsimem`) swath variable exposes its GEOLOCATION domain (#1053).

    `from_bytes` is the only in-memory NetCDF reader that works on Windows/macOS, so it is the
    sanctioned way to read a downloaded L2 swath granule. Before #1053 the `/vsimem` exclusion in
    `_geolocation_source` left an in-memory swath reporting no geolocation arrays — `has_geolocation`
    False, `.geolocation` None, `.geolocate()` raising — even though the on-disk read works. The fix
    mirrors #1050: resolve the classic handle via `_vsimem_path` and admit `/vsimem`.
    """

    @staticmethod
    def _bytes():
        with open(CURV, "rb") as fh:
            return fh.read()

    def test_from_bytes_variable_has_geolocation(self):
        """An unnamed from_bytes swath variable exposes the GEOLOCATION domain, like on-disk."""
        from pyramids.netcdf import NetCDF

        on_disk = NetCDF.read_file(CURV).get_variable("Tair")
        var = NetCDF.from_bytes(self._bytes()).get_variable("Tair")
        assert on_disk.has_geolocation is True, "fixture must carry geolocation arrays"
        assert var.has_geolocation is True, (
            "an in-memory swath must expose its GEOLOCATION domain (#1053)"
        )
        domain = var.geolocation
        assert domain is not None
        assert "X_DATASET" in domain
        assert "Y_DATASET" in domain

    def test_named_from_bytes_variable_has_geolocation(self):
        """A named from_bytes read resolves through `_vsimem_path`, not the cosmetic name (#1053)."""
        from pyramids.netcdf import NetCDF

        container = NetCDF.from_bytes(self._bytes(), name="swath.nc")
        assert container.file_name == "swath.nc", (
            "the cosmetic name must shadow file_name for this test to exercise the fix"
        )
        var = container.get_variable("Tair")
        assert var.has_geolocation is True, (
            "a named in-memory swath must still expose geolocation — the name must not shadow "
            "the /vsimem path"
        )

    def test_from_bytes_variable_geolocates(self):
        """geolocate() warps an in-memory swath variable to the requested grid as a base Dataset."""
        from pyramids.netcdf import NetCDF

        var = NetCDF.from_bytes(self._bytes()).get_variable("Tair")
        out = var.geolocate(to_epsg=4326)
        assert type(out) is Dataset
        assert out.epsg == 4326
        assert out.band_count == var.band_count
