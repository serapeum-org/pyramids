"""Unit tests for pyramids.stac._loader (STAC asset → Dataset/NetCDF dispatch)."""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF
from pyramids.stac import _loader
from pyramids.stac._loader import _engine_for, _resolve_asset, load_asset, which_engine
from pyramids.stac.signers import AWSRequesterPaysSigner

pytestmark = pytest.mark.core

_GEOTIFF = "tests/data/geotiff/era5_land_monthly_averaged.tif"
_NETCDF = "tests/data/netcdf/noah-precipitation-1979.nc"
_COG_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


class _AppendSigner:
    """Tiny non-mock signer that records the href and supplies a GDAL env."""

    def __init__(self, suffix: str = "", env: dict[str, str] | None = None):
        self.suffix = suffix
        self.seen: str | None = None
        self._env = env or {}

    def sign_href(self, href: str) -> str:
        """Record the href and append the configured suffix."""
        self.seen = href
        return f"{href}{self.suffix}"

    def gdal_env(self) -> dict[str, str]:
        """Return the configured GDAL config mapping (empty by default)."""
        return dict(self._env)


@pytest.fixture
def grib_asset(tmp_path):
    """Write a 1-band GRIB2 and return a raw STAC asset dict pointing at it.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        dict: `{"href": <path>, "type": "application/wmo-grib2"}`.
    """
    mem = gdal.GetDriverByName("MEM").Create("", 6, 4, 1, gdal.GDT_Float32)
    mem.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    mem.SetProjection(sr.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(np.full((4, 6), 280.0, "float32"))
    path = tmp_path / "x.grib2"
    dst = gdal.GetDriverByName("GRIB").CreateCopy(str(path), mem)
    dst.FlushCache()
    dst = None
    return {"href": str(path), "type": "application/wmo-grib2"}


class TestEngineFor:
    """Tests for _engine_for dispatch."""

    @pytest.mark.parametrize(
        "media_type, href, expected",
        [
            (_COG_TYPE, "x.tif", "gdal"),
            ("image/geotiff", "x", "gdal"),
            ("application/x-netcdf", "x", "netcdf"),
            ("application/netcdf", "x", "netcdf"),
            ("application/wmo-grib2", "x", "grib"),
            ("application/vnd+zarr", "x", "zarr"),
        ],
    )
    def test_media_type_wins(self, media_type, href, expected):
        """media_type selects the engine regardless of extension.

        Args:
            media_type: The asset media type.
            href: An href whose extension is intentionally uninformative.
            expected: Expected engine name.

        Test scenario:
            Each recognised media type maps to its reader.
        """
        assert _engine_for(media_type, href) == expected, f"{media_type} misrouted"

    @pytest.mark.parametrize(
        "href, expected",
        [
            ("s3://b/scene.tif", "gdal"),
            ("s3://b/scene.TIFF", "gdal"),
            ("https://h/x.nc", "netcdf"),
            ("https://h/gfs.f000.grib2", "grib"),
            ("https://h/x.grib", "grib"),
            ("s3://b/cube.zarr", "zarr"),
            ("s3://b/cube.zarr/", "zarr"),
            ("https://h/x.tif?token=abc", "gdal"),
        ],
    )
    def test_extension_fallback(self, href, expected):
        """With no media type, the href extension picks the engine.

        Args:
            href: Asset href (may carry a query string or trailing slash).
            expected: Expected engine name.

        Test scenario:
            Extension fallback ignores query strings and trailing slashes.
        """
        assert _engine_for(None, href) == expected, f"{href} misrouted"

    def test_unknown_raises(self):
        """An unrecognised type and extension raise ValueError.

        Test scenario:
            Neither media type nor extension identifies a reader.
        """
        with pytest.raises(ValueError, match="Cannot determine a reader"):
            _engine_for(None, "s3://b/data.unknown")


class TestResolveAsset:
    """Tests for _resolve_asset."""

    def test_bare_asset_dict(self):
        """A bare asset dict resolves to its href and type.

        Test scenario:
            asset_key=None treats the input as the asset itself.
        """
        href, mt = _resolve_asset({"href": "x.tif", "type": "image/tiff"}, None)
        assert (href, mt) == ("x.tif", "image/tiff")

    def test_item_with_asset_key(self):
        """An Item + key resolves the named asset.

        Test scenario:
            The asset under `assets[key]` is returned.
        """
        item = {"assets": {"B04": {"href": "b04.tif", "type": _COG_TYPE}}}
        href, mt = _resolve_asset(item, "B04")
        assert href == "b04.tif" and mt == _COG_TYPE

    def test_attribute_style_asset(self):
        """A pystac-like object exposing .href/.media_type is supported.

        Test scenario:
            getattr access path (not dict) resolves href + media_type.
        """
        from types import SimpleNamespace

        asset = SimpleNamespace(href="a.nc", media_type="application/x-netcdf")
        item = SimpleNamespace(assets={"data": asset})
        href, mt = _resolve_asset(item, "data")
        assert href == "a.nc" and mt == "application/x-netcdf"

    def test_missing_asset_raises_keyerror(self):
        """A missing asset key raises KeyError listing what's available.

        Test scenario:
            The requested key is absent from the item's assets.
        """
        with pytest.raises(KeyError, match="not found"):
            _resolve_asset({"assets": {"a": {"href": "x"}}}, "missing")

    def test_asset_without_href_raises_keyerror(self):
        """An asset lacking an href raises KeyError.

        Test scenario:
            The asset exists but carries no href.
        """
        with pytest.raises(KeyError, match="no 'href'"):
            _resolve_asset({"assets": {"a": {"type": "image/tiff"}}}, "a")


class TestWhichEngine:
    """Tests for which_engine."""

    def test_cog_item(self):
        """A COG asset reports the gdal engine.

        Test scenario:
            `which_engine` mirrors `_engine_for` for a real asset dict.
        """
        assert which_engine({"href": "s3://b/x.tif", "type": _COG_TYPE}) == "gdal"

    def test_zarr_by_media_type(self):
        """A Zarr asset reports the zarr engine without opening it.

        Test scenario:
            Zarr is recognised by media type (no .zarr fixture needed).
        """
        assert (
            which_engine({"href": "s3://b/c.zarr", "type": "application/zarr"})
            == "zarr"
        )


class TestLoadAsset:
    """Tests for load_asset."""

    def test_loads_geotiff_as_dataset(self):
        """A GeoTIFF asset opens as a Dataset.

        Test scenario:
            The local era5 GeoTIFF fixture loads with its 9 bands.
        """
        ds = load_asset({"href": _GEOTIFF, "type": "image/tiff"})
        assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds).__name__}"
        assert ds.band_count == 9, f"Expected 9 bands, got {ds.band_count}"

    def test_loads_netcdf_as_netcdf(self):
        """A NetCDF asset opens as a NetCDF container.

        Test scenario:
            An Item + key pointing at the noah NetCDF yields a NetCDF.
        """
        item = {"assets": {"v": {"href": _NETCDF, "type": "application/x-netcdf"}}}
        result = load_asset(item, "v")
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_loads_grib_via_open_grib(self, grib_asset):
        """A GRIB2 asset opens through open_grib as a Dataset.

        Args:
            grib_asset: Fixture providing a GRIB2 asset dict.

        Test scenario:
            The GRIB path routes to open_grib and returns a Dataset.
        """
        ds = load_asset(grib_asset)
        assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds).__name__}"
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"

    def test_extension_fallback_loads(self):
        """An asset with no media type loads via extension dispatch.

        Test scenario:
            A `.tif` href and no type still opens as a Dataset.
        """
        ds = load_asset({"href": _GEOTIFF})
        assert isinstance(ds, Dataset), "Extension fallback should open the GeoTIFF"

    def test_signer_rewrites_href(self):
        """The signer's sign_href is applied before opening.

        Test scenario:
            A fake signer records the original href; an identity-suffix
            keeps the path openable.
        """
        signer = _AppendSigner(suffix="")
        load_asset({"href": _GEOTIFF, "type": "image/tiff"}, signer=signer)
        assert signer.seen == _GEOTIFF, f"signer did not see the href: {signer.seen}"

    def test_signer_gdal_env_active_during_open(self, monkeypatch):
        """The signer's gdal_env is installed as GDAL config while opening.

        Test scenario:
            A reader stub captures `AWS_REQUEST_PAYER` at call time; with an
            `AWSRequesterPaysSigner` it must read `requester`.
        """
        captured: dict[str, str | None] = {}

        def fake_read_file(href, vsi=None):
            captured["payer"] = gdal.GetConfigOption("AWS_REQUEST_PAYER")
            return "DS"

        monkeypatch.setattr(_loader.Dataset, "read_file", staticmethod(fake_read_file))
        load_asset(
            {"href": "s3://usgs-landsat/x.tif", "type": "image/tiff"},
            signer=AWSRequesterPaysSigner(),
        )
        assert captured["payer"] == "requester", (
            f"signer gdal_env not active during open: {captured['payer']}"
        )

    def test_signer_gdal_env_restored_after_open(self):
        """The signer's GDAL config is torn down once the asset is opened.

        Test scenario:
            After a real load with an AWSRequesterPaysSigner, the global
            `AWS_REQUEST_PAYER` option is back to `None`.
        """
        assert gdal.GetConfigOption("AWS_REQUEST_PAYER") is None, "precondition: unset"
        load_asset({"href": _GEOTIFF, "type": "image/tiff"}, signer=AWSRequesterPaysSigner())
        assert gdal.GetConfigOption("AWS_REQUEST_PAYER") is None, "config not restored after open"

    def test_no_signer_applies_no_env(self, monkeypatch):
        """Without a signer, no extra GDAL config is set during the open.

        Test scenario:
            A reader stub sees `AWS_REQUEST_PAYER` unset when no signer is
            supplied.
        """
        captured: dict[str, str | None] = {}

        def fake_read_file(href, vsi=None):
            captured["payer"] = gdal.GetConfigOption("AWS_REQUEST_PAYER")
            return "DS"

        monkeypatch.setattr(_loader.Dataset, "read_file", staticmethod(fake_read_file))
        load_asset({"href": _GEOTIFF, "type": "image/tiff"})
        assert captured["payer"] is None, f"unexpected env without signer: {captured['payer']}"

    def test_signer_applies_both_sign_href_and_gdal_env(self, monkeypatch):
        """Both signer hooks fire: href rewrite and gdal_env install.

        Test scenario:
            An _AppendSigner with a custom env records the href AND its env
            option is active at open time.
        """
        captured: dict[str, str | None] = {}

        def fake_read_file(href, vsi=None):
            captured["href"] = href
            captured["sentinel"] = gdal.GetConfigOption("CPL_CURL_VERBOSE")
            return "DS"

        monkeypatch.setattr(_loader.Dataset, "read_file", staticmethod(fake_read_file))
        signer = _AppendSigner(suffix="?sig=x", env={"CPL_CURL_VERBOSE": "YES"})
        load_asset({"href": "s3://b/x.tif", "type": "image/tiff"}, signer=signer)
        assert signer.seen == "s3://b/x.tif", f"sign_href not called: {signer.seen}"
        assert captured["href"] == "s3://b/x.tif?sig=x", f"signed href not used: {captured['href']}"
        assert captured["sentinel"] == "YES", f"gdal_env not applied: {captured['sentinel']}"

    def test_missing_asset_raises(self):
        """Loading a missing asset raises KeyError.

        Test scenario:
            The requested asset key is absent.
        """
        with pytest.raises(KeyError):
            load_asset({"assets": {}}, "nope")

    def test_unknown_type_raises(self):
        """An asset with no recognisable type/extension raises ValueError.

        Test scenario:
            `data.bin` matches no reader.
        """
        with pytest.raises(ValueError, match="Cannot determine a reader"):
            load_asset({"href": "s3://b/data.bin"})
