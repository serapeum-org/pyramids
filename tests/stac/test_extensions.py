"""Unit tests for pyramids.stac._extensions (proj/raster/eo readers, PB-1).

Pure dict readers that turn STAC projection/raster/eo extension fields into a
grid + band-metadata dict, with no asset file opened. Tests use raw STAC JSON
dicts (the duck-typed contract) and pystac-like objects via SimpleNamespace.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from pyramids.base._errors import StacAssetError
from pyramids.stac._extensions import (
    affine_to_geotransform,
    parse_number,
    read_extension_metadata,
)

pytestmark = pytest.mark.core


class TestParseNumber:
    """Tests for parse_number."""

    @pytest.mark.parametrize(
        "value, expected",
        [(-9999, -9999.0), (0, 0.0), (1.5, 1.5), ("-9999", -9999.0), ("3.14", 3.14)],
    )
    def test_numeric_inputs(self, value, expected):
        """Numbers and numeric strings coerce to float.

        Args:
            value: Raw field value.
            expected: Expected float.

        Test scenario:
            Ints, floats, and numeric strings all become floats.
        """
        assert parse_number(value) == expected, f"{value!r} -> {expected}"

    @pytest.mark.parametrize("token", ["nan", "NaN", "-nan"])
    def test_nan_strings(self, token):
        """nan-family strings parse to float nan.

        Args:
            token: A nan-encoding string (case-insensitive).

        Test scenario:
            The raster extension may encode nodata as the string 'nan'.
        """
        assert math.isnan(parse_number(token)), f"{token!r} should be nan"

    @pytest.mark.parametrize("token, sign", [("inf", 1), ("-inf", -1), ("Infinity", 1)])
    def test_inf_strings(self, token, sign):
        """inf-family strings parse to signed infinity.

        Args:
            token: An inf-encoding string.
            sign: Expected sign (+1/-1).

        Test scenario:
            'inf' / '-inf' / 'Infinity' map to the right infinity.
        """
        result = parse_number(token)
        assert math.isinf(result) and (result > 0) == (sign > 0), f"{token!r} wrong"

    def test_none_returns_default(self):
        """None yields the supplied default.

        Test scenario:
            A missing field falls back to the default sentinel.
        """
        assert parse_number(None, default=42.0) == pytest.approx(42.0)

    def test_unparseable_returns_default(self):
        """An unparseable string yields the default.

        Test scenario:
            'n/a' cannot be parsed, so the default is returned.
        """
        assert parse_number("n/a", default=0.0) == pytest.approx(0.0)

    def test_bool_returns_default(self):
        """Booleans are not treated as numbers.

        Test scenario:
            True/False return the default rather than 1.0/0.0.
        """
        assert parse_number(True, default=-1.0) == -1.0


class TestAffineToGeotransform:
    """Tests for affine_to_geotransform."""

    def test_six_element_reorder(self):
        """A 6-element affine reorders to the GDAL geotransform.

        Test scenario:
            ``[a,b,c,d,e,f]`` -> ``(c,a,b,f,d,e)``.
        """
        result = affine_to_geotransform([30.0, 0.0, 224985.0, 0.0, -30.0, 6790215.0])
        assert result == (224985.0, 30.0, 0.0, 6790215.0, 0.0, -30.0), f"got {result}"

    def test_nine_element_drops_last_row(self):
        """A 9-element affine ignores the trailing [0,0,1] row.

        Test scenario:
            Only the first six coefficients are used.
        """
        result = affine_to_geotransform([10, 0, 100, 0, -10, 200, 0, 0, 1])
        assert result == (100.0, 10.0, 0.0, 200.0, 0.0, -10.0), f"got {result}"

    def test_too_short_raises(self):
        """Fewer than six coefficients raises ValueError.

        Test scenario:
            A 4-element transform is rejected.
        """
        with pytest.raises(ValueError, match="at least 6 coefficients"):
            affine_to_geotransform([1, 2, 3, 4])


class TestReadExtensionMetadata:
    """Tests for read_extension_metadata."""

    def _s2_item(self):
        """A Sentinel-2-style item dict with proj/raster/eo on the asset."""
        return {
            "properties": {"proj:epsg": 32633},
            "assets": {
                "B04": {
                    "href": "s3://b/B04.tif",
                    "proj:shape": [10980, 10980],
                    "proj:transform": [10.0, 0.0, 600000.0, 0.0, -10.0, 5300040.0],
                    "raster:bands": [{"nodata": 0, "scale": 0.0001, "offset": 0.0}],
                    "eo:bands": [{"name": "B04", "common_name": "red"}],
                }
            },
        }

    def test_full_metadata(self):
        """All fields populate from item + asset.

        Test scenario:
            crs from item-level epsg, geotransform from the asset transform,
            shape/raster_bands/eo_bands/band_names from the asset.
        """
        meta = read_extension_metadata(self._s2_item(), "B04")
        assert meta["epsg"] == 32633, f"epsg: {meta['epsg']}"
        assert meta["crs"] == "EPSG:32633", f"crs: {meta['crs']}"
        assert meta["geotransform"] == (600000.0, 10.0, 0.0, 5300040.0, 0.0, -10.0)
        assert meta["shape"] == [10980, 10980], f"shape: {meta['shape']}"
        assert meta["raster_bands"][0]["nodata"] == 0
        assert meta["band_names"] == ["B04"], f"band_names: {meta['band_names']}"

    def test_proj_code_preferred_over_epsg(self):
        """proj:code (v2) is used verbatim and supersedes proj:epsg.

        Test scenario:
            An asset carrying proj:code returns that string as crs.
        """
        item = {
            "assets": {
                "a": {"href": "x.tif", "proj:code": "EPSG:3035", "proj:epsg": 4326}
            }
        }
        assert read_extension_metadata(item, "a")["crs"] == "EPSG:3035"

    def test_asset_overrides_item(self):
        """An asset-level field overrides the item-level value.

        Test scenario:
            Asset proj:epsg=3857 wins over item proj:epsg=4326.
        """
        item = {
            "properties": {"proj:epsg": 4326},
            "assets": {"dem": {"href": "x.tif", "proj:epsg": 3857}},
        }
        assert read_extension_metadata(item, "dem")["epsg"] == 3857

    def test_no_transform_yields_none_geotransform(self):
        """A missing proj:transform leaves geotransform None.

        Test scenario:
            crs present but no transform -> geotransform is None.
        """
        item = {"assets": {"a": {"href": "x.tif", "proj:epsg": 4326}}}
        assert read_extension_metadata(item, "a")["geotransform"] is None

    def test_bare_asset_without_extensions(self):
        """A bare asset (asset_key=None) with no extension fields is all-empty.

        Test scenario:
            crs/geotransform/raster_bands are None.
        """
        meta = read_extension_metadata({"href": "x.tif"})
        assert (meta["crs"], meta["geotransform"], meta["raster_bands"]) == (
            None,
            None,
            None,
        )

    def test_band_names_none_when_incomplete(self):
        """band_names is None when any eo band lacks a name/common_name.

        Test scenario:
            One band has neither name nor common_name -> band_names None.
        """
        item = {
            "assets": {
                "a": {
                    "href": "x.tif",
                    "eo:bands": [{"name": "B01"}, {"center_wavelength": 0.5}],
                }
            }
        }
        assert read_extension_metadata(item, "a")["band_names"] is None

    def test_pystac_like_asset_extra_fields(self):
        """A pystac-like asset exposes extension fields via extra_fields.

        Test scenario:
            proj:epsg in extra_fields is read for an attribute-style asset.
        """
        asset = SimpleNamespace(href="x.tif", extra_fields={"proj:epsg": 25832})
        item = SimpleNamespace(properties={}, assets={"a": asset})
        assert read_extension_metadata(item, "a")["crs"] == "EPSG:25832"

    def test_missing_asset_raises(self):
        """A missing asset key raises StacAssetError.

        Test scenario:
            Reading metadata for an absent asset surfaces the shared error.
        """
        with pytest.raises(StacAssetError, match="not found"):
            read_extension_metadata({"assets": {"a": {"href": "x"}}}, "missing")
