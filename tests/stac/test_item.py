"""Unit tests for pyramids.stac._item (shared duck-typed STAC accessors, H1/M5).

These accessors back both :func:`pyramids.stac._loader._resolve_asset` and
:func:`pyramids.dataset._stac._resolve_asset_href`, so they must interpret a
STAC Item / Asset identically whether it is a raw JSON dict or a pystac-like
object (attribute access + ``extra_fields``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyramids.base._errors import StacAssetError, StacError, UnsupportedAssetError
from pyramids.stac._item import (
    asset_field,
    asset_href,
    asset_media_type,
    get_asset,
    get_assets,
    item_bbox,
    item_id,
    item_properties,
)

pytestmark = pytest.mark.core


class TestItemId:
    """Tests for item_id."""

    def test_dict_id(self):
        """A dict item's id is read from the ``id`` key.

        Test scenario:
            ``{"id": "scene-1"}`` → ``"scene-1"``.
        """
        assert item_id({"id": "scene-1"}) == "scene-1"

    def test_attribute_id(self):
        """A pystac-like item exposes ``.id``.

        Test scenario:
            An object with an ``id`` attribute resolves to it.
        """
        assert item_id(SimpleNamespace(id="abc")) == "abc"

    def test_missing_id_placeholder(self):
        """A missing id falls back to ``"?"``.

        Test scenario:
            A dict without ``id`` yields the placeholder.
        """
        assert item_id({"assets": {}}) == "?"


class TestItemProperties:
    """Tests for item_properties."""

    def test_dict_properties(self):
        """Item-level extension fields come from ``properties``.

        Test scenario:
            ``proj:epsg`` under ``properties`` is returned.
        """
        props = item_properties({"properties": {"proj:epsg": 32633}})
        assert props["proj:epsg"] == 32633, f"unexpected properties: {props}"

    def test_attribute_properties(self):
        """A pystac-like item exposes ``.properties``.

        Test scenario:
            The attribute mapping is returned as-is.
        """
        assert item_properties(SimpleNamespace(properties={"a": 1})) == {"a": 1}

    def test_missing_properties_empty(self):
        """An item without properties yields an empty mapping.

        Test scenario:
            ``{"id": "x"}`` → ``{}``.
        """
        assert item_properties({"id": "x"}) == {}


class TestItemBbox:
    """Tests for item_bbox."""

    def test_dict_bbox(self):
        """A raw JSON item exposes its bbox under the ``bbox`` key.

        Test scenario:
            A 2D ``[west, south, east, north]`` box is returned verbatim.
        """
        assert item_bbox({"bbox": [1.0, 2.0, 3.0, 4.0]}) == [1.0, 2.0, 3.0, 4.0]

    def test_attribute_bbox(self):
        """A pystac-like item exposes ``.bbox``.

        Test scenario:
            The attribute sequence is returned as-is.
        """
        item = SimpleNamespace(bbox=[10.0, 20.0, 11.0, 21.0])
        assert item_bbox(item) == [10.0, 20.0, 11.0, 21.0], "attribute bbox not read"

    def test_missing_bbox_none(self):
        """An item without a bbox yields None rather than raising.

        Test scenario:
            ``{"id": "x"}`` → ``None`` (callers treat this as "unknown extent").
        """
        assert item_bbox({"id": "x"}) is None, "a bbox-less item should give None"

    def test_three_dimensional_bbox_passthrough(self):
        """A 6-element 3D bbox is returned untouched.

        Test scenario:
            The accessor locates the bbox; interpreting its length is the
            caller's job (``_horizontal_bounds``).
        """
        box = [1.0, 2.0, 0.0, 3.0, 4.0, 100.0]
        assert item_bbox({"bbox": box}) == box, "3D bbox should pass through intact"

    def test_empty_bbox_passthrough(self):
        """An explicitly empty bbox is returned as-is, not coerced to None.

        Test scenario:
            ``[]`` is falsy but present; callers decide what that means.
        """
        assert item_bbox({"bbox": []}) == [], "an empty bbox should pass through"

    def test_non_dict_without_attribute_is_none(self):
        """An object with neither ``.bbox`` nor dict access yields None.

        Test scenario:
            A bare object is tolerated (duck typing, no AttributeError).
        """
        assert item_bbox(object()) is None, "an unrelated object should give None"


class TestGetAssets:
    """Tests for get_assets."""

    def test_dict_assets(self):
        """A raw item exposes its assets mapping.

        Test scenario:
            The asset keys are discoverable.
        """
        assets = get_assets({"assets": {"B04": {"href": "b04.tif"}}})
        assert list(assets) == ["B04"], f"unexpected keys: {list(assets)}"

    def test_attribute_assets(self):
        """A pystac-like item exposes ``.assets``.

        Test scenario:
            The attribute mapping is returned.
        """
        item = SimpleNamespace(assets={"data": SimpleNamespace(href="x.tif")})
        assert "data" in get_assets(item), "data asset should be present"

    def test_missing_assets_none(self):
        """An item without assets returns None.

        Test scenario:
            ``{"id": "x"}`` → ``None``.
        """
        assert get_assets({"id": "x"}) is None


class TestGetAsset:
    """Tests for get_asset."""

    def test_resolve_named_asset(self):
        """A present asset key resolves to its asset.

        Test scenario:
            ``B04`` resolves to the dict bearing its href.
        """
        asset = get_asset({"assets": {"B04": {"href": "b04.tif"}}}, "B04")
        assert asset["href"] == "b04.tif", f"unexpected asset: {asset}"

    def test_missing_asset_raises_stac_asset_error(self):
        """A missing asset raises StacAssetError listing what is available.

        Test scenario:
            ``B99`` absent → StacAssetError mentioning the available keys.
        """
        with pytest.raises(StacAssetError, match="not found") as exc:
            get_asset({"assets": {"B04": {"href": "x"}}}, "B99")
        assert "B04" in str(exc.value), f"available keys not listed: {exc.value}"

    def test_missing_asset_is_keyerror(self):
        """M5: StacAssetError remains catchable as KeyError (back-compat).

        Test scenario:
            ``except KeyError`` still catches the missing-asset error.
        """
        with pytest.raises(KeyError):
            get_asset({"assets": {}}, "missing")


class TestAssetHref:
    """Tests for asset_href."""

    def test_dict_href(self):
        """A dict asset's href is read from ``href``.

        Test scenario:
            ``{"href": "s3://b/x.tif"}`` → that string.
        """
        assert asset_href({"href": "s3://b/x.tif"}) == "s3://b/x.tif"

    def test_attribute_href(self):
        """A pystac-like asset exposes ``.href``.

        Test scenario:
            The attribute is returned (coerced to str).
        """
        assert asset_href(SimpleNamespace(href="a.tif")) == "a.tif"

    def test_missing_href_raises(self):
        """A href-less asset raises StacAssetError naming the key.

        Test scenario:
            The error message contains the asset key and ``has no 'href'``.
        """
        with pytest.raises(StacAssetError, match="has no 'href'") as exc:
            asset_href({"type": "image/tiff"}, asset_key="B04")
        assert "B04" in str(exc.value), f"key not in message: {exc.value}"


class TestAssetMediaType:
    """Tests for asset_media_type."""

    def test_dict_type(self):
        """A dict asset stores its media type under ``type``.

        Test scenario:
            ``{"type": "image/tiff"}`` → that string.
        """
        assert asset_media_type({"href": "x.tif", "type": "image/tiff"}) == "image/tiff"

    def test_attribute_media_type(self):
        """A pystac-like asset exposes ``.media_type``.

        Test scenario:
            The attribute is returned.
        """
        asset = SimpleNamespace(href="a.nc", media_type="application/x-netcdf")
        assert asset_media_type(asset) == "application/x-netcdf"

    def test_missing_type_none(self):
        """A typeless asset returns None.

        Test scenario:
            ``{"href": "x.tif"}`` → ``None``.
        """
        assert asset_media_type({"href": "x.tif"}) is None


class TestAssetField:
    """Tests for asset_field (PB-1 extension-field extension point)."""

    def test_dict_top_level_field(self):
        """Extension fields are top-level keys on a raw asset dict.

        Test scenario:
            ``proj:epsg`` is read directly off the dict.
        """
        assert asset_field({"href": "x.tif", "proj:epsg": 32633}, "proj:epsg") == 32633

    def test_pystac_extra_fields(self):
        """Extension fields live in ``extra_fields`` on a pystac-like asset.

        Test scenario:
            ``proj:transform`` is read from ``extra_fields``.
        """
        asset = SimpleNamespace(
            href="x.tif", extra_fields={"proj:transform": [10, 0, 0, 0, -10, 0]}
        )
        assert asset_field(asset, "proj:transform") == [10, 0, 0, 0, -10, 0]

    def test_default_when_absent(self):
        """An absent field returns the supplied default.

        Test scenario:
            A dict lacking the key yields the default sentinel.
        """
        assert asset_field({"href": "x.tif"}, "proj:epsg", default="none") == "none"


class TestErrorTaxonomy:
    """M5: the STAC error classes preserve backward-compatible bases."""

    def test_stac_asset_error_is_stac_error_and_keyerror(self):
        """StacAssetError is both a StacError and a KeyError.

        Test scenario:
            Subclass relationships keep ``except StacError`` and
            ``except KeyError`` working.
        """
        assert issubclass(StacAssetError, StacError), "should be a StacError"
        assert issubclass(StacAssetError, KeyError), "should stay a KeyError"

    def test_unsupported_asset_error_is_stac_error_and_valueerror(self):
        """UnsupportedAssetError is both a StacError and a ValueError.

        Test scenario:
            Subclass relationships keep ``except StacError`` and
            ``except ValueError`` working.
        """
        assert issubclass(UnsupportedAssetError, StacError), "should be a StacError"
        assert issubclass(UnsupportedAssetError, ValueError), "should stay a ValueError"
