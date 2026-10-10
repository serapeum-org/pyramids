"""Unit tests for the STAC `alternate-assets` preference (STAC-12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.base._errors import StacAssetError
from pyramids.dataset import Dataset
from pyramids.stac._item import (
    asset_alternate_href,
    preferred_asset_href,
    preferred_asset_source,
)
from pyramids.stac._loader import load_asset, resolved_href, which_engine

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data" / "geotiff"
_GEOTIFF = str(_DATA / "era5_land_monthly_averaged.tif")
_MISSING = "https://nowhere.invalid/not-there.tif"
_COG_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


class _Asset:
    """Minimal stand-in for a `pystac.Asset` (attributes + `extra_fields`)."""

    def __init__(self, href, media_type=None, extra_fields=None):
        self.href = href
        self.media_type = media_type
        self.extra_fields = extra_fields or {}


class _AppendSigner:
    """Signer that records the href it was handed and appends a token."""

    def __init__(self):
        self.seen = None

    def sign_href(self, href: str) -> str:
        """Record the href and append a fake token."""
        self.seen = href
        return f"{href}?sig=tok"

    def gdal_env(self) -> dict[str, str]:
        """Return no extra GDAL config."""
        return {}


@pytest.fixture
def alternate_asset():
    """Return an asset dict with an `s3` alternate beside the canonical href.

    Returns:
        dict: A raw STAC asset carrying the `alternate-assets` extension.
    """
    return {
        "href": "https://host/a.tif",
        "type": _COG_TYPE,
        "alternate": {
            "s3": {"href": "s3://bucket/a.tif", "title": "S3 mirror"},
            "broken": {"title": "no href here"},
        },
    }


class TestAssetAlternateHref:
    """`asset_alternate_href` reads the extension off both asset shapes."""

    def test_reads_a_dict_asset(self, alternate_asset):
        """A raw asset dict exposes its alternate href."""
        href = asset_alternate_href(alternate_asset, "s3")
        assert href == "s3://bucket/a.tif", f"wrong alternate href: {href}"

    def test_reads_a_pystac_style_asset(self):
        """A pystac-style Asset exposes the block through `extra_fields`."""
        asset = _Asset(
            "https://host/a.tif",
            extra_fields={"alternate": {"s3": {"href": "s3://bucket/a.tif"}}},
        )
        href = asset_alternate_href(asset, "s3")
        assert href == "s3://bucket/a.tif", f"extra_fields not consulted: {href}"

    def test_absent_key_is_none(self, alternate_asset):
        """An alternate key the asset does not publish answers `None`."""
        assert asset_alternate_href(alternate_asset, "gs") is None, "expected None"

    def test_entry_without_href_is_none(self, alternate_asset):
        """An alternate entry carrying no href answers `None`, not a crash."""
        assert asset_alternate_href(alternate_asset, "broken") is None, "expected None"

    def test_asset_without_alternate_block_is_none(self):
        """An asset with no `alternate` block at all answers `None`."""
        assert asset_alternate_href({"href": "x.tif"}, "s3") is None, "expected None"

    def test_non_mapping_alternate_is_none(self):
        """A malformed (non-mapping) `alternate` field answers `None`."""
        asset = {"href": "x.tif", "alternate": ["s3://bucket/a.tif"]}
        assert asset_alternate_href(asset, "s3") is None, "expected None"


class TestPreferredAssetHref:
    """`preferred_asset_href` prefers an alternate but never requires one."""

    def test_single_key_wins_over_canonical(self, alternate_asset):
        """A matched single-key preference replaces the canonical href."""
        href = preferred_asset_href(alternate_asset, "s3")
        assert href == "s3://bucket/a.tif", f"alternate not preferred: {href}"

    def test_first_matching_key_of_a_sequence_wins(self, alternate_asset):
        """With a preference list the first key that resolves is used."""
        href = preferred_asset_href(alternate_asset, ["gs", "s3"])
        assert href == "s3://bucket/a.tif", f"preference order not honoured: {href}"

    def test_unmatched_preference_falls_back_silently(self, alternate_asset):
        """An alternate the asset lacks falls back to the canonical href."""
        href = preferred_asset_href(alternate_asset, "gs")
        assert href == "https://host/a.tif", f"no silent fallback: {href}"

    def test_no_preference_keeps_the_canonical_href(self, alternate_asset):
        """The default (`None`) is today's behaviour: the canonical href."""
        href = preferred_asset_href(alternate_asset)
        assert href == "https://host/a.tif", f"default changed: {href}"

    def test_href_less_asset_still_raises(self):
        """A fallback onto a missing canonical href raises the usual error."""
        with pytest.raises(StacAssetError, match="has no 'href'"):
            preferred_asset_href({"type": _COG_TYPE}, "s3", asset_key="B04")

    def test_alternate_rescues_an_href_less_asset(self):
        """An asset with only an alternate resolves through it."""
        asset = {"type": _COG_TYPE, "alternate": {"s3": {"href": "s3://b/a.tif"}}}
        href = preferred_asset_href(asset, "s3", asset_key="B04")
        assert href == "s3://b/a.tif", f"alternate-only asset failed: {href}"


class TestResolvedHrefAlternate:
    """`resolved_href(alternate=...)` resolves and signs the chosen href."""

    def test_prefers_the_alternate(self, alternate_asset):
        """An explicit alternate is what would be opened."""
        assert resolved_href(alternate_asset, alternate="s3") == "s3://bucket/a.tif"

    def test_default_is_unchanged(self, alternate_asset):
        """Without the kwarg the canonical href is returned as before."""
        assert resolved_href(alternate_asset) == "https://host/a.tif"

    def test_unmatched_alternate_falls_back(self, alternate_asset):
        """An absent alternate falls back instead of raising."""
        assert resolved_href(alternate_asset, alternate="gs") == "https://host/a.tif"

    def test_signer_runs_on_the_alternate_href(self, alternate_asset):
        """The signer sees the chosen (alternate) href, not the canonical one."""
        signer = _AppendSigner()
        href = resolved_href(alternate_asset, signer=signer, alternate="s3")
        assert signer.seen == "s3://bucket/a.tif", f"signer saw {signer.seen}"
        assert href == "s3://bucket/a.tif?sig=tok", f"signed href wrong: {href}"

    def test_resolves_an_item_asset(self):
        """An item + asset key resolves the named asset's alternate."""
        item = {
            "assets": {
                "B04": {
                    "href": "https://host/B04.tif",
                    "alternate": {"s3": {"href": "s3://bucket/B04.tif"}},
                }
            }
        }
        href = resolved_href(item, "B04", alternate="s3")
        assert href == "s3://bucket/B04.tif", f"item path ignores alternate: {href}"


class TestWhichEngineAlternate:
    """`which_engine` dispatches on the href it would actually open."""

    def test_extension_fallback_uses_the_alternate(self):
        """With no media type the alternate's extension picks the reader."""
        asset = {
            "href": "https://host/a.tif",
            "alternate": {"nc": {"href": "s3://bucket/a.nc"}},
        }
        assert which_engine(asset, alternate="nc") == "netcdf", "wrong engine"
        assert which_engine(asset) == "gdal", "default dispatch changed"


class TestLoadAssetAlternate:
    """`load_asset(alternate=...)` opens the preferred href."""

    def test_opens_the_alternate(self):
        """An unreadable canonical href is bypassed by a readable alternate."""
        asset = {
            "href": _MISSING,
            "type": _COG_TYPE,
            "alternate": {"local": {"href": _GEOTIFF}},
        }
        dataset = load_asset(asset, alternate="local")
        assert isinstance(dataset, Dataset), f"not a Dataset: {type(dataset)}"
        assert dataset.band_count >= 1, "opened dataset has no bands"

    def test_default_opens_the_canonical_href(self):
        """Without the kwarg the canonical href is opened, as before."""
        asset = {
            "href": _GEOTIFF,
            "type": _COG_TYPE,
            "alternate": {"bogus": {"href": _MISSING}},
        }
        dataset = load_asset(asset)
        assert isinstance(dataset, Dataset), f"not a Dataset: {type(dataset)}"


class TestAlternateMediaType:
    """An alternate's own `type` chooses the reader for the href it supplies.

    Test scenario:
        The `alternate-assets` extension allows each alternate its own
        properties, `type` among them, so a mirror is free to publish the same
        data in another format — an `s3://` Zarr store beside an HTTPS COG.
        Dispatching on the canonical asset's media type then opens the mirror
        with the wrong reader, which the href-extension fallback cannot save
        because a declared type short-circuits it.
    """

    def test_an_alternate_type_picks_the_reader(self):
        """A Zarr alternate of a COG asset dispatches to the Zarr reader."""
        asset = {
            "href": "https://host/a.tif",
            "type": _COG_TYPE,
            "alternate": {
                "s3": {"href": "s3://bucket/a", "type": "application/vnd+zarr"}
            },
        }
        assert which_engine(asset, alternate="s3") == "zarr", (
            f"the alternate's own type was ignored: {which_engine(asset, alternate='s3')}"
        )
        assert which_engine(asset) == "gdal", "the canonical dispatch changed"

    def test_an_alternate_without_a_type_keeps_the_canonical_one(self):
        """A plain mirror inherits the asset's declared type, as before."""
        asset = {
            "href": "https://host/a",
            "type": _COG_TYPE,
            "alternate": {"s3": {"href": "s3://bucket/a"}},
        }
        assert which_engine(asset, alternate="s3") == "gdal", (
            "a typeless alternate should still dispatch on the asset's type"
        )

    def test_the_source_helper_reports_href_and_type_together(self):
        """`preferred_asset_source` answers the pair the dispatch needs."""
        asset = {
            "href": "https://host/a.tif",
            "type": _COG_TYPE,
            "alternate": {
                "s3": {"href": "s3://bucket/a", "type": "application/vnd+zarr"},
                "gs": {"href": "gs://bucket/a.tif"},
            },
        }
        assert preferred_asset_source(asset, "s3") == (
            "s3://bucket/a",
            "application/vnd+zarr",
        ), (
            f"wrong (href, type) for the typed alternate: {preferred_asset_source(asset, 's3')}"
        )
        assert preferred_asset_source(asset, "gs") == (
            "gs://bucket/a.tif",
            _COG_TYPE,
        ), (
            f"a typeless alternate should keep the asset's type: "
            f"{preferred_asset_source(asset, 'gs')}"
        )
        assert preferred_asset_source(asset) == ("https://host/a.tif", _COG_TYPE), (
            f"no preference should be unchanged: {preferred_asset_source(asset)}"
        )

    def test_a_pystac_style_alternate_type_is_read_too(self):
        """The block is found in `extra_fields`, so the type comes with it."""
        asset = _Asset(
            "https://host/a.tif",
            media_type=_COG_TYPE,
            extra_fields={
                "alternate": {
                    "s3": {"href": "s3://bucket/a", "type": "application/x-netcdf"}
                }
            },
        )
        assert preferred_asset_source(asset, "s3") == (
            "s3://bucket/a",
            "application/x-netcdf",
        ), f"pystac-style alternate type ignored: {preferred_asset_source(asset, 's3')}"

    def test_load_asset_opens_the_alternate_with_its_own_reader(self, tmp_path):
        """The engine chosen for a real open follows the alternate's type.

        Test scenario:
            The canonical href claims netCDF while the readable alternate is a
            GeoTIFF; opening it with the netCDF reader would fail or return the
            wrong wrapper.
        """
        asset = {
            "href": "https://nowhere.invalid/a.nc",
            "type": "application/x-netcdf",
            "alternate": {"local": {"href": _GEOTIFF, "type": _COG_TYPE}},
        }
        dataset = load_asset(asset, alternate="local")
        assert type(dataset) is Dataset, (
            f"the alternate's type should select the GDAL reader, got {type(dataset)}"
        )
