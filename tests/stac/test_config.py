"""Unit tests for pyramids.stac._config (rescale + stac_cfg overrides/aliases)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF
from pyramids.stac import load_asset
from pyramids.stac._config import (
    AssetMetadataWarning,
    AssetOverrides,
    band_packing,
    item_collection_id,
    materialise,
    resolve_alias,
    resolve_asset_metadata,
    resolve_overrides,
    warnings_ignored,
)

pytestmark = pytest.mark.core

_GEO = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)

# MODIS Sinusoidal: a real projection with no EPSG authority code, so
# `Dataset.epsg` is `None` for it and only the WKT identifies it.
_SINUSOIDAL_WKT = (
    'PROJCS["MODIS Sinusoidal",'
    'GEOGCS["Unknown datum based upon the custom spheroid",'
    'DATUM["Not specified",SPHEROID["Custom spheroid",6371007.181,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'
)


def _write(tmp_path, name, array, no_data_value):
    """Write a small raster and return its path.

    Args:
        tmp_path: pytest temp directory.
        name: File name to write under `tmp_path`.
        array: The pixel values.
        no_data_value: The no-data value the file declares.

    Returns:
        str: The written path.
    """
    path = str(tmp_path / name)
    Dataset.from_array(array, no_data_value=no_data_value, geo_ref=_GEO).to_file(path)
    return path


@pytest.fixture
def packed_counts(tmp_path):
    """An int16 raster of stored counts `[[0, 100], [200, 300]]`, nodata 0."""
    return _write(
        tmp_path, "packed.tif", np.array([[0, 100], [200, 300]], dtype="int16"), 0
    )


@pytest.fixture
def packed_asset(packed_counts):
    """A bare STAC asset over `packed_counts` declaring scale 0.01."""
    return {
        "href": packed_counts,
        "type": "image/tiff",
        "raster:bands": [{"scale": 0.01, "offset": 0.0, "nodata": 0}],
    }


@pytest.fixture
def nodataless(tmp_path):
    """An int16 raster `[[1, 2], [3, 4]]` declaring no no-data value at all."""
    return _write(
        tmp_path, "nodataless.tif", np.array([[1, 2], [3, 4]], dtype="int16"), None
    )


@pytest.fixture
def file_packed_counts(tmp_path):
    """The packed counts with the scale 0.01 written into the file itself.

    `packed_counts` deliberately leaves the file's packing at identity, so a
    rebuilt result's `scale == [1.0]` is true of the source too. Here the source
    declares `0.01`, so "the result declares identity packing" is a claim the
    two sides can disagree on.
    """
    path = str(tmp_path / "file_packed.tif")
    source = Dataset.from_array(
        np.array([[0, 100], [200, 300]], dtype="int16"), no_data_value=0, geo_ref=_GEO
    )
    source.scale = [0.01]
    source.to_file(path)
    return path


@pytest.fixture
def file_packed_asset(file_packed_counts):
    """A STAC asset over `file_packed_counts` declaring the same scale 0.01."""
    return {
        "href": file_packed_counts,
        "type": "image/tiff",
        "raster:bands": [{"scale": 0.01, "offset": 0.0, "nodata": 0}],
    }


@pytest.fixture
def sinusoidal_counts(tmp_path):
    """The packed counts on MODIS Sinusoidal — a CRS with no EPSG code."""
    path = str(tmp_path / "sinusoidal.tif")
    source = Dataset.from_array(
        np.array([[0, 100], [200, 300]], dtype="int16"),
        no_data_value=0,
        geo_ref=GeoReference(geo=(0.0, 1000.0, 0.0, 2000.0, 0.0, -1000.0), epsg=None),
    )
    source.crs = _SINUSOIDAL_WKT
    source.to_file(path)
    return path


@pytest.fixture
def named_bands(tmp_path):
    """A two-band raster carrying band names, band metadata and band units."""
    path = str(tmp_path / "named_bands.tif")
    source = Dataset.from_array(
        np.array([[[100, 100], [100, 100]], [[200, 200], [200, 200]]], dtype="int16"),
        no_data_value=None,
        geo_ref=_GEO,
    )
    source.band_names = ["red", "nir"]
    source.bands.metadata = [{"WAVELENGTH": "665"}, {"WAVELENGTH": "842"}]
    source.band_units = ["reflectance", "reflectance"]
    source.to_file(path)
    return path


class TestResolveAlias:
    """Tests for the band-alias map (cfg[collection]['aliases'])."""

    def test_configured_alias_resolves(self):
        """A configured alias maps to the real asset key.

        Test scenario:
            cfg maps "red" -> "B04" for collection "c".
        """
        cfg = {"c": {"aliases": {"red": "B04"}}}
        assert resolve_alias(cfg, "c", "red") == "B04", (
            "a configured alias must resolve to the real asset key"
        )

    def test_unaliased_name_passes_through(self):
        """A name with no alias entry is returned unchanged.

        Test scenario:
            "green" is absent from the alias map.
        """
        cfg = {"c": {"aliases": {"red": "B04"}}}
        assert resolve_alias(cfg, "c", "green") == "green", (
            "an unaliased name must be returned unchanged"
        )

    def test_no_cfg_passes_through(self):
        """Without a cfg nothing is aliased.

        Test scenario:
            cfg is None, and the collection is unconfigured.
        """
        assert resolve_alias(None, "c", "red") == "red", (
            "cfg=None must leave the asset key alone"
        )
        assert resolve_alias({"other": {"aliases": {"red": "B04"}}}, "c", "red") == (
            "red"
        ), "another collection's aliases must not apply"

    def test_collection_alias_wins_over_wildcard(self):
        """The collection's own alias map beats the '*' defaults.

        Test scenario:
            "*" maps red -> B04 and "c" maps red -> RED.
        """
        cfg = {"*": {"aliases": {"red": "B04"}}, "c": {"aliases": {"red": "RED"}}}
        assert resolve_alias(cfg, "c", "red") == "RED", (
            "the per-collection alias must win over the '*' section"
        )
        assert resolve_alias(cfg, "d", "red") == "B04", (
            "the '*' section must still apply to an unlisted collection"
        )


class TestResolveAssetMetadata:
    """Tests for merging the '*' asset defaults with a per-asset override."""

    def test_named_asset_overrides_wildcard(self):
        """A named-asset entry overrides the '*' asset defaults key by key.

        Test scenario:
            "*" sets nodata + data_type, "SCL" narrows data_type only.
        """
        cfg = {
            "c": {
                "assets": {
                    "*": {"nodata": 0, "data_type": "uint16"},
                    "SCL": {"data_type": "uint8"},
                }
            }
        }
        assert resolve_asset_metadata(cfg, "c", "SCL") == {
            "nodata": 0,
            "data_type": "uint8",
        }, "the named asset must override only the keys it declares"
        assert resolve_asset_metadata(cfg, "c", "B02") == {
            "nodata": 0,
            "data_type": "uint16",
        }, "an unnamed asset must get the '*' defaults"

    def test_unconfigured_resolves_to_nothing(self):
        """An unconfigured collection, or no cfg at all, yields an empty dict.

        Test scenario:
            A cfg for "c" is queried for "other", then cfg=None.
        """
        cfg = {"c": {"assets": {"*": {"nodata": 0}}}}
        assert resolve_asset_metadata(cfg, "other", "x") == {}, (
            "another collection's config must not leak"
        )
        assert resolve_asset_metadata(None, "c", "x") == {}, (
            "cfg=None must resolve to no overrides"
        )

    def test_wildcard_collection_supplies_defaults(self):
        """The '*' collection section applies under every collection id.

        Test scenario:
            "*" sets unit, "c" sets nodata; both are merged for "c".
        """
        cfg = {
            "*": {"assets": {"*": {"unit": "1"}}},
            "c": {"assets": {"*": {"nodata": 0}}},
        }
        assert resolve_asset_metadata(cfg, "c", "B02") == {"unit": "1", "nodata": 0}, (
            "the '*' collection defaults must merge under the named collection"
        )
        assert resolve_asset_metadata(cfg, None, "B02") == {"unit": "1"}, (
            "the '*' section must be the only one applying without a collection id"
        )

    def test_bare_asset_reads_wildcard_defaults_only(self):
        """A None asset key still picks up the '*' asset defaults.

        Test scenario:
            A bare asset (no key) under a configured collection.
        """
        cfg = {"c": {"assets": {"*": {"nodata": 0}, "B02": {"nodata": 1}}}}
        assert resolve_asset_metadata(cfg, "c", None) == {"nodata": 0}, (
            "a bare asset must read the '*' defaults and no named entry"
        )


class TestWarningsIgnored:
    """Tests for the per-collection `warnings: 'ignore'` switch."""

    def test_only_the_configured_collection_is_silenced(self):
        """`warnings: ignore` applies to its own collection only.

        Test scenario:
            Collection "c" opts out; "d" does not.
        """
        cfg = {"c": {"warnings": "ignore"}}
        assert warnings_ignored(cfg, "c") is True, (
            "the configured collection must be silenced"
        )
        assert warnings_ignored(cfg, "d") is False, (
            "an unconfigured collection must keep its warnings"
        )
        assert warnings_ignored(None, "c") is False, (
            "cfg=None must keep warnings enabled"
        )


class TestBandPacking:
    """Tests for reading per-band scale/offset out of `raster:bands`."""

    def test_scale_and_offset_are_parsed(self):
        """Declared scale/offset come back as floats, in band order.

        Test scenario:
            Two bands with distinct packings.
        """
        scales, offsets = band_packing(
            [{"scale": 0.01, "offset": 5}, {"scale": 2, "offset": -1}]
        )
        assert scales == (0.01, 2.0), f"expected both scales, got {scales}"
        assert offsets == (5.0, -1.0), f"expected both offsets, got {offsets}"

    def test_missing_entries_are_the_identity(self):
        """A band declaring neither key is the identity, and None is empty.

        Test scenario:
            A band carrying only nodata, then raster_bands=None.
        """
        assert band_packing([{"nodata": 0}]) == ((1.0,), (0.0,)), (
            "a band without packing must resolve to the identity"
        )
        assert band_packing(None) == ((), ()), (
            "a missing raster:bands must resolve to empty tuples"
        )

    def test_non_finite_spelling_is_coerced(self):
        """The raster extension's 'nan'/'inf' spellings are parsed as floats.

        Test scenario:
            A band declaring scale "inf".
        """
        scales, _ = band_packing([{"scale": "inf"}])
        assert np.isinf(scales[0]), f"expected an infinite scale, got {scales}"


class TestItemCollectionId:
    """Tests for the duck-typed collection-id accessor."""

    def test_raw_json_member(self):
        """Raw STAC JSON carries the collection as the `collection` member.

        Test scenario:
            An item dict with a collection.
        """
        assert item_collection_id({"collection": "c", "assets": {}}) == "c", (
            "the raw-JSON collection member must be read"
        )

    def test_attribute_form(self):
        """A pystac-style object exposes it as `collection_id`.

        Test scenario:
            A duck-typed object carrying collection_id.
        """

        class _Item:
            collection_id = "c"

        assert item_collection_id(_Item()) == "c", (
            "the pystac collection_id attribute must be read"
        )

    def test_absent_is_none(self):
        """A bare asset declares no collection.

        Test scenario:
            An asset dict with only an href.
        """
        assert item_collection_id({"href": "a.tif"}) is None, (
            "an asset without a collection must answer None"
        )


class TestAssetOverrides:
    """Tests for the resolved-override value object."""

    def test_empty_when_nothing_configured(self):
        """A default AssetOverrides has nothing to apply.

        Test scenario:
            No packing, no cfg values.
        """
        assert AssetOverrides().is_empty is True, (
            "an unconfigured override set must report itself empty"
        )

    def test_offset_only_declaration_still_rescales(self):
        """A band declaring only an offset is not the identity.

        Test scenario:
            offsets=(5.0,) with no scales at all.
        """
        overrides = AssetOverrides(offsets=(5.0,))
        assert overrides.rescales is True, (
            "an offset-only declaration must count as a rescale"
        )
        assert overrides.is_empty is False, (
            "an offset-only declaration must not look empty"
        )

    def test_packing_falls_back_to_identity_beyond_declared_bands(self):
        """A band beyond the declared raster:bands gets the identity.

        Test scenario:
            One declared band, band index 1 requested.
        """
        overrides = AssetOverrides(scales=(0.01,), offsets=(0.0,))
        assert overrides.packing(1) == (1.0, 0.0), (
            "an undeclared band must not borrow band 0's packing"
        )

    def test_resolve_overrides_reads_raster_bands(self, packed_asset):
        """`rescale=True` picks the packing out of the asset's raster:bands.

        Test scenario:
            The packed asset declares scale 0.01.
        """
        overrides = resolve_overrides(packed_asset, rescale=True)
        assert overrides.packing(0) == (0.01, 0.0), (
            f"expected the declared scale, got {overrides.packing(0)}"
        )

    def test_resolve_overrides_without_rescale_is_empty(self, packed_asset):
        """Without `rescale` the packing is not read at all.

        Test scenario:
            The same packed asset, rescale left off and no cfg.
        """
        assert resolve_overrides(packed_asset).is_empty is True, (
            "rescale=False must leave the declared packing unread"
        )

    def test_invalid_data_type_raises(self):
        """A cfg data_type numpy cannot parse is rejected up front.

        Test scenario:
            cfg declares data_type "not-a-dtype".
        """
        cfg = {"c": {"assets": {"*": {"data_type": "not-a-dtype"}}}}
        with pytest.raises(ValueError, match="not a numpy dtype name"):
            resolve_overrides({"href": "x.tif"}, cfg=cfg, collection_id="c")


class TestMaterialise:
    """Tests for the shared materialiser (mask -> scale -> fill)."""

    def test_identity_packing_keeps_the_handle(self, packed_counts):
        """Nothing configured means the opened dataset is returned as-is.

        Test scenario:
            An empty override set, and an all-identity packing.
        """
        dataset = Dataset.read_file(packed_counts)
        assert materialise(dataset, AssetOverrides()) is dataset, (
            "an empty override set must not rebuild the dataset"
        )
        identity = AssetOverrides(scales=(1.0,), offsets=(0.0,))
        assert materialise(dataset, identity) is dataset, (
            "an identity packing must not rebuild the dataset"
        )

    def test_masks_before_scaling(self, packed_counts):
        """The no-data sentinel is masked first, so it is never scaled.

        Test scenario:
            Counts [[0, 100], [200, 300]] with nodata 0, scale 0.01 and a
            **non-zero** offset of 10. The offset is what makes the two orders
            distinguishable: masking first leaves the sentinel cell NaN, while
            scaling first turns the stored `0` into `0 * 0.01 + 10 == 10.0`,
            which no longer equals the sentinel and so survives the fill as a
            plausible-looking physical value.
        """
        dataset = Dataset.read_file(packed_counts)
        result = materialise(dataset, AssetOverrides(scales=(0.01,), offsets=(10.0,)))
        values = result.read_array()
        assert np.isnan(values[0, 0]), (
            "mask -> scale must leave the sentinel NaN; scaling first would "
            f"have produced 0 * 0.01 + 10 = 10.0, got {values[0, 0]}"
        )
        assert values[0, 1] == pytest.approx(11.0), f"100 * 0.01 + 10 != {values[0, 1]}"
        assert values[1, 0] == pytest.approx(12.0), f"200 * 0.01 + 10 != {values[1, 0]}"
        assert values[1, 1] == pytest.approx(13.0), f"300 * 0.01 + 10 != {values[1, 1]}"

    def test_offset_is_applied(self, packed_counts):
        """The additive half of the packing is applied to valid pixels only.

        Test scenario:
            scale 0.01 with offset 10 on the packed counts.
        """
        dataset = Dataset.read_file(packed_counts)
        values = materialise(
            dataset, AssetOverrides(scales=(0.01,), offsets=(10.0,))
        ).read_array()
        assert values[1, 1] == pytest.approx(13.0), f"300 * 0.01 + 10 != {values[1, 1]}"
        assert np.isnan(values[0, 0]), "the sentinel must not pick up the offset"

    def test_result_declares_identity_packing(self, file_packed_counts):
        """The rebuilt raster drops the source's packing, so no read re-applies.

        Test scenario:
            The source file itself declares scale 0.01, and the STAC asset
            declares the same 0.01. The materialiser reads raw counts and
            applies the STAC packing once, so carrying the file's scale onto the
            result would make an unpacking read return 300 * 0.01 * 0.01 = 0.03
            instead of 3.0.
        """
        source = Dataset.read_file(file_packed_counts)
        assert source.scale == [0.01], (
            f"fixture must declare the scale in the file, got {source.scale}"
        )
        result = materialise(source, AssetOverrides(scales=(0.01,)))
        assert result.scale == [1.0], (
            f"the rebuild must not carry the source scale 0.01, got {result.scale}"
        )
        assert result.offset == [0.0], f"expected identity offset, got {result.offset}"
        assert result.read_array(unpack=True)[1, 1] == pytest.approx(3.0), (
            "an unpacking read of the result must not apply a scale a second "
            f"time, got {result.read_array(unpack=True)[1, 1]}"
        )
        assert np.isnan(result.no_data_value[0]), (
            f"expected NaN no-data, got {result.no_data_value}"
        )

    def test_wkt_only_crs_survives_the_rebuild(self, sinusoidal_counts):
        """A CRS with no EPSG code is carried through as WKT, not dropped.

        Test scenario:
            A MODIS Sinusoidal source (`epsg is None`) rescaled through the
            materialiser. Rebuilding from `epsg=dataset.epsg` alone leaves the
            result's CRS unset, which silently breaks every later `to_crs`,
            `crop(bbox)`, `align` and write.
        """
        source = Dataset.read_file(sinusoidal_counts)
        assert source.epsg is None, f"fixture must have no EPSG code, got {source.epsg}"
        assert source.crs, "fixture must carry a WKT projection"
        result = materialise(source, AssetOverrides(scales=(0.01,)))
        assert result.crs == source.crs, (
            f"the source projection must survive the rebuild, got {result.crs!r}"
        )
        assert "Sinusoidal" in result.crs, (
            f"the result must still be on the sinusoidal projection, got "
            f"{result.crs[:60]!r}"
        )

    def test_band_names_survive_the_rebuild(self, named_bands):
        """Band names are carried onto the rebuilt raster.

        Test scenario:
            A two-band source named ["red", "nir"] rescaled through the
            materialiser; `from_array` would otherwise name the bands
            "Band_1" / "Band_2".
        """
        source = Dataset.read_file(named_bands)
        result = materialise(source, AssetOverrides(scales=(0.01, 0.01)))
        assert result.band_names == ["red", "nir"], (
            f"expected the source band names, got {result.band_names}"
        )

    def test_band_metadata_survives_the_rebuild(self, named_bands):
        """Per-band metadata items are carried onto the rebuilt raster.

        Test scenario:
            The same two-band source, whose bands carry a WAVELENGTH item.
        """
        source = Dataset.read_file(named_bands)
        result = materialise(source, AssetOverrides(scales=(0.01, 0.01)))
        assert result.bands.metadata == [
            {"WAVELENGTH": "665"},
            {"WAVELENGTH": "842"},
        ], f"expected the source band metadata, got {result.bands.metadata}"

    def test_native_band_units_survive_the_rebuild(self, named_bands):
        """A source's own unit labels are kept when no unit override applies.

        Test scenario:
            The two-band source labels both bands "reflectance" and the
            overrides configure no unit at all.
        """
        source = Dataset.read_file(named_bands)
        result = materialise(source, AssetOverrides(scales=(0.01, 0.01)))
        assert result.band_units == ["reflectance", "reflectance"], (
            f"expected the source band units, got {result.band_units}"
        )

    def test_per_band_packing(self, tmp_path):
        """Each band gets its own scale, not band 0's.

        Test scenario:
            A 2-band raster with scales 0.01 and 10.
        """
        path = _write(
            tmp_path,
            "two_band.tif",
            np.array([[[100, 100], [100, 100]], [[2, 2], [2, 2]]], dtype="int16"),
            None,
        )
        result = materialise(
            Dataset.read_file(path), AssetOverrides(scales=(0.01, 10.0))
        )
        values = result.read_array()
        assert values[0][0, 0] == pytest.approx(1.0), (
            f"band 0 scaled wrong: {values[0]}"
        )
        assert values[1][0, 0] == pytest.approx(20.0), (
            f"band 1 must use its own scale, got {values[1]}"
        )

    def test_nodata_override_fills_a_gap(self, nodataless):
        """A configured nodata is stamped when the raster declares none.

        Test scenario:
            A raster with no no-data value and cfg nodata 1.
        """
        result = materialise(
            Dataset.read_file(nodataless), AssetOverrides(no_data_value=1.0)
        )
        assert result.no_data_value[0] == pytest.approx(1.0), (
            f"expected the configured no-data, got {result.no_data_value}"
        )
        assert result.read_array().tolist() == [[1, 2], [3, 4]], (
            "stamping no-data must not change any pixel value"
        )

    def test_nodata_override_does_not_replace_a_declared_value(self, packed_counts):
        """A declared no-data wins, and the skipped override warns.

        Test scenario:
            The packed raster declares nodata 0; cfg asks for 7.
        """
        dataset = Dataset.read_file(packed_counts)
        with pytest.warns(AssetMetadataWarning, match="already declares"):
            result = materialise(dataset, AssetOverrides(no_data_value=7.0))
        assert result is dataset, (
            "a skipped override must leave the opened handle untouched"
        )

    def test_quiet_suppresses_the_skip_warning(self, packed_counts, recwarn):
        """`warnings: ignore` silences the skipped-override warning.

        Test scenario:
            The same skipped nodata override with quiet=True.
        """
        dataset = Dataset.read_file(packed_counts)
        materialise(dataset, AssetOverrides(no_data_value=7.0, quiet=True))
        assert not [w for w in recwarn if w.category is AssetMetadataWarning], (
            "quiet=True must emit no AssetMetadataWarning"
        )

    def test_unit_override_is_stamped(self, nodataless):
        """A configured unit is written onto the materialised bands.

        Test scenario:
            An unlabelled raster and cfg unit "K".
        """
        result = materialise(Dataset.read_file(nodataless), AssetOverrides(unit="K"))
        assert result.band_units == ["K"], (
            f"expected the configured unit, got {result.band_units}"
        )

    def test_unit_override_does_not_replace_a_declared_unit(self):
        """A band that already carries a unit keeps it, and the skip warns.

        Test scenario:
            The raster declares "K" on its only band; cfg asks for "degC".
        """
        dataset = Dataset.from_array(
            np.array([[1, 2], [3, 4]], dtype="int16"), no_data_value=None, geo_ref=_GEO
        )
        dataset.band_units = ["K"]
        with pytest.warns(AssetMetadataWarning, match="already declares"):
            result = materialise(dataset, AssetOverrides(unit="degC"))
        assert result.band_units == ["K"], (
            f"the declared unit must win, got {result.band_units}"
        )

    def test_quiet_suppresses_the_unit_skip_warning(self, recwarn):
        """`warnings: ignore` silences the skipped-unit warning.

        Test scenario:
            The same skipped unit override with quiet=True.
        """
        dataset = Dataset.from_array(
            np.array([[1, 2], [3, 4]], dtype="int16"), no_data_value=None, geo_ref=_GEO
        )
        dataset.band_units = ["K"]
        materialise(dataset, AssetOverrides(unit="degC", quiet=True))
        assert not [w for w in recwarn if w.category is AssetMetadataWarning], (
            "quiet=True must emit no AssetMetadataWarning for a skipped unit"
        )

    def test_data_type_override_casts_the_counts(self, nodataless):
        """A configured data_type casts the stored values.

        Test scenario:
            An int16 raster configured as float32.
        """
        result = materialise(
            Dataset.read_file(nodataless), AssetOverrides(data_type="float32")
        )
        assert result.dtype == ["float32"], (
            f"expected the configured dtype, got {result.dtype}"
        )
        assert result.read_array().tolist() == [[1.0, 2.0], [3.0, 4.0]], (
            "a dtype cast must preserve the values"
        )

    def test_matching_data_type_is_a_no_op(self, nodataless):
        """A data_type equal to the native one rebuilds nothing.

        Test scenario:
            An int16 raster configured as int16.
        """
        dataset = Dataset.read_file(nodataless)
        assert materialise(dataset, AssetOverrides(data_type="int16")) is dataset, (
            "a data_type matching the native dtype must not rebuild"
        )

    def test_rotated_grid_survives_the_rebuild(self, tmp_path):
        """The rebuilt raster keeps the source geotransform and CRS.

        Test scenario:
            A rotated geotransform rescaled through the materialiser.
        """
        geo = (10.0, 1.0, 0.5, 20.0, 0.25, -1.0)
        path = str(tmp_path / "rotated.tif")
        Dataset.from_array(
            np.array([[1, 2], [3, 4]], dtype="int16"),
            no_data_value=None,
            geo_ref=GeoReference(geo=geo, epsg=32633),
        ).to_file(path)
        result = materialise(Dataset.read_file(path), AssetOverrides(scales=(2.0,)))
        assert tuple(result.geotransform) == pytest.approx(geo), (
            f"expected the source geotransform, got {result.geotransform}"
        )
        assert result.epsg == 32633, f"expected EPSG 32633, got {result.epsg}"


class TestLoadAssetRescale:
    """Tests for `load_asset(rescale=True)` (STAC-04)."""

    def test_rescale_applies_scale_offset(self, packed_asset):
        """The opt-in turns stored counts into physical units.

        Test scenario:
            A read-only on-disk asset declaring scale 0.01.
        """
        raw = load_asset(packed_asset).read_array()
        physical = load_asset(packed_asset, rescale=True).read_array()
        assert raw.tolist() == [[0, 100], [200, 300]], (
            f"rescale=False must keep the stored counts, got {raw.tolist()}"
        )
        assert physical[1, 0] == pytest.approx(2.0), f"200 * 0.01 != {physical[1, 0]}"
        assert np.isnan(physical[0, 0]), (
            "the no-data sentinel must stay no-data, not become 0.0"
        )

    def test_rescale_no_double_apply(self, file_packed_asset):
        """The rescaled dataset declares identity packing, not the file's.

        Test scenario:
            The asset's file declares scale 0.01 **and** the asset's
            raster:bands declares 0.01. `load_asset(rescale=True)` applies the
            packing exactly once, so the returned raster must declare identity:
            otherwise `read_array(unpack=True)` returns 0.03 for the cell whose
            physical value is 3.0.
        """
        raw = load_asset(file_packed_asset)
        assert raw.scale == [0.01], (
            f"the asset's file must declare the scale, got {raw.scale}"
        )
        dataset = load_asset(file_packed_asset, rescale=True)
        assert dataset.scale == [1.0], (
            f"the rescaled raster must not carry the file scale, got {dataset.scale}"
        )
        assert dataset.offset == [0.0], (
            f"expected identity offset, got {dataset.offset}"
        )
        assert dataset.read_array(unpack=True)[1, 1] == pytest.approx(3.0), (
            "an unpacking read must not apply the scale a second time, got "
            f"{dataset.read_array(unpack=True)[1, 1]}"
        )

    def test_rescale_without_raster_bands_is_a_no_op(self, packed_counts):
        """An asset declaring no raster:bands is read exactly as before.

        Test scenario:
            The same raster with no raster:bands on the asset.
        """
        asset = {"href": packed_counts, "type": "image/tiff"}
        dataset = load_asset(asset, rescale=True)
        assert dataset.dtype == ["int16"], (
            f"nothing to rescale must keep the native dtype, got {dataset.dtype}"
        )
        assert dataset.read_array().tolist() == [[0, 100], [200, 300]], (
            "nothing to rescale must keep the stored counts"
        )

    def test_rescale_leaves_non_gdal_engines_alone(self, tmp_path):
        """A netCDF asset is not rescaled — its reader owns CF packing.

        Test scenario:
            A netCDF asset carrying a raster:bands scale.
        """
        source = Dataset.from_array(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            no_data_value=-9999.0,
            geo_ref=_GEO,
        )
        path = str(tmp_path / "cube.nc")
        source.to_file(path)
        asset = {
            "href": path,
            "type": "application/x-netcdf",
            "raster:bands": [{"scale": 0.01}],
        }
        result = load_asset(asset, rescale=True)
        assert isinstance(result, NetCDF), (
            f"a netCDF asset must stay a NetCDF reader, got {type(result).__name__}"
        )
        assert type(result) is not Dataset, (
            "rescale must not materialise a plain Dataset out of a netCDF asset"
        )

    def test_default_is_unchanged(self, packed_asset):
        """Omitting both opt-ins reads byte-identically to before.

        Test scenario:
            The packed asset read with no new arguments.
        """
        dataset = load_asset(packed_asset)
        assert dataset.dtype == ["int16"], (
            f"the default read must keep the stored dtype, got {dataset.dtype}"
        )
        assert dataset.no_data_value[0] == pytest.approx(0.0), (
            f"the default read must keep the file's no-data, got {dataset.no_data_value}"
        )


class TestLoadAssetConfig:
    """Tests for `load_asset(cfg=...)` (STAC-07)."""

    def test_alias_resolves_the_asset_key(self, nodataless):
        """An alias is resolved before the href is looked up.

        Test scenario:
            cfg maps "rededge" -> "B05" for the item's collection.
        """
        item = {
            "collection": "c",
            "assets": {"B05": {"href": nodataless, "type": "image/tiff"}},
        }
        cfg = {"c": {"aliases": {"rededge": "B05"}}}
        dataset = load_asset(item, "rededge", cfg=cfg)
        assert dataset.read_array().tolist() == [[1, 2], [3, 4]], (
            "the aliased asset must be the one opened"
        )

    def test_unaliased_key_still_resolves(self, nodataless):
        """A cfg that aliases nothing leaves the asset key alone.

        Test scenario:
            The real key is passed while an unrelated alias is configured.
        """
        item = {
            "collection": "c",
            "assets": {"B05": {"href": nodataless, "type": "image/tiff"}},
        }
        cfg = {"c": {"aliases": {"red": "B04"}}}
        assert load_asset(item, "B05", cfg=cfg).read_array().tolist() == [
            [1, 2],
            [3, 4],
        ], "an unaliased key must still open its own asset"

    def test_missing_nodata_is_supplied(self, nodataless):
        """A cfg nodata fills the gap an item leaves.

        Test scenario:
            The raster declares no no-data; cfg supplies 1.
        """
        item = {
            "collection": "c",
            "assets": {"data": {"href": nodataless, "type": "image/tiff"}},
        }
        cfg = {"c": {"assets": {"*": {"nodata": 1}}}}
        dataset = load_asset(item, "data", cfg=cfg)
        assert dataset.no_data_value[0] == pytest.approx(1.0), (
            f"expected the configured no-data, got {dataset.no_data_value}"
        )

    def test_collection_id_argument_overrides_the_item(self, nodataless):
        """An explicit collection_id picks the cfg section to read.

        Test scenario:
            The item declares no collection, so one is passed in.
        """
        asset = {"href": nodataless, "type": "image/tiff"}
        cfg = {"c": {"assets": {"*": {"unit": "K"}}}}
        dataset = load_asset(asset, cfg=cfg, collection_id="c")
        assert dataset.band_units == ["K"], (
            f"expected the configured unit, got {dataset.band_units}"
        )

    def test_unconfigured_collection_changes_nothing(self, packed_counts):
        """A cfg for another collection leaves the read untouched.

        Test scenario:
            cfg configures "other"; the item is in "c".
        """
        item = {
            "collection": "c",
            "assets": {"data": {"href": packed_counts, "type": "image/tiff"}},
        }
        cfg = {"other": {"assets": {"*": {"nodata": 7}}}}
        dataset = load_asset(item, "data", cfg=cfg)
        assert dataset.no_data_value[0] == pytest.approx(0.0), (
            f"another collection's config must not apply, got {dataset.no_data_value}"
        )

    def test_warnings_ignore_silences_the_skip(self, packed_counts, recwarn):
        """`warnings: ignore` suppresses the skipped-override warning.

        Test scenario:
            A nodata override on a raster that already declares one.
        """
        item = {
            "collection": "c",
            "assets": {"data": {"href": packed_counts, "type": "image/tiff"}},
        }
        cfg = {"c": {"assets": {"*": {"nodata": 7}}, "warnings": "ignore"}}
        load_asset(item, "data", cfg=cfg)
        assert not [w for w in recwarn if w.category is AssetMetadataWarning], (
            "warnings: ignore must silence the skipped-override warning"
        )

    def test_rescale_and_cfg_combine(self, packed_counts):
        """Packing and a configured unit are applied in one rebuild.

        Test scenario:
            A packed asset with cfg supplying the unit.
        """
        item = {
            "collection": "c",
            "assets": {
                "data": {
                    "href": packed_counts,
                    "type": "image/tiff",
                    "raster:bands": [{"scale": 0.01}],
                }
            },
        }
        cfg = {"c": {"assets": {"*": {"unit": "K"}}}}
        dataset = load_asset(item, "data", rescale=True, cfg=cfg)
        assert dataset.band_units == ["K"], (
            f"expected the configured unit, got {dataset.band_units}"
        )
        assert dataset.read_array()[1, 1] == pytest.approx(3.0), (
            "the packing must still be applied alongside the cfg override"
        )
