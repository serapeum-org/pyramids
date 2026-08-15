"""Unit tests for the grouped COG option dataclasses and their validators.

Covers each ``__post_init__`` validation branch, ``Compression.coerce``, and the
per-group logic methods — ``_to_options`` (``Compression``/``Overviews``/
``Tiling``/``Layout``), ``BandSelection._needs_translate`` / ``_translate``, and
``Tags._has_any`` / ``_stamp`` — for `Compression`, `Overviews`, `Tiling`,
`BandSelection`, `Tags`, and `Layout`.
"""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids.base._errors import FailedToSaveError
from pyramids.dataset.cog import (
    PROFILES,
    BandSelection,
    Compression,
    Layout,
    Overviews,
    Tags,
    Tiling,
)

pytestmark = pytest.mark.core

_COERCED_PROFILE_KEYS = {"COMPRESS", "LEVEL", "QUALITY", "MAX_Z_ERROR"}


def _mem_dataset(
    gdal_dtype: int = gdal.GDT_Float32,
    n_bands: int = 1,
    color_table: bool = False,
) -> gdal.Dataset:
    """Build a tiny in-memory GDAL dataset for the method tests.

    Args:
        gdal_dtype: Band data type (drives the predictor / resampling defaults).
        n_bands: Number of bands to create.
        color_table: When ``True``, attach a 1-entry colour table to band 1 so
            the source reads as categorical.

    Returns:
        gdal.Dataset: A 4x4 MEM dataset the caller owns for the test's duration.
    """
    ds = gdal.GetDriverByName("MEM").Create("", 4, 4, n_bands, gdal_dtype)
    if color_table:
        ct = gdal.ColorTable()
        ct.SetColorEntry(0, (0, 0, 0, 255))
        ds.GetRasterBand(1).SetColorTable(ct)
    return ds


def test_all_profiles_use_only_coerced_keys():
    """Every PROFILES entry uses only keys `Compression.coerce` carries.

    Guards against a future profile adding a key (e.g. `PREDICTOR`, `NBITS`) that
    the string-coercion path would silently drop.
    """
    for name, opts in PROFILES.items():
        extra = set(opts) - _COERCED_PROFILE_KEYS
        assert not extra, f"profile {name!r} uses keys coerce would drop: {extra}"


class TestCompression:
    """Validation and coercion for `Compression`."""

    def test_defaults_are_none(self):
        """A bare `Compression` leaves every field `None`."""
        c = Compression()
        assert (c.compress, c.level, c.quality, c.predictor, c.max_z_error) == (
            None,
            None,
            None,
            None,
            None,
        )

    @pytest.mark.parametrize("quality", [0, 101, -1, 200])
    def test_quality_out_of_range_rejected(self, quality):
        """`quality` outside 1..100 raises `ValueError`.

        Args:
            quality: An out-of-range quality value.
        """
        with pytest.raises(ValueError, match="quality must be in 1..100"):
            Compression(quality=quality)

    @pytest.mark.parametrize("quality", [1, 50, 100])
    def test_quality_in_range_accepted(self, quality):
        """`quality` within 1..100 is accepted.

        Args:
            quality: An in-range quality value.
        """
        assert Compression(quality=quality).quality == quality

    @pytest.mark.parametrize(
        "predictor", [1, 2, 3, "YES", "NO", "STANDARD", "FLOATING_POINT"]
    )
    def test_valid_predictors_accepted(self, predictor):
        """Every GDAL predictor token is accepted.

        Args:
            predictor: A valid predictor value.
        """
        assert Compression(predictor=predictor).predictor == predictor

    @pytest.mark.parametrize("predictor", ["1", "2", "3"])
    def test_string_numeric_predictor_accepted(self, predictor):
        """String-numeric predictors (forwarded by the old flat API) are accepted.

        Args:
            predictor: A numeric predictor in string form.
        """
        assert Compression(predictor=predictor).predictor == predictor

    @pytest.mark.parametrize("predictor", ["bogus", 4, "maybe"])
    def test_invalid_predictor_rejected(self, predictor):
        """An unknown predictor raises `ValueError`.

        Args:
            predictor: An invalid predictor value.
        """
        with pytest.raises(ValueError, match="predictor must be one of"):
            Compression(predictor=predictor)

    def test_coerce_profile_string(self):
        """A profile-name string expands to its preset."""
        assert Compression.coerce("zstd") == Compression(compress="ZSTD", level=9)

    def test_coerce_lerc_carries_max_z_error(self):
        """The lerc profile carries `MAX_Z_ERROR` into `max_z_error`."""
        assert Compression.coerce("lerc").max_z_error == 0.0

    def test_coerce_none_passes_through(self):
        """`None` coerces to `None`."""
        assert Compression.coerce(None) is None

    def test_coerce_instance_is_identity(self):
        """An existing `Compression` is returned unchanged."""
        c = Compression(compress="LZW")
        assert Compression.coerce(c) is c

    def test_coerce_unknown_profile_rejected(self):
        """A non-profile string raises `ValueError`."""
        with pytest.raises(ValueError, match="unknown COG profile"):
            Compression.coerce("nope")

    def test_to_options_defaults_deflate_with_float_predictor(self):
        """A bare `Compression` yields DEFLATE with the float predictor.

        Test scenario:
            `Compression()._to_options(float_band)` defaults COMPRESS to DEFLATE,
            resolves PREDICTOR to 3 for a float source, and leaves LEVEL / QUALITY
            / MAX_Z_ERROR as `None`.
        """
        ds = _mem_dataset(gdal.GDT_Float32)
        band = ds.GetRasterBand(1)
        opts = Compression()._to_options(band)
        assert opts["COMPRESS"] == "DEFLATE", f"expected DEFLATE, got {opts['COMPRESS']}"
        assert opts["PREDICTOR"] == 3, f"float predictor should be 3, got {opts['PREDICTOR']}"
        assert opts["LEVEL"] is None and opts["QUALITY"] is None, f"unexpected: {opts}"
        assert opts["MAX_Z_ERROR"] is None, f"MAX_Z_ERROR should be None, got {opts['MAX_Z_ERROR']}"

    def test_to_options_integer_source_gets_predictor_2(self):
        """An integer source resolves PREDICTOR to 2.

        Test scenario:
            `Compression()._to_options(byte_band)` picks the horizontal-differencing
            predictor (2) suited to integer data.
        """
        ds = _mem_dataset(gdal.GDT_Byte)
        band = ds.GetRasterBand(1)
        assert Compression()._to_options(band)["PREDICTOR"] == 2, "integer predictor should be 2"

    def test_to_options_explicit_fields_passthrough(self):
        """Explicit compression fields are forwarded verbatim.

        Test scenario:
            `Compression(compress="ZSTD", level=18, predictor=1)` forwards
            COMPRESS/LEVEL and the explicit predictor (ZSTD honours PREDICTOR).
        """
        ds = _mem_dataset(gdal.GDT_Float32)
        band = ds.GetRasterBand(1)
        opts = Compression(compress="ZSTD", level=18, predictor=1)._to_options(band)
        assert (opts["COMPRESS"], opts["LEVEL"]) == ("ZSTD", 18), f"unexpected: {opts}"
        assert opts["PREDICTOR"] == 1, f"explicit predictor should win, got {opts['PREDICTOR']}"

    @pytest.mark.parametrize("compress", ["LERC", "NONE", "JPEG", "WEBP", "PACKBITS"])
    def test_to_options_predictor_dropped_for_non_predictor_compressor(self, compress):
        """PREDICTOR is omitted for methods that ignore it.

        Args:
            compress: A compression method outside the predictor-honouring set.

        Test scenario:
            Even with an explicit predictor, `_to_options` drops PREDICTOR (`None`)
            for LERC/NONE/JPEG/WEBP/PACKBITS, which GDAL would warn on.
        """
        ds = _mem_dataset(gdal.GDT_Byte)
        band = ds.GetRasterBand(1)
        opts = Compression(compress=compress, predictor=2)._to_options(band)
        assert opts["PREDICTOR"] is None, f"{compress} should drop PREDICTOR, got {opts['PREDICTOR']}"

    def test_to_options_max_z_error_carried(self):
        """`max_z_error` rides along as MAX_Z_ERROR.

        Test scenario:
            `Compression(compress="LERC", max_z_error=0.5)._to_options(band)`
            forwards the LERC tolerance.
        """
        ds = _mem_dataset(gdal.GDT_Float32)
        band = ds.GetRasterBand(1)
        opts = Compression(compress="LERC", max_z_error=0.5)._to_options(band)
        assert opts["MAX_Z_ERROR"] == 0.5, f"expected 0.5, got {opts['MAX_Z_ERROR']}"


class TestOverviews:
    """Validation for `Overviews`."""

    def test_negative_count_rejected(self):
        """A negative overview count raises `ValueError`."""
        with pytest.raises(ValueError, match="overview count must be >= 0"):
            Overviews(count=-1)

    def test_zero_count_accepted(self):
        """A zero overview count is accepted."""
        assert Overviews(count=0).count == 0

    def test_to_options_default_resampling_continuous(self):
        """A continuous (float, no palette) source defaults to averaging.

        Test scenario:
            `Overviews()._to_options(float_band)` resolves OVERVIEW_RESAMPLING to
            `average` and leaves count / compress as `None`.
        """
        ds = _mem_dataset(gdal.GDT_Float32)
        band = ds.GetRasterBand(1)
        opts = Overviews()._to_options(band)
        assert opts["OVERVIEW_RESAMPLING"] == "average", f"got {opts['OVERVIEW_RESAMPLING']}"
        assert opts["OVERVIEW_COUNT"] is None and opts["OVERVIEW_COMPRESS"] is None, f"unexpected: {opts}"

    def test_to_options_default_resampling_integer_is_mode(self):
        """An integer source defaults to the category-safe `mode` resampler.

        Test scenario:
            `Overviews()._to_options(byte_band)` never averages categorical data.
        """
        ds = _mem_dataset(gdal.GDT_Byte)
        band = ds.GetRasterBand(1)
        assert Overviews()._to_options(band)["OVERVIEW_RESAMPLING"] == "mode", "integer should use mode"

    def test_to_options_default_resampling_color_table_is_mode(self):
        """A palette source (float + colour table) still defaults to `mode`.

        Test scenario:
            A colour table marks the source categorical regardless of dtype.
        """
        ds = _mem_dataset(gdal.GDT_Float32, color_table=True)
        band = ds.GetRasterBand(1)
        assert Overviews()._to_options(band)["OVERVIEW_RESAMPLING"] == "mode", "palette should use mode"

    def test_to_options_explicit_fields_passthrough(self):
        """Explicit resampling / count / compress are forwarded verbatim.

        Test scenario:
            `Overviews(resampling="lanczos", count=4, compress="ZSTD")` maps each
            field to its OVERVIEW_* key without touching the source band.
        """
        ds = _mem_dataset(gdal.GDT_Byte)
        band = ds.GetRasterBand(1)
        opts = Overviews(resampling="lanczos", count=4, compress="ZSTD")._to_options(band)
        assert opts == {
            "OVERVIEW_RESAMPLING": "lanczos",
            "OVERVIEW_COUNT": 4,
            "OVERVIEW_COMPRESS": "ZSTD",
        }, f"unexpected: {opts}"


class TestTiling:
    """Validation for `Tiling`."""

    @pytest.mark.parametrize("strategy", ["auto", "lower", "upper"])
    def test_valid_strategies_accepted(self, strategy):
        """Every valid zoom-level strategy is accepted.

        Args:
            strategy: A valid zoom-level strategy.
        """
        assert Tiling(zoom_level_strategy=strategy).zoom_level_strategy == strategy

    def test_invalid_strategy_rejected(self):
        """An unknown zoom-level strategy raises `ValueError`."""
        with pytest.raises(ValueError, match="zoom_level_strategy must be"):
            Tiling(zoom_level_strategy="sideways")

    def test_to_options_plain_has_no_reprojection(self):
        """A bare `Tiling` emits no scheme, no target SRS, no warp resampling.

        Test scenario:
            `Tiling()._to_options()` leaves TILING_SCHEME `None`, drops
            WARP_RESAMPLING (not reprojecting), and omits the TARGET_SRS key.
        """
        opts = Tiling()._to_options()
        assert opts["TILING_SCHEME"] is None, f"expected no scheme, got {opts['TILING_SCHEME']}"
        assert opts["WARP_RESAMPLING"] is None, f"warp should be dropped, got {opts['WARP_RESAMPLING']}"
        assert "TARGET_SRS" not in opts, f"TARGET_SRS should be absent, got {opts}"

    def test_to_options_target_srs_int_formats_epsg(self):
        """An integer `target_srs` becomes `EPSG:<n>` and enables warp resampling.

        Test scenario:
            `Tiling(target_srs=3857, resampling="bilinear")._to_options()` formats
            the SRS and, because it reprojects, forwards WARP_RESAMPLING.
        """
        opts = Tiling(target_srs=3857, resampling="bilinear")._to_options()
        assert opts["TARGET_SRS"] == "EPSG:3857", f"got {opts['TARGET_SRS']}"
        assert opts["WARP_RESAMPLING"] == "bilinear", f"got {opts['WARP_RESAMPLING']}"

    def test_to_options_target_srs_string_verbatim(self):
        """A string `target_srs` is forwarded unchanged.

        Test scenario:
            `Tiling(target_srs="EPSG:3857")` is not re-formatted (already a string).
        """
        assert Tiling(target_srs="EPSG:3857")._to_options()["TARGET_SRS"] == "EPSG:3857"

    def test_to_options_scheme_sets_warp_no_target_srs(self):
        """A tiling scheme reprojects (warp on) but sets no TARGET_SRS.

        Test scenario:
            `Tiling(scheme="GoogleMapsCompatible", resampling="cubic")` forwards
            the scheme and WARP_RESAMPLING; TARGET_SRS stays absent.
        """
        opts = Tiling(scheme="GoogleMapsCompatible", resampling="cubic")._to_options()
        assert opts["TILING_SCHEME"] == "GoogleMapsCompatible", f"got {opts['TILING_SCHEME']}"
        assert opts["WARP_RESAMPLING"] == "cubic", f"got {opts['WARP_RESAMPLING']}"
        assert "TARGET_SRS" not in opts, f"TARGET_SRS should be absent, got {opts}"

    def test_to_options_scheme_and_target_srs_conflict_warns_scheme_wins(self):
        """When both scheme and target_srs are set, scheme wins with a warning.

        Test scenario:
            `Tiling(scheme=..., target_srs=3857)._to_options()` emits a
            `UserWarning`, drops TARGET_SRS, and keeps the scheme.
        """
        til = Tiling(scheme="GoogleMapsCompatible", target_srs=3857)
        with pytest.warns(UserWarning, match="scheme wins and target_srs is ignored"):
            opts = til._to_options()
        assert "TARGET_SRS" not in opts, f"target_srs should be dropped, got {opts}"
        assert opts["TILING_SCHEME"] == "GoogleMapsCompatible", f"scheme should win, got {opts}"

    def test_to_options_zoom_and_aligned_fields_passthrough(self):
        """Zoom / aligned-level knobs are forwarded verbatim.

        Test scenario:
            `zoom_level`, `zoom_level_strategy`, and `aligned_levels` map straight
            to their GDAL keys.
        """
        opts = Tiling(zoom_level=5, zoom_level_strategy="upper", aligned_levels=2)._to_options()
        assert (opts["ZOOM_LEVEL"], opts["ZOOM_LEVEL_STRATEGY"], opts["ALIGNED_LEVELS"]) == (
            5,
            "upper",
            2,
        ), f"unexpected: {opts}"


class TestBandSelection:
    """Validation for `BandSelection`."""

    def test_negative_index_rejected(self):
        """A negative band index raises `ValueError`."""
        with pytest.raises(ValueError, match="band indexes must be >= 0"):
            BandSelection(indexes=[0, -1])

    def test_zero_based_indexes_accepted(self):
        """Non-negative 0-based indices are accepted in order."""
        assert BandSelection(indexes=[2, 0, 1]).indexes == [2, 0, 1]

    def test_needs_translate_false_when_empty(self):
        """A bare `BandSelection` needs no pre-process."""
        assert BandSelection()._needs_translate() is False, "empty selection should not translate"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"indexes": [0]},
            {"out_dtype": "uint8"},
            {"nodata": 0},
        ],
    )
    def test_needs_translate_true_when_any_field_set(self, kwargs):
        """Any of indexes / out_dtype / nodata requires a translate.

        Args:
            kwargs: A single populated `BandSelection` field.
        """
        assert BandSelection(**kwargs)._needs_translate() is True, f"{kwargs} should translate"

    def test_translate_subsets_and_reorders_bands(self):
        """`_translate` keeps and reorders the requested bands.

        Test scenario:
            A 3-band source with `indexes=[2, 0]` yields a 2-band MEM dataset.
        """
        src = _mem_dataset(gdal.GDT_Byte, n_bands=3)
        mem = BandSelection(indexes=[2, 0])._translate(src)
        assert mem.RasterCount == 2, f"expected 2 bands, got {mem.RasterCount}"

    def test_translate_casts_output_dtype(self):
        """`_translate` casts to the requested output dtype.

        Test scenario:
            A float source with `out_dtype="uint8"` yields a Byte MEM band.
        """
        src = _mem_dataset(gdal.GDT_Float32)
        mem = BandSelection(out_dtype="uint8")._translate(src)
        assert mem.GetRasterBand(1).DataType == gdal.GDT_Byte, "output should be cast to Byte"

    def test_translate_sets_nodata(self):
        """`_translate` stamps the requested NoData value.

        Test scenario:
            `BandSelection(nodata=0)._translate(src)` sets NoData 0 on band 1.
        """
        src = _mem_dataset(gdal.GDT_Float32)
        mem = BandSelection(nodata=0)._translate(src)
        assert mem.GetRasterBand(1).GetNoDataValue() == 0, "NoData should be set to 0"

    def test_translate_raises_when_gdal_returns_none(self, monkeypatch):
        """`_translate` raises `FailedToSaveError` when GDAL yields no dataset.

        Args:
            monkeypatch: pytest fixture used to force `gdal.Translate` to `None`.

        Test scenario:
            A `None` return from `gdal.Translate` surfaces as a descriptive
            `FailedToSaveError` naming the requested fields.
        """
        src = _mem_dataset(gdal.GDT_Float32)
        monkeypatch.setattr(
            "pyramids.dataset.cog.options.gdal.Translate", lambda *a, **k: None
        )
        with pytest.raises(FailedToSaveError, match="pre-processed COG source") as exc:
            BandSelection(out_dtype="uint8")._translate(src)
        assert "out_dtype" in str(exc.value), f"message should name the fields, got: {exc.value}"


class TestLayout:
    """Validation for `Layout`."""

    def test_default_blocksize(self):
        """The default `Layout` uses a 512 blocksize."""
        assert Layout().blocksize == 512

    def test_bad_blocksize_rejected(self):
        """A non-power-of-2 blocksize raises `ValueError`."""
        with pytest.raises(ValueError, match="blocksize must be a power of 2"):
            Layout(blocksize=100)

    @pytest.mark.parametrize("bigtiff", ["IF_SAFER", "YES", "NO", "IF_NEEDED"])
    def test_valid_bigtiff_accepted(self, bigtiff):
        """Every valid BIGTIFF token is accepted.

        Args:
            bigtiff: A valid BIGTIFF value.
        """
        assert Layout(bigtiff=bigtiff).bigtiff == bigtiff

    def test_bad_bigtiff_rejected(self):
        """An unknown BIGTIFF value raises `ValueError`."""
        with pytest.raises(ValueError, match="bigtiff must be"):
            Layout(bigtiff="MAYBE")

    def test_to_options_defaults(self):
        """The default `Layout` serializes the house defaults.

        Test scenario:
            `Layout()._to_options()` emits BLOCKSIZE 512, BIGTIFF IF_SAFER,
            NUM_THREADS "ALL_CPUS", STATISTICS "YES", and drops the two unset
            toggles (`None`).
        """
        opts = Layout()._to_options()
        assert opts == {
            "BLOCKSIZE": 512,
            "BIGTIFF": "IF_SAFER",
            "NUM_THREADS": "ALL_CPUS",
            "ADD_ALPHA": None,
            "SPARSE_OK": None,
            "STATISTICS": "YES",
        }, f"unexpected defaults: {opts}"

    def test_to_options_num_threads_int_stringified(self):
        """An integer `num_threads` is stringified.

        Test scenario:
            `Layout(num_threads=4)._to_options()` forwards NUM_THREADS "4".
        """
        assert Layout(num_threads=4)._to_options()["NUM_THREADS"] == "4", "int threads should stringify"

    def test_to_options_toggles_map_to_yes_or_dropped(self):
        """The boolean toggles map to `True`/`None` (kept/dropped downstream).

        Test scenario:
            `Layout(add_mask=True, sparse_ok=True, statistics=False)` sets
            ADD_ALPHA / SPARSE_OK truthy and drops STATISTICS (`None`).
        """
        opts = Layout(add_mask=True, sparse_ok=True, statistics=False)._to_options()
        assert opts["ADD_ALPHA"] is True and opts["SPARSE_OK"] is True, f"toggles should be on: {opts}"
        assert opts["STATISTICS"] is None, f"statistics=False should drop, got {opts['STATISTICS']}"


class TestTags:
    """`Tags` is a plain carrier; `_has_any` / `_stamp` apply its contents."""

    def test_fields_round_trip(self):
        """The three fields are stored verbatim."""
        t = Tags(band_tags={0: {"name": "x"}}, metadata={"a": "b"})
        assert t.band_tags == {0: {"name": "x"}}
        assert t.metadata == {"a": "b"}

    def test_has_any_false_when_empty(self):
        """A bare `Tags` carries nothing to stamp."""
        assert Tags()._has_any() is False, "empty tags should have nothing to stamp"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"band_tags": {0: {"name": "x"}}},
            {"colormap": {0: (1, 2, 3, 4)}},
            {"metadata": {"a": "b"}},
        ],
    )
    def test_has_any_true_when_any_field_set(self, kwargs):
        """Any populated field makes `_has_any` true.

        Args:
            kwargs: A single populated `Tags` field.
        """
        assert Tags(**kwargs)._has_any() is True, f"{kwargs} should stamp"

    def test_stamp_metadata_sets_dataset_items(self):
        """`_stamp` writes dataset-level metadata as strings.

        Test scenario:
            `Tags(metadata={"source": "s2"})._stamp(ds)` sets the item on the MEM
            dataset.
        """
        ds = _mem_dataset(gdal.GDT_Byte)
        Tags(metadata={"source": "s2"})._stamp(ds)
        assert ds.GetMetadata().get("source") == "s2", f"metadata not stamped: {ds.GetMetadata()}"

    def test_stamp_band_tags_set_per_band(self):
        """`_stamp` writes per-band tags at the 1-based band number.

        Test scenario:
            `Tags(band_tags={0: {"name": "NDVI"}})._stamp(ds)` stamps band 1.
        """
        ds = _mem_dataset(gdal.GDT_Byte)
        Tags(band_tags={0: {"name": "NDVI"}})._stamp(ds)
        got = ds.GetRasterBand(1).GetMetadata().get("name")
        assert got == "NDVI", f"band tag not stamped, got {got!r}"

    def test_stamp_colormap_on_byte_builds_palette(self):
        """`_stamp` attaches a colour table to a Byte band.

        Test scenario:
            `Tags(colormap={0: (1, 2, 3, 4)})._stamp(byte_ds)` sets a colour table
            and flips the colour interpretation to palette.
        """
        ds = _mem_dataset(gdal.GDT_Byte)
        Tags(colormap={0: (1, 2, 3, 4)})._stamp(ds)
        band = ds.GetRasterBand(1)
        assert band.GetColorTable() is not None, "colour table should be attached"
        assert band.GetColorInterpretation() == gdal.GCI_PaletteIndex, "should be palette-interpreted"

    def test_stamp_colormap_rejects_non_byte_dtype(self):
        """`_stamp` rejects a colourmap on a non-Byte/UInt16 band.

        Test scenario:
            A float band with a colourmap raises `ValueError` naming the
            Byte/UInt16 constraint.
        """
        ds = _mem_dataset(gdal.GDT_Float32)
        with pytest.raises(ValueError, match="only supported on Byte/UInt16") as exc:
            Tags(colormap={0: (1, 2, 3, 4)})._stamp(ds)
        assert "Byte/UInt16" in str(exc.value), f"unexpected message: {exc.value}"
