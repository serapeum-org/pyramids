"""Tests for :meth:`pyramids.dataset.Dataset.from_band_files` and helpers.

Covers the band-stacking factory, the band-name derivation helper, the
grid-equality helper, and the :func:`pyramids.dataset.merge.stack_bands`
free-function alias.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import AlignmentError, CRSError
from pyramids.dataset import Dataset
from pyramids.dataset.dataset import _derive_band_names, _same_grid
from pyramids.dataset.merge import stack_bands

pytestmark = pytest.mark.core


def _make_band(
    tmp_path,
    name,
    value,
    *,
    dtype="int16",
    cell_size=1.0,
    top_left=(0.0, 0.0),
    epsg=4326,
    no_data_value=-9999,
    shape=(4, 5),
):
    """Write a single-band raster filled with a constant and return its path.

    Args:
        tmp_path: pytest temp directory.
        name: File name (including ``.tif``).
        value: Constant fill value.
        dtype: numpy dtype string for the band.
        cell_size: Pixel size.
        top_left: ``(x, y)`` of the top-left corner.
        epsg: EPSG code for the CRS.
        no_data_value: No-data value stamped on the band.
        shape: ``(rows, cols)``.

    Returns:
        str: Path to the written GeoTIFF.
    """
    path = os.path.join(str(tmp_path), name)
    Dataset.create_from_array(
        np.full(shape, value, dtype=dtype),
        top_left_corner=top_left,
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=no_data_value,
        path=path,
    ).close()
    return path


@pytest.fixture()
def band_files(tmp_path):
    """Three co-registered single-band GeoTIFFs named ``scene.B<n>.tif``.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        list[str]: Paths to ``scene.B2.tif`` (=2), ``scene.B3.tif`` (=3),
        ``scene.B4.tif`` (=4).
    """
    return [
        _make_band(tmp_path, "scene.B2.tif", 2),
        _make_band(tmp_path, "scene.B3.tif", 3),
        _make_band(tmp_path, "scene.B4.tif", 4),
    ]


class TestDeriveBandNames:
    """Tests for :func:`pyramids.dataset.dataset._derive_band_names`."""

    def test_earth_engine_pattern(self):
        """``<slug>.<band>.tif`` files yield the ``<band>`` token.

        Test scenario:
            ``["a.B2.tif", "a.B3.tif", "a.B4.tif"]`` — expected:
            ``["B2", "B3", "B4"]``.
        """
        names = _derive_band_names(["x/a.B2.tif", "x/a.B3.tif", "x/a.B4.tif"])
        assert names == ["B2", "B3", "B4"], f"unexpected names: {names}"

    def test_dotless_stem_kept_whole(self):
        """Files whose stem has no dot keep the whole stem as the band name.

        Test scenario:
            Landsat-style ``LC08_..._SR_B4.TIF`` — expected: the full stem.
        """
        names = _derive_band_names(
            ["dir/LC08_L2SP_SR_B4.TIF", "dir/LC08_L2SP_SR_B5.TIF"]
        )
        assert names == [
            "LC08_L2SP_SR_B4",
            "LC08_L2SP_SR_B5",
        ], f"unexpected names: {names}"

    def test_duplicate_names_get_suffix(self):
        """Colliding derived names get a ``_<n>`` disambiguating suffix.

        Test scenario:
            Two files in different dirs with the same name — expected: the
            second gets ``_1``.
        """
        names = _derive_band_names(
            ["a/scene.B2.tif", "b/scene.B2.tif", "c/scene.B2.tif"]
        )
        assert names == ["B2", "B2_1", "B2_2"], f"unexpected names: {names}"


class TestSameGrid:
    """Tests for :func:`pyramids.dataset.dataset._same_grid`."""

    def test_identical_grids(self, tmp_path):
        """Two rasters created with the same geo parameters compare equal.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Two ``4x5`` rasters, same cell size / corner / epsg — expected:
            ``_same_grid`` is ``True``.
        """
        a = Dataset.read_file(_make_band(tmp_path, "a.tif", 1))
        b = Dataset.read_file(_make_band(tmp_path, "b.tif", 2))
        assert _same_grid(a, b) is True, "identical grids should compare equal"

    @pytest.mark.parametrize(
        "kw",
        [
            {"cell_size": 2.0},
            {"top_left": (100.0, 100.0)},
            {"epsg": 3857},
            {"shape": (8, 9)},
        ],
        ids=["cell_size", "extent", "crs", "size"],
    )
    def test_mismatched_grids(self, tmp_path, kw):
        """Any of cell size / extent / CRS / size differing makes grids unequal.

        Args:
            tmp_path: pytest temp directory.
            kw: One geo parameter overridden on the second raster.

        Test scenario:
            Reference vs a raster differing in one property — expected:
            ``_same_grid`` is ``False``.
        """
        a = Dataset.read_file(_make_band(tmp_path, "a.tif", 1))
        b = Dataset.read_file(_make_band(tmp_path, "b.tif", 2, **kw))
        assert (
            _same_grid(a, b) is False
        ), f"grids differing in {list(kw)} should be unequal"


class TestFromBandFiles:
    """Tests for :meth:`Dataset.from_band_files`."""

    def test_basic_stack_derives_names_and_preserves_values(self, band_files):
        """Three single-band files stack into one 3-band dataset.

        Args:
            band_files: Three ``scene.B<n>.tif`` paths.

        Test scenario:
            ``Dataset.from_band_files(band_files)`` — expected: 3 bands,
            names ``["B2", "B3", "B4"]``, each band carries its source's
            constant value.
        """
        ds = Dataset.from_band_files(band_files)
        assert ds.band_count == 3, f"expected 3 bands, got {ds.band_count}"
        assert ds.band_names == ["B2", "B3", "B4"], f"unexpected names: {ds.band_names}"
        assert [int(ds.read_array(band=i).flat[0]) for i in range(3)] == [
            2,
            3,
            4,
        ], "per-band values were not preserved"
        assert ds.epsg == 4326, f"unexpected epsg: {ds.epsg}"

    def test_explicit_band_names(self, band_files):
        """``band_names=`` overrides the derived names.

        Args:
            band_files: Three band-file paths.

        Test scenario:
            ``from_band_files(..., band_names=["blue", "green", "red"])`` —
            expected: those names verbatim.
        """
        ds = Dataset.from_band_files(band_files, band_names=["blue", "green", "red"])
        assert ds.band_names == [
            "blue",
            "green",
            "red",
        ], f"names not applied: {ds.band_names}"

    def test_single_file_gives_one_band(self, band_files):
        """A one-element list yields a single-band dataset.

        Args:
            band_files: Three band-file paths (only the first is used).

        Test scenario:
            ``from_band_files([band_files[0]])`` — expected: 1 band named
            ``"B2"``.
        """
        ds = Dataset.from_band_files([band_files[0]])
        assert ds.band_count == 1, f"expected 1 band, got {ds.band_count}"
        assert ds.band_names == ["B2"], f"unexpected names: {ds.band_names}"

    def test_empty_files_raises_value_error(self):
        """An empty file list is rejected.

        Test scenario:
            ``from_band_files([])`` — expected: ``ValueError`` mentioning
            ``at least one``.
        """
        with pytest.raises(ValueError, match="at least one"):
            Dataset.from_band_files([])

    def test_band_names_length_mismatch_raises(self, band_files):
        """``band_names`` must have one entry per file.

        Args:
            band_files: Three band-file paths.

        Test scenario:
            ``from_band_files(3 files, band_names=2 names)`` — expected:
            ``ValueError``.
        """
        with pytest.raises(ValueError, match="band_names has 2 entries"):
            Dataset.from_band_files(band_files, band_names=["a", "b"])

    def test_multi_band_input_raises(self, tmp_path, band_files):
        """An input with more than one band is rejected.

        Args:
            tmp_path: pytest temp directory.
            band_files: A valid single-band file to pair it with.

        Test scenario:
            One 2-band input — expected: ``ValueError`` mentioning ``one
            band per file``.
        """
        mb = os.path.join(str(tmp_path), "mb.tif")
        Dataset.create_from_array(
            np.zeros((2, 4, 5), dtype="int16"),
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            path=mb,
        ).close()
        with pytest.raises(ValueError, match="one band per file"):
            Dataset.from_band_files([band_files[0], mb])

    def test_input_without_crs_raises_crs_error(self, tmp_path):
        """An input raster lacking a CRS is rejected.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A raw GeoTIFF created without a projection — expected:
            ``CRSError`` mentioning ``no CRS``.
        """
        nocrs = os.path.join(str(tmp_path), "nocrs.tif")
        drv = gdal.GetDriverByName("GTiff").Create(nocrs, 5, 4, 1, gdal.GDT_Int16)
        drv.GetRasterBand(1).Fill(1)
        drv = None
        with pytest.raises(CRSError, match="no CRS"):
            Dataset.from_band_files([nocrs, nocrs])

    def test_mismatched_grid_without_align_raises_alignment_error(
        self, tmp_path, band_files
    ):
        """Differing grids raise :class:`AlignmentError` unless ``align=True``.

        Args:
            tmp_path: pytest temp directory.
            band_files: Provides a reference-grid file.

        Test scenario:
            One file on a different (finer) grid — expected: ``AlignmentError``
            whose message points the user at ``align=True``.
        """
        odd = _make_band(tmp_path, "odd.tif", 0, cell_size=0.5, shape=(8, 9))
        with pytest.raises(AlignmentError, match="align=True"):
            Dataset.from_band_files([band_files[0], odd])

    def test_align_true_resamples_onto_first_grid(self, tmp_path, band_files):
        """``align=True`` resamples mismatched inputs onto ``files[0]``'s grid.

        Args:
            tmp_path: pytest temp directory.
            band_files: Provides the template-grid file.

        Test scenario:
            Stack a ``4x5`` template with an ``8x9`` finer raster, ``align=True``
            — expected: a 2-band result on the template's ``4x5`` grid.
        """
        odd = _make_band(tmp_path, "odd.tif", 7, cell_size=0.5, shape=(8, 9))
        template = Dataset.read_file(band_files[0])
        ds = Dataset.from_band_files([band_files[0], odd], align=True)
        assert ds.band_count == 2, f"expected 2 bands, got {ds.band_count}"
        assert (ds.rows, ds.columns) == (
            template.rows,
            template.columns,
        ), f"result grid {ds.rows}x{ds.columns} != template {template.rows}x{template.columns}"
        assert (
            ds.geotransform == template.geotransform
        ), "result geotransform != template"

    def test_mixed_dtypes_promote_and_preserve_float_values(self, tmp_path, band_files):
        """Inputs of different dtypes promote to a common dtype without truncation.

        Args:
            tmp_path: pytest temp directory.
            band_files: Provides an int16 band.

        Test scenario:
            Stack an int16 band with a float32 band of value ``1.5`` — expected:
            both output bands are float32 and the ``1.5`` survives (would be
            ``1`` if naively cast to int16).
        """
        f32 = _make_band(
            tmp_path, "scene.B8.tif", 1.5, dtype="float32", no_data_value=-9999
        )
        ds = Dataset.from_band_files([band_files[0], f32])
        assert ds.dtype == ["float32", "float32"], f"unexpected dtypes: {ds.dtype}"
        assert float(ds.read_array(band=1).flat[0]) == pytest.approx(
            1.5
        ), "float value truncated"

    def test_nodata_inherited_when_sources_agree(self, tmp_path):
        """When all sources share a no-data value, the output keeps it.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Two sources both with no-data ``-1`` — expected: output bands all
            have no-data ``-1``.
        """
        a = _make_band(tmp_path, "scene.B2.tif", 2, no_data_value=-1)
        b = _make_band(tmp_path, "scene.B3.tif", 3, no_data_value=-1)
        ds = Dataset.from_band_files([a, b])
        assert ds.no_data_value == (
            -1.0,
            -1.0,
        ), f"unexpected no-data: {ds.no_data_value}"

    def test_nodata_disagreement_warns_and_first_wins(self, tmp_path):
        """Disagreeing source no-data values trigger a warning; the first wins.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Sources with no-data ``-1`` and ``-2`` — expected: a ``UserWarning``
            and output no-data ``-1``.
        """
        a = _make_band(tmp_path, "scene.B2.tif", 2, no_data_value=-1)
        b = _make_band(tmp_path, "scene.B3.tif", 3, no_data_value=-2)
        with pytest.warns(UserWarning, match="disagree on no-data"):
            ds = Dataset.from_band_files([a, b])
        assert ds.no_data_value == (
            -1.0,
            -1.0,
        ), f"first source's no-data should win: {ds.no_data_value}"

    def test_no_source_nodata_yields_none(self, tmp_path):
        """If no source declares a no-data value, neither does the output.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Two sources created with ``no_data_value=None`` — expected: output
            bands have ``None`` no-data.
        """
        a = _make_band(tmp_path, "scene.B2.tif", 2, no_data_value=None)
        b = _make_band(tmp_path, "scene.B3.tif", 3, no_data_value=None)
        ds = Dataset.from_band_files([a, b])
        assert ds.no_data_value == (
            None,
            None,
        ), f"expected (None, None), got {ds.no_data_value}"

    def test_explicit_nodata_override(self, band_files):
        """An explicit ``no_data_value=`` overrides whatever the sources carry.

        Args:
            band_files: Sources with no-data ``-9999``.

        Test scenario:
            ``from_band_files(..., no_data_value=99)`` — expected: all output
            bands report ``99``.
        """
        ds = Dataset.from_band_files(band_files, no_data_value=99)
        assert ds.no_data_value == (
            99.0,
            99.0,
            99.0,
        ), f"override ignored: {ds.no_data_value}"

    def test_explicit_nodata_none_clears_it(self, tmp_path):
        """Passing ``no_data_value=None`` strips the inherited no-data.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Sources with no-data ``-1``, ``from_band_files(..., no_data_value=None)``
            — expected: output bands report ``None`` (the source value is
            dropped, not kept).
        """
        a = _make_band(tmp_path, "scene.B2.tif", 2, no_data_value=-1)
        b = _make_band(tmp_path, "scene.B3.tif", 3, no_data_value=-1)
        ds = Dataset.from_band_files([a, b], no_data_value=None)
        assert ds.no_data_value == (
            None,
            None,
        ), f"no-data should have been cleared: {ds.no_data_value}"

    def test_writes_to_disk_and_round_trips(self, tmp_path, band_files):
        """``path=`` writes a real GeoTIFF whose bands/names/values reload intact.

        Args:
            tmp_path: pytest temp directory.
            band_files: The three source bands.

        Test scenario:
            ``from_band_files(..., path=out.tif)`` then ``Dataset.read_file(out)``
            — expected: 3 bands, names ``["B2", "B3", "B4"]``, values ``[2, 3, 4]``.
        """
        out = os.path.join(str(tmp_path), "stacked.tif")
        Dataset.from_band_files(band_files, path=out)
        assert os.path.exists(out), "output file was not written"
        reloaded = Dataset.read_file(out)
        assert (
            reloaded.band_count == 3
        ), f"expected 3 bands on reload, got {reloaded.band_count}"
        assert reloaded.band_names == [
            "B2",
            "B3",
            "B4",
        ], f"band names lost: {reloaded.band_names}"
        assert [int(reloaded.read_array(band=i).flat[0]) for i in range(3)] == [
            2,
            3,
            4,
        ], "values changed on round-trip"

    def test_path_without_tif_extension_raises(self, band_files):
        """A non ``.tif`` output path is rejected.

        Args:
            band_files: The three source bands.

        Test scenario:
            ``from_band_files(..., path="out.png")`` — expected: ``ValueError``
            mentioning ``.tif``.
        """
        with pytest.raises(ValueError, match=r"\.tif"):
            Dataset.from_band_files(band_files, path="out.png")

    def test_align_with_mixed_dtypes(self, tmp_path, band_files):
        """``align=True`` and heterogeneous dtypes compose: resample then promote.

        Args:
            tmp_path: pytest temp directory.
            band_files: Provides an int16 template band.

        Test scenario:
            int16 ``4x5`` template + float32 ``8x9`` finer raster, ``align=True``
            — expected: 2-band float32 result on the template grid, float value
            preserved.
        """
        f32 = _make_band(
            tmp_path, "fine.tif", 0.25, dtype="float32", cell_size=0.5, shape=(8, 9)
        )
        template = Dataset.read_file(band_files[0])
        ds = Dataset.from_band_files([band_files[0], f32], align=True)
        assert ds.dtype == ["float32", "float32"], f"unexpected dtypes: {ds.dtype}"
        assert (ds.rows, ds.columns) == (
            template.rows,
            template.columns,
        ), "result not on template grid"
        assert float(ds.read_array(band=1).flat[0]) == pytest.approx(
            0.25
        ), "float value lost"

    def test_align_disagreeing_nodata_remaps_fringe_to_resolved(self, tmp_path):
        """``align=True`` with disagreeing source no-data: out-of-coverage fringe matches declared no-data.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Source A on a coarse grid with ``no_data_value=0``; source B on a
            FINER grid (so it covers a subset of A's extent) with
            ``no_data_value=65535``. The first-wins policy resolves to ``0``
            and a ``UserWarning`` fires. After ``align=True``, band 1 (from B)
            is aligned onto A's coarse grid — out-of-coverage cells used to
            carry B's ``65535`` sentinel while the output's declared no-data
            was ``0`` (mismatch). Expected after the fix: band 1's fringe is
            remapped to ``0`` so the array matches the declared no-data, while
            the in-coverage data cells keep their real value (``9``).
        """
        pA = _make_band(
            tmp_path,
            "a.tif",
            7,
            dtype="int32",
            top_left=(0.0, 0.0),
            cell_size=0.05,
            shape=(10, 10),
            no_data_value=0,
        )
        pB = _make_band(
            tmp_path,
            "b.tif",
            9,
            dtype="int32",
            top_left=(0.1, -0.1),
            cell_size=0.025,
            shape=(8, 8),
            no_data_value=65535,
        )
        with pytest.warns(UserWarning, match="disagree on no-data"):
            ds = Dataset.from_band_files([pA, pB], align=True)
        band1 = ds.read_array(band=1)
        uniques = sorted(set(band1.flatten().tolist()))
        assert ds.no_data_value == (0.0, 0.0), f"declared no-data: {ds.no_data_value}"
        assert (
            65535 not in uniques
        ), f"B's original no-data leaked into the aligned fringe: {uniques}"
        assert (
            0 in uniques and 9 in uniques
        ), f"expected the resolved no-data (0) and the real value (9): {uniques}"

    def test_align_agreeing_nodata_preserved(self, tmp_path):
        """``align=True`` with agreeing source no-data: fringe equals declared no-data (regression).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Both sources share ``no_data_value=0``. After ``align=True`` the
            aligned fringe must still be ``0`` (no extra remap, no regression
            from the disagreement fix above).
        """
        pA = _make_band(
            tmp_path,
            "a.tif",
            7,
            dtype="int32",
            top_left=(0.0, 0.0),
            cell_size=0.05,
            shape=(10, 10),
            no_data_value=0,
        )
        pB = _make_band(
            tmp_path,
            "b.tif",
            9,
            dtype="int32",
            top_left=(0.1, -0.1),
            cell_size=0.025,
            shape=(8, 8),
            no_data_value=0,
        )
        ds = Dataset.from_band_files([pA, pB], align=True)
        band1 = ds.read_array(band=1)
        uniques = sorted(set(band1.flatten().tolist()))
        assert ds.no_data_value == (0.0, 0.0), f"declared no-data: {ds.no_data_value}"
        assert uniques == [0, 9], f"aligned-agree case unexpected: {uniques}"

    def test_align_with_explicit_override_remaps_fringe(self, tmp_path):
        """An explicit ``no_data_value=`` override is propagated into every band's fringe.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Both sources use ``no_data_value=0``; caller passes
            ``no_data_value=42`` to override. After ``align=True`` band 1's
            fringe must contain the override value (``42``), not the source's
            (``0``).
        """
        pA = _make_band(
            tmp_path,
            "a.tif",
            7,
            dtype="int32",
            top_left=(0.0, 0.0),
            cell_size=0.05,
            shape=(10, 10),
            no_data_value=0,
        )
        pB = _make_band(
            tmp_path,
            "b.tif",
            9,
            dtype="int32",
            top_left=(0.1, -0.1),
            cell_size=0.025,
            shape=(8, 8),
            no_data_value=0,
        )
        ds = Dataset.from_band_files([pA, pB], align=True, no_data_value=42)
        band1 = ds.read_array(band=1)
        uniques = sorted(set(band1.flatten().tolist()))
        assert ds.no_data_value == (42.0, 42.0), f"declared no-data: {ds.no_data_value}"
        assert 0 not in uniques, f"source no-data leaked into fringe: {uniques}"
        assert uniques == [9, 42], f"unexpected band1 values: {uniques}"

    def test_all_nan_nodata_does_not_warn(self, tmp_path):
        """Float NaN sentinels in every source don't spuriously trigger the disagreement warning.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Both sources are float32 with ``no_data_value=NaN``. Even though
            ``NaN != NaN`` in Python, the "first-wins" reconciliation must
            treat both as equal — no ``UserWarning``, no remap needed.
        """
        pE = _make_band(
            tmp_path,
            "e.tif",
            1.5,
            dtype="float32",
            top_left=(0.0, 0.0),
            cell_size=0.05,
            shape=(10, 10),
            no_data_value=float("nan"),
        )
        pF = _make_band(
            tmp_path,
            "f.tif",
            2.5,
            dtype="float32",
            top_left=(0.1, -0.1),
            cell_size=0.025,
            shape=(8, 8),
            no_data_value=float("nan"),
        )
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            ds = Dataset.from_band_files([pE, pF], align=True)
        assert not any(
            "disagree on no-data" in str(w.message) for w in caught
        ), "spurious NaN-vs-NaN disagreement warning fired"
        band1 = ds.read_array(band=1)
        # The fringe must still be NaN (declared) and the centre = 2.5 (real).
        assert np.isnan(band1[0, 0]), f"fringe should be NaN, got {band1[0, 0]}"
        assert float(band1[5, 5]) == pytest.approx(2.5), f"centre lost: {band1[5, 5]}"


class TestStackBandsAlias:
    """Tests for :func:`pyramids.dataset.merge.stack_bands`."""

    def test_alias_delegates_to_from_band_files(self, band_files):
        """``stack_bands`` is a thin wrapper over ``Dataset.from_band_files``.

        Args:
            band_files: The three source bands.

        Test scenario:
            ``stack_bands(band_files)`` — expected: a :class:`Dataset` identical
            in shape/names to the classmethod's result.
        """
        ds = stack_bands(band_files)
        assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds)}"
        assert ds.band_count == 3 and ds.band_names == [
            "B2",
            "B3",
            "B4",
        ], f"unexpected result: {ds.band_count} bands, names {ds.band_names}"

    def test_alias_forwards_kwargs(self, tmp_path, band_files):
        """Keyword arguments pass straight through the alias.

        Args:
            tmp_path: pytest temp directory.
            band_files: The three source bands.

        Test scenario:
            ``stack_bands(..., band_names=..., no_data_value=..., path=...)`` —
            expected: all honoured.
        """
        out = os.path.join(str(tmp_path), "alias.tif")
        ds = stack_bands(
            band_files, band_names=["a", "b", "c"], no_data_value=7, path=out
        )
        assert ds.band_names == [
            "a",
            "b",
            "c",
        ], f"band_names not forwarded: {ds.band_names}"
        assert ds.no_data_value == (
            7.0,
            7.0,
            7.0,
        ), f"no_data_value not forwarded: {ds.no_data_value}"
        assert os.path.exists(out), "path not forwarded"
