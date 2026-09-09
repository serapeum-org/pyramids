"""Tests for :func:`pyramids.dataset.merge.merge_rasters` ``method=`` (PB-7) and
``dst_crs=`` cross-CRS auto-reproject (PY-M).

Covers the overlap-resolution rule added to ``merge_rasters``: the z-order
``first`` / ``last`` paths and the ``min`` / ``max`` / ``sum`` reduction paths
(via ``_merge_reduce``), no-coverage fill, the ``n`` source-nodata knob,
multi-band reduction, and the guard / error branches. The ``dst_crs=`` suite
covers the reproject-before-composite behaviour and its ``_prepare_sources`` /
``_as_srs`` helpers. Source rasters are written to ``tmp_path``.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

import pyramids.dataset.merge as merge_mod
from pyramids.base._domain import INHERIT_NO_DATA
from pyramids.base.crs import reproject_coordinates
from pyramids.base.georeference import GeoReference
from pyramids.base.remote import CloudConfig
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.merge import (
    _as_srs,
    _cloud_config,
    _merge_reduce,
    _mosaic_value_range,
    _prepare_sources,
    _requested_no_data,
    _source_bounds,
    _source_nodata,
    _sources_tile_their_union,
    _storable_marker,
    _unused_marker,
    merge_rasters,
    stack_bands,
)
from tests._helpers import traced_peak, write_raster

pytestmark = pytest.mark.core


class _FakeSigner:
    """Minimal signer stand-in exposing the ``sign_href`` and ``gdal_env`` hooks.

    Mirrors the two read-time hooks of :class:`pyramids.stac.signers.Signer`
    without pulling in the optional STAC dependency. ``sign_href`` records every
    href it is handed (in ``seen``) and returns it with ``suffix`` appended, so
    tests can assert the signer was applied to each source; the default empty
    suffix keeps local paths openable.

    Args:
        env: The GDAL config mapping the signer advertises.
        suffix: String appended to every href by ``sign_href`` (default ``""``
            → identity rewrite).
    """

    def __init__(self, env, *, suffix=""):
        self._env = dict(env)
        self.suffix = suffix
        self.seen: list[str] = []

    def sign_href(self, href):
        """Record ``href`` and return it with the configured suffix appended."""
        self.seen.append(href)
        return f"{href}{self.suffix}"

    def gdal_env(self):
        """Return the GDAL config mapping (fed into ``CloudConfig.extra``)."""
        return dict(self._env)


@pytest.fixture(scope="function")
def overlapping_pair(tmp_path):
    """Two 4x4 rasters overlapping in a 2-column strip on a shared 6x4 grid.

    Raster A (value 10) sits at columns 0..3; raster B (value 20) at columns
    2..5. Their union is 6 wide × 4 tall, with columns 2..3 overlapping.

    Returns:
        tuple[str, str]: (path_a, path_b).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = write_raster(tmp_path / "a.tif", a, (0, 4))
    pb = write_raster(tmp_path / "b.tif", b, (2, 4))
    return pa, pb


class TestMergeMethod:
    """Tests for the ``method=`` overlap rule of ``merge_rasters``."""

    @pytest.mark.parametrize(
        "method, expected_overlap",
        [("last", 20.0), ("first", 10.0), ("min", 10.0), ("max", 20.0), ("sum", 30.0)],
    )
    def test_overlap_resolution(
        self, overlapping_pair, tmp_path, method, expected_overlap
    ):
        """Each method resolves the overlap strip to the expected value.

        Args:
            method: The merge method under test.
            expected_overlap: The value the overlapping columns should hold.

        Test scenario:
            Columns 2..3 are covered by both A(10) and B(20); the non-overlap
            columns keep their single source's value for every method.
        """
        pa, pb = overlapping_pair
        out = tmp_path / f"out_{method}.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method=method)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.shape == (4, 6), f"Expected union shape (4, 6), got {arr.shape}"
        assert arr[0, 2] == expected_overlap and arr[0, 3] == expected_overlap, (
            f"{method} overlap should be {expected_overlap}, got {arr[0, 2]} / {arr[0, 3]}"
        )
        assert arr[0, 0] == pytest.approx(10.0), f"A-only column changed: {arr[0, 0]}"
        assert arr[0, 5] == pytest.approx(20.0), f"B-only column changed: {arr[0, 5]}"

    @pytest.mark.parametrize(
        "method, covered_a, covered_b",
        [("min", 10.0, 20.0), ("max", 10.0, 20.0), ("sum", 10.0, 20.0)],
    )
    def test_integer_sources_reduce_to_their_own_values(
        self, disjoint_pair, tmp_path, method, covered_a, covered_b
    ):
        """A reduction over integer tiles keeps their values instead of collapsing.

        Args:
            method: Each reduction rule.
            covered_a: What raster A's columns should still hold.
            covered_b: What raster B's columns should still hold.

        Test scenario:
            Two disjoint Int32 tiles. Each source used to be warped onto the strip
            in its own data type, where the NaN marking the area it does not cover
            cannot be stored and GDAL rounded it to 0. Those zeros then won every
            `fmin` and were added by every `sum`, so the whole mosaic came back as
            zeros -- data and gaps alike, whatever `no_data_value` said.
        """
        pa, pb = disjoint_pair
        out = tmp_path / f"int_{method}.tif"
        merge_rasters([pa, pb], out, method=method)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 0] == pytest.approx(covered_a), (
            f"{method}: raster A's own column reduced to {arr[0, 0]}"
        )
        assert arr[0, 11] == pytest.approx(covered_b), (
            f"{method}: raster B's own column reduced to {arr[0, 11]}"
        )
        assert arr[0, 5] == pytest.approx(-9999.0), (
            f"{method}: the uncovered column should hold the inherited marker, "
            f"got {arr[0, 5]}"
        )

    @pytest.mark.parametrize("method", ["min", "max", "sum"])
    def test_reduction_byte_identical_across_strip_sizes(
        self, tmp_path, monkeypatch, method
    ):
        """Striping the reduction (and its latitude prune) matches a single-pass merge.

        Test scenario:
            Two vertically-offset sources overlap in a middle row band. Merging with a
            1-row strip — forcing many strips and skipping each source over the strips
            it does not cover — must produce a byte-identical raster to the default
            single-pass merge.
        """
        top = write_raster(
            tmp_path / "top.tif", np.full((4, 4), 10.0, dtype="float32"), (0, 6)
        )
        bottom = write_raster(
            tmp_path / "bottom.tif", np.full((4, 4), 20.0, dtype="float32"), (0, 4)
        )
        single = tmp_path / f"single_{method}.tif"
        merge_rasters([top, bottom], single, no_data_value=-9999.0, method=method)
        monkeypatch.setattr(merge_mod, "_MERGE_STRIP_ROWS", 1)
        stripped = tmp_path / f"stripped_{method}.tif"
        merge_rasters([top, bottom], stripped, no_data_value=-9999.0, method=method)
        assert np.array_equal(
            Dataset.read_file(str(single)).read_array(),
            Dataset.read_file(str(stripped)).read_array(),
        ), f"{method}: 1-row-strip result differs from the single-pass merge"

    @pytest.mark.parametrize("method", ["min", "max", "sum"])
    def test_multiband_reduction_byte_identical_across_strip_sizes(
        self, tmp_path, monkeypatch, method
    ):
        """A multi-band striped reduction matches the single-pass merge on every band.

        Test scenario:
            Two vertically-offset 2-band sources; merging with a 1-row strip must equal
            the single-pass merge across both bands, exercising the per-band strip write
            (`reduced[band_index]` at the strip's row offset).
        """
        top = write_raster(
            tmp_path / "top.tif",
            np.stack(
                [np.full((4, 4), 10.0, "float32"), np.full((4, 4), 11.0, "float32")]
            ),
            (0, 6),
        )
        bottom = write_raster(
            tmp_path / "bottom.tif",
            np.stack(
                [np.full((4, 4), 20.0, "float32"), np.full((4, 4), 21.0, "float32")]
            ),
            (0, 4),
        )
        single = tmp_path / f"single_{method}.tif"
        merge_rasters([top, bottom], single, no_data_value=-9999.0, method=method)
        monkeypatch.setattr(merge_mod, "_MERGE_STRIP_ROWS", 1)
        stripped = tmp_path / f"stripped_{method}.tif"
        merge_rasters([top, bottom], stripped, no_data_value=-9999.0, method=method)
        single_arr = Dataset.read_file(str(single)).read_array()
        stripped_arr = Dataset.read_file(str(stripped)).read_array()
        assert single_arr.shape[0] == 2, f"expected 2 bands, got {single_arr.shape}"
        assert np.array_equal(single_arr, stripped_arr), (
            f"{method}: multi-band striped merge diverged from the single-pass merge"
        )

    def test_reduction_peak_memory_is_bounded_by_the_strip(self, tmp_path, monkeypatch):
        """The min/max/sum merge peaks far below a whole-union pass, proving the strip reduction.

        Test scenario:
            Merge two overlapping 8000x250 sources with 128-row strips and assert the traced
            Python peak stays well below the whole-union float64 byte size (~16 MB). A strip
            reduction holds a few 128-row strips; reading both sources whole and reducing the
            full union cube would peak at least the union size. The bound is the deterministic
            union byte size, not a second measured whole-union peak, so the verdict does not
            depend on the GDAL build's one-time read buffer landing in one measurement (see
            #1049; #1047 fixed the same flake for the reduce test).
        """
        rows, cols = 8000, 250
        pa = write_raster(
            tmp_path / "a.tif", np.ones((rows, cols), dtype="float32"), (0, rows)
        )
        pb = write_raster(
            tmp_path / "b.tif", np.full((rows, cols), 2.0, dtype="float32"), (0, rows)
        )

        # Warm the build's one-time GDAL read buffer with a small windowed read of each
        # source OUTSIDE the traced region so it is already live during the measurement; the
        # absolute ceiling below is the load-bearing backstop (see #1047 / #1049).
        Dataset.read_file(str(pa)).read_array(window=[0, 0, cols, 128])
        Dataset.read_file(str(pb)).read_array(window=[0, 0, cols, 128])

        # Stripped merge.
        monkeypatch.setattr(merge_mod, "_MERGE_STRIP_ROWS", 128)
        with traced_peak() as sp:
            merge_rasters(
                [pa, pb], tmp_path / "big.tif", no_data_value=-1.0, method="max"
            )
        stripped_peak = sp[0]

        # A strip reduction must peak far below a whole-union float64 pass
        # (rows * cols * float64 = ~16 MB); reading both sources whole would peak >= that.
        union_bytes = rows * cols * np.dtype("float64").itemsize
        assert stripped_peak < union_bytes // 2, (
            f"merge peaked at {stripped_peak / 1e6:.1f} MB, not far below the "
            f"{union_bytes / 1e6:.0f} MB whole-union pass — it did not stay under a full "
            "materialisation"
        )

    def test_default_method_is_last(self, overlapping_pair, tmp_path):
        """Omitting method defaults to last-wins (backward compatible).

        Test scenario:
            No method argument yields the same overlap as method='last'.
        """
        pa, pb = overlapping_pair
        out = tmp_path / "default.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(20.0), (
            f"Default should be last-wins (20), got {arr[0, 2]}"
        )

    def test_reduce_fills_uncovered_with_nodata(self, tmp_path):
        """Reduction methods write nodata where no source covers a pixel.

        Test scenario:
            A occupies the top-left 2x2, B the bottom-right 2x2 of a 4x4 union;
            the off-diagonal quadrants are covered by neither and become nodata
            even for 'sum' (which would otherwise yield 0).
        """
        a = np.full((2, 2), 5.0, dtype="float32")
        b = np.full((2, 2), 7.0, dtype="float32")
        pa = write_raster(tmp_path / "tl.tif", a, (0, 4))
        pb = write_raster(tmp_path / "br.tif", b, (2, 2))
        out = tmp_path / "gappy.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, method="sum")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 0] == pytest.approx(5.0), (
            f"Top-left should be A=5, got {arr[0, 0]}"
        )
        assert arr[3, 3] == pytest.approx(7.0), (
            f"Bottom-right should be B=7, got {arr[3, 3]}"
        )
        assert arr[0, 3] == -1.0, (
            f"Uncovered top-right should be nodata -1, got {arr[0, 3]}"
        )
        assert arr[3, 0] == -1.0, (
            f"Uncovered bottom-left should be nodata -1, got {arr[3, 0]}"
        )

    def test_reduce_multiband(self, tmp_path):
        """Reduction operates per band on multi-band sources.

        Test scenario:
            Two 2-band rasters fully overlapping: max picks the larger value in
            each band independently.
        """
        a = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 8.0)]).astype("float32")
        b = np.stack([np.full((3, 3), 4.0), np.full((3, 3), 2.0)]).astype("float32")
        pa = write_raster(tmp_path / "ma.tif", a, (0, 3))
        pb = write_raster(tmp_path / "mb.tif", b, (0, 3))
        out = tmp_path / "mmax.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method="max")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 1, 1] == pytest.approx(4.0), (
            f"Band 0 max should be 4, got {arr[0, 1, 1]}"
        )
        assert arr[1, 1, 1] == pytest.approx(8.0), (
            f"Band 1 max should be 8, got {arr[1, 1, 1]}"
        )

    def test_n_ignores_source_value_in_reduction(self, tmp_path):
        """The n knob makes a source pixel value count as no-data in reduction.

        Test scenario:
            A is all 10; B is all 20 but with n=20 ignored, so min over the
            overlap is 10 (B's 20 is excluded), not 10-vs-20.
        """
        a = np.full((4, 4), 10.0, dtype="float32")
        b = np.full((4, 4), 20.0, dtype="float32")
        pa = write_raster(tmp_path / "na.tif", a, (0, 4), nodata=-9999.0)
        pb = write_raster(tmp_path / "nb.tif", b, (2, 4), nodata=-9999.0)
        out = tmp_path / "n_min.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, n=20, method="min")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(10.0), (
            f"Overlap min ignoring 20 should be 10, got {arr[0, 2]}"
        )
        assert arr[0, 5] == -1.0, (
            f"B-only column was all-ignored -> nodata, got {arr[0, 5]}"
        )

    def test_invalid_method_raises(self, overlapping_pair, tmp_path):
        """An unknown method raises ValueError.

        Test scenario:
            'mean' is not a supported merge method.
        """
        pa, pb = overlapping_pair
        with pytest.raises(ValueError, match="method must be one of"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="mean")

    def test_failed_vrt_zorder_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from BuildVRT on the z-order path raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.BuildVRT to return None triggers the defensive
            guard in the last/first path.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = overlapping_pair
        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", lambda *a, **k: None)
        with pytest.raises(
            RuntimeError, match="building the source mosaic returned no raster"
        ):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="last")

    def test_failed_vrt_reduce_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from BuildVRT on the reduce path raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.BuildVRT to return None triggers the defensive
            guard inside _merge_reduce.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = overlapping_pair
        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", lambda *a, **k: None)
        with pytest.raises(RuntimeError) as excinfo:
            merge_rasters([pa, pb], tmp_path / "x.tif", method="sum")
        message = str(excinfo.value)
        assert "building the union mosaic returned no raster" in message, message
        assert Path(pa).name in message, f"sources are not named: {message}"
        assert "Swig Object" not in message, f"a SWIG proxy leaked: {message}"


@pytest.fixture(scope="function")
def disjoint_pair(tmp_path):
    """Two 4x4 int32 rasters with a 4-column gap between them.

    Raster A (value 10) covers columns 0..3 and raster B (value 20) covers
    columns 8..11 of the 12-wide union grid, leaving columns 4..7 with no
    source coverage.

    Returns:
        tuple[str, str]: (path_a, path_b).
    """
    a = np.full((4, 4), 10, dtype="int32")
    b = np.full((4, 4), 20, dtype="int32")
    pa = write_raster(tmp_path / "left.tif", a, (0, 4))
    pb = write_raster(tmp_path / "right.tif", b, (8, 4))
    return pa, pb


class TestMergeRastersInheritsNoData:
    """The default no-data is inherited from the sources, never invented (#1086)."""

    @staticmethod
    def _tiles(tmp_path, no_data, dtype="float32"):
        """Write two adjacent tiles whose 0.0 cells are real data."""
        west = np.array([[0.0, 0.0], [3.0, 7.0]], dtype=dtype)
        east = np.array([[0.0, 12.0], [9.0, 21.0]], dtype=dtype)
        paths = []
        for name, arr, x0 in (("w.tif", west, 0.0), ("e.tif", east, 2.0)):
            ds = Dataset.from_array(
                arr,
                geo_ref=GeoReference(
                    top_left_corner=(x0, 2.0), cell_size=1.0, epsg=4326
                ),
            )
            ds.to_file(tmp_path / name)
            handle = gdal.Open(str(tmp_path / name), gdal.GA_Update)
            band = handle.GetRasterBand(1)
            if no_data is None:
                band.DeleteNoDataValue()
            else:
                band.SetNoDataValue(no_data)
            handle.FlushCache()
            handle = None
            paths.append(tmp_path / name)
        return paths

    @staticmethod
    def _gapped_tiles(tmp_path, dtype="float32", no_data=None):
        """Write two tiles with a 2-column gap between them.

        The union grid is 6 wide and holds 1..8; columns 2..3 are covered by no
        source, which is what makes the mosaic's marker observable at all.

        Args:
            tmp_path: Directory to write the tiles into.
            dtype: The tiles' data type.
            no_data: What each tile declares, or `None` to declare nothing.
        """
        paths = []
        for name, values, x0 in (
            ("gw.tif", [[1, 2], [3, 4]], 0.0),
            ("ge.tif", [[5, 6], [7, 8]], 4.0),
        ):
            ds = Dataset.from_array(
                np.array(values, dtype=dtype),
                geo_ref=GeoReference(
                    top_left_corner=(x0, 2.0), cell_size=1.0, epsg=4326
                ),
            )
            ds.to_file(tmp_path / name)
            handle = gdal.Open(str(tmp_path / name), gdal.GA_Update)
            band = handle.GetRasterBand(1)
            if no_data is None:
                band.DeleteNoDataValue()
            else:
                band.SetNoDataValue(no_data)
            handle.FlushCache()
            handle = None
            paths.append(tmp_path / name)
        return paths

    @staticmethod
    def _masked_count(path):
        """Return how many cells of the mosaic read as masked."""
        ds = Dataset.read_file(str(path))
        masked = ds.read_array(masked=True)
        masked = masked[0] if masked.ndim == 3 else masked
        return int(masked.size - masked.count()), ds

    @staticmethod
    def _raw_marker(path):
        """Return the marker GDAL reports for band 1, without pyramids in between."""
        handle = gdal.Open(str(path))
        raw = handle.GetRasterBand(1).GetNoDataValue()
        handle = None
        return raw

    def test_agreeing_sources_are_inherited_and_real_zero_survives(self, tmp_path):
        """The declared -9999 is inherited, so genuine 0 m cells stay readable.

        Test scenario:
            The exact case from #1086: two elevation tiles declaring -9999 whose
            0.0 cells are sea-level land. The old default stamped 0, masking all
            three of them and making stats() report a minimum of 3.0.
        """
        out = tmp_path / "m.tif"
        merge_rasters(self._tiles(tmp_path, -9999.0), out)
        masked, ds = self._masked_count(out)
        assert ds.no_data_value[0] == pytest.approx(-9999.0), (
            f"the sources' -9999 should be inherited, got {ds.no_data_value[0]}"
        )
        assert masked == 0, f"no real cell should be masked, {masked} were"
        assert float(ds.stats(approx_ok=False)["min"].iloc[0]) == pytest.approx(0.0), (
            "stats() must report the true minimum of 0.0, not 3.0"
        )

    def test_sources_declaring_none_still_leave_real_zeros_readable(self, tmp_path):
        """No source declaring one must not cost the caller their real 0 cells.

        Test scenario:
            The Copernicus-DEM shape from #1086, whose tiles declare no no-data.
            The mosaic marks its uncovered pixels -- NaN here, since these tiles
            are float and NaN is what such a pixel would hold -- but the three
            genuine 0.0 cells are not among them.
        """
        out = tmp_path / "m.tif"
        merge_rasters(self._tiles(tmp_path, None), out)
        masked, ds = self._masked_count(out)
        raw = self._raw_marker(out)
        assert raw is not None and np.isnan(raw), (
            f"a float mosaic should declare the NaN its gaps hold, got {raw}"
        )
        assert masked == 0, f"no real cell should be masked, {masked} were"
        assert float(ds.stats(approx_ok=False)["min"].iloc[0]) == pytest.approx(0.0), (
            "stats() must still report the true minimum of 0.0"
        )

    def test_a_contiguous_integer_mosaic_is_left_unmarked_never_nan(self, tmp_path):
        """Tiles that leave no gap need no marker, and must not be given a NaN one.

        Test scenario:
            UInt16 tiles declaring no no-data and sharing an edge. There is no
            uncovered pixel for a marker to mark, so none is chosen -- and in
            particular not the NaN `init` puts in the VRT, which an integer band
            cannot store and which used to be the value that made gap pixels read
            as a real 0. The gapped case, where a marker *is* needed, is
            `test_a_gapped_mosaic_declares_what_its_gaps_hold`.
        """
        out = tmp_path / "m.tif"
        merge_rasters(self._tiles(tmp_path, None, dtype="uint16"), out)
        raw = self._raw_marker(out)
        masked, ds = self._masked_count(out)
        assert raw is None, f"a gapless mosaic needs no marker, got {raw}"
        assert masked == 0, f"no real cell should be masked, {masked} were"
        assert float(ds.stats(approx_ok=False)["min"].iloc[0]) == pytest.approx(0.0), (
            "the genuine 0 cells must still be readable"
        )

    def test_a_gapless_mosaic_is_not_surveyed(self, tmp_path, monkeypatch):
        """Choosing a marker reads every source, so it is not done when unnecessary.

        Test scenario:
            Contiguous UInt16 tiles declaring nothing. Proving a sentinel unused
            means an exact `ComputeRasterMinMax` over every source and, when no
            candidate clears that, reading the whole mosaic into memory -- billed
            per byte for a `/vsicurl/` or Requester-Pays source. With no gap to
            mark there is nothing to prove, and the survey must not run at all.
        """
        surveys = []
        monkeypatch.setattr(
            merge_mod,
            "_mosaic_value_range",
            lambda mosaic: surveys.append("range") or None,
        )
        monkeypatch.setattr(
            merge_mod,
            "_unused_marker",
            lambda *args, **kwargs: surveys.append("marker") or None,
        )
        merge_rasters(self._tiles(tmp_path, None, dtype="uint16"), tmp_path / "m.tif")
        assert surveys == [], f"a gapless mosaic was surveyed anyway: {surveys}"

    def test_the_survey_is_clipped_to_the_requested_window(self, tmp_path):
        """A window is a promise to read less, and the marker survey has to keep it.

        Test scenario:
            `bbox` restricts what is written, and choosing a marker reads every
            source to prove the value unused. Surveying the whole union would
            pull the bytes the window exists to avoid -- billed, for a remote
            source -- to answer about a raster that is never written.
        """
        surveyed = []
        real = merge_mod._mosaic_value_range
        try:
            merge_mod._mosaic_value_range = lambda mosaic: (
                surveyed.append((mosaic.RasterXSize, mosaic.RasterYSize))
                or real(mosaic)
            )
            merge_rasters(
                self._gapped_tiles(tmp_path, "uint16"),
                tmp_path / "windowed.tif",
                bbox=[0.0, 0.0, 1.0, 1.0],
            )
        finally:
            merge_mod._mosaic_value_range = real
        assert surveyed == [(1, 1)], (
            f"the survey should cover the window, not the union: {surveyed}"
        )

    def test_stack_bands_declares_nothing_where_a_mosaic_settles_on_a_marker(
        self, tmp_path
    ):
        """The two share the inheritance rule and part ways after it, deliberately.

        Test scenario:
            A stack covers one grid, so it has no uncovered pixel and nothing for
            a marker to mark; a gapped mosaic does have them. Two docstrings in
            this branch disagreed about which rule they share, so the difference
            is pinned on both sides.
        """
        same_grid = []
        for name, values in (
            ("s0.tif", [[1, 2], [3, 4]]),
            ("s1.tif", [[5, 6], [7, 8]]),
        ):
            Dataset.from_array(
                np.array(values, dtype="uint16"),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
                ),
            ).to_file(tmp_path / name)
            handle = gdal.Open(str(tmp_path / name), gdal.GA_Update)
            handle.GetRasterBand(1).DeleteNoDataValue()
            handle.FlushCache()
            handle = None
            same_grid.append(str(tmp_path / name))
        stacked = stack_bands(same_grid, path=tmp_path / "stacked.tif")
        gapped = tmp_path / "gapped.tif"
        merge_rasters(self._gapped_tiles(tmp_path, "uint16"), gapped)

        assert all(value is None for value in stacked.no_data_value), (
            f"a stack has no uncovered pixel to mark, got {stacked.no_data_value}"
        )
        assert self._raw_marker(gapped) == pytest.approx(65535), (
            "a gapped mosaic settles on a marker"
        )

    def test_a_numeric_init_the_data_uses_is_refused(self, tmp_path):
        """`init` is a preference, not an instruction, or it re-creates #1086.

        Test scenario:
            Gapped tiles holding a genuine 0.0, merged with `init=0`. Taken on
            trust that becomes the mosaic's marker and every real zero disappears
            from masked reads and from `stats()` -- the exact defect this branch
            closes, reached through a different argument. It is offered to the
            sentinel search as the preferred candidate instead, so the data vetoes
            it and the caller is told.
        """
        tiles = self._gapped_tiles(tmp_path, "float32")
        handle = gdal.Open(str(tiles[0]), gdal.GA_Update)
        handle.GetRasterBand(1).WriteArray(
            np.array([[0.0, 2.0], [3.0, 4.0]], "float32")
        )
        handle.FlushCache()
        handle = None

        out = tmp_path / "init_zero.tif"
        with pytest.warns(UserWarning, match="cannot mark the mosaic"):
            merge_rasters(tiles, out, init=0)
        masked, ds = self._masked_count(out)
        marker = self._raw_marker(out)
        values = np.asarray(ds.read_array(), dtype="float64")
        values = values[0] if values.ndim == 3 else values
        assert marker == pytest.approx(-9999.0), (
            f"a value the data does not use should be chosen, got {marker}"
        )
        assert masked == 4, f"only the four gap cells should be masked, {masked} were"
        assert values[0][0] == pytest.approx(0.0), (
            "the genuine 0.0 cell must survive as data"
        )

    def test_a_numeric_init_the_data_does_not_use_is_honoured(self, tmp_path):
        """A preference the data leaves free is still the caller's to make.

        Test scenario:
            The same gapped tiles with `init=77`, a value they do not hold. It is
            tried first and taken, so refusing one is about collision only.
        """
        out = tmp_path / "init_free.tif"
        merge_rasters(self._gapped_tiles(tmp_path, "float32"), out, init=77)
        masked, _ds = self._masked_count(out)
        assert self._raw_marker(out) == pytest.approx(77.0), (
            "an init the data does not use should be honoured"
        )
        assert masked == 4, f"the four gap cells should be masked, {masked} were"

    @pytest.mark.parametrize("dtype", ["int32", "float32"])
    def test_an_explicit_marker_fills_the_gaps_it_marks(self, tmp_path, dtype):
        """A marker the caller names has to reach the pixels it exists to mark.

        Args:
            dtype: A signed integer and a floating output, both of which can hold
                the marker.

        Test scenario:
            Gapped tiles merged with `no_data_value=-1`. The gaps used to be
            filled from `init` -- `0` on an integer band, `NaN` on a float one --
            while the band declared `-1`, so `read_array(masked=True)` masked
            nothing. The reduction methods already filled with the marker, so this
            is also what makes the two write paths agree.
        """
        out = tmp_path / f"explicit_{dtype}.tif"
        merge_rasters(self._gapped_tiles(tmp_path, dtype), out, no_data_value=-1)
        masked, _ds = self._masked_count(out)
        assert self._raw_marker(out) == pytest.approx(-1.0), "the marker was not kept"
        assert masked == 4, f"the four gap cells should be masked, {masked} were"

    def test_an_explicit_init_still_wins_over_an_explicit_marker(self, tmp_path):
        """Naming both means the caller decides what the gaps hold.

        Test scenario:
            `no_data_value=-1, init=0` asks for gaps of `0` under a declared `-1`.
            That masks nothing, but it is what was asked for, and overruling it
            would make `init` unusable.
        """
        out = tmp_path / "both.tif"
        merge_rasters(
            self._gapped_tiles(tmp_path, "int32"), out, no_data_value=-1, init=0
        )
        ds = Dataset.read_file(str(out))
        values = np.asarray(ds.read_array(), dtype="float64")
        values = values[0] if values.ndim == 3 else values
        assert self._raw_marker(out) == pytest.approx(-1.0), "the marker was not kept"
        assert values[0][2] == pytest.approx(0.0), (
            f"the caller's init should fill the gap, got {values[0][2]}"
        )

    def test_an_explicit_marker_the_band_cannot_store_warns(self, tmp_path):
        """A marker GDAL will drop should not be dropped silently.

        Test scenario:
            `no_data_value=-1` on a UInt16 mosaic. GDAL answers "Nodata value was
            not set to output band" and writes no marker at all, which is the
            defect this branch exists to close -- so pyramids says so first.
        """
        out = tmp_path / "unstorable.tif"
        with pytest.warns(UserWarning, match="cannot be stored in a uint16 band"):
            merge_rasters(self._gapped_tiles(tmp_path, "uint16"), out, no_data_value=-1)
        assert self._raw_marker(out) is None, (
            "GDAL should have dropped it -- the warning is the whole point"
        )

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float32", float("nan")), ("uint16", 65535.0), ("int16", -9999.0)],
    )
    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_a_gapped_mosaic_declares_what_its_gaps_hold(
        self, tmp_path, dtype, expected, method
    ):
        """Uncovered pixels are marked, so they never read back as measurements.

        Args:
            dtype: The source tiles' data type.
            expected: The marker a z-order mosaic of that dtype should declare.
            method: Each overlap-resolution rule merge_rasters offers.

        Test scenario:
            Two tiles holding 1..8 with a two-column gap between them. Left
            undeclared, an integer mosaic writes those gap pixels as a literal 0
            -- indistinguishable from data, and enough to drag stats() from a
            true mean of 4.5 down to 1.5. The reduction methods write Float64, so
            their marker is NaN whatever the sources' dtype.
        """
        out = tmp_path / f"gap_{dtype}_{method}.tif"
        merge_rasters(self._gapped_tiles(tmp_path, dtype), out, method=method)
        masked, ds = self._masked_count(out)
        raw = self._raw_marker(out)
        wanted = float("nan") if method in ("min", "max", "sum") else expected
        assert raw is not None, "a mosaic with gaps must declare a marker"
        if np.isnan(wanted):
            assert np.isnan(raw), f"expected a NaN marker, got {raw}"
        else:
            assert raw == pytest.approx(wanted), f"expected {wanted}, got {raw}"
        assert masked == 4, f"the four gap cells should be masked, {masked} were"
        stats = ds.stats(approx_ok=False)
        assert float(stats["min"].iloc[0]) == pytest.approx(1.0), (
            f"gap pixels leaked into stats(): min {stats['min'].iloc[0]}"
        )
        assert float(stats["mean"].iloc[0]) == pytest.approx(4.5), (
            f"gap pixels leaked into stats(): mean {stats['mean'].iloc[0]}"
        )

    @pytest.mark.parametrize("method", ["last", "min"])
    def test_an_explicit_none_asks_for_no_marker_on_either_path(self, tmp_path, method):
        """``no_data_value=None`` means "no marker", and means it on both paths.

        Args:
            method: One z-order and one reduction rule, which reach the two
                different write paths.

        Test scenario:
            Inheriting nothing chooses a marker; asking for none explicitly is a
            different request, and the two write paths used to answer it
            differently -- ``(None,)`` for z-order against ``(nan,)`` for the
            reduction.
        """
        out = tmp_path / f"none_{method}.tif"
        merge_rasters(
            self._gapped_tiles(tmp_path, "float32"),
            out,
            no_data_value=None,
            method=method,
        )
        assert self._raw_marker(out) is None, (
            f"method={method} stamped a marker the caller declined"
        )

    def test_data_using_every_spare_value_warns_instead_of_masking_it(self, tmp_path):
        """A mosaic with no free sentinel is written bare, and says so.

        Test scenario:
            A uint8 tile holding all 256 values leaves nothing that could mark a
            gap without also masking real data. Choosing one anyway would be the
            #1086 defect over again, so the marker is dropped and the caller is
            told.
        """
        saturated = np.arange(256, dtype="uint8").reshape(16, 16)
        Dataset.from_array(
            saturated,
            geo_ref=GeoReference(top_left_corner=(0.0, 16.0), cell_size=1.0, epsg=4326),
        ).to_file(tmp_path / "full.tif")
        Dataset.from_array(
            np.ones((2, 2), dtype="uint8"),
            geo_ref=GeoReference(top_left_corner=(40.0, 2.0), cell_size=1.0, epsg=4326),
        ).to_file(tmp_path / "far.tif")
        for name in ("full.tif", "far.tif"):
            handle = gdal.Open(str(tmp_path / name), gdal.GA_Update)
            handle.GetRasterBand(1).DeleteNoDataValue()
            handle.FlushCache()
            handle = None
        out = tmp_path / "saturated.tif"
        with pytest.warns(UserWarning, match="use every value that data type"):
            merge_rasters([tmp_path / "full.tif", tmp_path / "far.tif"], out)
        assert self._raw_marker(out) is None, (
            "no value was free, so none should have been stamped"
        )

    def test_the_gaps_hold_the_marker_the_sources_declared(self, tmp_path):
        """An inherited marker has to reach the pixels it exists to cover.

        Test scenario:
            Two float tiles declaring -9999 with a gap between them. The mosaic
            declared -9999 while its gaps still held the NaN `init` puts in the
            VRT, so the marker matched nothing and `read_array(masked=True)`
            masked nothing.
        """
        out = tmp_path / "declared_gap.tif"
        merge_rasters(self._gapped_tiles(tmp_path, "float32", -9999.0), out)
        masked, ds = self._masked_count(out)
        assert self._raw_marker(out) == pytest.approx(-9999.0), (
            "the sources' own value should still be inherited"
        )
        assert masked == 4, f"the four gap cells should be masked, {masked} were"
        assert float(ds.stats(approx_ok=False)["mean"].iloc[0]) == pytest.approx(4.5), (
            "the gaps leaked into stats()"
        )

    def test_an_unstorable_inherited_marker_is_replaced_and_warns(self, tmp_path):
        """A NaN inherited onto an integer band is refused by GDAL, so it is not used.

        Test scenario:
            `Dataset.no_data_value` reports NaN for an integer raster that was
            asked for one, so integer sources really can declare it. Handed
            straight to `gdal.Translate` it produced "Nodata value was not set to
            output band" and a mosaic marking nothing -- the same undeclared gaps
            #1086's fix set out to close, reached through the inherited path.
        """
        out = tmp_path / "nan_on_int.tif"
        with pytest.warns(UserWarning, match="cannot store"):
            merge_rasters(self._gapped_tiles(tmp_path, "uint16", float("nan")), out)
        masked, _ds = self._masked_count(out)
        marker = self._raw_marker(out)
        assert marker == pytest.approx(65535), (
            f"a storable marker should replace the NaN, got {marker}"
        )
        assert masked == 4, f"the four gap cells should be masked, {masked} were"

    def test_a_declared_no_data_cell_stays_no_data(self, tmp_path):
        """A source cell that IS no-data is not leaked into the mosaic as data.

        Test scenario:
            The other half of #1086: the sources' declared value used to be
            ignored on the way in too, so a -9999 hole arrived as a real -9999.
        """
        paths = self._tiles(tmp_path, -9999.0)
        holed = np.array([[-9999.0, 5.0], [3.0, 7.0]], dtype="float32")
        ds = Dataset.from_array(
            holed,
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )
        ds.to_file(paths[0])
        handle = gdal.Open(str(paths[0]), gdal.GA_Update)
        handle.GetRasterBand(1).SetNoDataValue(-9999.0)
        handle.FlushCache()
        handle = None

        out = tmp_path / "m.tif"
        merge_rasters(paths, out)
        masked, mosaic = self._masked_count(out)
        values = np.asarray(mosaic.read_array(), dtype="float64")
        values = values[0] if values.ndim == 3 else values
        # The masked count alone does not discriminate: the old `0` default
        # masked the one real 0.0 cell and produced the same total.
        assert mosaic.no_data_value[0] == pytest.approx(-9999.0), (
            f"the sources' marker should be inherited, got {mosaic.no_data_value[0]}"
        )
        assert masked == 1, f"the declared no-data cell should stay masked, {masked}"
        assert float(np.nanmin(values[values != -9999.0])) == pytest.approx(0.0), (
            "the real 0.0 cell must remain readable data"
        )

    def test_only_band_ones_marker_is_inherited(self, tmp_path):
        """One marker is stamped on every band, and band 1 decides which.

        Test scenario:
            Two-band VRT sources whose band 1 declares nothing and whose band 2
            declares -9999. Nothing is inherited, so the mosaic falls back to a
            chosen marker and band 2's value is dropped. The sources are VRTs
            because GeoTIFF cannot express the case at all -- its
            `TIFFTAG_GDAL_NODATA` holds one value for the whole dataset, and GDAL
            says so ("This value will be used for all bands on re-opening") --
            which is also why the output side of this cannot be fixed by reading
            more bands. Pinned so a future per-band implementation has something
            to flip.
        """
        paths = []
        for name, base, x0 in (("mw", 1.0, 0.0), ("me", 5.0, 2.0)):
            Dataset.from_array(
                np.stack(
                    [
                        np.full((2, 2), base, dtype="float32"),
                        np.full((2, 2), base + 10.0, dtype="float32"),
                    ]
                ),
                geo_ref=GeoReference(
                    top_left_corner=(x0, 2.0), cell_size=1.0, epsg=4326
                ),
            ).to_file(tmp_path / f"{name}.tif")
            source = gdal.Open(str(tmp_path / f"{name}.tif"))
            vrt = gdal.GetDriverByName("VRT").CreateCopy(
                str(tmp_path / f"{name}.vrt"), source
            )
            vrt.GetRasterBand(1).DeleteNoDataValue()
            vrt.GetRasterBand(2).SetNoDataValue(-9999.0)
            vrt.FlushCache()
            vrt = None
            source = None
            paths.append(tmp_path / f"{name}.vrt")

        handle = gdal.Open(str(paths[0]))
        declared = [
            handle.GetRasterBand(index + 1).GetNoDataValue() for index in range(2)
        ]
        handle = None
        assert declared == [None, -9999.0], (
            f"the source must actually declare per-band markers, got {declared}"
        )

        out = tmp_path / "multiband.tif"
        merge_rasters(paths, out)
        handle = gdal.Open(str(out))
        markers = [
            handle.GetRasterBand(index + 1).GetNoDataValue()
            for index in range(handle.RasterCount)
        ]
        handle = None
        assert len(markers) == 2, f"expected a two-band mosaic, got {len(markers)}"
        assert all(value is not None and np.isnan(value) for value in markers), (
            f"band 1 declares nothing, so both bands take the fallback: {markers}"
        )

    def test_disagreeing_sources_warn_and_take_the_first(self, tmp_path):
        """A disagreement is surfaced rather than silently resolved."""
        west, east = self._tiles(tmp_path, -9999.0)
        handle = gdal.Open(str(east), gdal.GA_Update)
        handle.GetRasterBand(1).SetNoDataValue(-32768.0)
        handle.FlushCache()
        handle = None
        out = tmp_path / "m.tif"
        with pytest.warns(UserWarning, match="disagree on no-data value"):
            merge_rasters([west, east], out)
        ds = Dataset.read_file(str(out))
        assert ds.no_data_value[0] == pytest.approx(-9999.0), (
            "the first source's value should win"
        )

    @pytest.mark.parametrize("method", ["last", "min"])
    def test_every_source_s_own_hole_is_skipped_not_just_the_winner_s(
        self, tmp_path, method
    ):
        """A hole is a hole in whichever source declared it.

        Args:
            method: One z-order and one reduction rule, so both write paths
                answer alike.

        Test scenario:
            Tile A declares -9999 and tile B declares -32768, and each has a
            hole. The z-order path passed `srcNodata="nan"` to `gdal.BuildVRT`,
            which *replaces* every source's own declaration -- so both holes
            composited as real measurements and only the winner's value was
            masked afterwards by the inherited marker. B's -32768 hole came out
            as a readable -32768.
        """
        west = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
        east = np.array([[5.0, -32768.0], [7.0, 8.0]], dtype="float32")
        paths = []
        for name, arr, x0, marker in (
            ("dw.tif", west, 0.0, -9999.0),
            ("de.tif", east, 2.0, -32768.0),
        ):
            Dataset.from_array(
                arr,
                geo_ref=GeoReference(
                    top_left_corner=(x0, 2.0), cell_size=1.0, epsg=4326
                ),
            ).to_file(tmp_path / name)
            handle = gdal.Open(str(tmp_path / name), gdal.GA_Update)
            handle.GetRasterBand(1).SetNoDataValue(marker)
            handle.FlushCache()
            handle = None
            paths.append(tmp_path / name)

        out = tmp_path / f"holes_{method}.tif"
        with pytest.warns(UserWarning, match="disagree on no-data value"):
            merge_rasters(paths, out, method=method)
        masked, ds = self._masked_count(out)
        values = np.asarray(ds.read_array(), dtype="float64")
        values = values[0] if values.ndim == 3 else values
        assert masked == 2, (
            f"both sources' holes should be masked, {masked} were: {values.tolist()}"
        )
        assert -32768.0 not in values, (
            f"the second source's hole leaked as data: {values.tolist()}"
        )

    def test_an_explicit_value_still_overrides(self, tmp_path):
        """Passing a value keeps working, masking whatever holds it."""
        out = tmp_path / "m.tif"
        merge_rasters(self._tiles(tmp_path, -9999.0), out, no_data_value=0)
        masked, ds = self._masked_count(out)
        assert ds.no_data_value[0] == pytest.approx(0.0), "explicit 0 must be honoured"
        assert masked == 3, f"an explicit 0 masks the three real zeros, got {masked}"

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_every_method_inherits(self, tmp_path, method):
        """Inheritance is not specific to the default z-order path.

        Args:
            method: Each overlap-resolution rule merge_rasters offers.
        """
        out = tmp_path / f"m_{method}.tif"
        merge_rasters(self._tiles(tmp_path, -9999.0), out, method=method)
        masked, ds = self._masked_count(out)
        assert ds.no_data_value[0] == pytest.approx(-9999.0), (
            f"method={method} did not inherit, got {ds.no_data_value[0]}"
        )
        assert masked == 0, f"method={method} masked {masked} real cells"

    def test_passing_the_sentinel_explicitly_matches_the_default(self, tmp_path):
        """``no_data_value=INHERIT_NO_DATA`` is the default spelled out.

        Test scenario:
            The sentinel is what tells "nothing was passed" apart from an
            explicit ``None``, so passing it has to inherit the sources' marker
            rather than decline one the way ``None`` does.
        """
        out = tmp_path / "sentinel.tif"
        merge_rasters(
            self._tiles(tmp_path, -9999.0), out, no_data_value=INHERIT_NO_DATA
        )
        assert self._raw_marker(out) == pytest.approx(-9999.0), (
            f"the sentinel should inherit, got {self._raw_marker(out)}"
        )

    def test_the_default_reads_as_inherit_in_the_signature(self):
        """The rendered default says what it does, for the generated API docs.

        Test scenario:
            A bare ``object()`` sentinel reaches the docs as
            ``no_data_value=<object object at 0x...>`` -- a line that says
            nothing and changes on every build.
        """
        rendered = str(inspect.signature(merge_rasters).parameters["no_data_value"])
        assert rendered.endswith("= inherit"), (
            f"the default should render as 'inherit', got {rendered!r}"
        )


def _mem_mosaic(bands, no_data=None):
    """Build an in-memory Float32 dataset holding `bands`, for the marker helpers.

    Args:
        bands: One 2-D array-like per band, all of the same shape.
        no_data: The marker every band declares, or `None` to declare none.

    Returns:
        gdal.Dataset: The in-memory dataset.
    """
    first = np.asarray(bands[0], dtype="float32")
    handle = gdal.GetDriverByName("MEM").Create(
        "", first.shape[1], first.shape[0], len(bands), gdal.GDT_Float32
    )
    for index, values in enumerate(bands):
        band = handle.GetRasterBand(index + 1)
        if no_data is not None:
            band.SetNoDataValue(float(no_data))
        band.WriteArray(np.asarray(values, dtype="float32"))
    return handle


@pytest.fixture(scope="function")
def unmarked_tiles(tmp_path):
    """Two adjacent float32 tiles declaring no no-data, opened for the marker helpers.

    The union grid is 6 wide and holds 1..8, with a two-column gap between the
    tiles -- the shape `_storable_marker` exists to mark.

    Returns:
        tuple[list, list[str]]: The open GDAL handles and their paths.
    """
    paths = []
    for name, values, x0 in (
        ("uw.tif", [[1, 2], [3, 4]], 0.0),
        ("ue.tif", [[5, 6], [7, 8]], 4.0),
    ):
        Dataset.from_array(
            np.array(values, dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(x0, 2.0), cell_size=1.0, epsg=4326),
        ).to_file(tmp_path / name)
        handle = gdal.Open(str(tmp_path / name), gdal.GA_Update)
        handle.GetRasterBand(1).DeleteNoDataValue()
        handle.FlushCache()
        handle = None
        paths.append(str(tmp_path / name))
    return [gdal.Open(path) for path in paths], paths


class TestSourceNodata:
    """Tests for ``_source_nodata``, which reads ``n=`` as an override or as none."""

    @pytest.mark.parametrize("value", ["nan", "NaN", float("nan"), np.float32("nan")])
    def test_a_nan_spelling_means_no_override(self, value):
        """Every spelling of NaN leaves each source with its own declared marker.

        Args:
            value: A way of spelling the default ``n="nan"``.

        Test scenario:
            A blanket ``srcNodata`` *replaces* what each source declares, so the
            default has to mean "no override" rather than "ignore NaN cells" --
            otherwise a mosaic of tiles declaring -9999 and -32768 composites
            both of their holes as real measurements.
        """
        resolved = _source_nodata(value)
        assert resolved is None, (
            f"{value!r} should mean 'no override', got {resolved!r}"
        )

    @pytest.mark.parametrize(
        "value, expected", [(-9999, -9999.0), ("-32768", -32768.0), (0, 0.0)]
    )
    def test_an_explicit_value_becomes_a_float_override(self, value, expected):
        """Anything else is an override, coerced to the float GDAL wants.

        Args:
            value: The caller's ``n=``, as a number or its string spelling.
            expected: The float the compositor should be handed.
        """
        resolved = _source_nodata(value)
        assert resolved == pytest.approx(expected), (
            f"{value!r} should override with {expected}, got {resolved!r}"
        )


class TestMosaicValueRange:
    """Tests for ``_mosaic_value_range``, the survey a sentinel is chosen against."""

    def test_the_range_spans_every_band(self):
        """The answer covers all bands, so a sentinel free in band 1 is not enough.

        Test scenario:
            Band 1 holds 1..4 and band 2 holds -5..10; only a value outside
            -5..10 is genuinely unused by the mosaic.
        """
        mosaic = _mem_mosaic([[[1.0, 2.0], [3.0, 4.0]], [[10.0, -5.0], [6.0, 7.0]]])
        assert _mosaic_value_range(mosaic) == (-5.0, 10.0), (
            f"the range must span both bands, got {_mosaic_value_range(mosaic)}"
        )

    def test_a_band_with_no_valid_cells_is_skipped(self):
        """A band that is entirely no-data constrains nothing rather than failing.

        Test scenario:
            GDAL raises instead of answering when a band holds no valid pixel to
            measure; the remaining band still has to decide the range.
        """
        mosaic = _mem_mosaic(
            [np.full((2, 2), -9999.0), [[1.0, 2.0], [3.0, 4.0]]], no_data=-9999.0
        )
        assert _mosaic_value_range(mosaic) == (1.0, 4.0), (
            f"the empty band should be skipped, got {_mosaic_value_range(mosaic)}"
        )

    def test_a_mosaic_with_no_valid_cells_has_no_range(self):
        """An all-no-data mosaic rules out no sentinel at all.

        Test scenario:
            Every band raises, so there is no minimum or maximum to test a
            candidate against and the caller has to fall back.
        """
        mosaic = _mem_mosaic([np.full((2, 2), -9999.0)], no_data=-9999.0)
        assert _mosaic_value_range(mosaic) is None, (
            f"an unmeasurable mosaic has no range, got {_mosaic_value_range(mosaic)}"
        )

    def test_a_read_failure_is_not_mistaken_for_an_empty_band(self):
        """Only "no valid pixels" means empty; anything else is a failed read.

        Test scenario:
            A truncated remote object or an expired credential raises from the
            same call as an all-no-data band. Swallowing those chose a marker
            from whatever bands happened to answer -- or from none at all.
        """

        class _Failing:
            """A band whose statistics call fails for a reason that is not emptiness."""

            def ComputeRasterMinMax(self, approx_ok):  # noqa: N802
                """Fail the way a truncated read does."""
                raise RuntimeError("IReadBlock failed at X offset 0, Y offset 0")

        class _Mosaic:
            """A one-band mosaic whose band cannot be measured."""

            RasterCount = 1

            def GetRasterBand(self, index):  # noqa: N802
                """Return the failing band."""
                return _Failing()

        with pytest.raises(RuntimeError, match="IReadBlock"):
            _mosaic_value_range(_Mosaic())


class TestUnusedMarker:
    """Tests for ``_unused_marker``, which picks a sentinel the data does not use."""

    def test_a_candidate_outside_the_data_is_chosen(self):
        """The range test alone clears the package default in the ordinary case.

        Test scenario:
            The mosaic holds 1..4, so -9999 cannot occur in it and is cleared
            without the mosaic ever being read into an array.
        """
        mosaic = _mem_mosaic([[[1.0, 2.0], [3.0, 4.0]]])
        chosen = _unused_marker(mosaic, np.dtype("float32"))
        assert chosen == pytest.approx(-9999.0), (
            f"a candidate outside 1..4 should be taken, got {chosen!r}"
        )

    def test_an_unmeasurable_mosaic_takes_the_first_candidate(self):
        """With no range to test against, the preferred candidate is taken as-is.

        Test scenario:
            An all-no-data mosaic gives no bounds, and a uint8 band prefers its
            own maximum over 0 -- the value a future write or fill would take.
        """
        mosaic = _mem_mosaic([np.full((2, 2), -9999.0)], no_data=-9999.0)
        chosen = _unused_marker(mosaic, np.dtype("uint8"))
        assert chosen == 255, f"uint8 should reach for 255 first, got {chosen!r}"

    def test_a_dtype_with_no_storable_candidate_yields_none(self):
        """A dtype that can hold no sentinel gets none, rather than an index error.

        Test scenario:
            ``bool`` offers neither the package default nor integer extremes, so
            the candidate list is empty. GDAL has no boolean band, so this pins
            the guard rather than a mosaic a caller could build.
        """
        mosaic = _mem_mosaic([np.full((2, 2), -9999.0)], no_data=-9999.0)
        chosen = _unused_marker(mosaic, np.dtype(bool))
        assert chosen is None, f"no candidate fits a bool band, got {chosen!r}"


class TestStorableMarker:
    """Tests for ``_storable_marker``, which settles what an inheriting mosaic declares."""

    @pytest.mark.parametrize("init", [None, "none", "not-a-number"])
    def test_an_uncoercible_init_still_yields_a_storable_sentinel(
        self, unmarked_tiles, init
    ):
        """An ``init`` that is not a number cannot leave the mosaic unmarked.

        Args:
            init: An uncovered-pixel value ``float()`` refuses.

        Test scenario:
            ``init`` is only usable as the marker when it coerces to a number the
            dtype can store; when it does not, the sentinel search has to run
            instead of the marker being dropped -- dropping it is what leaves gap
            pixels reading as ordinary data (#1086).
        """
        ordered, paths = unmarked_tiles
        marker = _storable_marker(ordered, paths, init, None)
        assert marker == pytest.approx(-9999.0), (
            f"init={init!r} should fall through to a storable sentinel, got {marker!r}"
        )


class TestRequestedNoData:
    """Tests for ``_requested_no_data``, the gate on what a caller may pass."""

    @pytest.mark.parametrize("spelling", ["inherit", "Inherit", " inherit "])
    def test_the_word_the_default_renders_as_means_the_default(self, spelling):
        """`no_data_value="inherit"` is the default spelled out, not a typo.

        Args:
            spelling: A way of writing the word the signature displays.

        Test scenario:
            The default renders as `no_data_value=inherit` in `help()` and in the
            generated docs, which invites passing the string. Handed to GDAL it
            produced a silently unmarked mosaic.
        """
        assert _requested_no_data(spelling) is INHERIT_NO_DATA, (
            f"{spelling!r} should resolve to the sentinel"
        )

    @pytest.mark.parametrize("value", ["nodata", "", "none "])
    def test_a_string_that_names_no_number_is_refused(self, value):
        """A value GDAL would drop is refused, rather than dropped.

        Args:
            value: A string that is neither the sentinel's word nor a number.

        Test scenario:
            `gdal.Translate` answers an unparsable `-a_nodata` with "Nodata value
            was not set to output band" and writes no marker -- the defect this
            module exists to close, arriving through a typo.
        """
        with pytest.raises(ValueError, match="is not a number"):
            _requested_no_data(value)

    @pytest.mark.parametrize("value", [0, -9999, -1.5, "0", None])
    def test_a_number_or_none_passes_through_unchanged(self, value):
        """Everything the output can actually carry is left alone.

        Args:
            value: A marker, or `None` for no marker at all.
        """
        assert _requested_no_data(value) == value or (
            value is None and _requested_no_data(value) is None
        ), f"{value!r} should pass through unchanged"


class TestSourcesTileTheirUnion:
    """Tests for ``_sources_tile_their_union``, which decides whether to survey."""

    @pytest.mark.parametrize(
        "bounds, tiled",
        [
            ([(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)], True),
            ([(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 5.0, 2.0)], False),
            ([(0.0, 0.0, 2.0, 2.0), (0.0, 2.0, 2.0, 4.0)], True),
            ([(0.0, 0.0, 4.0, 4.0), (1.0, 1.0, 3.0, 3.0)], True),
            ([(0.0, 0.0, 2.0, 2.0), (2.0, 2.0, 4.0, 4.0)], False),
            ([], False),
        ],
    )
    def test_footprints_answer_without_reading_a_pixel(self, bounds, tiled):
        """Coverage is decided from the rectangles alone.

        Args:
            bounds: The sources' extents.
            tiled: Whether they leave no gap between them.

        Test scenario:
            Includes the diagonal pair, whose bounding box is covered only in two
            of its four quadrants -- the case a bounding-box comparison gets
            wrong.
        """
        assert _sources_tile_their_union(bounds) is tiled, (
            f"{bounds} should answer {tiled}"
        )

    def test_too_many_sources_answer_cautiously(self, monkeypatch):
        """Past the cut-off the cheap answer is 'assume a gap', never 'assume none'.

        Test scenario:
            The cut grid is quadratic in the number of sources, so it stops being
            cheap; answering `False` costs a survey, while answering `True` would
            skip one that was needed.
        """
        monkeypatch.setattr(merge_mod, "_MAX_TILED_SOURCES", 1)
        bounds = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)]
        assert _sources_tile_their_union(bounds) is False, (
            "past the cut-off the answer must be the cautious one"
        )


class TestMergeRastersInputContracts:
    """Input/output contracts of ``merge_rasters`` beyond the overlap rule."""

    def test_unopenable_source_is_named_end_to_end(self, tmp_path):
        """A real failed open through ``merge_rasters`` names the source.

        Test scenario:
            No mocking -- real GDAL under ``gdal.UseExceptions()`` raises for a
            missing source, and the wrapper must name it and its ``1/2``
            position. This pins the catch type: if GDAL ever raised something
            other than ``RuntimeError`` the wrapper would stop catching it and
            this test would fail, which the monkeypatched tests cannot detect
            (#1107).
        """
        good = write_raster(
            tmp_path / "good.tif", np.ones((4, 4), dtype="float32"), (0, 4)
        )
        missing = str(tmp_path / "absent_tile.tif")
        with pytest.raises(RuntimeError) as excinfo:
            merge_rasters([missing, str(good)], tmp_path / "out.tif")
        message = str(excinfo.value)
        assert "absent_tile.tif" in message, f"source not named: {message}"
        assert "1/2" in message, f"source position not reported: {message}"

    def test_zorder_init_fills_uncovered_pixels(self, disjoint_pair, tmp_path):
        """``init`` fills pixels no source covers on the z-order path.

        Test scenario:
            Two disjoint tiles leave columns 4..7 uncovered; with
            ``init=-1.0`` / ``no_data_value=-1.0`` those pixels read -1 and
            the output advertises -1 as its nodata marker.
        """
        pa, pb = disjoint_pair
        out = tmp_path / "gap.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, init=-1.0, method="last")
        ds = Dataset.read_file(str(out))
        arr = ds.read_array()
        assert arr.shape == (4, 12), f"Expected union shape (4, 12), got {arr.shape}"
        assert arr[0, 5] == pytest.approx(-1), (
            f"Uncovered pixel should hold init=-1, got {arr[0, 5]}"
        )
        assert ds.no_data_value[0] == pytest.approx(-1.0), (
            f"Output nodata should be -1.0, got {ds.no_data_value[0]}"
        )

    def test_zorder_preserves_source_dtype(self, disjoint_pair, tmp_path):
        """The z-order path keeps the sources' integer dtype.

        Test scenario:
            int32 sources merged with method='last' produce an int32 output
            (BuildVRT + Translate copy the band type through).
        """
        pa, pb = disjoint_pair
        out = tmp_path / "dtype_zorder.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, init=-1.0, method="last")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.dtype == np.int32, f"z-order should preserve int32, got {arr.dtype}"

    def test_reduce_promotes_to_float64(self, disjoint_pair, tmp_path):
        """The reduction path writes Float64 regardless of the source dtype.

        Test scenario:
            int32 sources merged with method='max' produce a float64 output —
            the documented dtype contract of the NaN-aware reducer.
        """
        pa, pb = disjoint_pair
        out = tmp_path / "dtype_reduce.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, method="max")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.dtype == np.float64, f"reduce should write float64, got {arr.dtype}"

    def test_n_ignores_source_value_in_zorder(self, overlapping_pair, tmp_path):
        """``n`` marks a source value as nodata on the z-order path too.

        Test scenario:
            With n=20 every pixel of raster B (all 20s) is treated as source
            nodata: the overlap strip falls back to A's 10 and B-only columns
            become the init fill.
        """
        pa, pb = overlapping_pair
        out = tmp_path / "n_zorder.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, init=-1.0, n=20, method="last")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(10.0), (
            f"Overlap should fall back to A=10, got {arr[0, 2]}"
        )
        assert arr[0, 5] == pytest.approx(-1.0), (
            f"B-only column should be init=-1, got {arr[0, 5]}"
        )
        assert arr[0, 0] == pytest.approx(10.0), f"A-only column changed: {arr[0, 0]}"

    def test_path_object_inputs(self, disjoint_pair, tmp_path):
        """``src`` entries and ``dst`` may be ``pathlib.Path`` objects.

        Test scenario:
            The signature accepts str | Path; passing Path for every argument
            produces the same mosaic as the str form.
        """
        pa, pb = disjoint_pair
        out = tmp_path / "path_objects.tif"
        merge_rasters([Path(pa), Path(pb)], Path(out), no_data_value=-1.0, init=-1.0)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.shape == (4, 12), f"Expected union shape (4, 12), got {arr.shape}"
        assert arr[0, 0] == pytest.approx(10) and arr[0, 11] == pytest.approx(20), (
            f"Tile values lost: left={arr[0, 0]}, right={arr[0, 11]}"
        )


class TestDatasetCollectionMergeMethod:
    """Tests that DatasetCollection.merge threads method through."""

    def test_collection_merge_method(self, overlapping_pair, tmp_path):
        """DatasetCollection.merge forwards method= to merge_rasters.

        Test scenario:
            A file-backed collection of the two overlapping rasters merged with
            method='sum' yields 30 in the overlap.
        """
        from pyramids.dataset.collection import DatasetCollection

        pa, pb = overlapping_pair
        collection = DatasetCollection.read_multiple_files(
            [pa, pb], with_order=False, date=False
        )
        out = tmp_path / "coll_sum.tif"
        collection.merge(out, no_data_value=-9999.0, method="sum")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(30.0), (
            f"Collection sum overlap should be 30, got {arr[0, 2]}"
        )


@pytest.fixture(scope="function")
def shared_crs_pair(tmp_path):
    """Two 4x4 EPSG:4326 rasters overlapping in a 2-column strip (a shared CRS).

    Returns:
        tuple[str, str]: (path_a value 10, path_b value 20).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = write_raster(tmp_path / "sa.tif", a, (0, 4), epsg=4326)
    pb = write_raster(tmp_path / "sb.tif", b, (2, 4), epsg=4326)
    return pa, pb


@pytest.fixture(scope="function")
def disagree_pair(tmp_path):
    """Two overlapping rasters in *different* CRSs (EPSG:4326 and EPSG:3857).

    The first is written natively in 4326; the second is the same footprint
    reprojected to 3857 on disk, so the pair genuinely disagrees on CRS.

    Returns:
        tuple[str, str]: (path_4326, path_3857).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = write_raster(tmp_path / "da_4326.tif", a, (0, 4), epsg=4326)
    pb_4326 = write_raster(tmp_path / "db_4326.tif", b, (2, 4), epsg=4326)
    pb = str(tmp_path / "db_3857.tif")
    Dataset.read_file(pb_4326).to_crs(3857).to_file(pb)
    return pa, pb


class TestMergeRastersDstCrs:
    """Tests for the ``dst_crs=`` cross-CRS auto-reproject of ``merge_rasters`` (PY-M)."""

    def test_dst_crs_epsg_int_reprojects(self, shared_crs_pair, tmp_path):
        """An EPSG int ``dst_crs`` reprojects sources and stamps the target CRS.

        Test scenario:
            Two EPSG:4326 tiles merged with ``dst_crs=3857`` produce a mosaic
            whose CRS is EPSG:3857.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "int_crs.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, dst_crs=3857)
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"Expected output EPSG 3857, got {result.epsg}"

    def test_dst_crs_string_reprojects(self, shared_crs_pair, tmp_path):
        """A CRS string ``dst_crs`` is parsed and applied like the int form.

        Test scenario:
            ``dst_crs="EPSG:3857"`` yields the same EPSG:3857 mosaic as the int
            ``3857``.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "str_crs.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, dst_crs="EPSG:3857")
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"Expected output EPSG 3857, got {result.epsg}"

    def test_default_shared_crs_no_reproject(self, shared_crs_pair, tmp_path):
        """``dst_crs=None`` with a shared CRS keeps the previous behaviour.

        Test scenario:
            Omitting ``dst_crs`` leaves both EPSG:4326 tiles untouched: the
            output stays in EPSG:4326 and the last-wins overlap is 20.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "default_crs.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        result = Dataset.read_file(str(out))
        arr = result.read_array()
        assert result.epsg == 4326, f"Expected output EPSG 4326, got {result.epsg}"
        assert arr[0, 2] == pytest.approx(20.0), (
            f"Last-wins overlap should be 20, got {arr[0, 2]}"
        )

    def test_disagree_reprojects_onto_first_source_crs(self, disagree_pair, tmp_path):
        """Mismatched CRSs with ``dst_crs=None`` reproject onto the first source.

        Test scenario:
            A 4326 source and a 3857 source merged without ``dst_crs`` are
            composited in the first source's CRS, EPSG:4326.
        """
        pa, pb = disagree_pair
        out = tmp_path / "disagree.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        result = Dataset.read_file(str(out))
        assert result.epsg == 4326, (
            f"Disagreeing sources should mosaic in the first source CRS 4326, got {result.epsg}"
        )

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_dst_crs_with_each_method(self, shared_crs_pair, tmp_path, method):
        """Reproject composes with every overlap-resolution method.

        Args:
            method: The merge method combined with the reproject.

        Test scenario:
            ``dst_crs=3857`` plus each method produces a readable EPSG:3857
            mosaic (the z-order and reduce paths both honour the reproject).
        """
        pa, pb = shared_crs_pair
        out = tmp_path / f"crs_{method}.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method=method, dst_crs=3857)
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"{method}: expected EPSG 3857, got {result.epsg}"
        assert result.read_array().size > 0, f"{method}: mosaic is empty"

    def test_invalid_dst_crs_string_raises(self, shared_crs_pair, tmp_path):
        """An unparseable ``dst_crs`` string raises ValueError.

        Test scenario:
            ``dst_crs="not-a-crs"`` cannot be parsed and raises ValueError before
            any compositing happens.
        """
        pa, pb = shared_crs_pair
        with pytest.raises(ValueError, match="Could not parse dst_crs"):
            merge_rasters([pa, pb], tmp_path / "bad.tif", dst_crs="not-a-crs")

    def test_invalid_resampling_raises(self, shared_crs_pair, tmp_path):
        """An unsupported ``resampling`` value raises ValueError.

        Test scenario:
            ``resampling="sinc"`` is not in INTERPOLATION_METHODS and is rejected
            before any compositing.
        """
        pa, pb = shared_crs_pair
        with pytest.raises(ValueError, match="does not exist"):
            merge_rasters(
                [pa, pb], tmp_path / "bad.tif", dst_crs=3857, resampling="sinc"
            )

    @pytest.mark.parametrize("resampling", ["nearest neighbor", "bilinear", "cubic"])
    def test_resampling_methods_reproject(self, shared_crs_pair, tmp_path, resampling):
        """Each supported resampling method reprojects to ``dst_crs`` successfully.

        Args:
            resampling: The resampling method under test.

        Test scenario:
            A reproject to EPSG:3857 with each method produces a 3857 mosaic.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / f"r_{resampling.split()[0]}.tif"
        merge_rasters(
            [pa, pb], out, dst_crs=3857, resampling=resampling, no_data_value=-9999.0
        )
        assert Dataset.read_file(str(out)).epsg == 3857, (
            f"{resampling} did not reproject"
        )

    def test_warp_failure_raises(self, shared_crs_pair, tmp_path, monkeypatch):
        """A None from gdal.Warp during reproject raises RuntimeError.

        Test scenario:
            With ``dst_crs=3857`` forcing a reproject, monkeypatching gdal.Warp
            to return None trips the defensive guard in ``_prepare_sources``.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = shared_crs_pair
        monkeypatch.setattr(merge_mod.gdal, "Warp", lambda *a, **k: None)
        with pytest.raises(
            RuntimeError, match="reprojecting to the target CRS returned no raster"
        ):
            merge_rasters([pa, pb], tmp_path / "x.tif", dst_crs=3857)

    def test_raising_warp_names_the_source(
        self, shared_crs_pair, tmp_path, monkeypatch
    ):
        """A raising ``gdal.Warp`` names the source it could not reproject.

        Test scenario:
            Under ``gdal.UseExceptions()`` Warp raises rather than returning
            None, so the reproject half of ``_prepare_sources`` must name the
            source too -- the ``Raises:`` contract covers the whole function,
            not just the open (#1107).
        """
        pa, pb = shared_crs_pair

        def _raise(*_args, **_kwargs):
            raise RuntimeError("Too many points failed to transform")

        monkeypatch.setattr(merge_mod.gdal, "Warp", _raise)
        with pytest.raises(RuntimeError) as excinfo:
            merge_rasters([pa, pb], tmp_path / "x.tif", dst_crs=3857)
        message = str(excinfo.value)
        assert "reprojecting to the target CRS failed for source" in message, (
            f"unexpected message: {message}"
        )
        assert "failed to transform" in message, f"GDAL message not kept: {message}"

    def test_open_failure_raises(self, shared_crs_pair, tmp_path, monkeypatch):
        """A None from gdal.Open while reading source CRS raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.Open to return None trips the guard in the
            CRS-probe loop of ``_prepare_sources``.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = shared_crs_pair
        monkeypatch.setattr(merge_mod.gdal, "Open", lambda *a, **k: None)
        with pytest.raises(RuntimeError) as excinfo:
            merge_rasters([pa, pb], tmp_path / "x.tif")
        message = str(excinfo.value)
        assert "GDAL returned no dataset" in message, message
        assert Path(pa).name in message, f"the failing source is not named: {message}"
        assert "1/2" in message, f"the source position is missing: {message}"


class TestSourceBounds:
    """Tests for the ``_source_bounds`` extent helper used by the strip reduction."""

    def test_from_path(self, tmp_path):
        """A path resolves to its ``(west, south, east, north)`` extent.

        Test scenario:
            A 4x4 raster at top-left ``(0, 4)`` with unit cells spans x/y ``[0, 4]``.
        """
        path = write_raster(
            tmp_path / "s.tif", np.ones((4, 4), dtype="float32"), (0, 4)
        )
        assert _source_bounds(str(path)) == (0.0, 0.0, 4.0, 4.0), "wrong path bounds"

    def test_from_open_dataset(self, tmp_path):
        """An already-open ``gdal.Dataset`` is used directly, not reopened.

        Test scenario:
            Passing an open handle returns the same extent as passing its path.
        """
        path = write_raster(
            tmp_path / "s.tif", np.ones((4, 4), dtype="float32"), (0, 4)
        )
        assert _source_bounds(gdal.Open(str(path))) == (0.0, 0.0, 4.0, 4.0), (
            "wrong open-dataset bounds"
        )

    def test_unopenable_source_raises(self):
        """A source that cannot be opened raises a clear ``RuntimeError``.

        Test scenario:
            A non-existent path cannot be opened, so the extent lookup fails loudly
            rather than returning a bogus extent.
        """
        with pytest.raises(RuntimeError, match="could not open merge source"):
            _source_bounds("/no/such/raster/does-not-exist.tif")

    def test_raising_open_names_the_source(self, monkeypatch):
        """A raising ``gdal.Open`` still reports which source could not be opened.

        Test scenario:
            Under ``gdal.UseExceptions()`` (pyramids' default) ``gdal.Open``
            raises instead of returning None, so the ``is None`` guard never
            runs. For a remote source GDAL's message is a bare HTTP status that
            names nothing, so ``_source_bounds`` must add the source itself and
            chain GDAL's original message (#1107).
        """
        remote = "/vsicurl/https://example.invalid/tile_B04_0042.tif"

        def _raise(*_args, **_kwargs):
            raise RuntimeError("HTTP response code: 403")

        monkeypatch.setattr(merge_mod.gdal, "Open", _raise)
        with pytest.raises(RuntimeError) as excinfo:
            _source_bounds(remote)
        message = str(excinfo.value)
        assert "tile_B04_0042.tif" in message, f"source not named: {message}"
        assert "403" in message, f"GDAL's own message not preserved: {message}"
        cause = excinfo.value.__cause__
        assert cause is not None, "GDAL's own error should be chained as __cause__"
        assert "403" in str(cause), f"the chained cause lost GDAL's text: {cause!r}"

    def test_open_returning_none_raises(self, monkeypatch):
        """A ``None`` from ``gdal.Open`` is classified, not returned to the caller.

        Test scenario:
            A caller running with ``gdal.DontUseExceptions()`` gets ``None`` from a
            failed open rather than an exception. ``open_network_dataset`` brands
            that shape too, so ``_source_bounds`` never has to guard for it -- this
            pins that classification as seen from ``_source_bounds`` (#1107).
        """
        monkeypatch.setattr(merge_mod.gdal, "Open", lambda *a, **k: None)
        with pytest.raises(
            RuntimeError, match="GDAL returned no dataset for merge source"
        ):
            _source_bounds("/no/such/raster/does-not-exist.tif")


class TestPrepareSources:
    """Tests for the ``_prepare_sources`` reproject helper."""

    @pytest.mark.parametrize(
        "failing_index, expected_position", [(0, "1/2"), (1, "2/2")]
    )
    def test_raising_open_names_the_source_and_its_position(
        self, shared_crs_pair, monkeypatch, failing_index, expected_position
    ):
        """A raising ``gdal.Open`` names the failing source and how far the open got.

        Args:
            failing_index: Position of the unopenable remote source in ``src_paths``.
            expected_position: The ``n/total`` marker the message must carry.

        Test scenario:
            One of two sources is a remote tile whose open raises a bare
            ``HTTP response code: 403`` -- GDAL names no source for a
            ``/vsicurl/`` path. ``_prepare_sources`` must report the URL, its
            position in ``src_paths``, and chain GDAL's message (#1107). Both
            positions are exercised so the reported index tracks the real one.
        """
        pa, _pb = shared_crs_pair
        remote = "/vsicurl/https://example.invalid/tile_B04_0042.tif"
        paths = [pa, pa]
        paths[failing_index] = remote
        real_open = merge_mod.gdal.Open

        def _raise_for_remote(path, *args, **kwargs):
            if str(path) == remote:
                raise RuntimeError("HTTP response code: 403")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "Open", _raise_for_remote)
        with pytest.raises(RuntimeError) as excinfo:
            _prepare_sources(paths, None)
        message = str(excinfo.value)
        assert "tile_B04_0042.tif" in message, f"source not named: {message}"
        assert expected_position in message, f"wrong position marker: {message}"
        assert "403" in message, f"GDAL's own message not preserved: {message}"
        cause = excinfo.value.__cause__
        assert cause is not None, "GDAL's own error should be chained as __cause__"
        assert "403" in str(cause), f"the chained cause lost GDAL's text: {cause!r}"

    def test_unopenable_signed_source_does_not_leak_its_credential(
        self, shared_crs_pair, monkeypatch
    ):
        """The failure message keeps the URL but blanks the signed credential.

        Test scenario:
            ``merge_rasters`` signs every source before ``_prepare_sources`` sees
            it, so ``src_paths`` can hold a live SAS/presigned URL. A failed open
            must report the source -- that is the point of #1107 -- with the
            secret replaced by ``<redacted>``, never the token itself.
        """
        pa, _pb = shared_crs_pair
        signed = (
            "/vsicurl/https://acct.blob.core.windows.net/c/tile.tif?sig=SECRETTOKEN"
        )
        real_open = merge_mod.gdal.Open

        def _raise_for_signed(path, *args, **kwargs):
            if str(path) == signed:
                raise RuntimeError("HTTP response code: 403")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "Open", _raise_for_signed)
        with pytest.raises(RuntimeError) as excinfo:
            _prepare_sources([pa, signed], None)
        message = str(excinfo.value)
        assert "SECRETTOKEN" not in message, f"credential leaked: {message}"
        assert "<redacted>" in message, f"credential not redacted: {message}"
        assert "tile.tif" in message, f"source should still be named: {message}"

    def test_shared_crs_reuses_open_handles_no_reproject(self, shared_crs_pair):
        """A shared CRS with no ``dst_crs`` reuses the open handles (no reproject).

        Test scenario:
            When sources agree and ``dst_crs`` is None, no reproject happens.
            Each source is opened once and that same handle is returned as the
            compositor input (and held in keepalive) — so no path is opened
            twice. The handles are plain opens, not warped VRTs.
        """
        pa, pb = shared_crs_pair
        sources, keepalive = _prepare_sources([pa, pb], None)
        assert len(sources) == 2, f"Expected two sources, got {len(sources)}"
        assert all(isinstance(s, gdal.Dataset) for s in sources), (
            f"Cheap path should reuse open datasets, got {[type(s) for s in sources]}"
        )
        assert sources is keepalive, (
            "sources and keepalive should be the same held handles"
        )

    def test_dst_crs_materialises_all_as_datasets(self, shared_crs_pair):
        """An explicit ``dst_crs`` materialises every source as a dataset.

        Test scenario:
            With ``dst_crs`` set, BuildVRT cannot mix paths and datasets, so all
            sources are returned as gdal.Dataset objects and held in keepalive.
        """
        pa, pb = shared_crs_pair
        sources, keepalive = _prepare_sources([pa, pb], 3857)
        assert all(isinstance(s, gdal.Dataset) for s in sources), (
            f"All sources should be gdal.Dataset, got {[type(s) for s in sources]}"
        )
        assert len(keepalive) == len(sources), (
            f"Keepalive should hold every dataset, got {len(keepalive)} vs {len(sources)}"
        )

    def test_disagree_no_dst_crs_materialises_all(self, disagree_pair):
        """Disagreeing CRSs with no ``dst_crs`` still materialise all as datasets.

        Test scenario:
            The matching first source is opened and the mismatched one warped, so
            the returned list is homogeneous gdal.Dataset objects.
        """
        pa, pb = disagree_pair
        sources, keepalive = _prepare_sources([pa, pb], None)
        assert all(isinstance(s, gdal.Dataset) for s in sources), (
            f"Disagree path should yield datasets, got {[type(s) for s in sources]}"
        )
        assert len(keepalive) == 2, (
            f"Both datasets should be held, got {len(keepalive)}"
        )

    def test_crs_less_source_raises(self, shared_crs_pair, tmp_path):
        """A source with no CRS raises a clear ValueError.

        Test scenario:
            A bare GeoTIFF written without a projection is rejected (rather than
            silently mis-aligning the mosaic) when merged with a CRS-bearing one.
        """
        pa, _ = shared_crs_pair
        nocrs = str(tmp_path / "nocrs.tif")
        ds = gdal.GetDriverByName("GTiff").Create(nocrs, 4, 4, 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(np.zeros((4, 4), dtype="float32"))
        ds.FlushCache()
        ds = None
        with pytest.raises(ValueError, match="has no CRS"):
            _prepare_sources([pa, nocrs], None)


class TestAsSrs:
    """Tests for the ``_as_srs`` CRS-parsing helper."""

    def test_epsg_int(self):
        """An EPSG int builds a spatial reference with that authority code.

        Test scenario:
            ``_as_srs(4326)`` returns an SRS whose authority code is 4326.
        """
        srs = _as_srs(4326)
        assert srs.GetAuthorityCode(None) == "4326", (
            f"Expected authority code 4326, got {srs.GetAuthorityCode(None)}"
        )

    def test_crs_string(self):
        """A ``"EPSG:nnnn"`` string parses to the matching spatial reference.

        Test scenario:
            ``_as_srs("EPSG:3857")`` returns an SRS with authority code 3857.
        """
        srs = _as_srs("EPSG:3857")
        assert srs.GetAuthorityCode(None) == "3857", (
            f"Expected authority code 3857, got {srs.GetAuthorityCode(None)}"
        )

    def test_invalid_string_raises(self):
        """An unparseable CRS string raises ValueError.

        Test scenario:
            ``_as_srs("not-a-crs")`` cannot be parsed and raises ValueError.
        """
        with pytest.raises(ValueError, match="Could not parse dst_crs"):
            _as_srs("not-a-crs")

    def test_invalid_epsg_int_raises(self):
        """An invalid EPSG code raises ValueError.

        Test scenario:
            ``_as_srs(999999)`` is not a real EPSG code and raises ValueError.
        """
        with pytest.raises(ValueError, match="Could not parse dst_crs"):
            _as_srs(999999)


@pytest.fixture(scope="function")
def same_grid_bands(tmp_path):
    """Two single-band 4x4 EPSG:4326 rasters sharing one grid (for stacking).

    Returns:
        tuple[str, str]: (band_a value 1, band_b value 2) on identical grids.
    """
    a = np.full((4, 4), 1.0, dtype="float32")
    b = np.full((4, 4), 2.0, dtype="float32")
    pa = write_raster(tmp_path / "band_a.tif", a, (0, 4))
    pb = write_raster(tmp_path / "band_b.tif", b, (0, 4))
    return pa, pb


class TestCloudConfigHelper:
    """Tests for the ``_cloud_config`` signer-to-context helper (PY-N)."""

    def test_none_returns_nullcontext(self):
        """A ``None`` signer yields a no-op nullcontext.

        Test scenario:
            ``_cloud_config(None)`` installs no GDAL config, so callers see the
            previous behaviour unchanged.
        """
        ctx = _cloud_config(None)
        assert isinstance(ctx, nullcontext), f"Expected nullcontext, got {type(ctx)}"

    def test_signer_returns_seeded_cloudconfig(self):
        """A signer yields a CloudConfig carrying its ``gdal_env()`` mapping.

        Test scenario:
            ``_cloud_config(signer)`` returns a CloudConfig whose GDAL config
            equals the signer's advertised environment.
        """
        signer = _FakeSigner({"AWS_REGION": "us-west-2"})
        ctx = _cloud_config(signer)
        assert isinstance(ctx, CloudConfig), f"Expected CloudConfig, got {type(ctx)}"
        assert ctx.as_gdal_config() == {"AWS_REGION": "us-west-2"}, (
            f"CloudConfig should carry the signer env, got {ctx.as_gdal_config()}"
        )


class TestMergeRastersSigner:
    """Tests for the ``signer=`` cloud-config kwarg of ``merge_rasters`` (PY-N)."""

    def test_signer_none_unchanged(self, shared_crs_pair, tmp_path):
        """``signer=None`` leaves the merge result unchanged.

        Test scenario:
            Omitting ``signer`` produces the same last-wins overlap (20) as the
            signer-free baseline.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "no_signer.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, signer=None)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(20.0), (
            f"signer=None overlap should be 20, got {arr[0, 2]}"
        )

    def test_signer_produces_correct_output(self, shared_crs_pair, tmp_path):
        """A signer does not change the merge result for local inputs.

        Test scenario:
            Passing a signer (harmless local-read config) still yields the
            last-wins overlap of 20 — the config only affects cloud access.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "signed.tif"
        merge_rasters(
            [pa, pb],
            out,
            no_data_value=-9999.0,
            signer=_FakeSigner({"GDAL_HTTP_TIMEOUT": "30"}),
        )
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == pytest.approx(20.0), (
            f"Signed merge overlap should be 20, got {arr[0, 2]}"
        )

    def test_signer_config_active_during_merge(
        self, shared_crs_pair, tmp_path, monkeypatch
    ):
        """The signer's GDAL config is live while the mosaic is composited.

        Test scenario:
            A spy on gdal.BuildVRT reads the signer's sentinel config option at
            call time; it must see the value, proving CloudConfig was entered.
        """
        from pyramids.dataset import merge as merge_mod

        seen = {}
        real_build_vrt = merge_mod.gdal.BuildVRT

        def spy(*args, **kwargs):
            seen["value"] = gdal.GetConfigOption("PYRAMIDS_TEST_KEY")
            return real_build_vrt(*args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", spy)
        pa, pb = shared_crs_pair
        merge_rasters(
            [pa, pb],
            tmp_path / "active.tif",
            no_data_value=-9999.0,
            signer=_FakeSigner({"PYRAMIDS_TEST_KEY": "on"}),
        )
        assert seen["value"] == "on", (
            f"Signer config should be active during BuildVRT, got {seen.get('value')!r}"
        )

    def test_no_signer_config_absent_during_merge(
        self, shared_crs_pair, tmp_path, monkeypatch
    ):
        """Without a signer no extra config is installed for the merge.

        Test scenario:
            With ``signer=None`` the sentinel config option is unset (None) when
            gdal.BuildVRT runs — the nullcontext path installs nothing.
        """
        from pyramids.dataset import merge as merge_mod

        seen = {}
        real_build_vrt = merge_mod.gdal.BuildVRT

        def spy(*args, **kwargs):
            seen["value"] = gdal.GetConfigOption("PYRAMIDS_TEST_KEY")
            return real_build_vrt(*args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "BuildVRT", spy)
        pa, pb = shared_crs_pair
        merge_rasters([pa, pb], tmp_path / "plain.tif", no_data_value=-9999.0)
        assert seen["value"] is None, (
            f"No signer should leave the sentinel unset, got {seen.get('value')!r}"
        )

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_signer_with_each_method(self, shared_crs_pair, tmp_path, method):
        """Signer composes with every overlap-resolution method.

        Args:
            method: The merge method combined with the signer.

        Test scenario:
            Passing a signer with each method still yields a readable mosaic of
            the union shape (4, 6).
        """
        pa, pb = shared_crs_pair
        out = tmp_path / f"signer_{method}.tif"
        merge_rasters(
            [pa, pb], out, no_data_value=-9999.0, method=method, signer=_FakeSigner({})
        )
        arr = Dataset.read_file(str(out)).read_array()
        assert arr.shape == (
            4,
            6,
        ), f"{method}: expected union shape (4, 6), got {arr.shape}"

    def test_signer_with_dst_crs(self, shared_crs_pair, tmp_path):
        """Signer composes with ``dst_crs`` reprojection.

        Test scenario:
            ``signer`` plus ``dst_crs=3857`` reprojects and stamps EPSG:3857
            while the signer config is applied.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "signer_crs.tif"
        merge_rasters(
            [pa, pb], out, no_data_value=-9999.0, dst_crs=3857, signer=_FakeSigner({})
        )
        result = Dataset.read_file(str(out))
        assert result.epsg == 3857, f"Expected EPSG 3857, got {result.epsg}"

    def test_signer_sign_href_applied_to_each_source(self, shared_crs_pair, tmp_path):
        """H2: ``signer.sign_href`` is called once per source before compositing.

        Test scenario:
            An identity-rewrite signer records each href it signs; after the
            merge its ``seen`` list must equal the two source paths, proving the
            ``sign_href`` hook fires for every source (not only ``gdal_env``).
        """
        pa, pb = shared_crs_pair
        signer = _FakeSigner({})
        merge_rasters(
            [pa, pb], tmp_path / "signed_each.tif", no_data_value=-9999.0, signer=signer
        )
        assert signer.seen == [
            pa,
            pb,
        ], f"sign_href should see each source once, got {signer.seen}"

    def test_signed_href_reaches_mosaic(self, shared_crs_pair, tmp_path, monkeypatch):
        """H2: the *signed* href (not the raw path) is what reaches the mosaic.

        Test scenario:
            A signer that appends ``?sig=tok`` is used; ``_prepare_sources`` is
            stubbed to capture the paths it receives and abort. The captured
            paths must carry the suffix, proving signing happens before the
            GDAL mosaic step — the exact gap H2 fixes (a SAS signer whose
            credential rides the URL would otherwise be dropped).
        """
        captured: dict[str, list[str]] = {}

        class _Stop(Exception):
            pass

        def fake_prepare(src_paths, dst_crs, resampling):
            captured["paths"] = list(src_paths)
            raise _Stop()

        monkeypatch.setattr("pyramids.dataset.merge._prepare_sources", fake_prepare)
        pa, pb = shared_crs_pair
        signer = _FakeSigner({}, suffix="?sig=tok")
        with pytest.raises(_Stop):
            merge_rasters(
                [pa, pb], tmp_path / "x.tif", no_data_value=-9999.0, signer=signer
            )
        assert captured["paths"] == [
            f"{pa}?sig=tok",
            f"{pb}?sig=tok",
        ], f"signed hrefs should reach the mosaic step, got {captured['paths']}"

    def test_url_only_signer_empty_gdal_env(self, shared_crs_pair, tmp_path):
        """H2: a URL-signing signer with an empty ``gdal_env()`` still authenticates.

        Test scenario:
            A signer whose ``gdal_env()`` is ``{}`` (credential rides the href)
            must still have its ``sign_href`` applied to every source — this is
            the case that silently read unauthenticated before H2.
        """
        pa, pb = shared_crs_pair
        signer = _FakeSigner({})
        assert signer.gdal_env() == {}, "precondition: URL-only signer has no env"
        merge_rasters(
            [pa, pb], tmp_path / "url_only.tif", no_data_value=-9999.0, signer=signer
        )
        assert signer.seen == [
            pa,
            pb,
        ], f"URL-only signer's sign_href must still fire per source, got {signer.seen}"


class TestStackBandsSigner:
    """Tests for the ``signer=`` cloud-config kwarg of ``stack_bands`` (PY-N)."""

    def test_signer_none_band_count(self, same_grid_bands):
        """``signer=None`` stacks the inputs into one band per file.

        Test scenario:
            Two single-band rasters stack into a 2-band dataset with no signer.
        """
        pa, pb = same_grid_bands
        result = stack_bands([pa, pb], signer=None)
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"

    def test_signer_band_count(self, same_grid_bands):
        """A signer does not change the stacked band count.

        Test scenario:
            Passing a signer still yields one band per input file.
        """
        pa, pb = same_grid_bands
        result = stack_bands([pa, pb], signer=_FakeSigner({"GDAL_HTTP_TIMEOUT": "30"}))
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"

    def test_signer_config_active_during_stack(self, same_grid_bands, monkeypatch):
        """The signer's GDAL config is live while the bands are stacked.

        Test scenario:
            A spy on Dataset.from_band_files reads the sentinel config option at
            call time and must see it, proving CloudConfig was entered.
        """
        from pyramids.dataset import merge as merge_mod

        seen = {}
        real_from_band_files = merge_mod.Dataset.from_band_files

        def spy(*args, **kwargs):
            seen["value"] = gdal.GetConfigOption("PYRAMIDS_TEST_KEY")
            return real_from_band_files(*args, **kwargs)

        monkeypatch.setattr(merge_mod.Dataset, "from_band_files", spy)
        pa, pb = same_grid_bands
        stack_bands([pa, pb], signer=_FakeSigner({"PYRAMIDS_TEST_KEY": "on"}))
        assert seen["value"] == "on", (
            f"Signer config should be active during stacking, got {seen.get('value')!r}"
        )

    def test_signer_sign_href_applied_to_each_file(self, same_grid_bands):
        """H2: ``signer.sign_href`` fires once per input file before stacking.

        Test scenario:
            An identity-rewrite signer records each href; after the stack its
            ``seen`` list must equal the two input paths, proving ``stack_bands``
            applies the ``sign_href`` hook too (not only ``gdal_env``).
        """
        pa, pb = same_grid_bands
        signer = _FakeSigner({})
        result = stack_bands([pa, pb], signer=signer)
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"
        assert signer.seen == [
            pa,
            pb,
        ], f"sign_href should see each input once, got {signer.seen}"


@pytest.fixture
def uint16_mixed_res_bands(tmp_path):
    """A 10 m and a 20 m uint16 band on the same origin/CRS (nodata 0).

    Mirrors the Sentinel-2 case from issue #362: same unsigned dtype, mismatched
    resolution, so stacking requires align=True.

    Returns:
        tuple[str, str]: (path_10m, path_20m).
    """
    a = Dataset.from_array(
        np.arange(16, dtype="uint16").reshape(4, 4),
        no_data_value=0,
        geo_ref=GeoReference(top_left_corner=(0.0, 40.0), cell_size=10.0, epsg=32630),
    )
    b = Dataset.from_array(
        (np.arange(4, dtype="uint16") + 1).reshape(2, 2),
        no_data_value=0,
        geo_ref=GeoReference(top_left_corner=(0.0, 40.0), cell_size=20.0, epsg=32630),
    )
    pa, pb = str(tmp_path / "b10.tif"), str(tmp_path / "b20.tif")
    a.to_file(pa)
    b.to_file(pb)
    return pa, pb


class TestStackBandsUint16Align:
    """#362: align=True must not overflow on unsigned-dtype bands."""

    def test_stack_bands_uint16_align(self, uint16_mixed_res_bands):
        """stack_bands(align=True) stacks mixed-resolution uint16 bands.

        Test scenario:
            A 10 m + 20 m uint16 pair (nodata 0) stacks into one 2-band uint16
            dataset without the OverflowError from the -9999 template default.
        """
        pa, pb = uint16_mixed_res_bands
        result = stack_bands([pa, pb], align=True, no_data_value=0)
        assert result.band_count == 2, f"expected 2 bands, got {result.band_count}"
        assert result.dtype[0] == "uint16", f"expected uint16, got {result.dtype}"
        assert result.no_data_value[0] == 0, (
            f"nodata should be 0, got {result.no_data_value[0]}"
        )

    def test_from_band_files_uint16_align(self, uint16_mixed_res_bands):
        """from_band_files(align=True) (the underlying API) also succeeds.

        Test scenario:
            The same uint16 mixed-resolution stack via Dataset.from_band_files.
        """
        pa, pb = uint16_mixed_res_bands
        result = Dataset.from_band_files([pa, pb], align=True, no_data_value=0)
        assert result.band_count == 2, f"expected 2 bands, got {result.band_count}"
        assert result.dtype[0] == "uint16", f"expected uint16, got {result.dtype}"

    def test_uint16_align_grid_matches_first(self, uint16_mixed_res_bands):
        """The stacked grid matches the first (10 m) band, not the coarse one.

        Test scenario:
            align resamples the 20 m band onto the 4x4 10 m grid.
        """
        pa, pb = uint16_mixed_res_bands
        result = Dataset.from_band_files([pa, pb], align=True, no_data_value=0)
        assert (result.rows, result.columns) == (
            4,
            4,
        ), f"grid: {(result.rows, result.columns)}"

    def test_uint16_align_inherited_nodata(self, uint16_mixed_res_bands):
        """align=True works when nodata is inherited (not passed) from uint16 sources.

        Test scenario:
            Omitting no_data_value inherits 0 from the sources; the template must
            still not default to -9999 and overflow.
        """
        pa, pb = uint16_mixed_res_bands
        result = Dataset.from_band_files([pa, pb], align=True)
        assert result.band_count == 2, f"expected 2 bands, got {result.band_count}"
        assert result.no_data_value[0] == 0, (
            f"inherited nodata should be 0, got {result.no_data_value[0]}"
        )


class TestMergeNoneGuards:
    """gdal.Translate / gdal.Warp returning None raises a clear RuntimeError (ARC-22)."""

    def test_translate_none_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from the mosaic gdal.Translate raises RuntimeError, not AttributeError."""
        pa, pb = overlapping_pair
        monkeypatch.setattr(gdal, "Translate", lambda *a, **k: None)
        out = str(tmp_path / "o.tif")
        with pytest.raises(RuntimeError, match="writing the mosaic produced no output"):
            merge_rasters([pa, pb], out, no_data_value=-1.0, method="last")

    def test_reduce_warp_none_raises(self, overlapping_pair, tmp_path, monkeypatch):
        """A None from the per-source gdal.Warp raises RuntimeError in _merge_reduce."""
        pa, pb = overlapping_pair
        monkeypatch.setattr(gdal, "Warp", lambda *a, **k: None)
        out = str(tmp_path / "o.tif")
        with pytest.raises(RuntimeError) as excinfo:
            _merge_reduce([pa, pb], out, "min", -1.0, "nan")
        message = str(excinfo.value)
        assert "warping onto the union grid returned no raster" in message, message
        assert Path(pa).name in message, f"the failing source is not named: {message}"

    def test_reduce_path_names_the_failing_source(
        self, overlapping_pair, tmp_path, monkeypatch
    ):
        """The reduce methods name the source, like the z-order methods do.

        Test scenario:
            ``merge_rasters`` hands ``_merge_reduce`` open ``gdal.Dataset``
            handles, whose repr is a SWIG proxy address. Before the labels were
            threaded through, a failure on ``method="min"`` reported that proxy
            instead of the file -- #1107's own complaint surviving on half the
            public ``method`` surface. The message must name the file and carry
            its ``1/2`` position, and must not leak a proxy repr.
        """

        def _raise(*_args, **_kwargs):
            raise RuntimeError("Too many points failed to transform")

        pa, pb = overlapping_pair
        monkeypatch.setattr(merge_mod.gdal, "Warp", _raise)
        with pytest.raises(RuntimeError) as excinfo:
            merge_rasters([pa, pb], tmp_path / "o.tif", method="min")
        message = str(excinfo.value)
        assert Path(pa).name in message, f"the failing source is not named: {message}"
        assert "1/2" in message, f"the source position is missing: {message}"
        assert "Swig Object" not in message, f"a SWIG proxy leaked: {message}"


class TestMergeRastersBbox:
    """Tests for the ``bbox=`` / ``bbox_crs=`` window on ``merge_rasters`` (issue #1064).

    The fixture pair spans a 6x4 union grid on EPSG:4326 at 1.0-degree cells with
    its top-left at ``(0, 4)``, so a native window is easy to state in pixels: the
    bbox ``(1, 1, 4, 3)`` selects columns 1..4 and rows 1..3, i.e. 3x2.
    """

    WINDOW = (1.0, 1.0, 4.0, 3.0)

    @staticmethod
    def _grid(path):
        """Return ``(x_size, y_size, geotransform)`` of a written raster."""
        ds = gdal.Open(str(path))
        return ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform()

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_bbox_restricts_the_output_grid(self, overlapping_pair, tmp_path, method):
        """Every method writes only the windowed sub-grid.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: The overlap-resolution rule under test.

        Test scenario:
            Both code paths are covered - z-order via ``projWin`` and the reduction
            path via the clipped union grid - and both must yield the 3x2 window
            rather than the full 6x4 mosaic.
        """
        out = tmp_path / f"win_{method}.tif"
        merge_rasters(list(overlapping_pair), out, method=method, bbox=self.WINDOW)
        x_size, y_size, _ = self._grid(out)
        assert (x_size, y_size) == (3, 2), (
            f"{method}: expected the 3x2 window, got {x_size}x{y_size}"
        )

    @pytest.mark.parametrize("method", ["last", "min"])
    def test_bbox_output_matches_the_same_slice_of_the_full_merge(
        self, overlapping_pair, tmp_path, method
    ):
        """The window holds the same pixels the full merge puts there.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            Restricting the read must not shift or resample anything - a shape-only
            assertion would pass even if the window were taken from the wrong place.
        """
        full = tmp_path / f"full_{method}.tif"
        windowed = tmp_path / f"win_{method}.tif"
        merge_rasters(list(overlapping_pair), full, method=method)
        merge_rasters(list(overlapping_pair), windowed, method=method, bbox=self.WINDOW)

        full_arr = gdal.Open(str(full)).ReadAsArray()
        win_arr = gdal.Open(str(windowed)).ReadAsArray()
        assert np.allclose(win_arr, full_arr[1:3, 1:4], equal_nan=True), (
            f"{method}: window {win_arr.tolist()} != full slice "
            f"{full_arr[1:3, 1:4].tolist()}"
        )

    @pytest.mark.parametrize("method", ["last", "max"])
    def test_windowed_grid_stays_aligned_to_the_full_grid(
        self, overlapping_pair, tmp_path, method
    ):
        """The window's origin lands on a full-merge pixel edge, at the same scale.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            A window that shifted the origin off-grid or changed the cell size would
            silently resample; the snap is what keeps a windowed merge a strict
            sub-grid of the unwindowed one.
        """
        full = tmp_path / f"a_{method}.tif"
        windowed = tmp_path / f"b_{method}.tif"
        merge_rasters(list(overlapping_pair), full, method=method)
        merge_rasters(list(overlapping_pair), windowed, method=method, bbox=self.WINDOW)

        _, _, full_gt = self._grid(full)
        _, _, win_gt = self._grid(windowed)
        assert win_gt[1] == full_gt[1], f"{method}: x cell size changed"
        assert win_gt[5] == full_gt[5], f"{method}: y cell size changed"
        col = (win_gt[0] - full_gt[0]) / full_gt[1]
        row = (win_gt[3] - full_gt[3]) / full_gt[5]
        assert col == pytest.approx(round(col)), f"{method}: x origin off-grid ({col})"
        assert row == pytest.approx(round(row)), f"{method}: y origin off-grid ({row})"

    @pytest.mark.parametrize("method", ["last", "max"])
    def test_bbox_in_another_crs_selects_the_same_area(
        self, overlapping_pair, tmp_path, method
    ):
        """``bbox_crs=`` lets the bbox stay in the caller's own CRS.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared EPSG:4326 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            The same window is given in EPSG:3857 and must select the same area. A
            reprojected rectangle is a curved quadrilateral, so its envelope can be
            a pixel wider than the native one - the assertion allows that but not a
            full-extent result.
        """
        xs, ys = reproject_coordinates(
            [self.WINDOW[0], self.WINDOW[2]],
            [self.WINDOW[1], self.WINDOW[3]],
            from_crs=4326,
            to_crs=3857,
            precision=None,
        )
        bbox_3857 = (xs[0], ys[0], xs[1], ys[1])
        out = tmp_path / f"m_{method}.tif"
        merge_rasters(
            list(overlapping_pair), out, method=method, bbox=bbox_3857, bbox_crs=3857
        )
        x_size, y_size, _ = self._grid(out)
        assert 3 <= x_size <= 4, f"{method}: expected ~3 cols, got {x_size}"
        assert 2 <= y_size <= 3, f"{method}: expected ~2 rows, got {y_size}"

    @pytest.mark.parametrize("method", ["last", "first", "min", "max", "sum"])
    def test_disjoint_bbox_raises_for_every_method(
        self, overlapping_pair, tmp_path, method
    ):
        """A window that misses the mosaic fails loudly on every path.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: The overlap-resolution rule under test.

        Test scenario:
            GDAL does not treat a disjoint ``projWin`` as an error - it writes a 1x1
            no-data raster at the window's origin, which reads back as a successful
            merge of nothing. Both paths must reject it instead.
        """
        with pytest.raises(ValueError, match="does not overlap"):
            merge_rasters(
                list(overlapping_pair),
                tmp_path / f"x_{method}.tif",
                method=method,
                bbox=(100.0, 100.0, 101.0, 101.0),
            )

    @pytest.mark.parametrize("method", ["last", "sum"])
    def test_no_bbox_is_unchanged(self, overlapping_pair, tmp_path, method):
        """Omitting ``bbox`` merges the full extent, as before.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            The window is opt-in; the default path must not change.
        """
        out = tmp_path / f"f_{method}.tif"
        merge_rasters(list(overlapping_pair), out, method=method)
        x_size, y_size, _ = self._grid(out)
        assert (x_size, y_size) == (6, 4), (
            f"{method}: default merge should stay 6x4, got {x_size}x{y_size}"
        )


class TestRestrictGrid:
    """Unit tests for the grid-clipping helper behind the reduction path."""

    GEOTRANSFORM = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

    def test_snaps_outward_onto_the_grid(self):
        """A window inside a pixel grows to cover whole pixels.

        Test scenario:
            Rounding inward would silently drop a partially-covered edge pixel the
            caller asked for, so the clip floors the near edges and ceils the far
            ones.
        """
        gt, x_size, y_size = merge_mod._restrict_grid(
            self.GEOTRANSFORM, 6, 4, "", (1.4, 1.4, 3.6, 2.6), None
        )
        assert (x_size, y_size) == (3, 2), f"expected 3x2, got {x_size}x{y_size}"
        assert gt[0] == 1.0, f"x origin not snapped: {gt[0]}"
        assert gt[3] == 3.0, f"y origin not snapped: {gt[3]}"

    def test_clamps_to_the_grid_extent(self):
        """A window larger than the mosaic clips to the mosaic.

        Test scenario:
            Asking for more than exists must not produce a grid larger than the
            union, which would read outside every source.
        """
        _, x_size, y_size = merge_mod._restrict_grid(
            self.GEOTRANSFORM, 6, 4, "", (-50.0, -50.0, 50.0, 50.0), None
        )
        assert (x_size, y_size) == (6, 4), f"expected 6x4, got {x_size}x{y_size}"

    def test_disjoint_window_raises(self):
        """A non-overlapping window raises rather than returning an empty grid.

        Test scenario:
            A zero-sized grid would be written as a valid-looking empty raster.
        """
        with pytest.raises(ValueError, match="does not overlap"):
            merge_mod._restrict_grid(
                self.GEOTRANSFORM, 6, 4, "", (100.0, 100.0, 101.0, 101.0), None
            )


class TestBboxInProjection:
    """Unit tests for the bbox reprojection helper."""

    def test_passthrough_when_no_bbox_crs_given(self):
        """With ``bbox_crs=None`` the bbox is already in the target CRS.

        Test scenario:
            No reprojection should occur, and no CRS is needed to decide that.
        """
        result = merge_mod._bbox_in_projection((1.0, 2.0, 3.0, 4.0), None, "")
        assert result == (1.0, 2.0, 3.0, 4.0), f"passthrough changed the bbox: {result}"

    def test_unprojectable_bbox_raises(self):
        """A bbox outside the target CRS's domain raises instead of yielding inf.

        Test scenario:
            An orthographic projection can only represent the hemisphere it faces;
            the antipodal side reprojects to non-finite coordinates. Left unchecked
            those would flow into the grid arithmetic and produce a nonsense window
            rather than an error.
        """
        ortho = "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        with pytest.raises(ValueError, match="does not project"):
            merge_mod._bbox_in_projection((175.0, -5.0, 179.0, 5.0), 4326, ortho)


class TestBboxValidation:
    """`bbox` is validated once, up front, for both merge paths."""

    @pytest.mark.parametrize(
        ("bad", "exc", "why"),
        [
            ("1234", TypeError, "a 4-character string is not four coordinates"),
            (b"1234", TypeError, "bytes are not four coordinates"),
            (12.0, TypeError, "a scalar is not a sequence"),
            ((1.0, 2.0, 3.0), ValueError, "three values is not a bbox"),
            ((1.0, 2.0, 3.0, 4.0, 5.0), ValueError, "five values is not a bbox"),
            ((1.0, "south", 3.0, 4.0), TypeError, "a non-numeric element"),
            ((1.0, None, 3.0, 4.0), TypeError, "None is not a coordinate"),
            ((1.0, float("nan"), 3.0, 4.0), ValueError, "NaN is not a coordinate"),
            ((1.0, 2.0, float("inf"), 4.0), ValueError, "inf is not a coordinate"),
            ((4.0, 1.0, 1.0, 3.0), ValueError, "west > east is inverted"),
            ((1.0, 3.0, 4.0, 1.0), ValueError, "south > north is inverted"),
            ((1.0, 1.0, 1.0, 3.0), ValueError, "zero width selects nothing"),
            ((1.0, 1.0, 4.0, 1.0), ValueError, "zero height selects nothing"),
        ],
    )
    def test_rejects_malformed_bbox(self, bad, exc, why):
        """A malformed bbox is refused with a typed error rather than opaque fallout.

        Args:
            bad: The malformed bbox.
            exc: The exception type expected.
            why: What makes it malformed.

        Test scenario:
            Unvalidated, `"1234"` unpacked into four coordinates and silently became
            a window, and a NaN surfaced from deep in the grid arithmetic as
            "cannot convert float NaN to integer".
        """
        with pytest.raises(exc):
            merge_mod._validated_bbox(bad)

    @pytest.mark.parametrize(
        "accepted",
        [
            np.array([1.0, 1.0, 4.0, 3.0]),
            [1.0, 1.0, 4.0, 3.0],
            (1, 1, 4, 3),
        ],
        ids=["ndarray", "list", "ints"],
    )
    def test_accepts_any_iterable_of_four_numbers(self, accepted):
        """A bbox need not be a `Sequence` to be accepted.

        Args:
            accepted: A well-formed bbox in a container that should be allowed.

        Test scenario:
            `np.ndarray` does not register as a `collections.abc.Sequence`, so an
            isinstance check on that ABC rejected `GeoDataFrame.total_bounds` — the
            most natural way a caller in this codebase produces a bbox.
        """
        assert merge_mod._validated_bbox(accepted) == (1.0, 1.0, 4.0, 3.0), (
            f"{type(accepted).__name__} should be accepted as a bbox"
        )

    @pytest.mark.parametrize("method", ["last", "max"])
    def test_malformed_bbox_rejected_on_both_paths(
        self, overlapping_pair, tmp_path, method
    ):
        """Both merge paths reject the same malformed bbox the same way.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            method: One z-order and one reduction method.

        Test scenario:
            An inverted bbox used to be rejected on one path and silently normalised
            by GDAL on the other.
        """
        with pytest.raises(ValueError):
            merge_rasters(
                list(overlapping_pair),
                tmp_path / f"bad_{method}.tif",
                method=method,
                bbox=(4.0, 1.0, 1.0, 3.0),
            )


class TestBboxPathAgreement:
    """The z-order and reduction paths must select the identical window."""

    @staticmethod
    def _grid(path):
        """Return ``(x_size, y_size, origin_x, origin_y)`` of a written raster."""
        ds = gdal.Open(str(path))
        gt = ds.GetGeoTransform()
        return ds.RasterXSize, ds.RasterYSize, round(gt[0], 6), round(gt[3], 6)

    @pytest.mark.parametrize(
        ("label", "kwargs"),
        [
            ("native", {"bbox": (1.0, 1.0, 4.0, 3.0)}),
            ("reprojected", {"bbox": (1.0, 1.0, 4.0, 3.0), "bbox_crs": 4326}),
            (
                "dst_crs_reprojected",
                {"dst_crs": 3857, "bbox": (1.0, 1.0, 4.0, 3.0), "bbox_crs": 4326},
            ),
        ],
    )
    def test_paths_return_the_same_grid(
        self, overlapping_pair, tmp_path, label, kwargs
    ):
        """`last` and `max` produce the same output grid for the same window.

        Args:
            overlapping_pair: Two 4x4 rasters on a shared 6x4 union grid.
            tmp_path: pytest temp directory.
            label: Names the window form under test.
            kwargs: The window arguments passed to both methods.

        Test scenario:
            The two paths once reprojected the window independently and rounded to
            opposite sides of a pixel edge, returning different rasters for identical
            arguments (3x3 at x=111364.8 against 4x3 at x=0.0 under `dst_crs=3857`).
            This is the regression guard for that.
        """
        z_order = tmp_path / f"z_{label}.tif"
        reduce_ = tmp_path / f"r_{label}.tif"
        merge_rasters(list(overlapping_pair), z_order, method="last", **kwargs)
        merge_rasters(list(overlapping_pair), reduce_, method="max", **kwargs)
        assert self._grid(z_order) == self._grid(reduce_), (
            f"{label}: z-order {self._grid(z_order)} != reduce {self._grid(reduce_)}"
        )
        # Matching geometry is not enough: two paths can agree on the grid and still
        # disagree on which source won a pixel. Comparing content is meaningful for
        # this fixture because the later raster also holds the larger value (B=20
        # over A=10), so `last` and `max` must resolve the overlap identically.
        z_values = gdal.Open(str(z_order)).ReadAsArray()
        r_values = gdal.Open(str(reduce_)).ReadAsArray()
        assert np.array_equal(z_values, r_values, equal_nan=True), (
            f"{label}: the two paths agree on the grid but not on its content\n"
            f"z-order:\n{z_values}\nreduce:\n{r_values}"
        )


class TestBboxReprojectionCurvature:
    """The reprojected window must cover the whole requested area, not a chord."""

    def test_envelope_covers_a_curved_edge(self):
        """The envelope of a strongly curved reprojection is not the corner envelope.

        Test scenario:
            EPSG:3035 (Lambert azimuthal) into lon/lat bows each edge outward, so the
            extreme latitude lies in an edge's interior. A four-corner envelope fell
            0.035 deg (~4 km) short of the true north edge. Densely sampling the edges
            gives the bound the implementation must at least reach.
        """
        bbox = (3000000.0, 3000000.0, 4500000.0, 4000000.0)
        computed = merge_mod._bbox_in_projection(bbox, 3035, "EPSG:4326")

        samples_x, samples_y = [], []
        for step in np.linspace(0.0, 1.0, 200):
            samples_x += [bbox[0] + step * (bbox[2] - bbox[0])] * 2 + [bbox[0], bbox[2]]
            samples_y += [bbox[1], bbox[3]] + [bbox[1] + step * (bbox[3] - bbox[1])] * 2
        lons, lats = reproject_coordinates(
            samples_x, samples_y, from_crs=3035, to_crs=4326, precision=None
        )
        corner_north = max(
            reproject_coordinates(
                [bbox[0], bbox[2], bbox[0], bbox[2]],
                [bbox[1], bbox[3], bbox[3], bbox[1]],
                from_crs=3035,
                to_crs=4326,
                precision=None,
            )[1]
        )
        assert max(lats) - corner_north > 0.01, (
            "precondition: this CRS pair must actually curve, otherwise the test "
            "cannot distinguish a corner envelope from a densified one"
        )
        assert computed[3] > corner_north, (
            f"north edge {computed[3]} is only the corner bound {corner_north}; the "
            "envelope is not densified"
        )
        assert computed[3] == pytest.approx(max(lats), abs=0.005), (
            f"north edge {computed[3]} does not reach the densified bound {max(lats)}"
        )


class TestBboxGridGeometry:
    """`_restrict_grid` on grids that are not the north-up, axis-aligned default."""

    def test_south_up_grid_is_not_rejected(self):
        """A positive pixel height still resolves a window — a helper invariant only.

        Test scenario:
            Ordering the row offsets by value, rather than assuming north-up, is what
            stops a south-up grid raising a spurious "does not overlap".

            This pins `_restrict_grid` on its own terms and is NOT evidence that
            pyramids merges south-up rasters. The case cannot be reached end-to-end:
            the union grid always comes from `gdal.BuildVRT`, which skips south-up
            sources outright ("does not support positive NS resolution"), and
            `_merge_reduce`'s strip loop would hand `gdal.Warp` an `outputBounds` with
            `minY > maxY` if one ever did arrive. The ordering stays as defensive code
            so the helper remains correct in isolation.
        """
        gt, x_size, y_size = merge_mod._restrict_grid(
            (0.0, 1.0, 0.0, 0.0, 0.0, 1.0), 6, 4, "", (1.0, 1.0, 4.0, 3.0), None
        )
        assert (x_size, y_size) == (3, 2), f"expected 3x2, got {x_size}x{y_size}"
        assert gt[3] == 1.0, f"south-up origin should be the low edge, got {gt[3]}"

    @pytest.mark.parametrize(
        "geotransform",
        [
            (0.0, 1.0, 0.5, 4.0, 0.0, -1.0),
            (0.0, 1.0, 0.0, 4.0, 0.2, -1.0),
        ],
        ids=["row_skew", "col_skew"],
    )
    def test_rotated_or_sheared_grid_is_refused(self, geotransform):
        """A skewed geotransform is refused rather than silently mis-georeferenced.

        Args:
            geotransform: A grid carrying a non-zero skew term.

        Test scenario:
            The window arithmetic assumes an axis-aligned grid; applying it to a
            rotated one would place the output in the wrong location.
        """
        with pytest.raises(ValueError, match="rotated or sheared"):
            merge_mod._restrict_grid(geotransform, 6, 4, "", (1.0, 1.0, 4.0, 3.0), None)

    def test_zero_pixel_size_is_refused(self):
        """A degenerate geotransform is refused rather than dividing by zero.

        Test scenario:
            A zero pixel size would raise ZeroDivisionError from inside the offset
            arithmetic.
        """
        with pytest.raises(ValueError, match="zero pixel size"):
            merge_mod._restrict_grid(
                (0.0, 0.0, 0.0, 4.0, 0.0, -1.0), 6, 4, "", (1.0, 1.0, 4.0, 3.0), None
            )

    def test_degenerately_thin_window_is_diagnosed_as_thin(self):
        """A sub-tolerance window inside the mosaic is not called disjoint.

        Test scenario:
            The tolerance is applied inward at both edges, so a box starting on a
            pixel boundary and spanning less than it snaps to zero width while
            sitting squarely inside the mosaic. Reporting "does not overlap" sent
            the caller to inspect their extents instead of their box width.
        """
        with pytest.raises(ValueError, match="selects no whole pixel"):
            merge_mod._restrict_grid(
                (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
                6,
                4,
                "",
                (1.0, 1.0, 1.0000001, 3.0),
                None,
            )

    def test_disjoint_window_is_still_diagnosed_as_disjoint(self):
        """A box that genuinely misses the mosaic keeps the overlap message.

        Test scenario:
            The thin-window check runs first, so it must not swallow the disjoint
            case it was split out from.
        """
        with pytest.raises(ValueError, match="does not overlap"):
            merge_mod._restrict_grid(
                (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
                6,
                4,
                "",
                (100.0, 100.0, 104.0, 103.0),
                None,
            )

    def test_edge_on_a_pixel_boundary_adds_no_extra_pixel(self):
        """A window landing exactly on grid lines yields exactly that many pixels.

        Test scenario:
            Float noise puts an edge a few ulps past a boundary; snapping outward on
            that noise costs a spurious row or column and shifts the origin, which is
            how the two merge paths came to disagree.
        """
        nudged = (1.0 + 1e-12, 1.0 - 1e-12, 4.0 - 1e-12, 3.0 + 1e-12)
        _, x_size, y_size = merge_mod._restrict_grid(
            (0.0, 1.0, 0.0, 4.0, 0.0, -1.0), 6, 4, "", nudged, None
        )
        assert (x_size, y_size) == (3, 2), (
            f"a boundary-aligned window should be 3x2, got {x_size}x{y_size}"
        )


class TestCollectionMergeBbox:
    """`DatasetCollection.merge` forwards the window to `merge_rasters`."""

    @pytest.fixture
    def two_day_collection(self, tmp_path):
        """A file-backed collection of two 4x4 tiles on a shared 6x4 union grid."""
        for index, left in enumerate((0, 2)):
            write_raster(
                tmp_path / f"2024-01-0{index + 1}.tif",
                np.full((4, 4), 10.0 + index, dtype="float32"),
                (left, 4),
            )
        return DatasetCollection.from_files(
            str(tmp_path), glob="*.tif", date_format="%Y-%m-%d"
        )

    def test_merge_without_bbox_is_the_full_union(self, two_day_collection, tmp_path):
        """The default still merges the whole union grid.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            The window is opt-in; adding the parameter must not change the default.
        """
        out = tmp_path / "full.tif"
        two_day_collection.merge(out)
        ds = gdal.Open(str(out))
        assert (ds.RasterXSize, ds.RasterYSize) == (6, 4), (
            f"expected the full 6x4 union, got {ds.RasterXSize}x{ds.RasterYSize}"
        )

    def test_merge_with_bbox_restricts_the_output(self, two_day_collection, tmp_path):
        """A bbox reaches `merge_rasters` and restricts the merge.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            Without this the motivating STAC workflow could not use the window at
            all — the collection is how those mosaics are actually built.
        """
        out = tmp_path / "win.tif"
        two_day_collection.merge(out, bbox=(1.0, 1.0, 4.0, 3.0))
        ds = gdal.Open(str(out))
        assert (ds.RasterXSize, ds.RasterYSize) == (3, 2), (
            f"expected the 3x2 window, got {ds.RasterXSize}x{ds.RasterYSize}"
        )

    def test_merge_bbox_crs_is_forwarded(self, two_day_collection, tmp_path):
        """`bbox_crs` reaches `merge_rasters` rather than being dropped.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            A silently-ignored `bbox_crs` would read the window as if it were in the
            mosaic's CRS, selecting the wrong area instead of failing.
        """
        xs, ys = reproject_coordinates(
            [1.0, 4.0], [1.0, 3.0], from_crs=4326, to_crs=3857, precision=None
        )
        out = tmp_path / "win_crs.tif"
        two_day_collection.merge(out, bbox=(xs[0], ys[0], xs[1], ys[1]), bbox_crs=3857)
        ds = gdal.Open(str(out))
        assert 3 <= ds.RasterXSize <= 4, f"expected ~3 cols, got {ds.RasterXSize}"
        assert 2 <= ds.RasterYSize <= 3, f"expected ~2 rows, got {ds.RasterYSize}"

    def test_merge_rejects_a_malformed_bbox(self, two_day_collection, tmp_path):
        """Validation is not bypassed by going through the collection.

        Args:
            two_day_collection: Collection of two overlapping tiles.
            tmp_path: pytest temp directory.

        Test scenario:
            The collection forwards the window unchanged, so `merge_rasters`' checks
            must still apply.
        """
        with pytest.raises(ValueError):
            two_day_collection.merge(tmp_path / "bad.tif", bbox=(4.0, 1.0, 1.0, 3.0))


class TestBboxAntimeridian:
    """A window that wraps the antimeridian must be refused, not inverted."""

    ANTIMERIDIAN_BBOX = (700000.0, 100000.0, 950000.0, 300000.0)
    UTM_60N = 32660
    # A constant rather than a helper call: inside a `pytest.raises` block a second
    # invocation could itself be the one that raises, which is what the assertion is
    # meant to pin down.
    LONLAT = "EPSG:4326"

    def test_wrapped_envelope_is_refused(self):
        """A reprojection that wraps past 180 deg raises instead of inverting.

        Test scenario:
            `pyproj.transform_bounds` signals an antimeridian crossing by returning
            west > east rather than by widening the envelope. This UTM 60N window is
            about 2.25 deg wide and reprojects to (178.797, ..., -178.955); read at
            face value that is a box spanning the long way round.
        """
        with pytest.raises(ValueError, match="crosses the antimeridian"):
            merge_mod._bbox_in_projection(
                self.ANTIMERIDIAN_BBOX, self.UTM_60N, self.LONLAT
            )

    def test_wrapped_window_does_not_become_its_complement(self):
        """The wrapped window never resolves to the rest of the world.

        Test scenario:
            `_restrict_grid` sorts the two column offsets, so before this was caught
            the ~2.25 deg request resolved to 35776 columns of a global 0.01 deg grid
            — a 357.76 deg window, the complement of the one asked for. That is both
            the wrong area and a near-global read from a call made to bound one.
        """
        global_grid = (-180.0, 0.01, 0.0, 90.0, 0.0, -0.01)
        with pytest.raises(ValueError, match="crosses the antimeridian"):
            merge_mod._restrict_grid(
                global_grid,
                36000,
                18000,
                self.LONLAT,
                self.ANTIMERIDIAN_BBOX,
                self.UTM_60N,
            )

    def test_an_ordinary_reprojected_window_still_passes(self):
        """The guard rejects only wrapped envelopes, not every reprojection.

        Test scenario:
            A guard keyed on west > east would break every ordinary cross-CRS window
            if reprojection could produce that ordering for other reasons; this
            pins that it does not.
        """
        west, south, east, north = merge_mod._bbox_in_projection(
            (500000.0, 100000.0, 600000.0, 300000.0), 32633, self.LONLAT
        )
        assert west < east, f"west {west} should stay below east {east}"
        assert south < north, f"south {south} should stay below north {north}"


class TestBboxReprojectionFailure:
    """A CRS the transformer cannot build is reported against the bbox."""

    @pytest.mark.parametrize(
        "bad_crs",
        ["not-a-crs", 999999],
        ids=["garbage_string", "unknown_epsg"],
    )
    def test_unusable_bbox_crs_raises_a_clear_error(self, bad_crs):
        """An unusable `bbox_crs` raises rather than escaping as a pyproj error.

        Args:
            bad_crs: A CRS pyproj cannot resolve.

        Test scenario:
            The handler around the transform exists so the caller learns which bbox
            and which CRS failed. Left uncaught they would get a bare pyproj CRSError
            naming neither.
        """
        with pytest.raises(ValueError, match="could not be reprojected"):
            merge_mod._bbox_in_projection((1.0, 1.0, 4.0, 3.0), bad_crs, "EPSG:4326")


class TestBboxLongitudeConvention:
    """A lon/lat window is read in the mosaic's own longitude convention."""

    SIGNED_GRID = (-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    WRAPPED_GRID = (0.0, 1.0, 0.0, 90.0, 0.0, -1.0)

    def test_signed_window_on_a_wrapped_mosaic(self):
        """A -180..180 window resolves against a 0..360 mosaic.

        Test scenario:
            Global grids from climate NetCDF commonly run 0..360 while callers write
            signed bboxes. Untranslated, the two overlap only partially and the clamp
            returned that sliver as a success — the eastern part of the requested
            area, with no error.
        """
        clipped, x_size, y_size = merge_mod._restrict_grid(
            self.WRAPPED_GRID, 360, 180, "EPSG:4326", (-10.0, -5.0, -2.0, 5.0), None
        )
        assert (x_size, y_size) == (8, 10), f"expected 8x10, got {x_size}x{y_size}"
        assert clipped[0] == 350.0, f"expected origin 350.0, got {clipped[0]}"

    def test_wrapped_window_on_a_signed_mosaic(self):
        """A 0..360 window resolves against a -180..180 mosaic.

        Test scenario:
            The translation has to work in both directions, not just the one the
            climate-grid case motivates.
        """
        clipped, x_size, y_size = merge_mod._restrict_grid(
            self.SIGNED_GRID, 360, 180, "EPSG:4326", (350.0, -5.0, 358.0, 5.0), None
        )
        assert (x_size, y_size) == (8, 10), f"expected 8x10, got {x_size}x{y_size}"
        assert clipped[0] == -10.0, f"expected origin -10.0, got {clipped[0]}"

    def test_window_across_the_seam_is_refused(self):
        """A window spanning the mosaic's longitude seam raises.

        Test scenario:
            A signed window over the prime meridian becomes 350..10 in 0..360 — west
            past east. Sorting those edges would resolve it into the complement, the
            same trap the antimeridian guard exists for.
        """
        with pytest.raises(ValueError, match="crosses the seam"):
            merge_mod._restrict_grid(
                self.WRAPPED_GRID, 360, 180, "EPSG:4326", (-10.0, -5.0, 10.0, 5.0), None
            )

    def test_projected_mosaic_is_untouched(self):
        """Longitude rewriting never applies to a projected CRS.

        Test scenario:
            Easting values can legitimately be negative or exceed 180; treating them
            as longitudes would corrupt every projected window.
        """
        _, x_size, y_size = merge_mod._restrict_grid(
            (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
            6,
            4,
            "EPSG:3857",
            (1.0, 1.0, 4.0, 3.0),
            None,
        )
        assert (x_size, y_size) == (3, 2), f"expected 3x2, got {x_size}x{y_size}"


class TestReduceWindowPrunesSources:
    """The reduction path skips sources the window does not touch."""

    def test_narrow_window_does_not_warp_every_source(self, tmp_path, monkeypatch):
        """A narrow window warps only the sources it overlaps.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: Used to count `gdal.Warp` calls.

        Test scenario:
            A strip spans the windowed grid's width, but the per-source prune tested
            latitude only. On a wide east-west mosaic every source shares the same
            latitude band, so all of them were warped per strip however narrow the
            window — the exact cost the window exists to avoid.
        """
        paths = []
        for index in range(6):
            path = tmp_path / f"tile{index}.tif"
            write_raster(
                path, np.full((4, 4), float(index), dtype="float32"), (index * 4, 4)
            )
            paths.append(str(path))

        calls = []
        real_warp = gdal.Warp

        def counting_warp(*args, **kwargs):
            calls.append(kwargs.get("outputBounds"))
            return real_warp(*args, **kwargs)

        monkeypatch.setattr(merge_mod.gdal, "Warp", counting_warp)
        merge_rasters(
            paths, tmp_path / "win.tif", method="max", bbox=(1.0, 1.0, 3.0, 3.0)
        )
        assert len(calls) == 1, (
            f"a window inside one tile should warp one source, got {len(calls)} warps"
        )

    def test_pruning_does_not_drop_needed_sources(self, tmp_path):
        """A window spanning several tiles still reduces all of them.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            An east/west prune that is too eager would silently drop contributing
            sources, leaving no-data where real values belong.
        """
        for index in range(3):
            write_raster(
                tmp_path / f"t{index}.tif",
                np.full((4, 4), float(index + 1), dtype="float32"),
                (index * 4, 4),
            )
        out = tmp_path / "spanning.tif"
        merge_rasters(
            sorted(str(p) for p in tmp_path.glob("t*.tif")),
            out,
            method="max",
            bbox=(0.0, 0.0, 12.0, 4.0),
        )
        values = Dataset.read_file(str(out)).read_array()
        assert np.nanmax(values) == 3.0, (
            f"the easternmost tile must still contribute, got max {np.nanmax(values)}"
        )
