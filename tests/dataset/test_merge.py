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

from contextlib import nullcontext

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base.remote import CloudConfig
from pyramids.dataset import Dataset
from pyramids.dataset.merge import (
    _as_srs,
    _cloud_config,
    _prepare_sources,
    merge_rasters,
    stack_bands,
)

pytestmark = pytest.mark.core


class _FakeSigner:
    """Minimal signer stand-in exposing only ``gdal_env()``.

    Mirrors the ``gdal_env`` hook of :class:`pyramids.stac.signers.Signer`
    without pulling in the optional STAC dependency.

    Args:
        env: The GDAL config mapping the signer advertises.
    """

    def __init__(self, env):
        self._env = dict(env)

    def gdal_env(self):
        """Return the GDAL config mapping (fed into ``CloudConfig.extra``)."""
        return dict(self._env)


def _write(path, arr, top_left, *, epsg=4326, cell_size=1.0, nodata=-9999.0):
    """Write ``arr`` to ``path`` as a GeoTIFF and return the path string.

    Args:
        path: Output path for the GeoTIFF.
        arr: Array to write.
        top_left: Top-left corner of the raster.
        epsg: EPSG code of the source CRS (default 4326).
        cell_size: Pixel size in CRS units.
        nodata: No-data marker stamped on the output.

    Returns:
        str: The output path as a string.
    """
    ds = Dataset.create_from_array(
        arr, top_left_corner=top_left, cell_size=cell_size, epsg=epsg, no_data_value=nodata
    )
    ds.to_file(str(path))
    return str(path)


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
    pa = _write(tmp_path / "a.tif", a, (0, 4))
    pb = _write(tmp_path / "b.tif", b, (2, 4))
    return pa, pb


class TestMergeMethod:
    """Tests for the ``method=`` overlap rule of ``merge_rasters``."""

    @pytest.mark.parametrize(
        "method, expected_overlap",
        [("last", 20.0), ("first", 10.0), ("min", 10.0), ("max", 20.0), ("sum", 30.0)],
    )
    def test_overlap_resolution(self, overlapping_pair, tmp_path, method, expected_overlap):
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
        assert arr[0, 0] == 10.0, f"A-only column changed: {arr[0, 0]}"
        assert arr[0, 5] == 20.0, f"B-only column changed: {arr[0, 5]}"

    def test_default_method_is_last(self, overlapping_pair, tmp_path):
        """Omitting method defaults to last-wins (backward compatible).

        Test scenario:
            No method argument yields the same overlap as method='last'.
        """
        pa, pb = overlapping_pair
        out = tmp_path / "default.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0)
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == 20.0, f"Default should be last-wins (20), got {arr[0, 2]}"

    def test_reduce_fills_uncovered_with_nodata(self, tmp_path):
        """Reduction methods write nodata where no source covers a pixel.

        Test scenario:
            A occupies the top-left 2x2, B the bottom-right 2x2 of a 4x4 union;
            the off-diagonal quadrants are covered by neither and become nodata
            even for 'sum' (which would otherwise yield 0).
        """
        a = np.full((2, 2), 5.0, dtype="float32")
        b = np.full((2, 2), 7.0, dtype="float32")
        pa = _write(tmp_path / "tl.tif", a, (0, 4))
        pb = _write(tmp_path / "br.tif", b, (2, 2))
        out = tmp_path / "gappy.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, method="sum")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 0] == 5.0, f"Top-left should be A=5, got {arr[0, 0]}"
        assert arr[3, 3] == 7.0, f"Bottom-right should be B=7, got {arr[3, 3]}"
        assert arr[0, 3] == -1.0, f"Uncovered top-right should be nodata -1, got {arr[0, 3]}"
        assert arr[3, 0] == -1.0, f"Uncovered bottom-left should be nodata -1, got {arr[3, 0]}"

    def test_reduce_multiband(self, tmp_path):
        """Reduction operates per band on multi-band sources.

        Test scenario:
            Two 2-band rasters fully overlapping: max picks the larger value in
            each band independently.
        """
        a = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 8.0)]).astype("float32")
        b = np.stack([np.full((3, 3), 4.0), np.full((3, 3), 2.0)]).astype("float32")
        pa = _write(tmp_path / "ma.tif", a, (0, 3))
        pb = _write(tmp_path / "mb.tif", b, (0, 3))
        out = tmp_path / "mmax.tif"
        merge_rasters([pa, pb], out, no_data_value=-9999.0, method="max")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 1, 1] == 4.0, f"Band 0 max should be 4, got {arr[0, 1, 1]}"
        assert arr[1, 1, 1] == 8.0, f"Band 1 max should be 8, got {arr[1, 1, 1]}"

    def test_n_ignores_source_value_in_reduction(self, tmp_path):
        """The n knob makes a source pixel value count as no-data in reduction.

        Test scenario:
            A is all 10; B is all 20 but with n=20 ignored, so min over the
            overlap is 10 (B's 20 is excluded), not 10-vs-20.
        """
        a = np.full((4, 4), 10.0, dtype="float32")
        b = np.full((4, 4), 20.0, dtype="float32")
        pa = _write(tmp_path / "na.tif", a, (0, 4), nodata=-9999.0)
        pb = _write(tmp_path / "nb.tif", b, (2, 4), nodata=-9999.0)
        out = tmp_path / "n_min.tif"
        merge_rasters([pa, pb], out, no_data_value=-1.0, n=20, method="min")
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == 10.0, f"Overlap min ignoring 20 should be 10, got {arr[0, 2]}"
        assert arr[0, 5] == -1.0, f"B-only column was all-ignored -> nodata, got {arr[0, 5]}"

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
        with pytest.raises(RuntimeError, match="gdal.BuildVRT returned None"):
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
        with pytest.raises(RuntimeError, match="gdal.BuildVRT returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif", method="sum")


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
        assert arr[0, 2] == 30.0, f"Collection sum overlap should be 30, got {arr[0, 2]}"


@pytest.fixture(scope="function")
def shared_crs_pair(tmp_path):
    """Two 4x4 EPSG:4326 rasters overlapping in a 2-column strip (a shared CRS).

    Returns:
        tuple[str, str]: (path_a value 10, path_b value 20).
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = _write(tmp_path / "sa.tif", a, (0, 4), epsg=4326)
    pb = _write(tmp_path / "sb.tif", b, (2, 4), epsg=4326)
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
    pa = _write(tmp_path / "da_4326.tif", a, (0, 4), epsg=4326)
    pb_4326 = _write(tmp_path / "db_4326.tif", b, (2, 4), epsg=4326)
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
        assert arr[0, 2] == 20.0, f"Last-wins overlap should be 20, got {arr[0, 2]}"

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
        with pytest.raises(ValueError, match="resampling must be one of"):
            merge_rasters([pa, pb], tmp_path / "bad.tif", dst_crs=3857, resampling="sinc")

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
        merge_rasters([pa, pb], out, dst_crs=3857, resampling=resampling, no_data_value=-9999.0)
        assert Dataset.read_file(str(out)).epsg == 3857, f"{resampling} did not reproject"

    def test_warp_failure_raises(self, shared_crs_pair, tmp_path, monkeypatch):
        """A None from gdal.Warp during reproject raises RuntimeError.

        Test scenario:
            With ``dst_crs=3857`` forcing a reproject, monkeypatching gdal.Warp
            to return None trips the defensive guard in ``_prepare_sources``.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = shared_crs_pair
        monkeypatch.setattr(merge_mod.gdal, "Warp", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.Warp returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif", dst_crs=3857)

    def test_open_failure_raises(self, shared_crs_pair, tmp_path, monkeypatch):
        """A None from gdal.Open while reading source CRS raises RuntimeError.

        Test scenario:
            Monkeypatching gdal.Open to return None trips the guard in the
            CRS-probe loop of ``_prepare_sources``.
        """
        from pyramids.dataset import merge as merge_mod

        pa, pb = shared_crs_pair
        monkeypatch.setattr(merge_mod.gdal, "Open", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="gdal.Open returned None"):
            merge_rasters([pa, pb], tmp_path / "x.tif")


class TestPrepareSources:
    """Tests for the ``_prepare_sources`` reproject helper."""

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
        assert sources is keepalive, "sources and keepalive should be the same held handles"

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
        assert len(keepalive) == 2, f"Both datasets should be held, got {len(keepalive)}"

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
    pa = _write(tmp_path / "band_a.tif", a, (0, 4))
    pb = _write(tmp_path / "band_b.tif", b, (0, 4))
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
        assert arr[0, 2] == 20.0, f"signer=None overlap should be 20, got {arr[0, 2]}"

    def test_signer_produces_correct_output(self, shared_crs_pair, tmp_path):
        """A signer does not change the merge result for local inputs.

        Test scenario:
            Passing a signer (harmless local-read config) still yields the
            last-wins overlap of 20 — the config only affects cloud access.
        """
        pa, pb = shared_crs_pair
        out = tmp_path / "signed.tif"
        merge_rasters(
            [pa, pb], out, no_data_value=-9999.0, signer=_FakeSigner({"GDAL_HTTP_TIMEOUT": "30"})
        )
        arr = Dataset.read_file(str(out)).read_array()
        assert arr[0, 2] == 20.0, f"Signed merge overlap should be 20, got {arr[0, 2]}"

    def test_signer_config_active_during_merge(self, shared_crs_pair, tmp_path, monkeypatch):
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
            [pa, pb], tmp_path / "active.tif", no_data_value=-9999.0,
            signer=_FakeSigner({"PYRAMIDS_TEST_KEY": "on"}),
        )
        assert seen["value"] == "on", (
            f"Signer config should be active during BuildVRT, got {seen.get('value')!r}"
        )

    def test_no_signer_config_absent_during_merge(self, shared_crs_pair, tmp_path, monkeypatch):
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
        assert arr.shape == (4, 6), f"{method}: expected union shape (4, 6), got {arr.shape}"

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
