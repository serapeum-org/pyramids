"""Tests for :meth:`Dataset.to_zarr` / :meth:`Dataset.from_zarr`.

DASK-10: Zarr IO path. Parallel chunk writes (one file per dask chunk),
round-trip geobox metadata, fsspec store support, ``compute=False``
returns :class:`dask.delayed.Delayed`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask, import_zarr
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

try:
    import_dask("dask not installed")
    import_zarr("zarr not installed")
    import zarr
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_ZARR = False
else:
    HAS_ZARR = True
requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="dask + zarr not installed")


@pytest.fixture
def small_dataset(tmp_path):
    """Create + save a 5×6 float32 Dataset so its ``_file_name`` is set.

    Zarr IO goes through the lazy ``read_array(chunks=...)`` path which
    needs a real on-disk file to open inside the chunk reader.
    """
    arr = np.arange(30, dtype=np.float32).reshape(5, 6)
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 5.0),
        cell_size=1.0,
        epsg=4326,
    )
    src_path = str(tmp_path / "src.tif")
    ds.to_file(src_path)
    return Dataset.read_file(src_path)


class TestRoundtripEager:
    """Eager Dataset → Zarr → Dataset round-trip preserves values + geobox."""

    @requires_zarr
    def test_values_roundtrip(self, small_dataset, tmp_path):
        store = str(tmp_path / "roundtrip.zarr")
        small_dataset.to_zarr(store)
        reloaded = Dataset.from_zarr(store)
        original = small_dataset.read_array()
        roundtrip = reloaded.read_array()
        if original.ndim != roundtrip.ndim:
            original = np.atleast_3d(original)
            roundtrip = np.atleast_3d(roundtrip)
        np.testing.assert_array_equal(original.squeeze(), roundtrip.squeeze())

    @requires_zarr
    def test_epsg_roundtrip(self, small_dataset, tmp_path):
        store = str(tmp_path / "epsg.zarr")
        small_dataset.to_zarr(store)
        reloaded = Dataset.from_zarr(store)
        assert reloaded.epsg == small_dataset.epsg

    @requires_zarr
    def test_geotransform_roundtrip(self, small_dataset, tmp_path):
        store = str(tmp_path / "gt.zarr")
        small_dataset.to_zarr(store)
        reloaded = Dataset.from_zarr(store)
        assert reloaded.geotransform == small_dataset.geotransform

    @requires_zarr
    def test_per_band_nodata_roundtrip(self, tmp_path):
        """Per-band no-data values survive the round-trip (Z-4).

        Test scenario:
            A 2-band dataset with distinct per-band no-data ``[5.0, 6.0]`` is
            written and reopened. Both values must be recovered (the old reader
            took band 0 only).
        """
        arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326,
            no_data_value=[5.0, 6.0],
        )
        src_path = str(tmp_path / "nd_src.tif")
        ds.to_file(src_path)
        Dataset.read_file(src_path).to_zarr(str(tmp_path / "nd.zarr"))
        reloaded = Dataset.from_zarr(str(tmp_path / "nd.zarr"))
        assert tuple(reloaded.no_data_value) == (5.0, 6.0), (
            f"per-band no-data not recovered: {reloaded.no_data_value}"
        )

    @requires_zarr
    def test_absent_nodata_roundtrip(self, tmp_path):
        """A dataset with no no-data stays no-data after round-trip (Z-4).

        Test scenario:
            A source with ``no_data_value=None`` must round-trip to ``(None,)``,
            not the ``-9999`` sentinel the old reader substituted.
        """
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326,
            no_data_value=None,
        )
        src_path = str(tmp_path / "none_src.tif")
        ds.to_file(src_path)
        Dataset.read_file(src_path).to_zarr(str(tmp_path / "none.zarr"))
        reloaded = Dataset.from_zarr(str(tmp_path / "none.zarr"))
        assert tuple(reloaded.no_data_value) == (None,), (
            f"absent no-data not preserved: {reloaded.no_data_value}"
        )

    @requires_zarr
    def test_projected_crs_roundtrip(self, tmp_path):
        """A projected CRS survives the round-trip via the stored WKT (Z-3).

        Test scenario:
            A UTM (EPSG:32636) dataset is written and reopened. The reader now
            prefers the stored ``spatial_ref`` WKT over re-deriving from the
            ``epsg`` attr, so the full projection is recovered (previously the
            CRS was rebuilt from EPSG only).
        """
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=32636
        )
        src_path = str(tmp_path / "utm.tif")
        ds.to_file(src_path)
        Dataset.read_file(src_path).to_zarr(str(tmp_path / "utm.zarr"))
        reloaded = Dataset.from_zarr(str(tmp_path / "utm.zarr"))
        assert reloaded.epsg == 32636, f"projected CRS not recovered: {reloaded.epsg}"

    @requires_zarr
    def test_band_names_roundtrip(self, tmp_path):
        """Custom band names survive a Zarr round-trip (Z-5).

        Test scenario:
            A 2-band dataset with explicit band names ``['alpha', 'beta']`` is
            written and reopened. The reader must restore the names via the
            ``band_names`` setter (the previous code guarded on a non-existent
            ``set_band_names`` method, so names were silently dropped).
        """
        arr = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326
        )
        ds.band_names = ["alpha", "beta"]
        src_path = str(tmp_path / "bn_src.tif")
        ds.to_file(src_path)
        Dataset.read_file(src_path).to_zarr(str(tmp_path / "bn.zarr"))
        reloaded = Dataset.from_zarr(str(tmp_path / "bn.zarr"))
        assert reloaded.band_names == ["alpha", "beta"], (
            f"band names not restored: {reloaded.band_names}"
        )


class TestComputeFalseDefers:
    """``compute=False`` returns :class:`dask.delayed.Delayed`."""

    @requires_zarr
    def test_returns_delayed(self, small_dataset, tmp_path):
        from dask.delayed import Delayed

        store = str(tmp_path / "deferred.zarr")
        result = small_dataset.to_zarr(store, compute=False)
        assert isinstance(result, Delayed)

    @requires_zarr
    def test_delayed_compute_writes_data(self, small_dataset, tmp_path):
        store = str(tmp_path / "compute.zarr")
        delayed = small_dataset.to_zarr(store, compute=False)
        delayed.compute()
        reloaded = Dataset.from_zarr(store)
        np.testing.assert_array_equal(
            np.atleast_3d(reloaded.read_array()).squeeze(),
            np.atleast_3d(small_dataset.read_array()).squeeze(),
        )

    @requires_zarr
    def test_delayed_compute_finalizes_metadata(self, small_dataset, tmp_path):
        """Computing the deferred write also writes geobox attrs (Z-9).

        Test scenario:
            With ``compute=False`` the data write and the metadata finalize are
            bundled into one ``dask.delayed`` via ``_finalize_after_write`` so
            the attribute write runs *after* the data write. After ``.compute()``
            the ``data`` array must carry the geobox attrs (``epsg`` +
            ``GeoTransform``), proving the finalize step ran.
        """
        store = str(tmp_path / "compute_meta.zarr")
        small_dataset.to_zarr(store, compute=False).compute()
        attrs = dict(zarr.open_group(store, mode="r")["data"].attrs)
        assert int(attrs["epsg"]) == 4326, f"epsg attr not finalized: {attrs.get('epsg')}"
        assert "GeoTransform" in attrs, f"GeoTransform attr missing: {attrs}"


class TestChunksParameter:
    """``chunks=`` controls the underlying dask-array chunking."""

    @requires_zarr
    def test_custom_chunks_respected(self, small_dataset, tmp_path):
        store = str(tmp_path / "chunked.zarr")
        small_dataset.to_zarr(store, chunks=(1, 3, 3))
        root = zarr.open_group(store, mode="r")
        assert root["data"].chunks == (1, 3, 3)

    @requires_zarr
    def test_chunked_read_matches_eager(self, small_dataset, tmp_path):
        """from_zarr(chunks=...) parallel-reads identical values (FR-2).

        Test scenario:
            With ``chunks`` given the read goes through dask.array.from_zarr (a
            parallel chunked read) instead of one eager ``[:]``; values must
            match the eager read, and the result is a plain GDAL-backed Dataset
            (no leftover ``_backend`` flag — the old cosmetic marker, Z-2).
        """
        store = str(tmp_path / "readchunks.zarr")
        small_dataset.to_zarr(store)
        chunked = Dataset.from_zarr(store, chunks=(1, 2, 2))
        assert getattr(chunked, "_backend", None) != "dask", (
            "from_zarr must not pre-mark a materialised dataset as dask-backed "
            f"(got {getattr(chunked, '_backend', None)!r}); read_array owns that flag"
        )
        eager = Dataset.from_zarr(store)
        np.testing.assert_array_equal(
            np.atleast_3d(eager.read_array()).squeeze(),
            np.atleast_3d(chunked.read_array()).squeeze(),
        )


class TestImportErrorPath:
    """Missing zarr / dask surfaces actionable OptionalPackageDoesNotExist."""

    def test_raises_without_zarr(self, small_dataset, tmp_path, monkeypatch):
        """to_zarr with zarr absent raises OptionalPackageDoesNotExist (Z-11).

        Test scenario:
            Patch ``__import__`` so ``import zarr`` fails. ``Dataset.to_zarr``
            must raise the package-wide ``OptionalPackageDoesNotExist`` — not a
            bare ``ImportError`` — and the message must carry both the PyPI and
            conda-forge ``[lazy]`` install hints composed by ``lazy_extra_hint``.
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "zarr":
                raise ImportError("no zarr")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(OptionalPackageDoesNotExist) as exc_info:
            small_dataset.to_zarr(str(tmp_path / "nope.zarr"))
        message = str(exc_info.value)
        assert "pip install 'pyramids-gis[lazy]'" in message, (
            f"PyPI install hint missing from message: {message!r}"
        )
        assert "conda install -c conda-forge pyramids-lazy" in message, (
            f"conda-forge install hint missing from message: {message!r}"
        )


class TestGeoZarrLayout:
    """The written store follows the GeoZarr / CF convention (FR-1)."""

    @requires_zarr
    def test_store_has_geozarr_arrays(self, small_dataset, tmp_path):
        """A written store carries spatial_ref + x/y coords + grid_mapping (FR-1).

        Test scenario:
            ``to_zarr`` must emit the GeoZarr layout so standards-based readers
            georeference it: a ``spatial_ref`` grid-mapping array (with
            ``crs_wkt`` + ``GeoTransform``), 1-D ``x``/``y`` coordinate arrays,
            ``grid_mapping="spatial_ref"`` and ``_ARRAY_DIMENSIONS`` on ``data``.
        """
        store = str(tmp_path / "geozarr.zarr")
        small_dataset.to_zarr(store)
        group = zarr.open_group(store, mode="r")
        keys = set(group.array_keys())
        assert {"data", "spatial_ref", "x", "y"} <= keys, f"missing arrays: {keys}"
        assert group["data"].attrs["grid_mapping"] == "spatial_ref", (
            f"grid_mapping not set: {dict(group['data'].attrs)}"
        )
        assert group["data"].attrs["_ARRAY_DIMENSIONS"] == ["band", "y", "x"], (
            f"data dims wrong: {group['data'].attrs.get('_ARRAY_DIMENSIONS')}"
        )
        sr_attrs = dict(group["spatial_ref"].attrs)
        assert "crs_wkt" in sr_attrs and "GeoTransform" in sr_attrs, (
            f"spatial_ref attrs incomplete: {sorted(sr_attrs)}"
        )
        assert group["x"].shape == (small_dataset.columns,), "x length mismatch"
        assert group["y"].shape == (small_dataset.rows,), "y length mismatch"

    @requires_zarr
    def test_legacy_store_read_warns_and_recovers(self, tmp_path):
        """Reading a legacy flat-attr store warns but still recovers the geobox.

        Test scenario:
            A store written in the legacy layout (geo-referencing as flat attrs
            on ``data``, no ``spatial_ref`` array) must still open via
            ``from_zarr``, recover EPSG 4326, and emit a ``DeprecationWarning``.
        """
        from pyramids.base.crs import sr_from_epsg

        store = str(tmp_path / "legacy.zarr")
        group = zarr.open_group(store, mode="w")
        data = group.create_array(
            "data", data=np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        )
        data.attrs.update(
            {
                "spatial_ref": sr_from_epsg(4326).ExportToWkt(),
                "GeoTransform": "0.0 1.0 0.0 3.0 0.0 -1.0",
                "epsg": 4326,
                "no_data_value": [-9999.0],
                "band_names": [],
                "dtype": "float32",
                "shape": [1, 3, 4],
            }
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Consolidated metadata is currently not part"
            )
            zarr.consolidate_metadata(store)
        with pytest.warns(DeprecationWarning, match="legacy pyramids geobox"):
            reloaded = Dataset.from_zarr(store)
        assert reloaded.epsg == 4326, f"legacy geobox not recovered: {reloaded.epsg}"


class TestCompressor:
    """``compressor=`` controls the zarr codec on the data array (FR-4)."""

    @requires_zarr
    def test_custom_codec_applied(self, small_dataset, tmp_path):
        """A passed zarr-v3 codec is used for the data array (FR-4).

        Test scenario:
            Passing ``compressor=BloscCodec(cname='zstd')`` makes the written
            ``data`` array use that codec (read back via ``.compressors``);
            values still round-trip.
        """
        from zarr.codecs import BloscCodec

        store = str(tmp_path / "zstd.zarr")
        small_dataset.to_zarr(store, compressor=BloscCodec(cname="zstd"))
        compressors = zarr.open_group(store, mode="r")["data"].compressors
        assert any("zstd" in str(getattr(c, "cname", "")) for c in compressors), (
            f"zstd codec not applied: {compressors}"
        )
        np.testing.assert_array_equal(
            np.atleast_3d(Dataset.from_zarr(store).read_array()).squeeze(),
            np.atleast_3d(small_dataset.read_array()).squeeze(),
        )

    @requires_zarr
    def test_uncompressed(self, small_dataset, tmp_path):
        """``compressor=None`` writes an uncompressed data array (FR-4).

        Test scenario:
            Passing ``compressor=None`` yields a ``data`` array with no
            compressors.
        """
        store = str(tmp_path / "raw.zarr")
        small_dataset.to_zarr(store, compressor=None)
        assert zarr.open_group(store, mode="r")["data"].compressors == (), (
            "expected no compressors"
        )


class TestMultiscalePyramid:
    """``overview_factors=`` writes pyramid levels; ``level=`` reads them (FR-7)."""

    @pytest.fixture
    def big_dataset(self, tmp_path):
        arr = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 16.0), cell_size=1.0, epsg=4326
        )
        src = str(tmp_path / "big.tif")
        ds.to_file(src)
        return Dataset.read_file(src)

    @requires_zarr
    def test_writes_levels_and_multiscales_attr(self, big_dataset, tmp_path):
        """to_zarr(overview_factors=...) writes decimated levels + multiscales (FR-7).

        Test scenario:
            ``overview_factors=[2, 4]`` adds ``data_2`` (8x8) and ``data_4``
            (4x4) arrays and a root ``multiscales`` attribute listing the level
            paths and factors.
        """
        store = str(tmp_path / "ms.zarr")
        big_dataset.to_zarr(store, overview_factors=[2, 4])
        root = zarr.open_group(store, mode="r")
        assert {"data", "data_2", "data_4"} <= set(root.array_keys()), "levels missing"
        assert root["data_2"].shape == (1, 8, 8), f"data_2 {root['data_2'].shape}"
        assert root["data_4"].shape == (1, 4, 4), f"data_4 {root['data_4'].shape}"
        # OGC/OME-Zarr multiscales: list of multiscale defs with
        # `datasets[].coordinateTransformations[{type:scale, scale:[...]}]`.
        ms = root.attrs["multiscales"]
        assert isinstance(ms, list) and len(ms) == 1, f"multiscales not a list: {ms}"
        paths = [d["path"] for d in ms[0]["datasets"]]
        assert paths == ["data", "data_2", "data_4"], f"paths {paths}"
        scales = [d["coordinateTransformations"][0]["scale"][1] for d in ms[0]["datasets"]]
        assert scales == [1.0, 2.0, 4.0], f"scales {scales}"
        assert ms[0]["axes"][0]["name"] == "band" and ms[0]["axes"][2]["name"] == "x"

    @requires_zarr
    def test_read_level_scales_geobox(self, big_dataset, tmp_path):
        """from_zarr(level=f) reads the decimated level with cell size scaled (FR-7).

        Test scenario:
            Level 2 of a 16x16 / cell_size 1.0 store is 8x8 with cell_size 2.0
            and the same EPSG/origin; level 1 is the full-res 16x16.
        """
        store = str(tmp_path / "ms2.zarr")
        big_dataset.to_zarr(store, overview_factors=[2, 4])
        lvl2 = Dataset.from_zarr(store, level=2)
        assert (lvl2.rows, lvl2.columns) == (8, 8), f"level-2 dims {(lvl2.rows, lvl2.columns)}"
        assert lvl2.cell_size == 2.0, f"level-2 cell_size {lvl2.cell_size}"
        assert lvl2.epsg == 4326, f"level-2 epsg {lvl2.epsg}"
        assert Dataset.from_zarr(store).rows == 16, "level-1 should be full res"

    @requires_zarr
    def test_missing_level_raises(self, big_dataset, tmp_path):
        """Requesting an unwritten level raises a clear KeyError (FR-7).

        Test scenario:
            A store written without overviews has no ``data_8`` array, so
            ``from_zarr(level=8)`` raises KeyError naming the missing array.
        """
        store = str(tmp_path / "ms3.zarr")
        big_dataset.to_zarr(store)
        with pytest.raises(KeyError, match="data_8|overview level 8"):
            Dataset.from_zarr(store, level=8)

    @requires_zarr
    def test_overviews_require_compute(self, big_dataset, tmp_path):
        """overview_factors with compute=False raises (FR-7).

        Test scenario:
            Pyramid levels are built eagerly from GDAL overviews, so combining
            ``overview_factors`` with ``compute=False`` raises ValueError.
        """
        store = str(tmp_path / "ms4.zarr")
        with pytest.raises(ValueError, match="compute=True"):
            big_dataset.to_zarr(store, overview_factors=[2], compute=False)

    @requires_zarr
    def test_level_read_preserves_nodata_and_band_names(self, tmp_path):
        """Level reads carry nodata + band names from the base (M2).

        Test scenario:
            A 2-band dataset with explicit nodata + band names is written with
            overview_factors=[2]; ``from_zarr(store, level=2)`` recovers the
            same nodata tuple and band names rather than dropping them.
        """
        arr = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 8.0), cell_size=1.0, epsg=4326,
            no_data_value=[-1.0, -2.0],
        )
        ds.band_names = ["red", "nir"]
        src = str(tmp_path / "ms_meta.tif")
        ds.to_file(src)
        Dataset.read_file(src).to_zarr(
            str(tmp_path / "ms_meta.zarr"), overview_factors=[2]
        )
        lvl2 = Dataset.from_zarr(str(tmp_path / "ms_meta.zarr"), level=2)
        assert tuple(lvl2.no_data_value) == (-1.0, -2.0), (
            f"level nodata not preserved: {lvl2.no_data_value}"
        )
        assert lvl2.band_names == ["red", "nir"], (
            f"level band names not preserved: {lvl2.band_names}"
        )


class TestResolveStore:
    """Tests for the v3 store resolution in `_resolve_store` (M1)."""

    def test_str_path_passthrough_no_storage_options(self, tmp_path):
        """A local str/Path with no storage_options returns a bare string.

        Test scenario:
            With ``storage_options=None`` (or empty), ``_resolve_store`` should
            return the path/URL as a plain string so zarr-v3 resolves it
            directly (no fsspec mapper wrapping).
        """
        from pyramids.dataset.ops._zarr import _resolve_store

        result = _resolve_store(str(tmp_path / "s.zarr"), None)
        assert isinstance(result, str), f"expected str, got {type(result).__name__}"
        assert result.endswith("s.zarr"), f"unexpected return: {result}"

    @requires_zarr
    def test_url_with_storage_options_uses_fsspec_store(self, monkeypatch):
        """A URL + storage_options routes through `FsspecStore.from_url` (M1).

        Test scenario:
            ``_resolve_store("s3://...", {"anon": True})`` should call
            ``zarr.storage.FsspecStore.from_url`` with the URL and forwarded
            options — not silently drop them or wrap in an FSMap (which v3
            rejects with `component=`).
        """
        from pyramids.dataset.ops._zarr import _resolve_store
        from zarr.storage import FsspecStore

        calls: list = []

        def fake_from_url(url, storage_options=None):
            calls.append((url, dict(storage_options or {})))
            return "FAKE_FSSPEC_STORE"

        monkeypatch.setattr(FsspecStore, "from_url", staticmethod(fake_from_url))
        result = _resolve_store("s3://bucket/x.zarr", {"anon": True})
        assert result == "FAKE_FSSPEC_STORE", f"unexpected: {result}"
        assert calls == [("s3://bucket/x.zarr", {"anon": True})], f"calls={calls}"
