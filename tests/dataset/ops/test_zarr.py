"""Tests for :meth:`Dataset.to_zarr` / :meth:`Dataset.from_zarr`.

DASK-10: Zarr IO path. Parallel chunk writes (one file per dask chunk),
round-trip geobox metadata, fsspec store support, ``compute=False``
returns :class:`dask.delayed.Delayed`.
"""

from __future__ import annotations

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
