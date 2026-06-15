"""Tests for the zarr-v3-safe native kerchunk manifest builder (#530).

The native builder (`pyramids.netcdf._kerchunk_native.build_single_manifest`)
walks an HDF5/NetCDF4 file with h5py and emits a kerchunk v1 reference manifest
without instantiating a live zarr group, avoiding the kerchunk->zarr-v3 `sync()`
deadlock. These tests pin:

* byte-parity of chunk references against the legacy kerchunk translator,
* functional round-trip through `xr.open_dataset(engine="kerchunk")`,
* the chunked + deflate + shuffle path,
* dimension-name and fill-value semantics.

The builder needs only h5py; parity tests additionally need kerchunk, and
round-trip tests need xarray. Each is gated independently.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_kerchunk

pytestmark = pytest.mark.netcdf_lazy

h5py = pytest.importorskip("h5py")

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    HAS_XARRAY = False
else:
    HAS_XARRAY = True
try:
    import_kerchunk("kerchunk not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_KERCHUNK = False
else:
    HAS_KERCHUNK = True

requires_xarray = pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
requires_kerchunk = pytest.mark.skipif(
    not HAS_KERCHUNK, reason="kerchunk not installed"
)

FIXTURE = "tests/data/netcdf/pyramids-netcdf-3d.nc"


def _make_chunked_file(path: str) -> np.ndarray:
    """Write a chunked + gzip + shuffle dataset; return the source array."""
    rng = np.random.default_rng(0)
    data = rng.random((4, 20, 25)).astype("f4")
    with h5py.File(path, "w") as f:
        d = f.create_dataset(
            "temp",
            data=data,
            chunks=(1, 10, 13),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        d.attrs["_FillValue"] = np.array([-9999.0], dtype="f4")
        d.attrs["units"] = "K"
        f.attrs["Conventions"] = "CF-1.8"
    return data


def _make_time_file(path: str, index: int) -> np.ndarray:
    """Write an HDF5 file with a (time, lat, lon) var and proper dim scales."""
    data = (np.arange(2 * 5 * 6).reshape(2, 5, 6) + index * 1000).astype("f4")
    times = np.array([index * 2, index * 2 + 1], dtype="f8")
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("lat", data=np.arange(5, dtype="f4"))
        f.create_dataset("lon", data=np.arange(6, dtype="f4"))
        var = f.create_dataset("v", data=data, chunks=(1, 5, 6))
        for name in ("time", "lat", "lon"):
            f[name].make_scale(name)
        var.dims[0].attach_scale(f["time"])
        var.dims[1].attach_scale(f["lat"])
        var.dims[2].attach_scale(f["lon"])
    return data


def _codec_chain(zarray: dict) -> list:
    """Normalise (filters, compressor) into one ordered codec list.

    kerchunk and the native builder both round-trip correctly but distribute
    shuffle/zlib differently between `filters` and `compressor`; comparing the
    concatenated chain makes the legitimate representational choice irrelevant.
    """
    chain = list(zarray.get("filters") or [])
    if zarray.get("compressor"):
        chain.append(zarray["compressor"])
    return chain


class TestParityWithKerchunk:
    """Native manifest matches the kerchunk translator where it counts."""

    @requires_kerchunk
    def test_same_keys_and_chunk_refs_on_fixture(self, tmp_path):
        """Native + kerchunk agree on every ref key and chunk byte-range/inline."""
        from pyramids.netcdf._kerchunk import to_kerchunk
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        kref = to_kerchunk(FIXTURE, tmp_path / "k.json", backend="kerchunk")["refs"]
        nref = build_single_manifest(FIXTURE)["refs"]

        assert set(nref) == set(kref), "ref key sets must match kerchunk"
        for key in kref:
            if key.endswith((".zarray", ".zattrs", ".zgroup")):
                continue
            assert nref[key] == kref[key], f"chunk ref {key} must match kerchunk"

    @requires_kerchunk
    def test_zarray_codec_chain_matches(self, tmp_path):
        """Decoded codec chain matches kerchunk for the chunked compressed case."""
        from pyramids.netcdf._kerchunk import to_kerchunk
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        src = str(tmp_path / "chunked.nc")
        _make_chunked_file(src)
        kref = to_kerchunk(src, tmp_path / "k.json", backend="kerchunk")["refs"]
        nref = build_single_manifest(src)["refs"]

        k_arr = json.loads(kref["temp/.zarray"])
        n_arr = json.loads(nref["temp/.zarray"])
        assert _codec_chain(n_arr) == _codec_chain(k_arr), "codec chain must match"
        assert n_arr["chunks"] == [1, 10, 13], "chunk shape preserved"


class TestRoundTrip:
    """Native manifest opens correctly through xarray's kerchunk engine."""

    @requires_xarray
    @requires_kerchunk
    def test_fixture_values_match_direct_read(self, tmp_path):
        """Non-fill data round-trips; fill cells decode to NaN."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        manifest = tmp_path / "native.json"
        manifest.write_text(json.dumps(build_single_manifest(FIXTURE)))
        ds = xr.open_dataset(str(manifest), engine="kerchunk")
        try:
            got = np.asarray(ds["values"].values)
            with h5py.File(FIXTURE, "r") as f:
                truth = f["values"][...]
                fill = np.asarray(f["values"].attrs["_FillValue"]).reshape(-1)[0]
            mask = truth != fill
            np.testing.assert_allclose(got[mask], truth[mask])
            assert np.isnan(got[~mask]).all(), "fill cells must decode to NaN"
            assert list(ds["values"].dims) == ["bands", "y", "x"]
        finally:
            ds.close()

    @requires_xarray
    @requires_kerchunk
    def test_chunked_compressed_roundtrip(self, tmp_path):
        """A chunked + gzip + shuffle dataset round-trips bit-for-bit."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        src = str(tmp_path / "chunked.nc")
        data = _make_chunked_file(src)
        manifest = tmp_path / "native.json"
        manifest.write_text(json.dumps(build_single_manifest(src)))
        ds = xr.open_dataset(str(manifest), engine="kerchunk")
        try:
            np.testing.assert_array_equal(np.asarray(ds["temp"].values), data)
        finally:
            ds.close()


class TestMetadataSemantics:
    """Dimension names, fill values, and attribute hygiene."""

    def test_array_dimensions_resolved(self):
        """`_ARRAY_DIMENSIONS` come from the NetCDF dimension scales."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        refs = build_single_manifest(FIXTURE)["refs"]
        assert json.loads(refs["values/.zattrs"])["_ARRAY_DIMENSIONS"] == [
            "bands",
            "y",
            "x",
        ]
        assert json.loads(refs["x/.zattrs"])["_ARRAY_DIMENSIONS"] == ["x"]

    def test_fill_value_only_from_attribute(self):
        """`_FillValue` attr -> fill_value; HDF5 default fill -> null."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        refs = build_single_manifest(FIXTURE)["refs"]
        values = json.loads(refs["values/.zarray"])
        bands = json.loads(refs["bands/.zarray"])
        assert values["fill_value"] == pytest.approx(-3.4028230607370965e38)
        assert bands["fill_value"] is None, "no _FillValue attr -> null"

    def test_scalar_attrs_squeezed(self):
        """GDAL's 1-element CF attributes are emitted as scalars, not lists."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        refs = build_single_manifest(FIXTURE)["refs"]
        crs = json.loads(refs["transverse_mercator/.zattrs"])
        assert crs["longitude_of_central_meridian"] == pytest.approx(-75.0)

    def test_bookkeeping_attrs_stripped(self):
        """HDF5/NetCDF bookkeeping keys never leak into `.zattrs`."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        refs = build_single_manifest(FIXTURE)["refs"]
        for key in ("values/.zattrs", "x/.zattrs", ".zattrs"):
            attrs = json.loads(refs[key])
            assert not (
                {"CLASS", "NAME", "REFERENCE_LIST", "DIMENSION_LIST", "_FillValue"}
                & set(attrs)
            )


class TestCombine:
    """Native concat of per-file manifests along one dimension."""

    def test_single_manifest_passthrough(self):
        """Combining one manifest returns it unchanged (one file = no concat)."""
        from pyramids.netcdf._kerchunk_native import (
            build_single_manifest,
            combine_manifests,
        )

        one = build_single_manifest(FIXTURE)
        combined = combine_manifests([one], concat_dim="bands")
        assert combined["refs"].keys() == one["refs"].keys()

    def test_stacks_concat_variable_shape(self, tmp_path):
        """Every var with the concat dim is stacked; its shape sums over files."""
        from pyramids.netcdf._kerchunk_native import (
            build_single_manifest,
            combine_manifests,
        )

        paths = [str(tmp_path / f"f{i}.h5") for i in range(3)]
        for index, path in enumerate(paths):
            _make_time_file(path, index)
        per_file = [build_single_manifest(p) for p in paths]
        combined = combine_manifests(per_file, concat_dim="time")["refs"]

        assert json.loads(combined["v/.zarray"])["shape"] == [6, 5, 6]
        assert json.loads(combined["time/.zarray"])["shape"] == [6]
        assert json.loads(combined["lat/.zarray"])["shape"] == [5], "lat not stacked"

    @requires_xarray
    @requires_kerchunk
    def test_combined_data_equals_real_stack(self, tmp_path):
        """Round-trip: the combined cube equals np.concatenate of the inputs."""
        from pyramids.netcdf._kerchunk_native import (
            build_single_manifest,
            combine_manifests,
        )

        paths = [str(tmp_path / f"f{i}.h5") for i in range(3)]
        datas = [_make_time_file(p, i) for i, p in enumerate(paths)]
        per_file = [build_single_manifest(p) for p in paths]
        manifest = tmp_path / "combined.json"
        manifest.write_text(json.dumps(combine_manifests(per_file, concat_dim="time")))

        ds = xr.open_dataset(str(manifest), engine="kerchunk")
        try:
            np.testing.assert_array_equal(
                np.asarray(ds["v"].values), np.concatenate(datas, axis=0)
            )
            np.testing.assert_array_equal(ds["time"].values, np.arange(6))
        finally:
            ds.close()

    def test_misaligned_identical_dim_warns(self, tmp_path):
        """A non-concat coord that differs across files warns (not silently merged)."""
        from pyramids.netcdf._kerchunk_native import (
            build_single_manifest,
            combine_manifests,
        )

        a = str(tmp_path / "a.h5")
        b = str(tmp_path / "b.h5")
        _make_time_file(a, 0)
        _make_time_file(b, 1)
        # shift file b's lat so the inlined coordinate values diverge
        with h5py.File(b, "r+") as f:
            f["lat"][...] = np.arange(5, dtype="f4") + 0.5
        per_file = [build_single_manifest(a), build_single_manifest(b)]
        with pytest.warns(UserWarning, match="may not be co-registered"):
            combine_manifests(per_file, concat_dim="time")

    def test_differing_variable_sets_rejected(self, tmp_path):
        """Files with different variables cannot be combined."""
        from pyramids.netcdf._kerchunk_native import (
            build_single_manifest,
            combine_manifests,
        )

        good = str(tmp_path / "good.h5")
        _make_time_file(good, 0)
        extra = str(tmp_path / "extra.h5")
        _make_time_file(extra, 1)
        with h5py.File(extra, "a") as f:
            f.create_dataset("bonus", data=np.zeros(3, dtype="f4"))
        per_file = [build_single_manifest(good), build_single_manifest(extra)]
        with pytest.raises(ValueError, match="variable sets"):
            combine_manifests(per_file, concat_dim="time")


class TestFilterMapping:
    """HDF5 filter pipeline → zarr (compressor, filters) mapping."""

    def test_shuffle_after_deflate_rejected(self):
        """An unsupported shuffle-after-deflate order raises (not silent wrong data).

        h5py's high-level API only writes shuffle-before-deflate, so the reversed
        order is exercised with a faked filter pipeline.
        """
        from pyramids.netcdf._kerchunk_native import _compressor_and_filters

        dataset = Mock()
        dataset.name = "/v"
        dataset.dtype = np.dtype("f4")
        dcpl = dataset.id.get_create_plist.return_value
        dcpl.get_nfilters.return_value = 2
        # index 0 = deflate (id 1), index 1 = shuffle (id 2) — the reversed order
        dcpl.get_filter.side_effect = [
            (1, 0, (4,), b"deflate"),
            (2, 0, (), b"shuffle"),
        ]
        with pytest.raises(ValueError, match="shuffle after deflate"):
            _compressor_and_filters(dataset)


class TestUnsupportedFeatures:
    """Out-of-scope HDF5 features fail with a clear, actionable error."""

    def test_object_dtype_rejected(self, tmp_path):
        """A vlen/object dtype raises ValueError (caller can fall back)."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        src = str(tmp_path / "vlen.h5")
        with h5py.File(src, "w") as f:
            f.create_dataset("s", data=np.array(["a", "bb"], dtype=object),
                             dtype=h5py.string_dtype())
        with pytest.raises(ValueError, match="vlen|object"):
            build_single_manifest(src)

    def test_unopenable_source_raises_oserror(self, tmp_path):
        """A non-HDF5 / unreadable source raises OSError (not ValueError).

        `to_kerchunk` relies on catching this to fall back to the kerchunk
        translator for sources the local-only native builder cannot open
        (e.g. remote URLs).
        """
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        not_hdf5 = tmp_path / "plain.txt"
        not_hdf5.write_text("this is not an HDF5 file")
        with pytest.raises(OSError):
            build_single_manifest(str(not_hdf5))
