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
            native, kerch = nref[key], kref[key]
            if isinstance(kerch, list):
                # byte-range ref [url, offset, size]: native absolutises the url, so
                # compare offset + size (the bytes that matter), not the path string
                assert native[1:] == kerch[1:], f"chunk {key} offset/size must match"
            else:
                assert native == kerch, f"inlined chunk {key} must match kerchunk"

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

    def test_byte_range_url_is_absolute(self):
        """Byte-range refs absolutise a local source so manifests are CWD-independent."""
        import os

        from pyramids.netcdf._kerchunk_native import build_single_manifest

        refs = build_single_manifest(FIXTURE)["refs"]
        url = refs["values/0.0.0"][0]   # values is a byte-range ref on this fixture
        assert os.path.isabs(url), f"expected an absolute source path, got {url!r}"

    def test_src_url_override_is_used_verbatim(self):
        """An explicit src_url is written into refs unchanged (e.g. a cloud URL)."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        refs = build_single_manifest(FIXTURE, src_url="s3://bucket/cube.nc")["refs"]
        assert refs["values/0.0.0"][0] == "s3://bucket/cube.nc"

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

    def test_subgroup_metadata_preserved(self, tmp_path):
        """Combine keeps sub-group `.zgroup`/`.zattrs`, not just root + variables."""
        from pyramids.netcdf._kerchunk_native import (
            build_single_manifest,
            combine_manifests,
        )

        paths = [str(tmp_path / f"f{i}.h5") for i in range(2)]
        for index, path in enumerate(paths):
            _make_time_file(path, index)
            with h5py.File(path, "r+") as f:
                grp = f.create_group("meta")
                grp.attrs["source"] = "unit-test"
        per_file = [build_single_manifest(p) for p in paths]
        combined = combine_manifests(per_file, concat_dim="time")["refs"]

        assert "meta/.zattrs" in combined, "sub-group attrs must survive combine"
        assert json.loads(combined["meta/.zattrs"])["source"] == "unit-test"

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


def _mini_time_manifest(time_len: int, chunk: int) -> dict:
    """A minimal v1 manifest with a single 1-D ``time`` variable."""
    zarray = {
        "shape": [time_len],
        "chunks": [chunk],
        "dtype": "<f8",
        "fill_value": None,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
        "compressor": None,
        "zarr_format": 2,
    }
    refs = {
        ".zgroup": json.dumps({"zarr_format": 2}),
        "time/.zarray": json.dumps(zarray),
        "time/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["time"]}),
    }
    for index in range(-(-time_len // chunk)):  # one inlined chunk per grid cell
        refs[f"time/{index}"] = "base64:AAAAAAAAAAA="
    return {"version": 1, "refs": refs}


class TestToJsonable:
    """`_to_jsonable` attribute coercion via emitted `.zattrs`."""

    def test_vector_and_bool_and_bytes_attrs(self, tmp_path):
        """Multi-element vectors stay lists; bytes decode; bools/ints coerce."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        src = str(tmp_path / "attrs.h5")
        with h5py.File(src, "w") as f:
            d = f.create_dataset("v", data=np.zeros(3, dtype="f4"))
            d.attrs["bounds"] = np.array([1.0, 2.0, 3.0], dtype="f8")  # vector -> list
            d.attrs["flag"] = np.bool_(True)
            d.attrs["count"] = np.int32(7)
            d.attrs["label"] = b"hello"
        attrs = json.loads(build_single_manifest(src)["refs"]["v/.zattrs"])
        assert attrs["bounds"] == pytest.approx([1.0, 2.0, 3.0]), "vector kept as list"
        assert attrs["flag"] is True, f"bool not coerced: {attrs['flag']!r}"
        assert attrs["count"] == 7, f"int not coerced: {attrs['count']!r}"
        assert attrs["label"] == "hello", f"bytes not decoded: {attrs['label']!r}"


class TestEncodeFillValue:
    """`_encode_fill_value` across dtypes and special values."""

    def test_float_int_nan_inf_bytes_and_absent(self, tmp_path):
        """_FillValue encodes per dtype; NaN/Inf become strings; absent -> None."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        src = str(tmp_path / "fills.h5")
        with h5py.File(src, "w") as f:
            nan_v = f.create_dataset("nan_v", data=np.zeros(2, dtype="f4"))
            nan_v.attrs["_FillValue"] = np.array([np.nan], dtype="f4")
            inf_v = f.create_dataset("inf_v", data=np.zeros(2, dtype="f8"))
            inf_v.attrs["_FillValue"] = np.array([-np.inf], dtype="f8")
            int_v = f.create_dataset("int_v", data=np.zeros(2, dtype="i4"))
            int_v.attrs["_FillValue"] = np.array([-1], dtype="i4")
            str_v = f.create_dataset("str_v", data=np.zeros(2, dtype="S2"))
            str_v.attrs["_FillValue"] = np.bytes_(b"ab")
            f.create_dataset("plain_v", data=np.zeros(2, dtype="f4"))
        refs = build_single_manifest(src)["refs"]
        import base64 as _b64

        assert json.loads(refs["nan_v/.zarray"])["fill_value"] == "NaN"
        assert json.loads(refs["inf_v/.zarray"])["fill_value"] == "-Infinity"
        assert json.loads(refs["int_v/.zarray"])["fill_value"] == -1
        encoded = json.loads(refs["str_v/.zarray"])["fill_value"]
        assert _b64.b64decode(encoded) == b"ab", f"S-fill base64 wrong: {encoded!r}"
        assert json.loads(refs["plain_v/.zarray"])["fill_value"] is None


class TestFilterMappingExtra:
    """Filter-pipeline edge cases not reachable from the parity fixtures."""

    def test_fletcher32_is_dropped(self, tmp_path):
        """A fletcher32 checksum filter is silently dropped (data still readable)."""
        from pyramids.netcdf._kerchunk_native import build_single_manifest

        src = str(tmp_path / "fletcher.h5")
        with h5py.File(src, "w") as f:
            f.create_dataset("v", data=np.ones((2, 4), dtype="f4"),
                             chunks=(1, 4), fletcher32=True)
        zarray = json.loads(build_single_manifest(src)["refs"]["v/.zarray"])
        assert zarray["filters"] is None, f"fletcher32 should be dropped: {zarray}"
        assert zarray["compressor"] is None, "no compressor expected"

    def test_unknown_filter_id_raises(self):
        """An unrecognised HDF5 filter id raises a clear ValueError."""
        from pyramids.netcdf._kerchunk_native import _compressor_and_filters

        dataset = Mock()
        dataset.name = "/v"
        dataset.dtype = np.dtype("f4")
        dcpl = dataset.id.get_create_plist.return_value
        dcpl.get_nfilters.return_value = 1
        dcpl.get_filter.side_effect = [(99999, 0, (), b"weird")]
        with pytest.raises(ValueError, match="unsupported HDF5 filter id"):
            _compressor_and_filters(dataset)


class TestEmitChunkRefs:
    """Direct `_emit_chunk_refs` behaviour for the non-inlining branch."""

    def test_remote_handle_none_never_inlines(self):
        """With src_handle=None (remote), small chunks are byte-range, not inlined."""
        from pyramids.netcdf._kerchunk_native import _emit_chunk_refs

        refs: dict = {}
        with h5py.File(FIXTURE, "r") as f:
            bands = f["bands"]  # tiny: would inline if a local handle were given
            _emit_chunk_refs(
                refs,
                dataset=bands,
                name="bands",
                src_url="s3://bucket/cube.nc",
                src_handle=None,
                chunks=[3],
                inline_threshold=500,
            )
        assert isinstance(refs["bands/0"], list), "remote chunk must be byte-range"
        assert refs["bands/0"][0] == "s3://bucket/cube.nc"


class TestCombineEdgeCases:
    """`combine_manifests` validation and flat-manifest handling."""

    def test_empty_input_raises(self):
        """An empty manifest list raises ValueError."""
        from pyramids.netcdf._kerchunk_native import combine_manifests

        with pytest.raises(ValueError, match="at least one manifest"):
            combine_manifests([], concat_dim="time")

    def test_flat_v0_manifest_passthrough(self):
        """A single flat (v0) manifest with no 'refs' key passes through."""
        from pyramids.netcdf._kerchunk_native import combine_manifests

        flat = {".zgroup": json.dumps({"zarr_format": 2})}
        out = combine_manifests([flat], concat_dim="time")
        assert out["refs"][".zgroup"] == flat[".zgroup"]

    def test_inconsistent_concat_chunk_size_raises(self):
        """Differing chunk size on the concat axis cannot form a uniform array."""
        from pyramids.netcdf._kerchunk_native import combine_manifests

        per_file = [_mini_time_manifest(2, 2), _mini_time_manifest(2, 1)]
        with pytest.raises(ValueError, match="inconsistent chunk size"):
            combine_manifests(per_file, concat_dim="time")

    def test_non_final_partial_chunk_raises(self):
        """A non-final file whose length is not a chunk multiple is rejected."""
        from pyramids.netcdf._kerchunk_native import combine_manifests

        per_file = [_mini_time_manifest(3, 2), _mini_time_manifest(2, 2)]
        with pytest.raises(ValueError, match="multiple of its chunk size"):
            combine_manifests(per_file, concat_dim="time")
