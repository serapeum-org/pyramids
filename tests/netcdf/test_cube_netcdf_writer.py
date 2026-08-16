"""Unit tests for :class:`pyramids.netcdf._cube_netcdf_writer.CubeNetCDFWriter`.

The writer is exercised in isolation with a duck-typed mock collection and a
patched streaming writer, so its three phases (guard + derived state, schema
assembly, streaming write) are pinned independently of real NetCDF I/O. The
end-to-end round-trip is covered by ``tests/dataset/collection/test_to_netcdf.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.dataset._cube_time import TimeAxis
from pyramids.netcdf._cube_netcdf_writer import CubeNetCDFWriter

_WRITER_MODULE = "pyramids.netcdf._cube_netcdf_writer.open_streaming_multidim_netcdf"


def _schema_writer(
    *,
    nodata=(-9999,),
    crs=None,
    epsg=4326,
    geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    band_count=2,
    names=("b1", "b2"),
    var_dtype="int16",
    rows=4,
    cols=5,
):
    """Build a writer with derived state set, for `_build_schema` unit tests.

    Returns:
        CubeNetCDFWriter: a writer whose ``_meta`` / ``band_count`` / ``names`` /
        ``var_dtype`` and base ``y`` / ``x`` axes are populated directly.
    """
    base = SimpleNamespace(y=np.arange(rows), x=np.arange(cols))
    writer = CubeNetCDFWriter(SimpleNamespace(_base=base))
    writer._meta = SimpleNamespace(
        nodata=nodata, crs=crs, epsg=epsg, geotransform=geotransform
    )
    writer.band_count = band_count
    writer.names = list(names)
    writer.var_dtype = np.dtype(var_dtype)
    return writer


def _stream_writer(
    *, datasets, files, band_count=1, names=("b1",), var_dtype="float32"
):
    """Build a writer over mock datasets/files, for `_stream` unit tests.

    Returns:
        CubeNetCDFWriter: a writer whose collection exposes ``datasets`` / ``files``.
    """
    writer = CubeNetCDFWriter(SimpleNamespace(datasets=datasets, files=files))
    writer.band_count = band_count
    writer.names = list(names)
    writer.var_dtype = np.dtype(var_dtype)
    return writer


class TestCubeNetCDFWriterWrite:
    """Tests for ``__init__`` and ``write`` (guard + wiring)."""

    def test_init_stores_collection(self):
        """__init__ captures the collection without touching its metadata.

        Test scenario:
            Construction is side-effect-free; no meta is read (an empty collection
            can still be constructed).
        """
        collection = SimpleNamespace(time_length=0)
        writer = CubeNetCDFWriter(collection)
        assert writer._collection is collection, (
            "collection must be stored on the writer"
        )

    def test_write_empty_collection_raises_before_touching_meta(self):
        """write() raises ValueError for an empty collection, before reading meta.

        Test scenario:
            ``time_length == 0`` raises with the ``to_netcdf`` message; ``_meta`` is
            never accessed (the namespace has none, so accessing it would AttributeError).
        """
        writer = CubeNetCDFWriter(SimpleNamespace(time_length=0))
        with pytest.raises(ValueError, match="cannot write an empty collection"):
            writer.write("out.nc")

    def test_write_derives_state_and_streams(self):
        """write() derives band state, opens the writer, and streams each timestep.

        Test scenario:
            A 2-timestep single-band collection: ``band_count``/``names``/``var_dtype``
            are derived from meta, ``open_streaming_multidim_netcdf`` is opened with
            the output path, and one slab per band per timestep is written.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((4, 5))
        meta = SimpleNamespace(
            shape=(1, 4, 5),
            band_names=["only"],
            dtype="float32",
            nodata=(None,),
            crs=None,
            epsg=None,
            geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        )
        collection = SimpleNamespace(
            time_length=2,
            time=None,
            datasets=[ds, ds],
            files=["f0.tif", "f1.tif"],
            _meta=meta,
            _base=SimpleNamespace(y=np.arange(4), x=np.arange(5)),
        )
        writer = CubeNetCDFWriter(collection)
        sink = Mock()
        with patch(_WRITER_MODULE) as mock_open:
            mock_open.return_value.__enter__.return_value = sink
            writer.write("out.nc", var_per_band=True)

        assert writer.band_count == 1, (
            f"band_count must derive to 1, got {writer.band_count}"
        )
        assert writer.names == ["only"], (
            f"names must come from band_names, got {writer.names}"
        )
        assert mock_open.call_args.args[0] == "out.nc", (
            "writer must open at the given path"
        )
        assert sink.write_slab.call_count == 2, (
            f"1 band x 2 timesteps => 2 slab writes, got {sink.write_slab.call_count}"
        )


class TestCubeNetCDFWriterBuildSchema:
    """Tests for ``_build_schema``."""

    def test_var_per_band_true_shapes_and_specs(self):
        """_build_schema(var_per_band=True) makes one 3-D variable per band.

        Test scenario:
            dims are ``time``/``y``/``x`` (no ``band``); each band name maps to a
            ``(time, y, x)`` spec; time coord carries the axis values + attrs.
        """
        writer = _schema_writer(nodata=None, band_count=2, names=("b1", "b2"))
        axis = TimeAxis(np.array([0, 1, 2]), {"note": "n"})
        dims, coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert dims == {"time": 3, "y": 4, "x": 5}, f"unexpected dims: {dims}"
        assert "band" not in dims, "var_per_band=True must not add a band dim"
        assert set(var_specs) == {"b1", "b2"}, (
            f"expected one var per band, got {set(var_specs)}"
        )
        assert var_specs["b1"][0] == ("time", "y", "x"), (
            "band var dims must be (time, y, x)"
        )
        assert coords["time"][0] is axis.values, "time coord must be the axis values"
        assert coords["time"][1] == {"note": "n"}, (
            "time coord must carry the axis attrs"
        )

    def test_var_per_band_false_single_data_var_with_band_dim(self):
        """_build_schema(var_per_band=False) makes one 4-D `data` variable.

        Test scenario:
            A ``band`` dim/coord is added and a single ``data`` variable spans
            ``(time, band, y, x)``; the human names ride along as a root attr.
        """
        writer = _schema_writer(nodata=None, band_count=2, names=("b1", "b2"))
        axis = TimeAxis(np.array([0, 1]), {})
        dims, coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=False
        )
        assert dims["band"] == 2, "var_per_band=False must add a band dim of size 2"
        assert set(var_specs) == {"data"}, (
            f"expected a single data var, got {set(var_specs)}"
        )
        assert var_specs["data"][0] == ("time", "band", "y", "x"), "data var dims wrong"
        assert root_attrs["band_names"] == "b1,b2", (
            f"band names must ride on the root attr, got {root_attrs.get('band_names')}"
        )

    def test_nodata_present_sets_var_and_root_attr(self):
        """_build_schema surfaces a typed nodata on each var and the root group.

        Test scenario:
            A ``nodata`` of -9999 is cast to the var dtype and attached to both the
            variable spec's attrs and the root attrs.
        """
        writer = _schema_writer(
            nodata=(-9999,), var_dtype="int16", band_count=1, names=("b1",)
        )
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert var_specs["b1"][2]["nodata"] == -9999, (
            "nodata must be on the variable attrs"
        )
        assert root_attrs["nodata"] == -9999, "nodata must be on the root attrs"

    def test_no_nodata_omits_the_attr(self):
        """_build_schema omits nodata attrs entirely when the source has none.

        Test scenario:
            ``nodata`` resolving to None leaves the variable attrs empty and adds no
            root ``nodata`` key.
        """
        writer = _schema_writer(nodata=(None,), band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert var_specs["b1"][2] == {}, (
            f"no nodata => empty var attrs, got {var_specs['b1'][2]}"
        )
        assert "nodata" not in root_attrs, "no nodata => no root nodata attr"

    def test_geobox_root_attrs(self):
        """_build_schema always writes CF-1.8 + GeoTransform, and crs/epsg when present.

        Test scenario:
            A crs with ``to_wkt`` and an epsg populate ``crs_wkt`` / ``epsg``;
            ``Conventions`` and ``GeoTransform`` are always present.
        """
        crs = SimpleNamespace(to_wkt=lambda: "WKT-HERE")
        writer = _schema_writer(crs=crs, epsg=4326, band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, _var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert root_attrs["Conventions"] == "CF-1.8", "must declare CF-1.8"
        assert root_attrs["crs_wkt"] == "WKT-HERE", (
            "crs_wkt must come from crs.to_wkt()"
        )
        assert root_attrs["epsg"] == 4326, "epsg must be written"
        assert root_attrs["GeoTransform"] == "0.0 1.0 0.0 0.0 0.0 -1.0", (
            "GeoTransform wrong"
        )

    def test_crs_none_omits_crs_wkt(self):
        """_build_schema omits crs_wkt when the meta has no CRS.

        Test scenario:
            ``crs=None`` leaves no ``crs_wkt`` root attr (GeoTransform still present).
        """
        writer = _schema_writer(crs=None, band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, _var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert "crs_wkt" not in root_attrs, "no crs => no crs_wkt attr"

    def test_crs_without_to_wkt_is_swallowed(self):
        """_build_schema swallows a non-None CRS that lacks a to_wkt() (AttributeError guard).

        Test scenario:
            A crs object without a ``to_wkt`` attribute triggers the
            ``except AttributeError`` guard, leaving no ``crs_wkt`` root attr.
        """
        writer = _schema_writer(crs=object(), band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, _var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert "crs_wkt" not in root_attrs, (
            "a crs without to_wkt() must be swallowed, not written"
        )


class TestCubeNetCDFWriterStream:
    """Tests for ``_stream``."""

    def test_stream_var_per_band_writes_one_slab_per_band(self):
        """_stream(var_per_band=True) writes each band of each timestep as a slab.

        Test scenario:
            Two single-band timesteps => two ``write_slab`` calls addressed to the
            band's variable name with the (rows, cols) plane.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((4, 5))
        writer = _stream_writer(datasets=[ds, ds], files=["f0", "f1"])
        sink = Mock()
        writer._stream(sink, dims={"y": 4, "x": 5}, var_per_band=True)
        assert sink.write_slab.call_count == 2, (
            f"1 band x 2 timesteps => 2 writes, got {sink.write_slab.call_count}"
        )
        assert sink.write_slab.call_args_list[0].args[0] == "b1", (
            "slab must address band var"
        )

    def test_stream_var_per_band_false_writes_the_data_var(self):
        """_stream(var_per_band=False) writes the whole block to the `data` var.

        Test scenario:
            The (band, rows, cols) block is written under the ``data`` variable name.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((4, 5))
        writer = _stream_writer(datasets=[ds], files=["f0"])
        sink = Mock()
        writer._stream(sink, dims={"y": 4, "x": 5}, var_per_band=False)
        assert sink.write_slab.call_args.args[0] == "data", (
            "single-var write must target 'data'"
        )

    def test_stream_mismatched_shape_raises_alignment_error_naming_file(self):
        """_stream raises AlignmentError naming the offending file on a shape mismatch.

        Test scenario:
            A timestep whose band count differs from the template raises, and the
            message names the source file.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((2, 4, 5))
        writer = _stream_writer(datasets=[ds], files=["bad.tif"], band_count=1)
        with pytest.raises(AlignmentError, match="bad.tif"):
            writer._stream(Mock(), dims={"y": 4, "x": 5}, var_per_band=True)

    def test_stream_mismatch_without_files_names_timestep(self):
        """_stream falls back to 'timestep N' in the error when files is None.

        Test scenario:
            With no ``files`` list, a shape mismatch names the positional timestep.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((3, 4, 5))
        writer = _stream_writer(datasets=[ds], files=None, band_count=1)
        with pytest.raises(AlignmentError, match="timestep 0"):
            writer._stream(Mock(), dims={"y": 4, "x": 5}, var_per_band=True)
