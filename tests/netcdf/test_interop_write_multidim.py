"""Unit tests for the GDAL-native multidim writer in
:mod:`pyramids.netcdf.engines.interop`.

`write_multidim_netcdf` (plus the `_build_multidim` / `_apply_md_array_attrs` /
`_create_copy_to_netcdf` helpers) assembles a NetCDF from plain ``numpy`` arrays
through GDAL's multidimensional API. `DatasetCollection.to_netcdf` and
`NetCDF.from_xarray` both route through it, but they never exercise a few
branches directly (a coordinate that is not a dimension, the ``units`` ->
``SetUnit`` routing, an empty-attrs array, a raw ``datetime64`` axis, and the
write-failure path) — this module covers those with a ``core`` suite that needs
only GDAL + NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.engines import interop
from pyramids.netcdf.engines.interop import (
    _apply_md_array_attrs,
    _build_multidim,
    _create_copy_to_netcdf,
    open_streaming_multidim_netcdf,
    write_multidim_netcdf,
)

pytestmark = pytest.mark.core


def _root(path: str):
    """Return the root group of a NetCDF opened in multidim mode.

    Args:
        path: Path to the ``.nc`` file.

    Returns:
        gdal.Group: The root group.
    """
    return gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER).GetRootGroup()


def _array_names(path: str) -> set[str]:
    """Return the set of MD-array names in a NetCDF's root group.

    Args:
        path: Path to the ``.nc`` file.

    Returns:
        set[str]: Every multidimensional array name.
    """
    return set(_root(path).GetMDArrayNames())


def _values(path: str, name: str) -> np.ndarray:
    """Read one MD-array's values.

    Args:
        path: Path to the ``.nc`` file.
        name: MD-array name.

    Returns:
        np.ndarray: The array values.
    """
    return np.asarray(_root(path).OpenMDArray(name).ReadAsArray())


def _unit(path: str, name: str) -> str:
    """Read one MD-array's CF unit slot.

    Args:
        path: Path to the ``.nc`` file.
        name: MD-array name.

    Returns:
        str: The unit string (empty when none was set).
    """
    return _root(path).OpenMDArray(name).GetUnit()


def _attrs(path: str, name: str) -> dict:
    """Read one MD-array's attribute dict.

    Args:
        path: Path to the ``.nc`` file.
        name: MD-array name.

    Returns:
        dict: ``{attr_name: value}`` for the array.
    """
    arr = _root(path).OpenMDArray(name)
    return {a.GetName(): a.Read() for a in arr.GetAttributes()}


def _root_attrs(path: str) -> dict:
    """Read the root-group attribute dict.

    Args:
        path: Path to the ``.nc`` file.

    Returns:
        dict: ``{attr_name: value}`` for the root group.
    """
    return {a.GetName(): a.Read() for a in _root(path).GetAttributes()}


def _stream_one_slab_then_raise(path, dims, coords, var_specs) -> None:
    """Enter the streaming writer, write one slab, then raise mid-stream.

    Args:
        path: Output ``.nc`` path.
        dims: Dimension name to length.
        coords: Coordinate spec.
        var_specs: Variable spec.

    Raises:
        RuntimeError: Always, after one slab is written, to exercise the
            mid-stream cleanup path with a single throwing call at the call site.
    """
    with open_streaming_multidim_netcdf(path, dims, coords, var_specs, {}) as writer:
        writer.write_slab("v", 0, np.array([[1]], dtype="int16"))
        raise RuntimeError("boom")


class TestWriteMultidimNetcdf:
    """Tests for :func:`write_multidim_netcdf`."""

    def test_writes_variable_and_coords_that_round_trip(self, tmp_path):
        """A (y, x) variable and its coords are written and read back verbatim.

        Test scenario:
            One 2-D ``temp`` variable over ``y``/``x`` — expected: the three
            MD-arrays exist and ``temp`` reads back equal to the source array.
        """
        data = np.arange(6, dtype="int16").reshape(2, 3)
        out = tmp_path / "grid.nc"
        write_multidim_netcdf(
            out,
            dims={"y": 2, "x": 3},
            coords={
                "y": (np.array([10.0, 20.0]), {}),
                "x": (np.array([1.0, 2.0, 3.0]), {}),
            },
            data_vars={"temp": (("y", "x"), data, {})},
            global_attrs={},
        )
        assert {"temp", "y", "x"}.issubset(_array_names(str(out))), (
            f"missing arrays: {_array_names(str(out))}"
        )
        assert np.array_equal(_values(str(out), "temp"), data), (
            "temp did not round-trip"
        )
        assert np.array_equal(_values(str(out), "y"), [10.0, 20.0]), "y coord wrong"

    def test_global_attrs_written_to_root(self, tmp_path):
        """Root-group attributes are written and read back.

        Test scenario:
            ``global_attrs={"Conventions": "CF-1.8", "epsg": 4326}`` — expected:
            both appear on the root group with the given values.
        """
        out = tmp_path / "attrs.nc"
        write_multidim_netcdf(
            out,
            dims={"x": 2},
            coords={"x": (np.array([0.0, 1.0]), {})},
            data_vars={"v": (("x",), np.array([1.0, 2.0]), {})},
            global_attrs={"Conventions": "CF-1.8", "epsg": 4326},
        )
        attrs = _root_attrs(str(out))
        assert attrs.get("Conventions") == "CF-1.8", f"bad root attrs: {attrs}"
        assert attrs.get("epsg") == 4326, f"bad epsg: {attrs.get('epsg')}"

    def test_empty_global_attrs_adds_no_caller_root_attrs(self, tmp_path):
        """An empty ``global_attrs`` mapping injects none of the caller's attrs.

        Test scenario:
            ``global_attrs={}`` — expected: the write succeeds and the variable
            round-trips; GDAL stamps its own default ``Conventions`` (``CF-1.6``),
            but the caller's ``CF-1.8`` marker is never added.
        """
        out = tmp_path / "noattrs.nc"
        write_multidim_netcdf(
            out,
            dims={"x": 1},
            coords={"x": (np.array([0.0]), {})},
            data_vars={"v": (("x",), np.array([9.0]), {})},
            global_attrs={},
        )
        assert _values(str(out), "v")[0] == 9.0, "variable did not round-trip"
        assert _root_attrs(str(out)).get("Conventions") != "CF-1.8", (
            "caller Conventions was injected despite an empty global_attrs"
        )

    def test_coord_not_in_dims_is_skipped(self, tmp_path):
        """A coordinate whose name is not a declared dimension is skipped.

        Test scenario:
            ``coords`` carries a ``bogus`` entry absent from ``dims`` — expected:
            it is silently skipped and never becomes an MD-array.
        """
        out = tmp_path / "skip.nc"
        write_multidim_netcdf(
            out,
            dims={"x": 2},
            coords={
                "x": (np.array([0.0, 1.0]), {}),
                "bogus": (np.array([5.0, 6.0]), {}),
            },
            data_vars={"v": (("x",), np.array([1.0, 2.0]), {})},
            global_attrs={},
        )
        assert "bogus" not in _array_names(str(out)), "non-dim coord was written"

    def test_units_attr_is_routed_to_the_unit_slot(self, tmp_path):
        """A ``units`` attribute is applied via ``SetUnit``, not as a plain attr.

        Test scenario:
            A variable with ``{"units": "kelvin", "long_name": "T"}`` — expected:
            ``GetUnit()`` returns ``kelvin`` and ``units`` is not among the plain
            attributes, while ``long_name`` is.
        """
        out = tmp_path / "units.nc"
        write_multidim_netcdf(
            out,
            dims={"x": 2},
            coords={"x": (np.array([0.0, 1.0]), {})},
            data_vars={
                "temp": (
                    ("x",),
                    np.array([1.0, 2.0]),
                    {"units": "kelvin", "long_name": "T"},
                )
            },
            global_attrs={},
        )
        assert _unit(str(out), "temp") == "kelvin", "units not routed to the unit slot"
        attrs = _attrs(str(out), "temp")
        assert "units" not in attrs, f"units leaked into plain attrs: {attrs}"
        assert attrs.get("long_name") == "T", f"long_name missing: {attrs}"

    def test_empty_var_attrs_are_a_noop(self, tmp_path):
        """A variable with an empty attribute dict writes cleanly.

        Test scenario:
            ``data_vars`` value carries ``{}`` attrs — expected: the write
            succeeds and the variable carries no plain attributes.
        """
        out = tmp_path / "empty.nc"
        write_multidim_netcdf(
            out,
            dims={"x": 1},
            coords={"x": (np.array([0.0]), {})},
            data_vars={"v": (("x",), np.array([3.0]), {})},
            global_attrs={},
        )
        assert _attrs(str(out), "v") == {}, "unexpected attributes on v"

    def test_datetime_coord_is_cf_encoded(self, tmp_path):
        """A raw ``datetime64`` coordinate is CF-encoded to numeric seconds.

        Test scenario:
            ``time`` is a ``datetime64[ns]`` array — expected: it is written as a
            numeric axis whose unit slot carries a CF ``seconds since`` unit.
        """
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
        out = tmp_path / "time.nc"
        write_multidim_netcdf(
            out,
            dims={"time": 2},
            coords={"time": (time, {})},
            data_vars={"v": (("time",), np.array([1.0, 2.0]), {})},
            global_attrs={},
        )
        assert "seconds since" in _unit(str(out), "time"), "time not CF-encoded"
        vals = _values(str(out), "time")
        assert vals.shape == (2,), f"time axis wrong shape: {vals.shape}"
        assert np.issubdtype(vals.dtype, np.number), f"time axis not numeric: {vals!r}"

    def test_coord_length_mismatch_raises_value_error(self, tmp_path):
        """A coordinate whose length differs from its dimension size raises.

        Test scenario:
            ``dims={"x": 3}`` but the ``x`` coord array has length 2 — expected:
            a ``ValueError`` naming the coordinate and the length mismatch.
        """
        path = tmp_path / "badcoord.nc"
        dims = {"x": 3}
        coords = {"x": (np.array([0.0, 1.0]), {})}
        data_vars = {"v": (("x",), np.array([1.0, 2.0, 3.0]), {})}
        with pytest.raises(ValueError, match="dimension is length"):
            write_multidim_netcdf(path, dims, coords, data_vars, {})

    def test_variable_shape_mismatch_raises_value_error(self, tmp_path):
        """A variable whose shape differs from its declared dims raises.

        Test scenario:
            ``dims={"y": 2, "x": 3}`` but ``v`` is shape ``(2, 2)`` — expected: a
            ``ValueError`` naming the variable and the implied shape.
        """
        path = tmp_path / "badvar.nc"
        dims = {"y": 2, "x": 3}
        coords = {
            "y": (np.array([0.0, 1.0]), {}),
            "x": (np.array([0.0, 1.0, 2.0]), {}),
        }
        data_vars = {"v": (("y", "x"), np.zeros((2, 2)), {})}
        with pytest.raises(ValueError, match="imply"):
            write_multidim_netcdf(path, dims, coords, data_vars, {})

    def test_unknown_variable_dimension_raises_value_error(self, tmp_path):
        """A variable over a dimension absent from ``dims`` raises ``ValueError``.

        Test scenario:
            ``data_vars`` references dimension ``z`` not present in ``dims`` —
            expected: ``ValueError`` naming the offending variable and dimension,
            instead of an opaque ``KeyError``.
        """
        path = tmp_path / "bad.nc"
        dims = {"x": 2}
        coords = {"x": (np.array([0.0, 1.0]), {})}
        data_vars = {"v": (("z",), np.array([1.0, 2.0]), {})}
        with pytest.raises(ValueError, match="unknown dimension"):
            write_multidim_netcdf(path, dims, coords, data_vars, {})

    def test_write_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        """A ``None`` from the netCDF driver's ``CreateCopy`` raises ``RuntimeError``.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Force the netCDF driver's ``CreateCopy`` to return ``None`` — expected:
            ``RuntimeError`` whose message names the target path.
        """
        real = gdal.GetDriverByName

        class _NullCopyDriver:
            def CreateCopy(self, *args, **kwargs):
                return None

        def _fake(name):
            return _NullCopyDriver() if name == "netCDF" else real(name)

        path = tmp_path / "boom.nc"
        dims = {"x": 1}
        coords = {"x": (np.array([0.0]), {})}
        data_vars = {"v": (("x",), np.array([1.0]), {})}
        monkeypatch.setattr(interop.gdal, "GetDriverByName", _fake)
        with pytest.raises(RuntimeError, match="Failed to write NetCDF"):
            write_multidim_netcdf(path, dims, coords, data_vars, {})


class TestOpenStreamingMultidimNetcdf:
    """Tests for :func:`open_streaming_multidim_netcdf` (per-slab streaming)."""

    def test_streams_variable_and_coords_round_trip(self, tmp_path):
        """A ``(time, y, x)`` variable written one slab per timestep round-trips.

        Test scenario:
            Two timesteps written via ``write_slab`` — expected: the arrays exist,
            each timestep reads back its own constant block, and the coords are
            verbatim.
        """
        out = tmp_path / "stream.nc"
        dims = {"time": 2, "y": 2, "x": 3}
        coords = {
            "time": (np.array([0, 1], dtype="int64"), {}),
            "y": (np.array([10.0, 20.0]), {}),
            "x": (np.array([1.0, 2.0, 3.0]), {}),
        }
        var_specs = {"temp": (("time", "y", "x"), np.dtype("int16"), {})}
        with open_streaming_multidim_netcdf(
            out, dims, coords, var_specs, {"Conventions": "CF-1.8"}
        ) as writer:
            for t in range(2):
                writer.write_slab("temp", t, np.full((2, 3), t + 1, dtype="int16"))
        assert {"temp", "time", "y", "x"}.issubset(_array_names(str(out))), (
            f"missing arrays: {_array_names(str(out))}"
        )
        vals = _values(str(out), "temp")
        assert vals.shape == (2, 2, 3), f"unexpected shape: {vals.shape}"
        assert np.array_equal(vals[0], np.full((2, 3), 1)), "timestep 0 wrong"
        assert np.array_equal(vals[1], np.full((2, 3), 2)), "timestep 1 wrong"
        assert np.array_equal(_values(str(out), "y"), [10.0, 20.0]), "y coord wrong"
        assert _root_attrs(str(out)).get("Conventions") == "CF-1.8", "root attr wrong"

    def test_multiple_variables_streamed_with_attrs(self, tmp_path):
        """Two variables stream in parallel and their attributes round-trip.

        Test scenario:
            Per-timestep ``write_slab`` for ``b1`` (with a ``nodata`` attr) and
            ``b2`` — expected: both round-trip and ``b1`` carries ``nodata``.
        """
        out = tmp_path / "multi.nc"
        dims = {"time": 2, "y": 2, "x": 2}
        coords = {
            "time": (np.array([0, 1], dtype="int64"), {}),
            "y": (np.array([0.0, 1.0]), {}),
            "x": (np.array([0.0, 1.0]), {}),
        }
        var_specs = {
            "b1": (("time", "y", "x"), np.dtype("int16"), {"nodata": -1}),
            "b2": (("time", "y", "x"), np.dtype("int16"), {}),
        }
        with open_streaming_multidim_netcdf(out, dims, coords, var_specs, {}) as writer:
            for t in range(2):
                writer.write_slab("b1", t, np.full((2, 2), t, dtype="int16"))
                writer.write_slab("b2", t, np.full((2, 2), 10 + t, dtype="int16"))
        assert np.array_equal(_values(str(out), "b1")[1], np.full((2, 2), 1)), (
            "b1 wrong"
        )
        assert np.array_equal(_values(str(out), "b2")[0], np.full((2, 2), 10)), (
            "b2 wrong"
        )
        assert _attrs(str(out), "b1").get("nodata") == -1, "nodata attr not written"

    def test_datetime_coord_is_cf_encoded(self, tmp_path):
        """A raw ``datetime64`` time coordinate is CF-encoded to a numeric axis.

        Test scenario:
            ``time`` is ``datetime64[ns]`` — expected: a numeric axis whose unit
            slot carries a CF ``since`` unit.
        """
        out = tmp_path / "dt.nc"
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
        dims = {"time": 2, "x": 1}
        coords = {"time": (time, {}), "x": (np.array([0.0]), {})}
        var_specs = {"v": (("time", "x"), np.dtype("float32"), {})}
        with open_streaming_multidim_netcdf(out, dims, coords, var_specs, {}) as writer:
            for t in range(2):
                writer.write_slab("v", t, np.array([float(t)], dtype="float32"))
        assert "since" in _unit(str(out), "time"), "time coord not CF-encoded"

    def test_coord_not_in_dims_is_skipped(self, tmp_path):
        """A coordinate whose name is not a declared dimension is skipped.

        Test scenario:
            A ``bogus`` coord absent from ``dims`` — expected: it never becomes an
            MD-array.
        """
        out = tmp_path / "skip.nc"
        dims = {"time": 1, "x": 2}
        coords = {
            "x": (np.array([0.0, 1.0]), {}),
            "bogus": (np.array([5.0, 6.0]), {}),
        }
        var_specs = {"v": (("time", "x"), np.dtype("float64"), {})}
        with open_streaming_multidim_netcdf(out, dims, coords, var_specs, {}):
            pass
        assert "bogus" not in _array_names(str(out)), "non-dim coord was written"

    def test_coord_length_mismatch_raises_value_error(self, tmp_path):
        """A coordinate length that differs from its dimension size raises.

        Test scenario:
            ``dims={"x": 3}`` but the ``x`` coord has length 2 — expected: a
            ``ValueError`` naming the length mismatch, raised on context entry.
        """
        out = tmp_path / "badcoord.nc"
        with pytest.raises(ValueError, match="dimension is length"):
            with open_streaming_multidim_netcdf(
                out,
                {"x": 3},
                {"x": (np.array([0.0, 1.0]), {})},
                {"v": (("x",), np.dtype("float64"), {})},
                {},
            ):
                pass
        assert not out.exists(), "a setup-phase error must leave no partial file"

    def test_unknown_variable_dimension_raises_value_error(self, tmp_path):
        """A variable over a dimension absent from ``dims`` raises ``ValueError``.

        Test scenario:
            A variable references dimension ``z`` not in ``dims`` — expected:
            ``ValueError`` naming the unknown dimension, raised on context entry.
        """
        out = tmp_path / "badvar.nc"
        with pytest.raises(ValueError, match="unknown dimension"):
            with open_streaming_multidim_netcdf(
                out,
                {"x": 2},
                {"x": (np.array([0.0, 1.0]), {})},
                {"v": (("z",), np.dtype("float64"), {})},
                {},
            ):
                pass
        assert not out.exists(), "a setup-phase error must leave no partial file"

    def test_create_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        """A ``None`` from ``CreateMultiDimensional`` raises ``RuntimeError``.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Force the netCDF driver's ``CreateMultiDimensional`` to return ``None``
            — expected: ``RuntimeError`` naming the target path.
        """
        real = gdal.GetDriverByName

        class _NullCreateDriver:
            def CreateMultiDimensional(self, *args, **kwargs):
                return None

        monkeypatch.setattr(
            interop.gdal,
            "GetDriverByName",
            lambda name: _NullCreateDriver() if name == "netCDF" else real(name),
        )
        out = tmp_path / "boom.nc"
        with pytest.raises(RuntimeError, match="Failed to create NetCDF"):
            with open_streaming_multidim_netcdf(
                out,
                {"x": 1},
                {"x": (np.array([0.0]), {})},
                {"v": (("x",), np.dtype("float64"), {})},
                {},
            ):
                pass
        assert not out.exists(), "no file should exist when creation fails"

    def test_partial_file_removed_on_mid_stream_exception(self, tmp_path):
        """A mid-stream exception removes the partial file (atomic write).

        Test scenario:
            Write one slab, then raise inside the ``with`` block — expected: the
            error propagates and the partially written file is cleaned up, so a
            surviving file always means a complete write (no silently truncated
            NetCDF left behind).
        """
        out = tmp_path / "err.nc"
        dims = {"time": 2, "y": 1, "x": 1}
        coords = {
            "time": (np.array([0, 1], dtype="int64"), {}),
            "y": (np.array([0.0]), {}),
            "x": (np.array([0.0]), {}),
        }
        var_specs = {"v": (("time", "y", "x"), np.dtype("int16"), {})}
        with pytest.raises(RuntimeError, match="boom"):
            _stream_one_slab_then_raise(out, dims, coords, var_specs)
        assert not out.exists(), "a mid-stream error must leave no partial file"


class TestApplyMdArrayAttrs:
    """Tests for :func:`_apply_md_array_attrs`."""

    @staticmethod
    def _scalar_array():
        """Build a tiny in-memory MDArray to attach attributes to.

        Returns:
            gdal.MDArray: A length-1 ``float64`` array named ``a``.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("t")
        root = src.GetRootGroup()
        dim = root.CreateDimension("d", "", "", 1)
        ext = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        arr = root.CreateMDArray("a", [dim], ext)
        arr._keepalive = src  # keep the container alive for the test's lifetime
        return arr

    def test_empty_attrs_is_a_noop(self):
        """An empty attribute mapping neither sets a unit nor any attribute.

        Test scenario:
            ``attrs={}`` — expected: the array's unit stays empty.
        """
        arr = self._scalar_array()
        _apply_md_array_attrs(arr, {})
        assert arr.GetUnit() == "", f"unexpected unit: {arr.GetUnit()!r}"

    def test_units_key_sets_the_unit_slot(self):
        """A ``units`` key is applied through ``SetUnit`` and dropped from attrs.

        Test scenario:
            ``attrs={"units": "m"}`` — expected: ``GetUnit()`` returns ``m`` and no
            plain ``units`` attribute is written.
        """
        arr = self._scalar_array()
        _apply_md_array_attrs(arr, {"units": "m"})
        assert arr.GetUnit() == "m", "unit not set from the units key"
        assert all(a.GetName() != "units" for a in arr.GetAttributes()), (
            "units leaked into plain attributes"
        )

    def test_non_units_attrs_are_written(self):
        """Non-``units`` attributes are written as plain MD-array attributes.

        Test scenario:
            ``attrs={"note": "hi"}`` — expected: a ``note`` attribute reads back
            as ``hi``.
        """
        arr = self._scalar_array()
        _apply_md_array_attrs(arr, {"note": "hi"})
        got = {a.GetName(): a.Read() for a in arr.GetAttributes()}
        assert got.get("note") == "hi", f"note attribute missing: {got}"


class TestCreateCopyToNetcdf:
    """Tests for :func:`_create_copy_to_netcdf`."""

    def test_none_result_raises_runtime_error(self, tmp_path, monkeypatch):
        """A ``None`` ``CreateCopy`` result raises a ``RuntimeError`` naming the path.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Patch the netCDF driver so ``CreateCopy`` returns ``None`` — expected:
            ``RuntimeError`` mentioning the destination path.
        """
        mem = _build_multidim({"x": 1}, {"x": (np.array([0.0]), {})}, {}, {})
        real = gdal.GetDriverByName

        class _NullCopyDriver:
            def CreateCopy(self, *args, **kwargs):
                return None

        monkeypatch.setattr(
            interop.gdal,
            "GetDriverByName",
            lambda name: _NullCopyDriver() if name == "netCDF" else real(name),
        )
        target = str(tmp_path / "x.nc")
        with pytest.raises(RuntimeError, match="Failed to write NetCDF"):
            _create_copy_to_netcdf(mem, target)
