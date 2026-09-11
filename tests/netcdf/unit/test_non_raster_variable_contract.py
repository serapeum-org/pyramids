"""What a variable with no raster plane promises once it is a `LabeledArray` (#1126).

The sibling suites assert the *type* that comes back from `get_variable` and the values it
carries. Four things they do not reach are asserted here.

*The declaration.* `_has_raster_plane` predicts, from an array's rank and dtype class alone, what
`get_variable` will return for it. The container CRS walk, the raster guard and the fan-out
classifier all act on that prediction instead of reading the array, so it is checked against the
type `get_variable` really returns, and the reads it exists to save are counted.

*The refusal.* `_require_raster_variable` is the door every internal caller that needs geometry
now goes through. It is exercised from the public methods that let a user name the variable --
`crop_variable`, `reproject_variable`, `resample_variable` and `plot(variable=)` -- because a
refusal is only useful if it survives the call the user actually makes.

*The labels.* `LabeledArray` grew `name` / `unit` / `no_data_value` / `attributes`, then `scale` /
`offset`. Without them a
`-9999.0` sitting in `values` is indistinguishable from real data, so what the wrapper copies off
the array -- and what it defaults to when the array declares nothing -- is the difference between
the wrapper being a better answer than the handle it replaced and merely a safer one.

*The empty-extent dtype.* `_numpy_dtype_of` / `_compound_dtype` describe a variable GDAL declines
to read at all (`count[0] = 0 is invalid`). No driver here will build such a store for a compound
or string type -- MEM refuses a zero-length dimension outright -- so the two helpers are asserted
directly, on the `ExtendedDataType` they take.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.netcdf import LabeledArray
from pyramids.netcdf.netcdf import (
    Container,
    NetCDF,
    _compound_dtype,
    _has_raster_plane,
    _labeled_array_from_md_array,
    _numpy_dtype_of,
)

pytestmark = pytest.mark.core

PADDED_RECORD_SIZE = 24
PADDED_OFFSETS = [0, 8, 16]
# Opened classically (`NETCDF:file:var`), this store has no multidimensional group at all.
CLASSIC_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__6v__1d2-2d4__geog__y-asc.nc"
)
# Fills a double's 53-bit mantissa cannot hold: through a double each one compares equal to
# its neighbour as well as to itself, so masking by it would hide real data too.
INT64_FILL = -9223372036854775806
UINT64_FILL = 18446744073709551614


def _mem_store():
    """Create an empty in-memory multidimensional store.

    Returns:
        gdal.Dataset: The store; keep the reference alive while its arrays are used.
    """
    return gdal.GetDriverByName("MEM").CreateMultiDimensional("test")


def _series_store(values=None, dtype=None):
    """Build a store holding one 1-D array named ``series``.

    Args:
        values: The values to write, or None to leave the array unwritten.
        dtype: The `gdal.ExtendedDataType` for the array; Float64 when omitted.

    Returns:
        tuple[gdal.Dataset, gdal.MDArray]: The store and the array, both kept by the caller.
    """
    store = _mem_store()
    group = store.GetRootGroup()
    size = 3 if values is None else len(values)
    dimension = group.CreateDimension("n", None, None, size)
    array = group.CreateMDArray(
        "series", [dimension], dtype or gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    if values is not None:
        array.Write(values)
    return store, array


def _padded_record():
    """Build a compound type whose fields are padded apart and after the last one.

    Returns:
        gdal.ExtendedDataType: An `Int32` at 0, a `Float64` at 8 and an `Int32` at 16, declared
        24 bytes wide -- the layout a C compiler produces for that field order.
    """
    return gdal.ExtendedDataType.CreateCompound(
        "record_t",
        PADDED_RECORD_SIZE,
        [
            gdal.EDTComponent.Create(
                name, offset, gdal.ExtendedDataType.Create(gdal_type)
            )
            for name, offset, gdal_type in zip(
                ["a", "b", "c"],
                PADDED_OFFSETS,
                [gdal.GDT_Int32, gdal.GDT_Float64, gdal.GDT_Int32],
            )
        ],
    )


def _data_type(kind: str):
    """Build the extended type a test names.

    Args:
        kind: `"float64"`; `"string"`; `"record"`, two `Int32` fields; or
            `"record_with_text"`, an `Int32` and a string field -- the layout of a C struct
            holding an int and a `char *`.

    Returns:
        gdal.ExtendedDataType: The type.
    """
    int32 = gdal.ExtendedDataType.Create(gdal.GDT_Int32)
    if kind == "float64":
        data_type = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    elif kind == "string":
        data_type = gdal.ExtendedDataType.CreateString()
    elif kind == "record":
        data_type = gdal.ExtendedDataType.CreateCompound(
            "pair_t",
            8,
            [
                gdal.EDTComponent.Create("a", 0, int32),
                gdal.EDTComponent.Create("b", 4, int32),
            ],
        )
    else:
        data_type = gdal.ExtendedDataType.CreateCompound(
            "station_t",
            16,
            [
                gdal.EDTComponent.Create("id", 0, int32),
                gdal.EDTComponent.Create(
                    "label", 8, gdal.ExtendedDataType.CreateString()
                ),
            ],
        )
    return data_type


def _one_array_store(name, sizes, kind, group_name=None):
    """Build a store holding one unwritten array, at the root or in a sub-group.

    Args:
        name: The array's name.
        sizes: Its dimension sizes, outermost first; the dimensions are `d0`, `d1`, ...
        kind: The data type, as `_data_type` names it.
        group_name: The sub-group to create the array in, or None for the root.

    Returns:
        gdal.Dataset: The store; keep the reference alive while its arrays are used.
    """
    store = _mem_store()
    group = store.GetRootGroup()
    if group_name is not None:
        group = group.CreateGroup(group_name)
    dimensions = [
        group.CreateDimension(f"d{index}", None, None, size)
        for index, size in enumerate(sizes)
    ]
    group.CreateMDArray(name, dimensions, _data_type(kind))
    return store


def _axis_before_a_projected_grid():
    """Build a store that lists a 1-D `aaa_axis` ahead of a `zzz_grid` declaring EPSG:3857.

    Returns:
        gdal.Dataset: The store; keep the reference alive while its arrays are used.
    """
    store = _mem_store()
    group = store.GetRootGroup()
    axis = group.CreateMDArray(
        "aaa_axis",
        [group.CreateDimension("n", None, None, 3)],
        gdal.ExtendedDataType.Create(gdal.GDT_Float64),
    )
    axis.Write(np.array([1.0, 2.0, 3.0]))
    grid = group.CreateMDArray(
        "zzz_grid",
        [
            group.CreateDimension("y", "projection_y_coordinate", "Y", 2),
            group.CreateDimension("x", "projection_x_coordinate", "X", 3),
        ],
        gdal.ExtendedDataType.Create(gdal.GDT_Float64),
    )
    grid.Write(np.arange(6, dtype="float64").reshape(2, 3))
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(3857)
    grid.SetSpatialRef(reference)
    return store


@pytest.fixture(scope="function")
def array_reads(monkeypatch):
    """Record the name of every array whose values are read while the test runs.

    Both GDAL readers are wrapped -- `ReadAsArray` serves numeric and compound arrays, `Read`
    string ones -- and each still performs the read, so the code under test behaves exactly
    as it would unobserved. Request it before building the store to see every read.

    Args:
        monkeypatch: Pytest's patcher; it puts both readers back on teardown.

    Returns:
        list[str]: The names read, appended to as the test runs.
    """
    names: list[str] = []
    for method in ("ReadAsArray", "Read"):
        original = getattr(gdal.MDArray, method)

        def recording(self, *args, _original=original, **kwargs):
            names.append(self.GetName())
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(gdal.MDArray, method, recording)
    return names


class _StubDimension:
    """A dimension that answers a name and a size, standing in for `gdal.Dimension`."""

    def __init__(self, name: str, size: int):
        """Record the name and size this dimension reports.

        Args:
            name: The dimension name.
            size: The extent it declares.
        """
        self._name = name
        self._size = size

    def GetName(self) -> str:  # noqa: N802 - mirrors the GDAL SWIG spelling
        """The dimension's name."""
        return self._name

    def GetSize(self) -> int:  # noqa: N802 - mirrors the GDAL SWIG spelling
        """The extent the dimension declares."""
        return self._size


class _StubAttribute:
    """An attribute whose `Read` either answers a value or raises, as the test asks."""

    def __init__(self, name: str, value=None, error: Exception | None = None):
        """Record what this attribute answers.

        Args:
            name: The attribute name.
            value: The value `Read` returns when no error is configured.
            error: An exception `Read` raises instead of answering.
        """
        self._name = name
        self._value = value
        self._error = error

    def GetName(self) -> str:  # noqa: N802 - mirrors the GDAL SWIG spelling
        """The attribute's name."""
        return self._name

    def Read(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """The attribute's value, or the configured failure."""
        if self._error is not None:
            raise self._error
        return self._value


class _StubMDArray:
    """A numeric `gdal.MDArray` whose read result the test chooses.

    Only the surface `_labeled_array_from_md_array` touches is implemented. It exists to
    decouple the read from the declared dimensions, which no real GDAL array lets a test do.
    """

    def __init__(self, dims, read_result, attributes=()):
        """Record the declared dimensions and what the read answers with.

        Args:
            dims: `(name, size)` pairs, outermost first.
            read_result: The array `ReadAsArray` returns.
            attributes: The attributes `GetAttributes` returns.
        """
        self._dims = [_StubDimension(name, size) for name, size in dims]
        self._read_result = read_result
        self._attributes = list(attributes)

    def GetDimensions(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """The dimensions this array declares."""
        return self._dims

    def GetDataType(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """A numeric extended type, so the `ReadAsArray` branch is taken."""
        return gdal.ExtendedDataType.Create(gdal.GDT_Float64)

    def ReadAsArray(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """Whatever the test configured, shape included."""
        return self._read_result

    def GetAttributes(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """The attributes this array carries."""
        return self._attributes

    def GetUnit(self) -> str:  # noqa: N802 - mirrors the GDAL SWIG spelling
        """No unit declared."""
        return ""

    def GetScale(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """No `scale_factor`: the array is not packed."""
        return None

    def GetOffset(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """No `add_offset`: the array is not packed."""
        return None

    def GetNoDataValueAsDouble(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """No fill value declared."""
        return None


class _FailingStubMDArray(_StubMDArray):
    """A numeric array whose read fails the way an I/O or decompression error does."""

    def ReadAsArray(self):  # noqa: N802 - mirrors the GDAL SWIG spelling
        """Fail as a corrupt chunk would, with nothing to do with the type."""
        raise RuntimeError("simulated decompression failure")


def _grid_beside_a_flag(flag_type):
    """Build a store pairing an EPSG:4326 `t2m(y, x)` with a `flag(y, x)` of `flag_type`.

    Args:
        flag_type: The `gdal.ExtendedDataType` for `flag`, left unwritten.

    Returns:
        gdal.Dataset: The store; keep the reference alive while its arrays are used.
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    store = _mem_store()
    group = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    rows = group.CreateDimension("y", "HORIZONTAL_Y", None, 4)
    cols = group.CreateDimension("x", "HORIZONTAL_X", None, 5)
    y_axis = group.CreateMDArray("y", [rows], f64)
    y_axis.Write(np.array([40.0, 39.0, 38.0, 37.0]))
    rows.SetIndexingVariable(y_axis)
    x_axis = group.CreateMDArray("x", [cols], f64)
    x_axis.Write(np.array([10.0, 11.0, 12.0, 13.0, 14.0]))
    cols.SetIndexingVariable(x_axis)
    grid = group.CreateMDArray("t2m", [rows, cols], f64)
    grid.Write(np.arange(20, dtype="float64").reshape(4, 5))
    grid.SetSpatialRef(srs)
    group.CreateMDArray("flag", [rows, cols], flag_type)
    return store


def _grid_with_an_auxiliary(aux_type, aux_values=None, unit=None):
    """Build a store pairing an EPSG:4326 `t2m(y, x)` with a 1-D auxiliary `aux(k)`.

    Args:
        aux_type: The `gdal.ExtendedDataType` for `aux`.
        aux_values: Values to write into `aux`, or None to leave it unwritten -- an
            unwritten string array reads back as `None` in every entry, which is what a
            NULL looks like to the carry.
        unit: A unit to declare on `aux`, or None.

    Returns:
        gdal.Dataset: The store; keep the reference alive while its arrays are used.
    """
    store = _grid_beside_a_flag(gdal.ExtendedDataType.Create(gdal.GDT_Float64))
    group = store.GetRootGroup()
    group.DeleteMDArray("flag")
    aux = group.CreateMDArray(
        "aux", [group.CreateDimension("k", None, None, 2)], aux_type
    )
    if aux_values is not None:
        aux.Write(aux_values)
    if unit is not None:
        aux.SetUnit(unit)
    return store


class TestARasterOnlyOperationNamesTheVariableItRefuses:
    """The convenience methods that take a variable name must say why they cannot serve it."""

    @pytest.fixture(scope="function")
    def container(self):
        """A container whose only variable is a 1-D axis, so every op below meets one.

        Yields:
            Container: The open store; closed on teardown.
        """
        store, _ = _series_store(np.array([1.0, 2.0, 3.0]))
        container = Container(store)
        yield container
        container.close()

    @pytest.mark.parametrize(
        "method, arguments",
        [
            ("crop_variable", ("series", None)),
            ("reproject_variable", ("series", 3857)),
            ("resample_variable", ("series", 0.5)),
        ],
        ids=["crop", "reproject", "resample"],
    )
    def test_a_variable_named_operation_refuses_with_the_dimensions(
        self, container, method, arguments
    ):
        """Each `*_variable` convenience method refuses, naming the variable and its axes.

        Args:
            container: Fixture holding a store whose only variable is 1-D.
            method: The convenience method under test.
            arguments: Its positional arguments, the variable name first.

        Test scenario:
            All three are `get_variable` followed by a geometry operation, so before the
            guard each died inside GDAL's own object with `AttributeError: 'MDArray' object
            has no attribute 'crop'` -- a message naming neither the variable nor the reason.
            Expected: a `ValueError` that names `series`, prints the dimensions that make it
            non-raster, and points at the accessor that does work.
        """
        with pytest.raises(ValueError) as excinfo:
            getattr(container, method)(*arguments)

        message = str(excinfo.value)
        assert "series" in message, f"the refusal must name the variable: {message}"
        assert "'n'" in message, f"the refusal must print the dimensions: {message}"
        assert "get_variable" in message, (
            f"the refusal must point at the accessor that works: {message}"
        )

    def test_plotting_a_named_non_raster_variable_is_refused_the_same_way(
        self, container
    ):
        """`plot(variable=...)` goes through the same guard, one module away.

        Args:
            container: Fixture holding a store whose only variable is 1-D.

        Test scenario:
            The plot engine resolves the variable itself rather than reusing the container's
            result, so a guard placed only in `netcdf.py` would leave this route uncovered.
            Expected: the same refusal, raised before any rendering is attempted -- so it does
            not depend on the plotting extra being installed.
        """
        with pytest.raises(ValueError) as excinfo:
            container.plot(variable="series")

        message = str(excinfo.value)
        assert "no raster plane" in message, (
            f"the plot route must give the same refusal: {message}"
        )
        assert "series" in message, f"the refusal must name the variable: {message}"

    def test_a_raster_variable_is_handed_through_unchanged(self):
        """The guard must not stand between a caller and a variable that does have a plane.

        Test scenario:
            A refusal that fired for everything would satisfy the tests above, so the same
            container is asked for a gridded variable. Expected: the `NetCDF` subset itself,
            identical to what `get_variable` answers.
        """
        store = _mem_store()
        group = store.GetRootGroup()
        lat = group.CreateDimension("lat", "latitude", "Y", 2)
        lon = group.CreateDimension("lon", "longitude", "X", 3)
        grid = group.CreateMDArray(
            "t2m", [lat, lon], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        grid.Write(np.arange(6, dtype="float64").reshape(2, 3))
        container = Container(store)
        try:
            required = container._require_raster_variable("t2m")

            assert not isinstance(required, LabeledArray), (
                f"a gridded variable must not be refused, got {type(required).__name__}"
            )
            assert required.shape == container.get_variable("t2m").shape, (
                "the guard must hand back exactly what `get_variable` answers"
            )
        finally:
            container.close()

    def test_the_refusal_is_decided_without_reading_the_array(
        self, container, array_reads
    ):
        """A refusal must cost nothing: the declaration says "no raster plane" before any read.

        Args:
            container: Fixture holding a store whose only variable is 1-D.
            array_reads: Fixture recording every array read during the test.

        Test scenario:
            The guard used to ask `get_variable`, which materialises the whole array, only to
            discard the values and refuse -- a full read of a long series spent on an error.
            It now reads the rank and dtype class off the declaration. Both routes give the
            same refusal, so only the reads tell them apart. Expected: the refusal, with
            `series` never read.
        """
        with pytest.raises(ValueError, match="no raster plane"):
            container.crop_variable("series", None)

        assert "series" not in array_reads, (
            f"the refusal read the array it refused: {array_reads}"
        )

    def test_the_streaming_fan_out_refuses_by_name_too(self, container, tmp_path):
        """The streaming path guards the variables it is handed, not only the classifier.

        Args:
            container: Fixture holding a store whose only variable is 1-D.
            tmp_path: Where the streamed file would have been written.

        Test scenario:
            `_stream_apply_to_file` was the one fan-out site still calling `get_variable`
            directly. The classifier upstream now keeps a non-raster variable out of
            `spatial_vars`, so one is handed in directly here -- the case the local guard
            exists for. Without it the stream failed on `_band_dim_names` with an
            `AttributeError` naming neither the variable nor the reason. Expected: the named
            refusal, and no file left behind.
        """
        destination = tmp_path / "streamed.nc"

        with pytest.raises(ValueError, match="series has no raster plane"):
            container._stream_apply_to_file(
                "crop", {"mask": None}, ["series"], [], destination
            )

        assert not destination.exists(), "a refused stream must not write a file"

    def test_a_store_with_no_multidimensional_group_is_served_by_get_variable(self):
        """With no declaration to consult, the guard falls back to what `get_variable` answers.

        Test scenario:
            A store opened classically (`NETCDF:file:var`) has no multidimensional group, so
            there is no array declaration to classify -- and every variable on that path is a
            raster. The declaration check must step aside rather than dereference an array it
            could not open. Expected: the raster subset, with the shape `get_variable` gives
            the same name.
        """
        store = NetCDF.read_file(
            str(CLASSIC_SAMPLE), read_only=True, open_as_multi_dimensional=False
        )
        try:
            assert store._working_group() is None, "fixture changed: no classic store"
            name = store.variable_names[0]

            required = store._require_raster_variable(name)

            assert isinstance(required, NetCDF), (
                f"a classic variable must be handed through, got {type(required).__name__}"
            )
            assert required.shape == store.get_variable(name).shape, (
                "the guard must hand back exactly what `get_variable` answers"
            )
        finally:
            store.close()


class TestReadArrayOnAVariableWithNoRasterPlane:
    """`read_array` reuses the values `get_variable` already materialised; unpacking still works."""

    @pytest.fixture(scope="function")
    def packed(self):
        """A container holding a 1-D `Int16` series with a CF scale and offset.

        Yields:
            Container: The open store; closed on teardown.
        """
        store, array = _series_store(
            np.array([10, 20, 30], dtype="int16"),
            gdal.ExtendedDataType.Create(gdal.GDT_Int16),
        )
        array.SetScale(0.5)
        array.SetOffset(100.0)
        container = Container(store)
        yield container
        container.close()

    def test_unpack_applies_the_arrays_own_scale_and_offset(self, packed):
        """A non-raster variable has no band to carry the packing, so the array's own is used.

        Args:
            packed: Fixture holding a scaled 1-D `Int16` series.

        Test scenario:
            `[10, 20, 30]` stored with `scale_factor=0.5` and `add_offset=100.0`. The values now
            come from the already-materialised wrapper rather than a second read, so the
            unpacking has to be re-applied to them; a reuse that dropped it would answer with
            the raw integers and look like a plain successful read. Expected: `[105.0, 110.0,
            115.0]`, and the storage integers when unpacking is not asked for.
        """
        unpacked = packed.read_array(variable="series", unpack=True)
        stored = packed.read_array(variable="series", unpack=False)

        assert np.array_equal(unpacked, [105.0, 110.0, 115.0]), (
            f"scale/offset must be applied, got {unpacked}"
        )
        assert np.array_equal(stored, [10, 20, 30]), (
            f"the default read must stay in storage units, got {stored}"
        )

    def test_a_second_read_is_not_scaled_twice(self, packed):
        """Reusing the resolved values must not mean unpacking them in place.

        Args:
            packed: Fixture holding a scaled 1-D `Int16` series.

        Test scenario:
            The read reuses the values it resolved rather than going back to GDAL, so an
            in-place unpack would be a real risk if those values were ever shared. They are
            not on this path -- `read_array` builds a fresh wrapper each call and never touches
            the `variables` cache -- so this pins repeatability, not aliasing; the shared object
            is covered by the next test. Expected: the second unpacked read equals the first,
            and a plain read after it still answers the storage integers.
        """
        first = packed.read_array(variable="series", unpack=True)
        second = packed.read_array(variable="series", unpack=True)

        assert np.array_equal(first, second), (
            f"repeating the read must repeat the answer, got {first} then {second}"
        )
        assert np.array_equal(packed.read_array(variable="series"), [10, 20, 30]), (
            "an unpacked read must not rewrite the stored values"
        )

    def test_the_cached_wrapper_is_shared_so_it_cannot_be_written(self, packed):
        """`variables` hands every caller the same array, so it must not be mutable.

        Args:
            packed: Fixture holding a scaled 1-D `Int16` series.

        Test scenario:
            `variables` caches per key, so `variables["series"]` is one object for every
            caller -- unlike `get_variable`, which builds a fresh one each time. Left
            writeable, a single `values += 1` made the mapping answer `[11, 21, 31]` while
            `get_variable` and `read_array` still said `[10, 20, 30]`. Expected: the same
            object on repeat access, a mutation refused at the point it is attempted, and all
            three accessors still agreeing afterwards.
        """
        first = packed.variables["series"]
        second = packed.variables["series"]

        assert first is second, "the mapping must cache the wrapper, not rebuild it"
        assert not first.values.flags.writeable, "the shared array must be read-only"
        with pytest.raises(ValueError, match="read-only"):
            first.values += 1

        cached = packed.variables["series"].values.tolist()
        fresh = packed.get_variable("series").values.tolist()
        read = np.asarray(packed.read_array(variable="series")).tolist()
        assert cached == fresh == read == [10, 20, 30], (cached, fresh, read)

    def test_a_fresh_wrapper_stays_writeable(self, packed):
        """Only the shared object is locked; a caller's own copy is theirs to change.

        Args:
            packed: Fixture holding a scaled 1-D `Int16` series.

        Test scenario:
            `get_variable` returns a new wrapper on every call, so nothing else holds its
            array and there is no reason to refuse a write to it.
        """
        fresh = packed.get_variable("series")

        assert fresh.values.flags.writeable


class TestTheLabelsTheWrapperCarries:
    """`values` alone cannot say whether a `-9999.0` is data; the labels around it can."""

    def test_it_copies_the_name_unit_fill_value_and_attributes(self):
        """Everything the raw `MDArray` used to expose is carried onto the wrapper.

        Test scenario:
            An array declaring a unit, a fill value and an attribute. These were reachable on
            the handle that used to come back (`GetUnit`, `GetNoDataValueAsDouble`,
            `GetAttributes`), so dropping them would have made the wrapper a regression for a
            caller who read them. Expected: all four present, with `values` left unmasked --
            the fill value is reported, not applied.
        """
        store, array = _series_store(np.array([1.0, -9999.0, 3.0]))
        array.SetUnit("K")
        array.SetNoDataValueDouble(-9999.0)
        attribute = array.CreateAttribute(
            "long_name", [], gdal.ExtendedDataType.CreateString()
        )
        attribute.Write("a series")
        container = Container(store)
        try:
            variable = container.get_variable("series")

            assert variable.name == "series", f"unexpected name {variable.name!r}"
            assert variable.unit == "K", f"unexpected unit {variable.unit!r}"
            assert variable.no_data_value == -9999.0, (
                f"unexpected fill value {variable.no_data_value!r}"
            )
            assert variable.attributes == {"long_name": "a series"}, (
                f"unexpected attributes {variable.attributes!r}"
            )
            assert np.array_equal(variable.values, [1.0, -9999.0, 3.0]), (
                f"the fill value must be reported, not applied: {variable.values}"
            )
        finally:
            container.close()

    def test_a_packed_array_carries_the_scale_and_offset_it_needs(self):
        """A unit without its packing parameters labels the wrong numbers.

        Test scenario:
            GDAL lifts `scale_factor` / `add_offset` out of the attribute list, so they
            never reach `attributes`. Before they were carried, a packed series came back
            as `[10, 15]` labelled `K` with nothing on the object to say the true values
            were 105 and 110. `values` stays packed -- `read_array`'s `unpack=False`
            default -- so the wrapper has to carry what unpacking needs.
        """
        store, array = _series_store(np.array([10.0, 15.0]))
        array.SetUnit("K")
        array.SetScale(1.0)
        array.SetOffset(95.0)
        container = Container(store)
        try:
            series = container.get_variable("series")

            assert series.values.tolist() == [10.0, 15.0], "values must stay packed"
            assert (series.scale, series.offset) == (1.0, 95.0)
            unpacked = series.values * series.scale + series.offset
            assert unpacked.tolist() == [105.0, 110.0]
        finally:
            container.close()

    def test_an_array_that_declares_none_of_them_gets_the_documented_defaults(self):
        """A bare array must answer empty labels rather than GDAL's placeholders.

        Test scenario:
            An array with no unit, no fill value and no attributes. Expected: `unit` an empty
            string, `no_data_value` `None` -- not the `0.0` a numeric default would make
            indistinguishable from a declared zero fill -- and `attributes` an empty dict.
        """
        store, _ = _series_store(np.array([1.0, 2.0, 3.0]))
        container = Container(store)
        try:
            variable = container.get_variable("series")

            assert variable.unit == "", f"unexpected unit {variable.unit!r}"
            assert variable.no_data_value is None, (
                f"an undeclared fill value must stay None, got {variable.no_data_value!r}"
            )
            assert variable.attributes == {}, (
                f"unexpected attributes {variable.attributes!r}"
            )
        finally:
            container.close()

    def test_an_unpacked_array_reports_no_scale_or_offset(self):
        """No packing declared must read as `None`, not as the identity transform.

        Test scenario:
            `scale` / `offset` are there so a caller can tell packed values from plain ones.
            Defaulting them to `1.0` / `0.0` would keep `values * scale + offset` computable
            but make "is this packed?" unanswerable -- the reason `no_data_value` stays `None`
            rather than `0.0`. Expected: both `None` on an array that declares neither.
        """
        store, _ = _series_store(np.array([1.0, 2.0, 3.0]))
        container = Container(store)
        try:
            variable = container.get_variable("series")

            assert (variable.scale, variable.offset) == (None, None), (
                f"an unpacked array must report no packing, got "
                f"{(variable.scale, variable.offset)}"
            )
        finally:
            container.close()

    @pytest.mark.parametrize(
        "gdal_type, numpy_type, setter, fill",
        [
            (gdal.GDT_Int64, "int64", "SetNoDataValueInt64", INT64_FILL),
            (gdal.GDT_UInt64, "uint64", "SetNoDataValueUInt64", UINT64_FILL),
        ],
        ids=["int64", "uint64"],
    )
    def test_a_64_bit_integer_fill_value_keeps_every_digit(
        self, gdal_type, numpy_type, setter, fill
    ):
        """A 64-bit fill read through a double matches its neighbour as well as itself.

        Args:
            gdal_type: The 64-bit integer type the series is declared with.
            numpy_type: The matching NumPy dtype for the written values.
            setter: The `MDArray` method that declares a fill of that type.
            fill: A fill a double's 53-bit mantissa cannot hold exactly.

        Test scenario:
            `GetNoDataValueAsDouble` rounds `-9223372036854775806` to
            `-9.223372036854776e+18`, and the UInt64 fill up to `2**64`. numpy compares a
            64-bit integer array against that double by converting the integers, which
            loses the same precision, so the rounded fill equals both the fill and the value
            next to it -- masking by it would hide a real datum. Expected: the fill as an
            exact `int` that selects the one element carrying it and not its neighbour.
        """
        # The value next to the fill is the one a double cannot tell apart from it.
        neighbour = fill + 1 if fill < 0 else fill - 1
        store, array = _series_store(
            np.array([1, fill, neighbour, 3], dtype=numpy_type),
            gdal.ExtendedDataType.Create(gdal_type),
        )
        getattr(array, setter)(fill)
        container = Container(store)
        try:
            variable = container.get_variable("series")

            assert isinstance(variable.no_data_value, int), (
                f"a 64-bit fill must stay an int, got {variable.no_data_value!r}"
            )
            assert variable.no_data_value == fill, (
                f"expected the fill {fill}, got {variable.no_data_value!r}"
            )
            matches = (variable.values == variable.no_data_value).tolist()
            assert matches == [False, True, False, False], (
                f"the fill must pick out exactly the element that carries it: {matches}"
            )
        finally:
            container.close()

    @pytest.mark.parametrize(
        "gdal_type", [gdal.GDT_Int64, gdal.GDT_UInt64], ids=["int64", "uint64"]
    )
    def test_a_64_bit_array_with_no_fill_value_answers_none(self, gdal_type):
        """The 64-bit accessors must say "none declared" as `None`, as the double one does.

        Args:
            gdal_type: The 64-bit integer type the series is declared with.

        Test scenario:
            The defaults test above reaches only the double accessor; these two types read
            their fill through accessors of their own. `0` is a legal integer fill, so an
            absent one coming back as a number would read as declared, and one that failed
            on the absence would make every such variable unreadable. Expected: `None`, with
            the values intact.
        """
        store, _ = _series_store(
            np.array([1, 2, 3], dtype="int64"), gdal.ExtendedDataType.Create(gdal_type)
        )
        container = Container(store)
        try:
            variable = container.get_variable("series")

            assert variable.no_data_value is None, (
                f"an undeclared fill must stay None, got {variable.no_data_value!r}"
            )
            assert variable.values.tolist() == [1, 2, 3], (
                f"unexpected values {variable.values}"
            )
        finally:
            container.close()

    def test_each_wrapper_gets_its_own_attribute_dict(self):
        """The default must not be a dict shared between instances.

        Test scenario:
            `attributes` defaults to a fresh mapping, so writing into one wrapper's must not
            appear on the next. Expected: the second wrapper still empty after the first is
            written to.
        """
        first = LabeledArray(np.zeros(2), ("n",), (2,))
        first.attributes["written"] = 1
        second = LabeledArray(np.zeros(2), ("n",), (2,))

        assert second.attributes == {}, (
            f"the default dict must not be shared, got {second.attributes!r}"
        )

    @pytest.mark.parametrize(
        "name, expected",
        [("series", "LabeledArray('series', dims=('n',), shape=(3,))"), ("", None)],
        ids=["named", "unnamed"],
    )
    def test_repr_names_the_variable_only_when_it_has_one(self, name, expected):
        """The name earns its place in the repr; an empty one must not leave a stray comma.

        Args:
            name: The name to construct with.
            expected: The exact repr, or None when only the absence is asserted.

        Test scenario:
            `LabeledDataset["var"]` constructs without a name, so both spellings are live.
            Expected: the named form quotes it ahead of `dims`; the unnamed one carries neither
            the name nor its separator.
        """
        array = LabeledArray(np.zeros(3), ("n",), (3,), name=name)

        if expected is not None:
            assert repr(array) == expected, f"unexpected repr {array!r}"
        else:
            assert repr(array) == "LabeledArray(dims=('n',), shape=(3,))", (
                f"an unnamed array must not carry a separator: {array!r}"
            )


class TestTheDtypeOfAVariableWithNoRecords:
    """`_numpy_dtype_of` / `_compound_dtype` describe an array GDAL will not let anyone read.

    An unlimited dimension with no records written declines the read outright (`count[0] = 0 is
    invalid`), so the empty array is built from the declared type. MEM refuses to create such a
    dimension at all -- `RuntimeError: Illegal dimension size 0` -- which leaves the two helpers
    reachable end-to-end only for the numeric case a sibling test already covers. They are
    asserted here on the type itself.
    """

    @pytest.mark.parametrize(
        "gdal_type, expected",
        [
            (gdal.GDT_Float64, np.float64),
            (gdal.GDT_Int16, np.int16),
            (gdal.GDT_Byte, np.uint8),
        ],
        ids=["float64", "int16", "byte"],
    )
    def test_a_numeric_type_maps_to_its_numpy_dtype(self, gdal_type, expected):
        """The empty array must have the dtype the read would have produced.

        Args:
            gdal_type: The declared GDAL numeric type.
            expected: The NumPy dtype it must map to.

        Test scenario:
            Answering `float64` for everything would go unnoticed on a float variable and
            silently widen an integer one, so signed, unsigned and floating types are each
            checked.
        """
        dtype = _numpy_dtype_of(gdal.ExtendedDataType.Create(gdal_type))

        assert dtype == np.dtype(expected), f"expected {expected}, got {dtype}"

    def test_a_string_type_maps_to_object(self):
        """An empty string variable gets the dtype a populated one does.

        Test scenario:
            A non-empty string array is read as `object`, because `Read` returns a missing
            entry as `None` and a `<U` array cannot hold one. Letting NumPy pick made the
            dtype depend on the data -- `<U` for a fully written column, `object` the moment
            one entry was missing. Expected: `object` for the empty case too, so
            `values.dtype.kind` is the same answer whether or not any record was written.
        """
        dtype = _numpy_dtype_of(gdal.ExtendedDataType.CreateString())

        assert dtype == np.dtype(object), f"expected object, got {dtype!r}"

    def test_a_compound_type_keeps_the_declared_offsets_and_record_size(self):
        """The record layout is read from GDAL, not inferred from the field formats.

        Test scenario:
            A record padded between its fields and after the last one. Both paddings are
            load-bearing and each is lost by a different shortcut: packing the formats puts the
            fields at 0/4/12, and inferring the stride from the offsets gives 20 rather than the
            declared 24. Either slides every record after the first against the buffer.
            Expected: the declared offsets and the declared record size, and a layout that
            differs from the naively packed one -- otherwise the assertion proves nothing.
        """
        dtype = _compound_dtype(_padded_record())

        assert dtype.names == ("a", "b", "c"), f"unexpected fields {dtype.names}"
        assert dtype.itemsize == PADDED_RECORD_SIZE, (
            f"the record stride must be the declared {PADDED_RECORD_SIZE}, got {dtype.itemsize}"
        )
        assert [dtype.fields[name][1] for name in dtype.names] == PADDED_OFFSETS, (
            f"the fields must sit at their declared offsets {PADDED_OFFSETS}, got "
            f"{[dtype.fields[name][1] for name in dtype.names]}"
        )
        packed = np.dtype([("a", "<i4"), ("b", "<f8"), ("c", "<i4")])
        assert dtype.itemsize != packed.itemsize, (
            "the fixture must actually be padded, or this asserts nothing"
        )

    def test_the_compound_branch_is_reached_through_the_dispatcher(self):
        """`_numpy_dtype_of` must route a compound type to `_compound_dtype`, not to a number.

        Test scenario:
            The dispatcher is what the empty-extent path calls; a compound falling through to
            its numeric branch would raise on `GetNumericDataType()` rather than describing the
            record. Expected: the structured dtype, identical to the direct call.
        """
        record = _padded_record()

        assert _numpy_dtype_of(record) == _compound_dtype(record), (
            "the dispatcher must hand a compound type to the compound builder"
        )

    def test_a_component_that_is_not_itself_numeric_is_refused(self):
        """A nested record has no numeric type code, so no NumPy field can be built for it.

        Test scenario:
            A compound whose first component is itself a compound. `GetNumericDataType()`
            reports `GDT_Unknown` for it, which maps to no NumPy type. Expected: a `ValueError`
            rather than a silently wrong field -- the message is GDAL's type code, which
            `_labeled_array_from_md_array` re-raises with the variable's name attached.
        """
        nested = gdal.ExtendedDataType.CreateCompound(
            "outer",
            32,
            [
                gdal.EDTComponent.Create("inner", 0, _padded_record()),
                gdal.EDTComponent.Create(
                    "c", 24, gdal.ExtendedDataType.Create(gdal.GDT_Int32)
                ),
            ],
        )

        with pytest.raises(ValueError) as excinfo:
            _compound_dtype(nested)

        assert "not supported" in str(excinfo.value), (
            f"unexpected refusal: {excinfo.value}"
        )


class TestAReadThatDisagreesWithTheDeclaredDimensions:
    """The two shapes come from different places, so they can disagree.

    `shape` is what the dimensions declare and `values.shape` is what the read returned. No real
    GDAL array lets a test drive the two apart, so a stub stands in for the handle.
    """

    def test_a_mis_shaped_read_is_refused_rather_than_wrapped(self):
        """A wrapper whose `shape` and `values.shape` disagree would mislead every later caller.

        Test scenario:
            An array declaring `(record: 3, level: 2)` whose read answers five values. Without
            the check the wrapper reports a shape its own values do not have, and the caller
            finds out somewhere far from the read. Expected: a `ValueError` naming the variable,
            the declared dimensions and both shapes.
        """
        stub = _StubMDArray(
            [("record", 3), ("level", 2)], np.arange(5, dtype="float64")
        )

        with pytest.raises(ValueError) as excinfo:
            _labeled_array_from_md_array(stub, "obs")

        message = str(excinfo.value)
        assert "obs" in message, f"the refusal must name the variable: {message}"
        assert "(3, 2)" in message, (
            f"the refusal must print the declared shape: {message}"
        )
        assert "(5,)" in message, f"the refusal must print the read shape: {message}"

    def test_a_matching_read_is_wrapped(self):
        """The check must not stand in the way of a read that agrees with the dimensions.

        Test scenario:
            The same declared dimensions, read back at `(3, 2)`. Expected: the wrapper, with
            the dimension names and the shape carried through -- a guard that refused both
            shapes would pass the test above and break every read.
        """
        stub = _StubMDArray(
            [("record", 3), ("level", 2)], np.arange(6, dtype="float64").reshape(3, 2)
        )

        wrapped = _labeled_array_from_md_array(stub, "obs")

        assert wrapped.dims == ("record", "level"), f"unexpected dims {wrapped.dims}"
        assert wrapped.shape == (3, 2), f"unexpected shape {wrapped.shape}"

    def test_an_unreadable_attribute_does_not_cost_the_caller_the_values(self):
        """One attribute GDAL cannot decode must not turn a good read into an exception.

        Test scenario:
            Two attributes, the second raising `RuntimeError` from `Read()` -- what GDAL does
            for a type it has no Python mapping for. Expected: the values and the readable
            attribute come back, and the unreadable one is simply absent.
        """
        stub = _StubMDArray(
            [("n", 3)],
            np.arange(3, dtype="float64"),
            attributes=[
                _StubAttribute("good", value="kept"),
                _StubAttribute("bad", error=RuntimeError("cannot read")),
            ],
        )

        wrapped = _labeled_array_from_md_array(stub, "obs")

        assert wrapped.attributes == {"good": "kept"}, (
            f"the readable attribute must survive alone, got {wrapped.attributes!r}"
        )
        assert np.array_equal(wrapped.values, [0.0, 1.0, 2.0]), (
            f"the values must not be lost with the attribute, got {wrapped.values}"
        )


class TestACompoundRecordWithAStringField:
    """A record type GDAL's Python bindings cannot read, through either of their readers."""

    def test_get_variable_refuses_it_by_name(self):
        """The refusal names the variable and the cause, not just GDAL's bare message.

        Test scenario:
            `ReadAsArray` raises "String buffer data type not supported in SWIG bindings" and
            `Read` "non-numeric buffer data type not supported", so there is nothing to
            materialise. Passed through, that message names neither the variable nor why a
            name the store lists will not read. Expected: a `ValueError` naming `stations`
            and saying it cannot be read, with GDAL's own error kept as the cause.
        """
        container = Container(_one_array_store("stations", (2,), "record_with_text"))
        try:
            with pytest.raises(ValueError) as excinfo:
                container.get_variable("stations")
        finally:
            container.close()

        message = str(excinfo.value)
        assert "stations" in message, f"the refusal must name the variable: {message}"
        assert "cannot be read" in message, f"the refusal must say why: {message}"
        assert isinstance(excinfo.value.__cause__, RuntimeError), (
            f"GDAL's error must stay reachable as the cause: {excinfo.value.__cause__!r}"
        )

    def test_a_raster_only_operation_refuses_it_for_having_no_plane(self):
        """Asked to crop one, the guard answers the question asked: it is not a raster.

        Test scenario:
            `stations(d0, d1)` is 2-D, but a record is not a raster at any rank. Asking
            `get_variable` first -- what the guard used to do -- surfaced the unreadable-type
            error instead, an answer to a question the caller did not ask: the record could
            not be cropped even if it were readable. The declaration answers without a read.
            Expected: the "no raster plane" refusal, printing both dimensions.
        """
        container = Container(_one_array_store("stations", (2, 3), "record_with_text"))
        try:
            with pytest.raises(ValueError) as excinfo:
                container.crop_variable("stations", None)
        finally:
            container.close()

        message = str(excinfo.value)
        assert "no raster plane" in message, f"unexpected refusal: {message}"
        assert "('d0', 'd1')" in message, (
            f"the refusal must print the dimensions: {message}"
        )


class TestTheDeclarationAgreesWithWhatGetVariableReturns:
    """`_has_raster_plane` predicts `get_variable`'s return type without reading; it must not err.

    Three callers act on the prediction rather than on the result: the container CRS walk skips
    on it, the raster guard refuses on it, and the fan-out classifier excludes on it. A wrong
    "no" refuses a variable `get_variable` would have served as a raster; a wrong "yes" hands
    those callers a `LabeledArray` where they expect geometry.
    """

    @pytest.mark.parametrize(
        "sizes, kind, expected",
        [
            ((3,), "float64", False),
            ((2, 3), "float64", True),
            ((2, 2, 3), "float64", True),
            ((3,), "string", False),
            ((2, 3), "string", False),
            ((2, 3), "record", False),
        ],
        ids=["1d", "2d", "3d", "1d-string", "2d-string", "2d-record"],
    )
    def test_the_prediction_matches_the_returned_type(self, sizes, kind, expected):
        """Rank and dtype class decide it, and both have to be consulted.

        Args:
            sizes: The array's dimension sizes.
            kind: Its data type, as `_data_type` names it.
            expected: Whether it has a raster plane.

        Test scenario:
            A 1-D numeric array fails on rank alone and a 2-D string one on dtype alone, so a
            predicate checking only one of the two gets exactly one of those rows wrong.
            Expected: the module-level predicate, the method the callers use, and the type
            `get_variable` actually returns all give the same answer.
        """
        store = _one_array_store("v", sizes, kind)
        container = Container(store)
        try:
            predicted = _has_raster_plane(store.GetRootGroup().OpenMDArray("v"))
            declared = container._declares_raster_plane("v")
            returned = container.get_variable("v")

            assert predicted is expected, f"_has_raster_plane said {predicted}"
            assert declared is expected, f"_declares_raster_plane said {declared}"
            assert isinstance(returned, NetCDF) is expected, (
                f"get_variable returned a {type(returned).__name__}"
            )
        finally:
            container.close()

    @pytest.mark.parametrize(
        "sizes, expected", [((3,), False), ((2, 3), True)], ids=["1d", "2d"]
    )
    def test_a_group_qualified_name_is_classified_in_its_own_group(
        self, sizes, expected
    ):
        """The CRS walk meets `forecast/v` names and must classify the array they point at.

        Args:
            sizes: The array's dimension sizes.
            expected: Whether it has a raster plane.

        Test scenario:
            `variable_names` qualifies a nested variable with its group path, and the CRS walk
            classifies every name it lists. The root group holds no array called
            `forecast/v` -- GDAL answers `Array forecast/v does not exist` -- so the name has
            to be walked down to its group before the declaration can be read. Expected: the
            same answer `get_variable` gives through the group, for a series and a grid.
        """
        container = Container(_one_array_store("v", sizes, "float64", "forecast"))
        try:
            assert "forecast/v" in container.variable_names, "fixture changed"

            declared = container._declares_raster_plane("forecast/v")
            returned = container.get_variable("forecast/v")

            assert declared is expected, f"_declares_raster_plane said {declared}"
            assert isinstance(returned, NetCDF) is expected, (
                f"get_variable returned a {type(returned).__name__}"
            )
        finally:
            container.close()


class TestTheContainerCrsSkipsAVariableWithNoRasterPlane:
    """A container borrows its CRS from its variables, and one of them may have no geometry."""

    def test_a_leading_non_raster_variable_does_not_hide_a_projected_crs(self):
        """The skip is visible on the public `epsg`, not only on the walk that implements it.

        Test scenario:
            A store enumerating a 1-D `aaa_axis` before a gridded `zzz_grid` that declares
            EPSG:3857. Asking a `LabeledArray` for `.crs` raises, and the walk's `except` wraps
            the whole loop, so without the skip the first variable would suppress the CRS of the
            second. A projected CRS is used deliberately: an empty answer on a geographic store
            is rescued by the CF lat/lon fallback, which would hide the failure. Expected:
            EPSG:3857, the same code the gridded variable itself reports.
        """
        store = _mem_store()
        group = store.GetRootGroup()
        axis_dim = group.CreateDimension("n", None, None, 3)
        axis = group.CreateMDArray(
            "aaa_axis", [axis_dim], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        axis.Write(np.array([1.0, 2.0, 3.0]))
        y = group.CreateDimension("y", "projection_y_coordinate", "Y", 2)
        x = group.CreateDimension("x", "projection_x_coordinate", "X", 3)
        y_values = group.CreateMDArray(
            "y", [y], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        y_values.Write(np.array([200.0, 100.0]))
        x_values = group.CreateMDArray(
            "x", [x], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        x_values.Write(np.array([0.0, 100.0, 200.0]))
        grid = group.CreateMDArray(
            "zzz_grid", [y, x], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        grid.Write(np.arange(6, dtype="float64").reshape(2, 3))
        reference = osr.SpatialReference()
        reference.ImportFromEPSG(3857)
        grid.SetSpatialRef(reference)
        container = Container(store)
        try:
            assert isinstance(
                container.get_variable(container.variable_names[0]), LabeledArray
            ), "the non-raster variable must come first, or this asserts nothing"

            assert container.epsg == 3857, (
                f"the leading non-raster variable suppressed the CRS: {container.epsg}"
            )
        finally:
            container.close()

    def test_the_skipped_variable_is_never_read(self, array_reads):
        """Stepping over a non-raster variable must not cost a read of it.

        Args:
            array_reads: Fixture recording every array read during the test.

        Test scenario:
            The walk used to ask `get_variable` for each name and skip whatever came back as
            a `LabeledArray` -- correct, but `get_variable` materialises, so it read a whole
            series only to learn it had no CRS. It now decides on the declaration. Both find
            the same CRS, so only the reads tell them apart; the recorder is armed before the
            store is even built, so a read at construction would be caught too. Expected:
            EPSG:3857 from the grid, and `aaa_axis` never read.
        """
        container = Container(_axis_before_a_projected_grid())
        try:
            assert container.variable_names[0] == "aaa_axis", "fixture changed"

            assert container.epsg == 3857, f"unexpected EPSG {container.epsg}"
            assert "aaa_axis" not in array_reads, (
                f"the CRS walk read the variable it skips: {array_reads}"
            )
        finally:
            container.close()

    def test_a_store_with_no_multidimensional_group_still_borrows_a_crs(self):
        """A name the declaration check cannot open must count as a raster, not be skipped.

        Test scenario:
            A classically opened store has no multidimensional group, so the check opens none
            of its arrays -- yet every variable on that path is a raster. Reading "could not
            classify" as "no raster plane" skips all of them, and the root of such a store
            carries no projection of its own, so it would report no CRS at all. Expected: the
            WGS 84 its variables declare.
        """
        store = NetCDF.read_file(
            str(CLASSIC_SAMPLE), read_only=True, open_as_multi_dimensional=False
        )
        try:
            assert store._working_group() is None, "fixture changed: no classic store"

            assert store.epsg == 4326, f"unexpected EPSG {store.epsg}"
        finally:
            store.close()


@pytest.mark.core
class TestTheRound2FixesTheirOwnTestPassFound:
    """Four defects the round-2 test pass found in the round-2 fixes themselves."""

    @pytest.mark.parametrize("kind", ["string", "compound"])
    def test_a_flag_the_carry_cannot_write_is_absent_not_empty(self, kind):
        """A variable that cannot be carried is dropped whole, with the true reason.

        Args:
            kind: The non-numeric dtype class of `flag`.

        Test scenario:
            Classifying `flag(y, x)` as auxiliary sends it to the carry, and GDAL's
            Python bindings cannot write a string array of rank >= 2 or any compound.
            The string branch created the array before writing it, so a failed write
            left `flag` listed in the result with every value `None`. And the demotion
            warning told the user to rename axes that already are `y` / `x`. Expected:
            `flag` absent, the "could not carry" warning present, no rename advice.
        """
        flag_type = (
            gdal.ExtendedDataType.CreateString()
            if kind == "string"
            else _data_type("compound")
        )
        container = Container(_grid_beside_a_flag(flag_type))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = container.to_crs(3857)
        messages = [str(warning.message) for warning in caught]

        assert "flag" not in result.variable_names, "a half-built flag must not remain"
        assert "t2m" in result.variable_names, "the raster must still be reprojected"
        assert any("could not carry" in message for message in messages), messages
        assert not any("rename the axes" in message for message in messages), messages

    def test_a_coordinate_name_gets_the_refusal_get_variable_would_give(self):
        """A name `get_variable` rejects is not refused for having no raster plane.

        Test scenario:
            A coordinate array has a declaration, so judging it on that refused `x`
            for having "no raster plane" and pointed at `get_variable` -- which then
            rejected the name outright. Expected: `get_variable`'s own refusal.
        """
        container = Container(_grid_beside_a_flag(gdal.ExtendedDataType.CreateString()))

        with pytest.raises(ValueError, match="not a valid variable name"):
            container._require_raster_variable("x")

    def test_a_read_failure_on_an_ordinary_array_keeps_its_own_message(self):
        """Only a compound read failure is blamed on a string field.

        Test scenario:
            Every `RuntimeError` from `ReadAsArray` used to be reworded as "a compound
            type with a string field is the usual cause", including an I/O or
            decompression failure on a plain float series -- which sent the reader to
            look for a type problem that was not there. Expected: GDAL's error, as-is.
        """
        failing = _FailingStubMDArray([("n", 3)], None)

        with pytest.raises(RuntimeError, match="simulated decompression failure"):
            _labeled_array_from_md_array(failing, "series")

    @pytest.mark.parametrize(
        "change",
        [
            lambda entry: setattr(entry, "unit", "m"),
            lambda entry: entry.attributes.__setitem__("added", 1),
        ],
        ids=["unit", "attributes"],
    )
    def test_the_cached_entry_is_frozen_whole_not_just_its_array(self, change):
        """Locking only `values` let the labels drift out of step the same way.

        Args:
            change: A mutation of one of the entry's labels.

        Test scenario:
            The cache hands one object to every caller. With only its array locked,
            `.unit = "m"` or a new attribute stuck in the cache while `get_variable`
            still said `K` and `{}` -- the disagreement the lock was added to prevent.
            Expected: the change refused, and the cache still agreeing with a fresh read.
        """
        store, array = _series_store(np.array([10.0, 20.0, 30.0]))
        array.SetUnit("K")
        container = Container(store)
        try:
            cached = container.variables["series"]

            with pytest.raises((AttributeError, TypeError)):
                change(cached)

            fresh = container.get_variable("series")
            assert (cached.unit, dict(cached.attributes)) == (
                fresh.unit,
                fresh.attributes,
            )
        finally:
            container.close()

    def test_a_copy_of_the_cached_entry_is_independent_and_mutable(self):
        """`.copy()` is the documented way out of the frozen entry.

        Test scenario:
            A caller who needs to modify the values takes a copy. It must be writeable
            and must not reach back into the cache.
        """
        store, array = _series_store(np.array([10.0, 20.0, 30.0]))
        array.SetUnit("K")
        container = Container(store)
        try:
            own = container.variables["series"].copy()
            own.unit = "m"
            own.values += 1

            cached = container.variables["series"]
            assert (own.unit, own.values.tolist()) == ("m", [11.0, 21.0, 31.0])
            assert (cached.unit, cached.values.tolist()) == ("K", [10.0, 20.0, 30.0])
        finally:
            container.close()


@pytest.mark.core
class TestTheCarryAndTheLockTheDocstringPassFound:
    """Four defects the round-2 docstring pass found; the first two predate this branch."""

    def test_a_string_auxiliary_with_a_null_entry_is_dropped_not_fatal(self):
        """One NULL string entry must not fail the whole operation.

        Test scenario:
            The carry reads a string auxiliary with `Read`, which returns a NULL entry as
            `None`, and writing that list raises `TypeError: sequence must contain
            strings`. The carry caught only `RuntimeError` and `ValueError`, so a single
            such auxiliary failed the whole `to_crs` -- on `main` too. Expected: the
            raster reprojected, the auxiliary dropped with a warning naming it.
        """
        container = Container(
            _grid_with_an_auxiliary(gdal.ExtendedDataType.CreateString())
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = container.to_crs(3857)

        assert "t2m" in result.variable_names
        assert "aux" not in result.variable_names
        assert any("could not carry" in str(warning.message) for warning in caught)

    def test_a_compound_auxiliary_does_not_fail_a_streamed_write(self, tmp_path):
        """The streamed arm must decline a compound auxiliary, as the in-memory one drops it.

        Args:
            tmp_path: Destination for the streamed output.

        Test scenario:
            `_stream_feasible` screened out string auxiliaries but not compound ones, and
            the streamed write maps each through `numpy_to_gdal_dtype`, which has no entry
            for a structured dtype. So `to_crs(path=...)` failed with "numpy data type is
            not supported" while the same call in memory succeeded -- on `main` too.
            Expected: the streamed call succeeds.
        """
        container = Container(_grid_with_an_auxiliary(_data_type("compound")))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            container.to_crs(3857, path=str(tmp_path / "out.nc"))

        assert (tmp_path / "out.nc").exists()

    @pytest.mark.parametrize(
        "attempt",
        [
            lambda entry: entry.values.setflags(write=True),
            lambda entry: delattr(entry, "unit"),
            lambda entry: entry.attributes["flags"].append("c"),
        ],
        ids=["reopen-the-array", "delete-a-label", "mutate-a-list-attribute"],
    )
    def test_nothing_reopens_a_frozen_entry(self, attempt):
        """Every route around the lock is closed, not just assignment.

        Args:
            attempt: A way of changing the cached entry without assigning to it.

        Test scenario:
            numpy lets anyone call `setflags(write=True)` on an array that owns its data,
            which re-opened the cache; `__setattr__` alone left `del entry.unit` free; and
            a multi-valued attribute came back as a list the read-only mapping around it
            did not protect. Each let the cache disagree with `get_variable` again.
            Expected: each refused, and the cache still matching a fresh read.
        """
        store, array = _series_store(np.array([10.0, 20.0, 30.0]))
        array.SetUnit("K")
        flags = array.CreateAttribute(
            "flags", [2], gdal.ExtendedDataType.CreateString()
        )
        flags.Write(["a", "b"])
        container = Container(store)
        try:
            cached = container.variables["series"]

            with pytest.raises((ValueError, AttributeError, TypeError)):
                attempt(cached)

            fresh = container.get_variable("series")
            assert cached.values.tolist() == fresh.values.tolist()
            assert cached.unit == fresh.unit
        finally:
            container.close()

    def test_a_string_auxiliary_keeps_its_unit_through_the_carry(self):
        """The string branch of the carry keeps the labels the numeric branch does.

        Test scenario:
            The numeric branch sets unit and spatial reference on the copy; the string
            branch set neither, so a string auxiliary came out of a crop with a unit of
            '' where it went in with '1'. Expected: the unit carried.
        """
        container = Container(
            _grid_with_an_auxiliary(
                gdal.ExtendedDataType.CreateString(), aux_values=["ok", "ok"], unit="1"
            )
        )

        result = container.to_crs(3857)

        assert result._working_group().OpenMDArray("aux").GetUnit() == "1"
