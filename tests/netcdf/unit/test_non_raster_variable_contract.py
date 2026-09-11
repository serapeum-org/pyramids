"""What a variable with no raster plane promises once it is a `LabeledArray` (#1126).

The sibling suites assert the *type* that comes back from `get_variable` and the values it
carries. Three things they do not reach are asserted here.

*The refusal.* `_require_raster_variable` is the door every internal caller that needs geometry
now goes through. It is exercised from the public methods that let a user name the variable --
`crop_variable`, `reproject_variable`, `resample_variable` and `plot(variable=)` -- because a
refusal is only useful if it survives the call the user actually makes.

*The labels.* `LabeledArray` grew `name` / `unit` / `no_data_value` / `attributes`. Without them a
`-9999.0` sitting in `values` is indistinguishable from real data, so what the wrapper copies off
the array -- and what it defaults to when the array declares nothing -- is the difference between
the wrapper being a better answer than the handle it replaced and merely a safer one.

*The empty-extent dtype.* `_numpy_dtype_of` / `_compound_dtype` describe a variable GDAL declines
to read at all (`count[0] = 0 is invalid`). No driver here will build such a store for a compound
or string type -- MEM refuses a zero-length dimension outright -- so the two helpers are asserted
directly, on the `ExtendedDataType` they take.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.netcdf import LabeledArray
from pyramids.netcdf.netcdf import (
    Container,
    _compound_dtype,
    _labeled_array_from_md_array,
    _numpy_dtype_of,
)

pytestmark = pytest.mark.core

PADDED_RECORD_SIZE = 24
PADDED_OFFSETS = [0, 8, 16]


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
            The read no longer goes back to GDAL, so an in-place unpack would leave the scaled
            numbers where the next caller finds them -- and `variables` caches the wrapper, so
            they would survive the call. Expected: the second unpacked read equals the first,
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

    def test_the_cached_mapping_hands_back_the_same_wrapper(self, packed):
        """`variables` caches per key, and a `LabeledArray` now lives in that cache.

        Args:
            packed: Fixture holding a scaled 1-D `Int16` series.

        Test scenario:
            `variables` is documented as loading on first access and caching after, which used
            to be a statement about `NetCDF` subsets only. Expected: the same object on repeat
            access, and its values untouched by an intervening unpacked `read_array` -- the
            aliasing the reuse change could have introduced.
        """
        first = packed.variables["series"]
        packed.read_array(variable="series", unpack=True)
        second = packed.variables["series"]

        assert first is second, "the mapping must cache the wrapper, not rebuild it"
        assert np.array_equal(second.values, [10, 20, 30]), (
            f"the cached values must stay in storage units, got {second.values}"
        )


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
