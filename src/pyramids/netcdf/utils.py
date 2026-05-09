from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeAlias, cast

from osgeo import gdal, osr

from pyramids.base._utils import gdal_to_numpy_dtype, import_cftime

# Keep simple, JSON-serializable attribute values only
AttributeScalar: TypeAlias = bool | int | float | str
AttributeVector: TypeAlias = list[AttributeScalar]
AttributeValue: TypeAlias = AttributeScalar | AttributeVector
_ORIGIN_RE = re.compile(r'^\s*([A-Za-z]+)\s+since\s+(.+?)\s*$', re.IGNORECASE)


def _full_name_with_fallback(group: gdal.Group, default_name: str | None = None) -> str:
    """Get the full hierarchical name of a GDAL group with fallback.

    Attempts `group.GetFullName()` first, then falls back to
    `"/<name>"` using `GetName()` or the provided default.

    Args:
        group: A GDAL multidimensional group object.
        default_name: Name to use when both `GetFullName`
            and `GetName` fail. Defaults to `None`,
            which produces `"/"`.

    Returns:
        The full hierarchical path string (e.g., `"/root/sub"`),
        or `"/"` for root / unnamed groups.
    """
    try:
        result = str(group.GetFullName())
    except Exception:
        # Root or fallback to "/<name>"
        try:
            gname = str(group.GetName())
        except Exception:
            gname = default_name or ""
        result = "/" if not gname else f"/{gname}"
    return result


def _get_group_name(group: gdal.Group) -> str:
    """Get the short name of a GDAL multidimensional group.

    Args:
        group: A GDAL multidimensional group object.

    Returns:
        The group name string, or `""` if the name
        cannot be retrieved.
    """
    try:
        gname = str(group.GetName())
    except Exception:
        gname = ""
    return gname


def _safe_array_names(group: gdal.Group) -> list[str]:
    """List multidimensional array names in a group, sorted.

    Args:
        group: A GDAL multidimensional group object.

    Returns:
        Sorted list of array name strings. Returns an empty
        list if the query fails or the group has no arrays.
    """
    try:
        names = group.GetMDArrayNames() or []
    except Exception:
        names = []
    return sorted(list(names))


def _safe_group_names(group: gdal.Group) -> list[str]:
    """List sub-group names in a group, sorted.

    Args:
        group: A GDAL multidimensional group object.

    Returns:
        Sorted list of sub-group name strings. Returns an
        empty list if the query fails or there are no
        sub-groups.
    """
    try:
        names = group.GetGroupNames() or []
    except Exception:
        names = []
    return sorted(list(names))


def _get_root_group(dataset: gdal.Dataset) -> gdal.Group | None:
    """Get the root group of a GDAL multidimensional dataset.

    Args:
        dataset: An opened GDAL dataset (must support
            the multidimensional API).

    Returns:
        The root `gdal.Group`, or `None` if the dataset
        does not expose a multidimensional group hierarchy.
    """
    try:
        return dataset.GetRootGroup()
    except Exception:
        return None


def _get_driver_name(dataset: gdal.Dataset) -> str:
    """Get the short driver name for a GDAL dataset.

    Args:
        dataset: An opened GDAL dataset.

    Returns:
        Driver short name (e.g., `"netCDF"`, `"GTiff"`),
        or `"UNKNOWN"` if retrieval fails.
    """
    try:
        result = str(dataset.GetDriver().ShortName)
    except Exception:
        result = "UNKNOWN"
    return result


def _export_srs(srs: osr.SpatialReference | None) -> tuple[str | None, str | None]:
    """Export a spatial reference to WKT and PROJJSON strings.

    Args:
        srs: An OSR spatial reference object, or `None`.

    Returns:
        A two-element tuple `(wkt, projjson)` where each
        element is a string or `None` if the export failed
        or *srs* was `None`.
    """
    if not srs:
        return None, None
    wkt = None
    projjson = None
    try:
        wkt = srs.ExportToWkt()
    except Exception:
        pass  # nosec B110
    try:
        projjson = srs.ExportToJSON()
    except Exception:
        pass  # nosec B110
    return wkt, projjson


def _get_array_nodata(
    mdarr: gdal.MDArray, attrs: dict[str, AttributeValue]
) -> int | float | str | None:
    """Determine the no-data value for a multidimensional array.

    Checks CF-standard attributes (`_FillValue`,
    `missing_value`) first, then falls back to the GDAL
    driver API methods.

    Args:
        mdarr: A GDAL multidimensional array object.
        attrs: Pre-read attribute dictionary for the array.

    Returns:
        The no-data value as an `int`, `float`, or
        `str`, or `None` if none is defined.
    """
    # Precedence: CF _FillValue, then missing_value, then driver API
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            v = attrs[key]
            if isinstance(v, list):
                return v[0] if v else None
            return v  # type: ignore[return-value]
    # Try driver API
    for meth in (
        "GetNoDataValueAsDouble",
        "GetNoDataValueAsInt64",
        "GetNoDataValueAsString",
    ):
        if hasattr(mdarr, meth):
            try:
                v = getattr(mdarr, meth)()
                # Some GDAL versions return (value, hasval)
                if (
                    isinstance(v, (list, tuple))
                    and len(v) == 2
                    and isinstance(v[1], (bool, int))
                ):
                    if v[1]:
                        return cast(int | float | str | None, _to_py_scalar(v[0]))
                    continue
                return cast(int | float | str | None, _to_py_scalar(v))
            except Exception:
                continue  # nosec B112
    return None


def _get_array_scale_offset(
    mdarr: gdal.MDArray, attrs: dict[str, AttributeValue]
) -> tuple[float | None, float | None]:
    """Extract scale and offset for packed data.

    Reads CF `scale_factor` / `add_offset` attributes first,
    then checks the GDAL driver API. The unpacking formula is:
    `value = packed * scale + offset`.

    Args:
        mdarr: A GDAL multidimensional array object.
        attrs: Pre-read attribute dictionary for the array.

    Returns:
        A tuple `(scale, offset)` where each element is a
        `float` or `None` if not defined.
    """
    scale = None
    offset = None
    # CF attributes first
    scale_raw = attrs.get("scale_factor")
    if isinstance(scale_raw, (int, float)):
        scale = float(scale_raw)
    offset_raw = attrs.get("add_offset")
    if isinstance(offset_raw, (int, float)):
        offset = float(offset_raw)
    # GDAL API may also expose
    if hasattr(mdarr, "GetScale"):
        try:
            s = mdarr.GetScale()
            if s is not None:
                scale = float(s)
        except Exception:
            pass  # nosec B110
    if hasattr(mdarr, "GetOffset"):
        try:
            o = mdarr.GetOffset()
            if o is not None:
                offset = float(o)
        except Exception:
            pass  # nosec B110
    return scale, offset


def _get_block_size(mdarr: gdal.MDArray) -> list[int] | None:
    """Get the block (chunk) size of a multidimensional array.

    Args:
        mdarr: A GDAL multidimensional array object.

    Returns:
        A list of integers representing the block size along
        each dimension, or `None` if unavailable.
    """
    try:
        bs = mdarr.GetBlockSize()
        if bs:
            return [int(b) for b in bs]
    except Exception:
        pass  # nosec B110
    return None


def _get_coord_variable_names(mdarr: gdal.MDArray) -> list[str]:
    """Get the names of coordinate variables for an array.

    Retrieves the full or short names of each coordinate
    variable associated with the given multidimensional array.

    Args:
        mdarr: A GDAL multidimensional array object.

    Returns:
        A list of coordinate variable name strings.
        Returns an empty list if none are found or the
        query fails.
    """
    names: list[str] = []
    try:
        cvs = mdarr.GetCoordinateVariables()
    except Exception:
        cvs = None
    if not cvs:
        return names
    for cv in cvs:
        try:
            # Some GDAL versions return MDArray objects, others names
            if hasattr(cv, "GetFullName"):
                names.append(cv.GetFullName())  # type: ignore[attr-defined]
            elif hasattr(cv, "GetName"):
                names.append(cv.GetName())
            else:
                names.append(str(cv))
        except Exception:
            # Fallback
            names.append(str(cv))
    return names


def _normalize_origin_string(origin: str) -> str:
    """Normalize a CF time origin into a zero-padded datetime string.

    Handles abbreviated origins such as `"1-1-1 0:0:0"` or
    `"1-1-1T0:0:0"` and pads them into the canonical form
    `"0001-01-01 00:00:00"` that `datetime.fromisoformat`
    can parse.

    Args:
        origin: A date or datetime string from a CF `units`
            attribute. May use `T` or space as the
            date/time separator, and components need not be
            zero-padded.

    Returns:
        A zero-padded datetime string in the format
        `"YYYY-MM-DD HH:MM:SS"` (with optional fractional
        seconds preserved).

    Examples:
        - Pad a minimal origin:
            ```python
            >>> from pyramids.netcdf.utils import (
            ...     _normalize_origin_string,
            ... )
            >>> _normalize_origin_string("1-1-1 0:0:0")
            '0001-01-01 00:00:00'

            ```

        - Handle ISO `T` separator:
            ```python
            >>> _normalize_origin_string("1979-1-1T0:0:0")
            '1979-01-01 00:00:00'

            ```

        - Date-only input gets midnight time:
            ```python
            >>> _normalize_origin_string("2000-6-15")
            '2000-06-15 00:00:00'

            ```
    """
    origin = origin.strip().replace("T", " ")
    if " " in origin:
        date_part, time_part = origin.split(" ", 1)
    else:
        date_part, time_part = origin, "0:0:0"

    ymd = date_part.strip().split("-")
    while len(ymd) < 3:
        ymd.append("1")
    y, m, d = ymd[:3]
    y = y.zfill(4)
    m = m.zfill(2)
    d = d.zfill(2)

    hms = time_part.strip().split(":")
    while len(hms) < 3:
        hms.append("0")
    H, M, S = (hms[0].zfill(2), hms[1].zfill(2), hms[2].zfill(2))

    # Support fractional seconds
    if "." in S:
        # Keep fractional seconds as-is; datetime.fromisoformat can handle it
        return f"{y}-{m}-{d} {H}:{M}:{S}"
    else:
        return f"{y}-{m}-{d} {H}:{M}:{S}"


def _parse_units_origin(units: str) -> tuple[str, datetime]:
    """Parse a CF time-units string into unit name and origin.

    Splits a string like `"days since 1979-01-01"` into
    the lowercase unit name and the origin as a `datetime`.

    Args:
        units: CF time-units string in the format
            `"<unit> since <origin>"`.

    Returns:
        A tuple `(unit, origin_datetime)` where *unit* is
        a lowercase string (e.g., `"days"`) and
        *origin_datetime* is a `datetime` instance.

    Raises:
        ValueError: If *units* does not match the expected
            `"<unit> since <origin>"` pattern.

    Examples:
        - Parse a standard day-based unit string:
            ```python
            >>> from pyramids.netcdf.utils import (
            ...     _parse_units_origin,
            ... )
            >>> unit, origin = _parse_units_origin(
            ...     "days since 1979-01-01"
            ... )
            >>> unit
            'days'
            >>> origin.year
            1979

            ```

        - Abbreviated origins are accepted:
            ```python
            >>> unit, origin = _parse_units_origin(
            ...     "hours since 1-1-1 0:0:0"
            ... )
            >>> unit
            'hours'
            >>> origin.year
            1

            ```

    See Also:
        _normalize_origin_string: Normalizes the origin
            portion of the string.
    """
    m = _ORIGIN_RE.match(units)
    if not m:
        raise ValueError(f"Unrecognized time units: {units!r}")

    unit, origin_raw = m.groups()
    origin_norm = _normalize_origin_string(origin_raw)

    # Try ISO-style parsing
    try:
        origin_dt = datetime.fromisoformat(origin_norm)
    except ValueError:
        # Fallback to explicit format if needed
        origin_dt = datetime.strptime(origin_norm, "%Y-%m-%d %H:%M:%S")

    return unit.lower(), origin_dt


def create_time_conversion_func(
    units: str,
    out_format: str = "%Y-%m-%d %H:%M:%S",
    calendar: str = "standard",
) -> Callable:
    """Create a converter that maps numeric CF time offsets to date strings.

    Parses CF-compliant time unit strings (e.g.,
    `"days since 1979-01-01"`) and returns a callable that
    converts numeric offsets to formatted date strings.

    For standard/proleptic_gregorian calendars, uses Python's
    `datetime` + `timedelta`. For non-standard calendars
    (`360_day`, `noleap`, `all_leap`, `julian`), uses
    `cftime.num2date()` (optional dependency).

    Args:
        units: CF time unit string in the format
            `"<unit> since <origin>"`. Supported units are
            days, hours, minutes, and seconds.
        out_format: strftime format for the output strings.
            Defaults to `"%Y-%m-%d %H:%M:%S"`.
        calendar: CF calendar type. Defaults to `"standard"`.
            Non-standard calendars require the `cftime` package.

    Returns:
        Callable: A function that takes a numeric value and
            returns a formatted date string.

    Raises:
        ValueError: If the unit string cannot be parsed or
            uses an unsupported time unit.
        ImportError: If a non-standard calendar is requested
            and `cftime` is not installed.

    Examples:
        - Convert day offsets from a 1979 origin:
            ```python
            >>> from pyramids.netcdf.utils import (
            ...     create_time_conversion_func,
            ... )
            >>> convert = create_time_conversion_func(
            ...     "days since 1979-01-01"
            ... )
            >>> convert(0)
            '1979-01-01 00:00:00'
            >>> convert(365)
            '1980-01-01 00:00:00'

            ```

        - Use hour-based units with a custom format:
            ```python
            >>> convert = create_time_conversion_func(
            ...     "hours since 2000-01-01",
            ...     out_format="%Y-%m-%d",
            ... )
            >>> convert(24)
            '2000-01-02'
            >>> convert(0)
            '2000-01-01'

            ```

    See Also:
        _parse_units_origin: Parses the unit string.
    """
    converter = None

    if calendar.lower() not in ("standard", "proleptic_gregorian", "gregorian"):
        import_cftime(
            f"Calendar '{calendar}' requires the cftime package. "
            f"Install it with: pixi add cftime"
        )
        import cftime

        def convert_cftime(value):
            dt = cftime.num2date(value, units, calendar)
            return dt.strftime(out_format)

        converter = convert_cftime
    else:
        unit, origin = _parse_units_origin(units)

        if unit.startswith("day"):
            scale = timedelta(days=1)
        elif unit.startswith("hour"):
            scale = timedelta(hours=1)
        elif unit.startswith("min"):
            scale = timedelta(minutes=1)
        elif unit.startswith("sec"):
            scale = timedelta(seconds=1)
        else:
            raise ValueError(f"Unsupported time unit: {unit!r}")

        def convert(value):
            dt = origin + value * scale
            return dt.strftime(out_format)

        converter = convert

    return converter


def _dtype_to_str(dt: Any) -> str:
    """Convert a GDAL extended data type to a numpy dtype string.

    Tries `dt.GetName()` first (works for string types), then
    `dt.GetNumericDataType()` which returns a GDAL code that
    `gdal_to_numpy_dtype()` converts to a name like `"float32"`.

    Args:
        dt: A GDAL `ExtendedDataType` or similar object.

    Returns:
        A numpy-compatible dtype name (e.g. `"float32"`,
        `"int16"`), or `"unknown"` if conversion fails.
    """
    result = "unknown"
    try:
        # gdal.ExtendedDataType in MDIM (works for string types)
        name = dt.GetName()
        if isinstance(name, str) and name:
            result = name.lower()
    except Exception:
        pass  # nosec B110
    if result == "unknown":
        try:
            # Numeric types: GetName() returns "" but GetNumericDataType()
            # gives the GDAL code (e.g. 6 = GDT_Float32)
            gdal_code = dt.GetNumericDataType()
            result = gdal_to_numpy_dtype(gdal_code)
        except Exception:
            pass  # nosec B110
    return result


def _to_py_scalar(x: Any) -> Any:
    """Convert a value to a native JSON-serializable Python type.

    Handles numpy scalars (via `.item()`), `bytes`
    (decoded as UTF-8), and passes through native Python
    scalars unchanged. Non-convertible values fall back to
    `str()`.

    Args:
        x: Any value, typically a numpy scalar, `bytes`,
            or a native Python scalar.

    Returns:
        A JSON-serializable Python value (`bool`, `int`,
        `float`, `str`, or `None`).

    Examples:
        - Native scalars pass through unchanged:
            ```python
            >>> from pyramids.netcdf.utils import _to_py_scalar
            >>> _to_py_scalar(42)
            42
            >>> _to_py_scalar(3.14)
            3.14
            >>> _to_py_scalar(None) is None
            True

            ```

        - Bytes are decoded to strings:
            ```python
            >>> _to_py_scalar(b"hello")
            'hello'

            ```
    """
    try:
        # numpy scalar
        if hasattr(x, "item") and callable(x.item):
            return x.item()
    except Exception:
        pass  # nosec B110

    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return x.decode("utf-8", errors="ignore")

    # Already a JSON-friendly scalar
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x

    # Fallback to string representation to avoid breaking JSON dump
    return str(x)


def _normalize_attr_value(val: Any) -> AttributeValue:
    """Normalize an attribute value to a JSON-serializable form.

    Converts lists/tuples element-wise and scalars directly
    using `_to_py_scalar`.

    Args:
        val: Raw attribute value from GDAL, which may be a
            list, tuple, numpy array, scalar, or `bytes`.

    Returns:
        A normalized `AttributeValue`: either a list of
        JSON-friendly scalars or a single scalar.
    """
    # Vector
    if isinstance(val, (list, tuple)):
        return [_to_py_scalar(v) for v in val]

    # Scalar
    return cast(AttributeValue, _to_py_scalar(val))


def _read_attribute_value(attr: gdal.Attribute) -> AttributeValue:
    """Read a single GDAL attribute and normalize its value.

    Tries the generic `attr.Read()` first, then falls back
    to type-specific readers (`ReadAsInt64`,
    `ReadAsDouble`, `ReadAsString`, etc.).

    Args:
        attr: A GDAL `Attribute` object.

    Returns:
        The attribute value as a JSON-serializable scalar or
        list of scalars.
    """
    # Try the generic Read() first; it often returns appropriate Python types
    val: Any
    try:
        val = attr.Read()
    except Exception:
        # try type-specifics
        for meth in (
            "ReadAsInt64",
            "ReadAsInt64Array",
            "ReadAsDouble",
            "ReadAsDoubleArray",
            "ReadAsString",
            "ReadAsStringArray",
        ):
            if hasattr(attr, meth):
                try:
                    val = getattr(attr, meth)()
                    break
                except Exception:
                    continue  # nosec B112
        else:
            val = None
    return _normalize_attr_value(val)


def _read_attributes(obj: Any) -> dict[str, AttributeValue]:
    """Read all attributes from a GDAL object into a dictionary.

    Iterates over attributes exposed by
    `obj.GetAttributes()` and normalizes each value. Skips
    attributes whose names cannot be retrieved and falls
    back gracefully for unreadable values.

    Args:
        obj: Any GDAL object that supports
            `GetAttributes()` (e.g., `gdal.Group`,
            `gdal.MDArray`).

    Returns:
        A dictionary mapping attribute names to their
        normalized JSON-serializable values.
    """
    attrs: dict[str, AttributeValue] = {}
    try:
        att_list = obj.GetAttributes()
    except Exception:
        att_list = None
    if not att_list:
        return attrs
    for att in att_list:
        try:
            name = att.GetName()
        except Exception:
            continue  # nosec B112
        try:
            attrs[name] = _read_attribute_value(att)
        except Exception:
            # Be robust; don't crash on odd attribute types
            attrs[name] = _normalize_attr_value(None)
    return attrs
