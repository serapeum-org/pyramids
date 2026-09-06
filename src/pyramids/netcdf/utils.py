from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeAlias, cast

import cftime
import numpy as np
import pandas as pd
from osgeo import gdal, osr

# Re-exported, not defined here. They moved down to `base` because
# `dataset/_cube_time.py` is one of the two writers that stamps them, and
# reaching into `netcdf` for them pointed `dataset` back up at a package
# that imports it. `decode_cf_time` below is what reads them back, so the
# names stay importable from here for every caller that already does.
from pyramids.base._cf_epoch import CF_EPOCH, CF_EPOCH_CALENDAR, cf_epoch_units
from pyramids.base._utils import gdal_to_numpy_dtype

# Keep simple, JSON-serializable attribute values only
AttributeScalar: TypeAlias = bool | int | float | str
AttributeVector: TypeAlias = list[AttributeScalar]
AttributeValue: TypeAlias = AttributeScalar | AttributeVector
# Matched against the stripped string. Written to be linear-time: `\s+` borders
# only `\S`-led groups (no overlap, so no backtracking ambiguity — S5852), and
# the class is single-cased because IGNORECASE already covers A-Z (S5869).
_ORIGIN_RE = re.compile(r"^([a-z]+)\s+since\s+(\S.*)$", re.IGNORECASE)


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


def resolve_full_name(obj: Any, group_full_name: str, fallback_name: str) -> str:
    """Full hierarchical name of a GDAL dimension / array, with a qualified fallback.

    Tries ``obj.GetFullName()``; when that raises (some drivers don't expose it on every
    object), composes the path from the parent group's full name and ``fallback_name`` —
    ``"<group>/<name>"``, or ``"/<name>"`` when the parent is the root group.

    Args:
        obj: A GDAL dimension or MDArray.
        group_full_name: Full name of the parent group (e.g. ``"/"`` or ``"/grp"``).
        fallback_name: The object's short name, used to build the fallback path.

    Returns:
        The resolved full-name string.
    """
    try:
        return cast("str", obj.GetFullName())
    except Exception:
        if group_full_name != "/":
            return f"{group_full_name}/{fallback_name}"
        return f"/{fallback_name}"


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
    return sorted(names)


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
    return sorted(names)


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


CF_NODATA_KEYS: tuple[str, ...] = ("_FillValue", "missing_value", "nodata")


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
    # Precedence: CF _FillValue, then missing_value, then `nodata`, then the
    # driver API. `nodata` is what `DatasetCollection.to_netcdf` writes, and it
    # was known only to the MDArray reader -- so the attribute path returned
    # None for a file whose only sentinel was written by pyramids itself.
    for key in CF_NODATA_KEYS:
        if key in attrs:
            v = attrs[key]
            if isinstance(v, list):
                return v[0] if v else None
            return v  # type: ignore[return-value]
    return _nodata_from_driver_api(mdarr)


def _nodata_from_driver_api(mdarr: gdal.MDArray) -> int | float | str | None:
    """Read a no-data value from the GDAL MDArray driver API.

    Tries the double / int64 / string accessors in order. Some GDAL versions
    return a ``(value, has_value)`` pair; the value is used only when the flag
    is set, otherwise the next accessor is tried. Any accessor that raises is
    skipped.

    Args:
        mdarr: A GDAL multidimensional array object.

    Returns:
        The no-data value as an ``int`` / ``float`` / ``str``, or ``None`` if
        no accessor yields one.
    """
    for meth in (
        "GetNoDataValueAsDouble",
        "GetNoDataValueAsInt64",
        "GetNoDataValueAsString",
    ):
        if not hasattr(mdarr, meth):
            continue
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

    # Keep fractional seconds as-is; datetime.fromisoformat can handle them.
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
    m = _ORIGIN_RE.match(units.strip())
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


def _is_standard_calendar(calendar: str | None) -> bool:
    """True for the Gregorian family (`standard` / `gregorian` / `proleptic_gregorian`), case-insensitive.

    Single source for the standard-vs-non-standard calendar split so the CF time helpers cannot drift on
    the accepted set or its casing (ARC-69). `create_time_conversion_func` lowercased its check but
    `decode_cf_time` did not, so a capitalised `"Gregorian"`/`"Standard"` was mis-classified as
    non-standard by the latter.

    Args:
        calendar: The CF calendar name (``None`` is treated as ``"standard"``).

    Returns:
        bool: `True` when `calendar` names a proleptic-Gregorian-compatible calendar.
    """
    return (calendar or "standard").lower() in (
        "standard",
        "gregorian",
        "proleptic_gregorian",
    )


# Nanoseconds in one unit, keyed by the prefix a CF unit name starts with, so
# `day`/`days`, `sec`/`second`/`seconds` and the rest all resolve. One table for
# both readers below: keeping a scale factor each is how they came to accept
# different sets of resolutions in the first place.
_NS_PER_CF_UNIT: dict[str, int] = {
    "day": 86_400_000_000_000,
    "hour": 3_600_000_000_000,
    "min": 60_000_000_000,
    "sec": 1_000_000_000,
    "millisecond": 1_000_000,
    "microsecond": 1_000,
    "nanosecond": 1,
}


def _ns_per_cf_unit(unit: str) -> int | None:
    """Nanoseconds in one `unit`, or `None` when the name is not one we can scale.

    Args:
        unit: The lowercased period name from a CF `units` string, e.g. `"days"`.

    Returns:
        int | None: The exact nanosecond count, or `None` for a period this
            package does not scale (`"months"`, `"common_years"`, …), which the
            callers hand to `cftime` instead.
    """
    return next(
        (ns for prefix, ns in _NS_PER_CF_UNIT.items() if unit.startswith(prefix)),
        None,
    )


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
            `"<unit> since <origin>"`. Supported units are days, hours,
            minutes, seconds, milliseconds, microseconds, and nanoseconds.
            Sub-second units decode at microsecond resolution (Python
            `datetime`'s finest), so a nanosecond axis is rounded to the
            nearest microsecond — exact for date/second output, and visible
            only in a `%f` (microsecond) `out_format`.
        out_format: strftime format for the output strings.
            Defaults to `"%Y-%m-%d %H:%M:%S"`.
        calendar: CF calendar type. Defaults to `"standard"`.
            Non-standard calendars are decoded with `cftime` (a core
            dependency).

    Returns:
        Callable: A function that takes a numeric value and
            returns a formatted date string.

    Raises:
        ValueError: If the unit string cannot be parsed or
            uses an unsupported time unit.

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

    if not _is_standard_calendar(calendar):

        def convert_cftime(value):
            dt = cftime.num2date(value, units, calendar)
            return dt.strftime(out_format)

        converter = convert_cftime
    else:
        unit, origin = _parse_units_origin(units)

        # datetime/timedelta is microsecond-resolution, so the offset is resolved as
        # ``origin + timedelta(microseconds=value * factor)``: day/hour/minute/second
        # stay exact over any realistic date range, while the sub-second units (notably
        # the ``nanoseconds`` axis DatasetCollection.to_netcdf writes) round to the
        # nearest microsecond — exact for date/second output, with any residual only in
        # the microsecond digit of a ``%f`` format. The scale comes from the shared
        # nanosecond table so this reader and ``decode_cf_time`` cannot drift on which
        # unit names they accept.
        nanos = _ns_per_cf_unit(unit)
        if nanos is None:
            raise ValueError(f"Unsupported time unit: {unit!r}")
        factor = nanos / 1000.0

        def convert(value):
            dt = origin + timedelta(microseconds=float(value) * factor)
            return dt.strftime(out_format)

        converter = convert

    return converter


# CF's time form is "<period> since <timestamp>". Matched case-insensitively,
# with `since` a whitespace-separated word carrying text on both sides, so a
# unit that merely contains the letters (or that is only the word) is not one.
_CF_TIME_UNITS = re.compile(r"\S\s+since\s+\S", re.IGNORECASE)


def _epoch_units_round_trip() -> None:
    """`cf_epoch_units` produces what `decode_cf_time` parses.

    The two halves live in different packages now -- the writer's constants in
    :mod:`pyramids.base._cf_epoch`, the reader here -- so the property that
    made sharing them worthwhile is asserted where both are in scope. Named
    with a leading underscore because it exists for its doctest.

    Examples:
        - A seconds axis written from the shared epoch decodes back to the
          dates it was written from:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.utils import cf_epoch_units, decode_cf_time
            >>> units = cf_epoch_units("seconds")
            >>> decode_cf_time(np.array([0.0, 86400.0]), units).astype(str).tolist()
            ['1970-01-01T00:00:00.000000000', '1970-01-02T00:00:00.000000000']

            ```
        - And so does the nanosecond axis the collection writer emits, which
          is the resolution that used to be produced and then not read:
            ```python
            >>> units = cf_epoch_units("nanoseconds")
            >>> offsets = np.array([0, 86_400_000_000_000], dtype="int64")
            >>> decode_cf_time(offsets, units).astype(str).tolist()
            ['1970-01-01T00:00:00.000000000', '1970-01-02T00:00:00.000000000']

            ```
    """


def cf_units_text(units: Any) -> str | None:
    """The `units` attribute as text, or `None` when it is not text at all.

    A `units` read straight out of an HDF5 / netCDF attribute can arrive as
    `bytes`. That is the same unit, undecoded -- not a different kind of thing
    -- so both the predicate below and the decoder that gates on it resolve it
    here rather than each deciding for itself. Answering "not a time unit" for
    bytes put them back in the hole the predicate exists to fill; decoding them
    in the predicate alone would have let `decode_cf_time` hand the raw bytes
    to `cftime`, which raises.

    Args:
        units: The CF `units` attribute -- a string, the bytes an undecoded
            attribute arrives as, `None`, or whatever a malformed file put
            there. GDAL normalises attributes into scalars *or lists*, so a
            `units` written as a one-element array is a real input.

    Returns:
        str | None: The text, or `None` for a non-text value and for bytes
            that are not valid UTF-8.

    Examples:
        - A string is itself, and bytes are decoded:
            ```python
            >>> from pyramids.netcdf.utils import cf_units_text
            >>> cf_units_text("days since 1970-01-01")
            'days since 1970-01-01'
            >>> cf_units_text(b"days since 1970-01-01")
            'days since 1970-01-01'

            ```
        - Anything that is not text at all answers `None`, rather than
          raising the way a bare `" since " in unit` would:
            ```python
            >>> from pyramids.netcdf.utils import cf_units_text
            >>> print(cf_units_text(None), cf_units_text(1), cf_units_text(["a"]))
            None None None

            ```

    See Also:
        is_cf_time_units: Asks whether that text names a CF time axis.
    """
    if isinstance(units, str):
        text: str | None = units
    elif isinstance(units, (bytes, bytearray)):
        try:
            text = bytes(units).decode("utf-8")
        except UnicodeDecodeError:
            text = None
    else:
        text = None
    return text


def is_cf_time_units(units: str | bytes | None) -> bool:
    """True when `units` declares a CF time axis.

    Four places asked this and answered differently. Axis *detection* matched a
    bare lowercased `"since"` substring; *decoding* required a case-sensitive
    `" since "`. A file whose units read `"Days SINCE 1970-01-01"` -- which
    `cftime` parses perfectly well -- was therefore reported as a time axis and
    then handed back as raw numbers, with nothing raised. This is the one rule
    both questions now use.

    Total: anything that is not a string is not a time unit. GDAL attributes
    are normalised into scalars *or lists*, so a `units` written as a
    one-element array is a real input, and the CF compliance checker exists to
    *report* on malformed files rather than to crash on one.

    Args:
        units: The CF `units` attribute -- a string, the bytes an undecoded
            attribute arrives as, `None`, or whatever a malformed file put
            there.

    Returns:
        bool: True when `units` is a string with the `<period> since
            <timestamp>` shape.

    Examples:
        - The ordinary form, and the uppercase one that used to decode wrong:
            ```python
            >>> from pyramids.netcdf.utils import is_cf_time_units
            >>> is_cf_time_units("days since 1970-01-01")
            True
            >>> is_cf_time_units("Days SINCE 1970-01-01")
            True

            ```
        - A spatial unit is not a time axis, nor is a missing one:
            ```python
            >>> from pyramids.netcdf.utils import is_cf_time_units
            >>> is_cf_time_units("degrees_north"), is_cf_time_units(None)
            (False, False)

            ```
        - Neither is a `units` a malformed file wrote as a list or a number:
            ```python
            >>> from pyramids.netcdf.utils import is_cf_time_units
            >>> is_cf_time_units(["days since 1970-01-01"])
            False
            >>> is_cf_time_units(1)
            False

            ```
        - `since` has to be a word with a period and an epoch around it, so a
          unit that merely contains the letters is not matched:
            ```python
            >>> from pyramids.netcdf.utils import is_cf_time_units
            >>> is_cf_time_units("sincerity"), is_cf_time_units("since")
            (False, False)

            ```
        - Bytes are decoded first, so an attribute that arrived undecoded
          still names the axis it names:
            ```python
            >>> from pyramids.netcdf.utils import is_cf_time_units
            >>> is_cf_time_units(b"days since 1970-01-01")
            True

            ```

    See Also:
        decode_cf_time: Decodes the axes this identifies.
    """
    text = cf_units_text(units)
    return text is not None and _CF_TIME_UNITS.search(text) is not None


# The Gregorian reform. `standard` / `gregorian` count Julian days before this
# instant and Gregorian days after it, while `proleptic_gregorian` -- and
# `datetime`, and `datetime64` -- run the Gregorian rules all the way back. An
# origin at or after it therefore puts every offset on the side where the two
# agree, which is what makes the integer path below safe to take; an earlier
# origin on a mixed calendar stays on `cftime`, which knows about the ten
# missing days.
_GREGORIAN_CUTOVER = datetime(1582, 10, 15)
# Bound on the nanosecond magnitudes the integer path will handle. Deliberately
# under `int64`'s 9.223e18 so the check -- made in float64, where the rounding
# error at this scale is ~1e3 ns -- cannot pass a value that then overflows.
_NS_LIMIT = 9.0e18


def _gregorian_scale_and_origin(
    units: str, calendar: str | None
) -> tuple[int, int] | None:
    """`(nanoseconds per unit, origin in nanoseconds since 1970)`, or `None`.

    `None` means the exact integer decode does not apply and the caller should
    fall back to `cftime`: an origin this package cannot parse (a `"1970-01-01
    00:00:00 UTC"` suffix, say), a period it does not scale (`"months"`), or a
    pre-1582 origin on a mixed Julian/Gregorian calendar, where `cftime`'s
    answer and `datetime`'s differ by the reform's ten days.

    Args:
        units: A CF `"<period> since <origin>"` string.
        calendar: The CF calendar name; `None` is read as `"standard"`.

    Returns:
        tuple[int, int] | None: The scale and origin, or `None` to fall back.
    """
    resolved: tuple[int, int] | None = None
    try:
        unit, origin = _parse_units_origin(units)
    except ValueError:
        unit, origin = "", None
    nanos = _ns_per_cf_unit(unit) if origin is not None else None
    if origin is not None and nanos is not None:
        # An origin written with a zone (`...T00:00:00Z`) parses tz-aware; shift
        # it to UTC so the arithmetic below stays naive, as `cftime` reads it.
        utc_offset = origin.utcoffset()
        if utc_offset is not None:
            origin = origin.replace(tzinfo=None) - utc_offset
        proleptic = (calendar or "standard").lower() == "proleptic_gregorian"
        if proleptic or origin >= _GREGORIAN_CUTOVER:
            elapsed = origin - datetime(1970, 1, 1)
            resolved = (
                nanos,
                elapsed.days * 86_400_000_000_000
                + elapsed.seconds * 1_000_000_000
                + elapsed.microseconds * 1_000,
            )
    return resolved


def _decode_gregorian_ns(
    values: np.ndarray, units: str, calendar: str | None
) -> np.typing.NDArray | None:
    """Add CF offsets to their origin as `datetime64[ns]`, or `None` to fall back.

    The scaling is done in integer nanoseconds rather than by handing the string
    to `cftime`, whose finest unit is the microsecond. That is what lets a
    `"nanoseconds since ..."` axis -- exactly what `DatasetCollection.to_netcdf`
    writes -- decode at all, and it keeps the sub-microsecond digits that
    resolution exists to preserve instead of rounding them away.

    Args:
        values: The numeric offsets read for the coordinate.
        units: The CF `"<period> since <origin>"` string.
        calendar: The CF calendar name; only the Gregorian family reaches here.

    Returns:
        np.ndarray | None: `datetime64[ns]` instants, or `None` when the offsets
            are not numeric or do not fit the type, so the caller falls back.
    """
    decoded: np.typing.NDArray | None = None
    scale = _gregorian_scale_and_origin(units, calendar)
    array = np.asarray(values)
    if scale is not None and array.dtype.kind in "iuf":
        nanos, origin_ns = scale
        # NaN is the interop writer's spelling of NaT, so it is a real offset
        # rather than a malformed one; zero it for the range test and restore it
        # as NaT below. An infinity is malformed, and fails `isfinite`.
        missing = (
            np.isnan(array)
            if array.dtype.kind == "f"
            else np.zeros_like(array, dtype=bool)
        )
        scaled = np.where(missing, 0.0, array.astype("float64") * float(nanos))
        largest = float(np.max(np.abs(scaled))) if scaled.size else 0.0
        if bool(np.all(np.isfinite(scaled))) and largest + abs(origin_ns) <= _NS_LIMIT:
            offsets = _exact_ns_offsets(array, nanos, missing)
            instants = (offsets + np.int64(origin_ns)).astype("datetime64[ns]")
            # `"NaT"` with an explicit unit: the bare spelling is generic, which
            # NumPy deprecates in arithmetic and comparison against a typed array.
            decoded = np.where(missing, np.datetime64("NaT", "ns"), instants)
    return decoded


def _exact_ns_offsets(
    array: np.typing.NDArray, nanos: int, missing: np.typing.NDArray
) -> np.typing.NDArray:
    """Scale `array` to integer nanoseconds, exactly where the input allows.

    An integer axis is scaled in `int64`, so an `int64` count of nanoseconds --
    the lossless encoding of `datetime64[ns]` -- survives untouched. A float axis
    is scaled in `float64` and rounded to the nearest nanosecond, which is the
    input's own precision limit rather than one this function imposes.

    Args:
        array: The numeric offsets, already known to fit the type.
        nanos: Nanoseconds in one unit of `array`.
        missing: Mask of `NaN` entries, zeroed before the float cast.

    Returns:
        np.ndarray: The offsets in `int64` nanoseconds.
    """
    if array.dtype.kind == "f":
        offsets = np.rint(np.where(missing, 0.0, array) * float(nanos)).astype("int64")
    else:
        offsets = array.astype("int64") * np.int64(nanos)
    return offsets


def decode_cf_time(
    values: np.ndarray,
    unit: str | bytes | None,
    calendar: str = "standard",
) -> np.typing.NDArray:
    """Decode numeric CF time offsets to datetimes.

    Standard / gregorian / proleptic_gregorian calendars yield ``datetime64[ns]`` when
    representable; non-standard calendars (``360_day`` / ``noleap`` …) yield ``cftime``
    objects. Arrays whose ``unit`` is not a ``"<interval> since <origin>"`` string are
    returned unchanged.

    A Gregorian-family axis is decoded by adding its offsets to the origin in integer
    nanoseconds, rather than through ``cftime``. Behaviour differs from earlier releases
    in two ways, both of them consequences of dropping ``cftime``'s microsecond floor:
    a ``"nanoseconds since …"`` axis decodes instead of raising, and a ``NaN`` offset
    decodes to ``NaT`` instead of to the origin. Anything the integer path cannot take
    exactly -- an unparseable origin, a period such as ``"months"``, an instant outside
    ``datetime64[ns]``'s 1678-2262 range, or a pre-1582 origin on a mixed
    Julian/Gregorian calendar -- still goes to ``cftime``, unchanged.

    Args:
        values: The numeric values already read for the coordinate.
        unit: The coordinate's CF unit string (e.g. ``"days since 1979-01-01"``),
            or the bytes an undecoded attribute arrives as.
        calendar: The CF calendar name. Defaults to ``"standard"``.

    Returns:
        np.ndarray: Decoded datetimes for a time axis, else ``values`` unchanged.

    Examples:
        - The resolution the collection writer counts in, read back with its
          sub-microsecond digits intact:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.utils import cf_epoch_units, decode_cf_time
            >>> offsets = np.array([0, 1577836800123456789], dtype="int64")
            >>> decoded = decode_cf_time(
            ...     offsets, cf_epoch_units("nanoseconds"), "proleptic_gregorian"
            ... )
            >>> decoded.astype(str).tolist()
            ['1970-01-01T00:00:00.000000000', '2020-01-01T00:00:00.123456789']

            ```

    See Also:
        is_cf_time_units: The predicate this gates on. It accepts every
            ``"<period> since <origin>"`` string, so a resolution it admits and
            this cannot decode is a defect -- which is what a ``nanoseconds``
            axis was, accepted by the predicate and refused by ``cftime``.
    """
    # Through the same normaliser the predicate uses: `cftime` needs `str`, and
    # a `units` the predicate accepted as bytes would otherwise raise here.
    text = cf_units_text(unit)
    decoded: np.typing.NDArray
    if not is_cf_time_units(text):
        decoded = values
    else:
        standard = _is_standard_calendar(calendar)
        text = cast("str", text)
        exact = _decode_gregorian_ns(values, text, calendar) if standard else None
        if exact is not None:
            decoded = exact
        else:
            decoded = np.asarray(
                cftime.num2date(
                    values, text, calendar, only_use_cftime_datetimes=not standard
                )
            )
            if standard:
                try:
                    decoded = decoded.astype("datetime64[ns]")
                except (ValueError, TypeError):
                    pass
    return decoded


def encode_cf_time(value: Any, unit: str, calendar: str = "standard") -> float:
    """Convert a date string / datetime to a coordinate's numeric CF scale.

    The inverse of :func:`decode_cf_time` for a single value: used to translate a
    user-supplied selection bound back to the time axis's stored numbers.

    Args:
        value: A date string, :class:`datetime`, or anything :class:`pandas.Timestamp`
            accepts.
        unit: The CF unit string the numeric scale is expressed in.
        calendar: The CF calendar name.

    Returns:
        float: ``value`` expressed in ``unit`` on the given ``calendar``.
    """
    year, month, day, hour, minute, second, microsecond = _datetime_components(value)
    dt = cftime.datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        calendar=calendar,
    )
    return float(cftime.date2num(dt, unit, calendar))


def _datetime_components(value: Any) -> tuple[int, int, int, int, int, int, int]:
    """Extract ``(year, month, day, hour, minute, second, microsecond)`` without Gregorian validation.

    ``encode_cf_time`` funnelled its input through ``pandas.Timestamp``, which is proleptic-Gregorian
    and nanosecond-bounded, so a date valid only on a ``360_day`` / ``noleap`` calendar (e.g. month
    day 30 in February) raised before it ever reached ``cftime`` (ARC-30). Pull the calendar fields
    out directly instead: from an object exposing ``year``/``month``/``day`` (``datetime`` /
    ``cftime.datetime`` / ``pandas.Timestamp``), or by parsing an ISO ``YYYY-MM-DD[ HH:MM:SS]`` string
    without any Gregorian range check. Only genuinely non-ISO strings fall back to ``pandas``.

    Args:
        value: A date string, ``datetime``-like object, or ``cftime.datetime``.

    Returns:
        The seven calendar components as ints.
    """
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return (
            int(value.year),
            int(value.month),
            int(value.day),
            int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)),
            int(getattr(value, "second", 0)),
            int(getattr(value, "microsecond", 0)),
        )
    match = re.match(
        r"\s*(\d{1,4})-(\d{1,2})-(\d{1,2})"
        r"(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?)?\s*$",
        str(value),
    )
    if match is None:
        ts = pd.Timestamp(value)
        return (
            ts.year,
            ts.month,
            ts.day,
            ts.hour,
            ts.minute,
            ts.second,
            ts.microsecond,
        )
    seconds_float = float(match.group(6)) if match.group(6) else 0.0
    whole_seconds = int(seconds_float)
    # Clamp so a value like "59.9999995" whose fraction rounds up to 1_000_000 does not exceed
    # cftime's microsecond range (0..999_999) and raise (review N1).
    microsecond = min(int(round((seconds_float - whole_seconds) * 1_000_000)), 999_999)
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)) if match.group(4) else 0,
        int(match.group(5)) if match.group(5) else 0,
        whole_seconds,
        microsecond,
    )


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


def _merge_unit(
    attrs: dict[str, AttributeValue], gdal_obj: Any
) -> dict[str, AttributeValue]:
    """Fold GDAL's ``GetUnit()`` back into ``attrs`` as a CF ``units`` entry.

    Prefer :func:`read_cf_attributes`, which pairs this with the attribute read; call this
    directly only when the attributes were obtained some other way.

    GDAL normalises a CF ``units`` attribute onto the MDArray / indexing-variable **unit
    slot** and drops it from the attribute list. That is not driver-specific — the netCDF
    and HDF5 drivers both do it — so a CF consumer reading attributes alone sees
    ``calendar`` and ``standard_name`` but never ``units`` (#1078).

    Existing ``units`` in ``attrs`` win, so a file that really does carry the attribute keeps
    its own value. That incumbent is taken as-is while the slot value is type-checked, which is
    deliberate but worth naming: the incumbent arrived through :func:`_read_attributes`, which
    has already normalised it, whereas the slot value goes in unmediated.

    The ``isinstance`` test enforces this mapping's declared value type. Real GDAL returns
    ``str`` here, so on a live handle the check never fires; what it actually rejects is a
    non-GDAL stand-in — a ``MagicMock`` in the unit tests will happily return a mock object
    from ``GetUnit()``, and that must not end up in metadata that is later serialised.

    A raising ``GetUnit`` is treated as "no unit". Reading metadata degrades rather than
    fails throughout this module, and a unit slot that errors should not take the caller's
    whole attribute read down with it.

    Args:
        attrs: Attributes already read from ``gdal_obj``; mutated in place.
        gdal_obj: Any GDAL object exposing ``GetUnit()`` (MDArray, indexing variable).

    Returns:
        dict[str, AttributeValue]: ``attrs``, with ``units`` added when the slot held one.
    """
    try:
        unit = gdal_obj.GetUnit()
    except (RuntimeError, AttributeError):
        unit = None
    if isinstance(unit, str) and unit and "units" not in attrs:
        attrs["units"] = unit
    return attrs


def read_cf_attributes(obj: Any) -> dict[str, AttributeValue]:
    """Read a GDAL object's attributes the way a CF consumer needs them.

    The CF reader for every consumer of ``units`` / ``calendar`` / ``standard_name`` /
    ``axis``. Use this rather than :func:`_read_attributes` wherever the attributes are
    interpreted as CF: attributes alone omit ``units``, because GDAL moves it to the object's
    unit slot, so a site that reads them raw silently loses it — which is how a time axis came
    back undecodable, CF axis detection by units never fired, and a UGRID variable's unit
    vanished on a round trip (#1078).

    :func:`_read_attributes` remains the right call for a verbatim view of what the file
    declares (serialisation, round-trip writers); this one is for interpreting CF.

    The slot is the CF ``units`` for the netCDF and HDF5 drivers, which is where this matters.
    Another multidim driver (Zarr, HDF-EOS) may fill it from a format-specific field, in which
    case the reported ``units`` is that driver's unit rather than a CF attribute the file
    literally declares — a reason to read attributes raw when the question is "what does this
    file say", not "what does this axis mean".

    Args:
        obj: Any GDAL object exposing ``GetAttributes()`` and ``GetUnit()``.

    Returns:
        dict[str, AttributeValue]: The object's attributes, including ``units``.
    """
    return _merge_unit(_read_attributes(obj), obj)


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


def _read_dim_names(md_arr: Any) -> list[str]:
    """Read the ordered dimension names of a GDAL MDArray.

    Prefers each dimension's full name and falls back to its short name.
    Any failure to enumerate the dimensions yields an empty list so the
    caller degrades gracefully.

    Args:
        md_arr: A GDAL `MDArray` supporting `GetDimensions()`.

    Returns:
        Ordered list of dimension names (full names where available),
        empty if the dimensions cannot be read.
    """
    dim_names: list[str] = []
    try:
        for d in md_arr.GetDimensions() or []:
            try:
                dim_names.append(d.GetFullName())
            except Exception:
                dim_names.append(d.GetName())
    except Exception:
        return []
    return dim_names


# Every public name this module offers: the CF epoch constants and helpers it
# re-exports from `base._cf_epoch`, plus every public function, type alias and
# constant it defines itself. The list is the module's declared surface -- a
# star-import and mkdocstrings' public-member detection both read it -- so a
# name left out is a name withdrawn, not merely one that is undocumented.
#
# Measured, not assumed: griffe reports `mod.exports` for this module and calls
# exactly those members public, so the docs page renders this list and nothing
# else. The first version listed only functions, which silently dropped the
# three attribute type aliases -- `AttributeValue` is in `NetCDFVariable`'s own
# signature, via `netcdf/models.py` -- and `CF_NODATA_KEYS` from the page.
__all__ = [
    "AttributeScalar",
    "AttributeValue",
    "AttributeVector",
    "CF_EPOCH",
    "CF_EPOCH_CALENDAR",
    "CF_NODATA_KEYS",
    "cf_epoch_units",
    "cf_units_text",
    "create_time_conversion_func",
    "decode_cf_time",
    "encode_cf_time",
    "is_cf_time_units",
    "read_cf_attributes",
    "resolve_full_name",
]
