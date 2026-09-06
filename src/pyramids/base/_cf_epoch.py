"""The epoch every pyramids writer counts time from, and the calendar it declares.

Two writers stamp these onto the arrays they emit -- the collection's netCDF
time axis (`dataset/_cube_time.py`) and the xarray interop path
(`netcdf/engines/interop.py`) -- and `pyramids.netcdf.utils.decode_cf_time`
reads them back. Sharing them is what keeps a written axis decodable: a writer
that drifted to `"standard"`, or to a different epoch, would produce a file the
reader silently decodes to the wrong dates.

They live in `base` rather than beside that reader because `dataset` is one of
the writers, and `pyramids.netcdf.netcdf` imports `pyramids.dataset` --
so importing them from `netcdf.utils` pointed `dataset` back up at `netcdf`
and inverted the package's own layering. `netcdf.utils` re-exports them, so the
names callers already use still resolve from there.

Two constants and a formatter, with no netCDF dependency of their own.

Attributes:
    CF_EPOCH (str): The instant offsets are counted from, `"1970-01-01
        00:00:00"`. Written into every `units` string this module builds, and
        parsed back out of it by the reader.
    CF_EPOCH_CALENDAR (str): The calendar the writers declare,
        `"proleptic_gregorian"`. Deliberately not `"standard"`: the standard
        (mixed Julian/Gregorian) calendar has no 1582-10-05 through 1582-10-14,
        so a date in that window is unrepresentable and dates before it shift
        by ten days. The proleptic calendar extends the Gregorian rules
        backwards without a gap, which is what `datetime64` and `datetime`
        already do -- so declaring it is what makes the round trip exact rather
        than merely usually right.

Examples:
    - The two halves a writer stamps onto a time axis:
        ```python
        >>> from pyramids.base._cf_epoch import CF_EPOCH_CALENDAR, cf_epoch_units
        >>> attributes = {
        ...     "units": cf_epoch_units("seconds"),
        ...     "calendar": CF_EPOCH_CALENDAR,
        ... }
        >>> attributes["units"]
        'seconds since 1970-01-01 00:00:00'
        >>> attributes["calendar"]
        'proleptic_gregorian'

        ```
"""

from __future__ import annotations

#: The instant every pyramids time axis counts its offsets from.
CF_EPOCH = "1970-01-01 00:00:00"
#: Proleptic, not "standard": the mixed Julian/Gregorian calendar drops
#: 1582-10-05 through 1582-10-14 entirely, so a date in that window cannot be
#: represented and earlier dates shift by ten days.
CF_EPOCH_CALENDAR = "proleptic_gregorian"


def cf_epoch_units(resolution: str) -> str:
    """The CF `units` string for offsets counted from the pyramids epoch.

    The writers disagree, deliberately, on the resolution they count in: the
    collection axis uses `nanoseconds` because an `int64` count at that scale
    round-trips `datetime64[ns]` exactly, while the xarray interop path uses
    fractional `seconds` because it must also carry `NaT` as `NaN`. What they
    must not disagree on is the epoch, which is why only that half is shared.

    Args:
        resolution: A CF time unit name, e.g. `"nanoseconds"` or `"seconds"`.

    Returns:
        str: `"<resolution> since <epoch>"`.

    Note:
        Not every string this builds is readable by
        :func:`pyramids.netcdf.utils.decode_cf_time`. That reader delegates to
        `cftime`, whose finest unit is microseconds, so it raises `ValueError`
        on `"nanoseconds since ..."` -- which is exactly what
        `DatasetCollection.to_netcdf` writes, because an `int64` count of
        nanoseconds is what round-trips `datetime64[ns]` without loss. The
        reader that handles every resolution this produces is
        :func:`pyramids.netcdf.utils.create_time_conversion_func`.

        The docstring used to claim `decode_cf_time` parses all of them. It
        was never true, and the test that should have caught it asserted
        `cf_epoch_units(r) == f"{r} since {CF_EPOCH}"` against a function whose
        body is that f-string -- so it could not fail, whatever the string was
        for.

    Examples:
        - The collection axis counts in nanoseconds:
            ```python
            >>> from pyramids.base._cf_epoch import cf_epoch_units
            >>> cf_epoch_units("nanoseconds")
            'nanoseconds since 1970-01-01 00:00:00'

            ```
        - The interop path counts in seconds, from the same epoch -- which is
          the whole reason the epoch is shared and the resolution is not:
            ```python
            >>> from pyramids.base._cf_epoch import cf_epoch_units
            >>> cf_epoch_units("seconds")
            'seconds since 1970-01-01 00:00:00'

            ```
        - Whatever the resolution, the epoch behind it is the same one, which
          is what lets two axes written at different scales be compared:
            ```python
            >>> from pyramids.base._cf_epoch import cf_epoch_units
            >>> epochs = {
            ...     cf_epoch_units(res).split(" since ")[1]
            ...     for res in ("seconds", "nanoseconds", "days")
            ... }
            >>> sorted(epochs)
            ['1970-01-01 00:00:00']

            ```

    See Also:
        pyramids.netcdf.utils.create_time_conversion_func: Reads any of these
            strings back, nanoseconds included.
        pyramids.netcdf.utils.decode_cf_time: Reads them back through `cftime`,
            which stops at microsecond resolution -- see the note above.
    """
    return f"{resolution} since {CF_EPOCH}"


__all__ = ["CF_EPOCH", "CF_EPOCH_CALENDAR", "cf_epoch_units"]
