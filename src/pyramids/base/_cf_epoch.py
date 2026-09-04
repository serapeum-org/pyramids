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

Three constants and a formatter, with no netCDF dependency of their own.
"""

from __future__ import annotations

CF_EPOCH = "1970-01-01 00:00:00"
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
        str: `"<resolution> since <epoch>"`, in the form
            :func:`pyramids.netcdf.utils.decode_cf_time` parses.

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

    See Also:
        pyramids.netcdf.utils.decode_cf_time: Reads these strings back.
    """
    return f"{resolution} since {CF_EPOCH}"


__all__ = ["CF_EPOCH", "CF_EPOCH_CALENDAR", "cf_epoch_units"]
