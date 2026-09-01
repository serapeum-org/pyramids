"""Resolve the GDAL output driver from a destination path.

The raster constructors used to take a `driver_type` alongside `path`, but the
two could only agree (noise) or disagree (an error state): with no `path` there
is nowhere to write, so MEM is forced regardless of what was asked, and with a
`path` the extension already names the format. Removing the parameter makes the
incoherent combination unrepresentable, which is better than validating it.

The extension is looked up in :class:`~pyramids.base._utils.Catalog`, the
package's existing driver table, rather than a hardcoded map. Two failures are
kept distinct, because they need different answers from the caller:

* the catalog has never heard of the extension -> :class:`DriverNotExistError`
* the driver exists but cannot ``Create`` -> :class:`FileFormatNotSupportedError`

The second case is real: `PNG` and `JP2OpenJPEG` are write-by-copy only, so a
constructor that builds with `Create` — `Dataset.from_array`,
`Dataset.create_empty` — cannot produce them. That is the *default* gate, not
the whole story: the package has as many write paths that go through
`CreateCopy` or `gdal.Translate` (`Dataset.copy`, `Dataset.to_file`,
`Vectorize.translate`, `Bands.change_no_data_value`, `IO.to_terrain_rgb`), and
those copy a copy-only format perfectly well. They pass `for_copy=True` to
skip the `Create` check while keeping the extension lookup and its
unknown-format error identical, so one resolver serves both kinds of writer and
a given path names the same format whichever one is used.
"""

from __future__ import annotations

from pathlib import Path

from pyramids.base._errors import DriverNotExistError, FileFormatNotSupportedError
from pyramids.base._utils import get_catalog

MEMORY_DRIVER = "MEM"

# The shared catalog. `Catalog.__init__` opens and parses the driver YAML, which
# costs ~17 ms — and this module sits on the write path of every disk-backed
# raster, so constructing one per call added that to each `to_file`-style
# operation (twice for `create_empty` / `empty_like`, which resolve the driver
# again to decide on the GTiff-only creation options). `get_catalog` is cached,
# so this instance is the same object `abstract_dataset.CATALOG` holds rather
# than a third parse of a file already in memory.
_CATALOG = get_catalog()


def copy_yields_writable(driver_name: str) -> bool:
    """Whether `CreateCopy` to this driver returns a handle that can be written.

    A driver that cannot `Create` builds its output in one shot and hands back
    a read-only dataset -- `PNG.CreateCopy` is the common case. Callers that
    label the result `access="write"` on that basis are wrong twice over: the
    package's own `ReadOnlyError` guard never fires, and the user gets a raw
    GDAL "attempt to write to dataset opened in read-only mode" instead.

    Args:
        driver_name: The GDAL driver short name, as returned by
            :func:`resolve_output_driver`.

    Returns:
        bool: `True` when the copy is writable. Unknown drivers answer `True`,
        which preserves the historical behaviour for anything the catalog does
        not describe.
    """
    key = _CATALOG.get_driver_name(driver_name)
    if key is None:
        # Unreachable from `resolve_output_driver`, whose return value always
        # names a catalogued driver. Kept for a caller that passes a GDAL name
        # from elsewhere: assuming writable preserves the behaviour every
        # caller had before this helper existed.
        return True
    entry = _CATALOG.get_driver(key) or {}
    return bool(entry.get("Creation"))


def resolve_output_driver(path: str | Path | None, *, for_copy: bool = False) -> str:
    """Return the GDAL driver name for a destination path.

    Args:
        path: Where the raster will be written, or `None` for an in-memory
            raster.
        for_copy: Whether the caller writes with `CreateCopy` rather than
            `Create`. The `Creation` flag records whether a driver supports
            `Create`, so a copy-based writer must not inherit that refusal:
            `PNG` and `JP2OpenJPEG` cannot be built band-by-band but copy
            perfectly well. Defaults to `False`, the stricter check.

    Returns:
        str: The GDAL driver short name — `"MEM"` when `path` is `None`,
            otherwise the driver the catalog associates with the extension
            (e.g. `"GTiff"` for `.tif`).

    Raises:
        TypeError: `path` is neither a `str` nor a `Path`.
        DriverNotExistError: `path` has no extension at all, or one the
            catalog does not know.
        FileFormatNotSupportedError: The format is write-by-copy only, so it
            cannot be built with `Create`. Never raised when `for_copy` is
            `True`.

    Examples:
        - No path means an in-memory raster:
            ```python
            >>> from pyramids.dataset._driver import resolve_output_driver
            >>> resolve_output_driver(None)
            'MEM'

            ```

        - The extension selects the driver:
            ```python
            >>> resolve_output_driver("out.tif")
            'GTiff'
            >>> resolve_output_driver("out.nc")
            'netCDF'

            ```

        - Sibling spellings of one format agree, because the catalog matches
          aliases as well as the canonical extension:
            ```python
            >>> resolve_output_driver("out.tiff")
            'GTiff'
            >>> resolve_output_driver("OUT.TIF")
            'GTiff'

            ```

        - A path with no suffix names nothing to resolve, and says so:
            ```python
            >>> resolve_output_driver("out")
            Traceback (most recent call last):
            pyramids.base._errors.DriverNotExistError: 'out' has no file extension, so the \
output format cannot be determined. Give the path a suffix naming the format (e.g. '.tif'), \
or pass path=None for an in-memory raster.

            ```

        - A copy-only format is refused up front rather than inside GDAL:
            ```python
            >>> resolve_output_driver("out.png")
            Traceback (most recent call last):
            pyramids.base._errors.FileFormatNotSupportedError: '.png' maps to the \
PNG driver, which cannot create a raster directly (write-by-copy only). Build the \
raster in memory or as GTiff, then convert.

            ```

        - A caller that writes with `CreateCopy` asks for the same lookup
          without the `Create` gate, so the same path resolves instead of
          raising:
            ```python
            >>> resolve_output_driver("out.png", for_copy=True)
            'PNG'

            ```

        - `for_copy` relaxes only that gate; an unknown extension is still an
          error either way:
            ```python
            >>> from pyramids.errors import DriverNotExistError
            >>> try:
            ...     resolve_output_driver("out.zzz", for_copy=True)
            ... except DriverNotExistError as error:
            ...     print(str(error).split(" is not")[0])
            The given extension: zzz

            ```
    """
    if path is None:
        driver = MEMORY_DRIVER
    else:
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"The path input should be string or Path type, given: {type(path)}"
            )
        extension = Path(path).suffix.lstrip(".").lower()
        if not extension:
            # Without a suffix there is nothing to resolve, and the generic
            # "the given extension:  is not associated with any driver"
            # message names an empty string. Say what is actually wrong.
            raise DriverNotExistError(
                f"'{path}' has no file extension, so the output format cannot "
                "be determined. Give the path a suffix naming the format "
                "(e.g. '.tif'), or pass path=None for an in-memory raster."
            )
        # Raises DriverNotExistError when the extension is unknown, which is the
        # right error for "pyramids has never heard of this format".
        key = _CATALOG.get_driver_name_by_extension(extension)
        entry = _CATALOG.get_driver(key)
        driver = str(entry["GDAL Name"])
        # `for_copy` selects a *different* capability, not the absence of one.
        # Skipping the check entirely let `.vrt` through every copy-based
        # writer, which then emitted a file GDAL could not reopen.
        if for_copy:
            # Unreachable through an extension today: the only `Copy: No` rows
            # (`hdf5`, `eedi`) carry no extension or alias, so the lookup above
            # can never return them. Kept because it is the question this
            # branch must ask -- if either row is ever given its real extension
            # (`.h5`), the refusal is already correct, and HDF5 is read-only in
            # GDAL so the message is too.
            if not entry.get("Copy"):
                raise FileFormatNotSupportedError(
                    f"'.{extension}' maps to the {driver} driver, which cannot be "
                    "written by copy. Write a GTiff and convert it with an external "
                    "tool."
                )
        elif not entry.get("Creation"):
            raise FileFormatNotSupportedError(
                f"'.{extension}' maps to the {driver} driver, which cannot create a "
                "raster directly (write-by-copy only). Build the raster in memory or "
                "as GTiff, then convert."
            )
        # Refused on both paths: a driver that writes a reference rather than
        # a self-contained raster produces a file that is unusable (VRT from
        # an in-memory source) or silently coupled to the source it points at.
        if entry.get("Self-contained") is False:
            raise FileFormatNotSupportedError(
                f"'.{extension}' maps to the {driver} driver, which writes a "
                "reference to other rasters rather than a self-contained file. "
                "Write a format that owns its pixels (e.g. '.tif')."
            )
    return driver


__all__ = ["MEMORY_DRIVER", "copy_yields_writable", "resolve_output_driver"]
