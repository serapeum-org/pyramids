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

The second case is real: `PNG` and `JP2OpenJPEG` are write-by-copy only, and
the constructors here build with ``Create``.
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
        if not for_copy and not entry.get("Creation"):
            raise FileFormatNotSupportedError(
                f"'.{extension}' maps to the {driver} driver, which cannot create a "
                "raster directly (write-by-copy only). Build the raster in memory or "
                "as GTiff, then convert."
            )
    return driver


__all__ = ["MEMORY_DRIVER", "resolve_output_driver"]
