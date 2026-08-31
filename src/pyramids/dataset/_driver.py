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

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.base._utils import Catalog

MEMORY_DRIVER = "MEM"


def resolve_output_driver(path: str | Path | None) -> str:
    """Return the GDAL driver name for a destination path.

    Args:
        path: Where the raster will be written, or `None` for an in-memory
            raster.

    Returns:
        str: The GDAL driver short name — `"MEM"` when `path` is `None`,
            otherwise the driver the catalog associates with the extension
            (e.g. `"GTiff"` for `.tif`).

    Raises:
        TypeError: `path` is neither a `str` nor a `Path`.
        DriverNotExistError: The extension is not in the catalog.
        FileFormatNotSupportedError: The format is write-by-copy only, so it
            cannot be built with `Create`.

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
        return MEMORY_DRIVER
    if not isinstance(path, (str, Path)):
        raise TypeError(
            f"The path input should be string or Path type, given: {type(path)}"
        )
    extension = Path(path).suffix.lstrip(".").lower()
    catalog = Catalog(raster_driver=True)
    # Raises DriverNotExistError when the extension is unknown, which is the
    # right error for "pyramids has never heard of this format".
    key = catalog.get_driver_name_by_extension(extension)
    entry = catalog.get_driver(key)
    gdal_name = entry["GDAL Name"]
    if not entry.get("Creation"):
        raise FileFormatNotSupportedError(
            f"'.{extension}' maps to the {gdal_name} driver, which cannot create a "
            "raster directly (write-by-copy only). Build the raster in memory or as "
            "GTiff, then convert."
        )
    return gdal_name


__all__ = ["MEMORY_DRIVER", "resolve_output_driver"]
