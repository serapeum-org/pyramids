"""Argument checks shared by every engine in :mod:`pyramids.dataset.engines`.

Each of these existed in several places at once, hand-rolled per call site
with the same message spelled slightly differently. That is how the band-range
check ended up written seven times and the window-bounds message four, one of
them missing a space. A check that lives in one place is a check that reads the
same everywhere, and can be corrected once.

Only genuinely repeated shapes belong here. Errors whose message names the
specific condition that failed -- "the bbox does not intersect the raster",
"N of M points fall outside the extent" -- are deliberately left where they are
raised: flattening them into a shared helper would trade an informative message
for a generic one.
"""

from __future__ import annotations

from osgeo import gdal

from pyramids.base._errors import OutOfBoundsError


def validate_band_index(
    band: int | None, band_count: int, *, name: str = "band", hint: str = ""
) -> None:
    """Reject a band index outside a dataset's band range.

    Args:
        band: Zero-based band index. `None` passes -- callers that accept it
            resolve their own default afterwards.
        band_count: Number of bands on the dataset.
        name: What to call the argument in the message. Defaults to `"band"`;
            pass the parameter's own name when a method takes more than one
            (e.g. `"u_band"`).
        hint: Optional sentence appended to the message, for callers that can
            say something useful about why they need the band.

    Raises:
        ValueError: `band` is negative or not below `band_count`.

    Examples:
        - A band inside the range passes silently, one outside does not:
            ```python
            >>> from pyramids.dataset.engines._validate import validate_band_index
            >>> validate_band_index(0, 3)
            >>> validate_band_index(3, 3)
            Traceback (most recent call last):
            ValueError: band 3 is out of range for a 3-band dataset.

            ```
    """
    if band is not None and (band < 0 or band >= band_count):
        message = f"{name} {band} is out of range for a {band_count}-band dataset."
        if name != "band":
            message = f"{name}={band} is out of range for a {band_count}-band dataset."
        raise ValueError(f"{message}{hint}")


def resolve_band_indices(
    bands: int | list[int] | None, band_count: int
) -> tuple[list[int], bool]:
    """Normalise a `bands=` argument to a validated list of indices.

    Args:
        bands: `None` for every band, an `int` for one, or a sequence of them.
        band_count: Number of bands on the dataset.

    Returns:
        tuple[list[int], bool]: The resolved indices, and whether the caller
            asked for a single band (so the result should be squeezed).

    Raises:
        ValueError: Any index is outside the dataset's band range.

    Examples:
        - `None` means every band and does not squeeze; a bare `int` squeezes:
            ```python
            >>> from pyramids.dataset.engines._validate import resolve_band_indices
            >>> resolve_band_indices(None, 3)
            ([0, 1, 2], False)
            >>> resolve_band_indices(1, 3)
            ([1], True)

            ```
    """
    if bands is None:
        band_list = list(range(band_count))
        squeeze = False
    elif isinstance(bands, int):
        band_list = [bands]
        squeeze = True
    else:
        band_list = list(bands)
        squeeze = False
    for band in band_list:
        validate_band_index(band, band_count)
    return band_list, squeeze


def world_to_pixel(
    geotransform: tuple[float, ...], x: float, y: float
) -> tuple[float, float]:
    """Convert a world coordinate to a fractional pixel coordinate.

    Inverts the geotransform and applies it, which is the only correct way to
    do this on a rotated or sheared grid -- dividing by `gt[1]`/`gt[5]` ignores
    the `gt[2]`/`gt[4]` rotation terms.

    Args:
        geotransform: The dataset's GDAL geotransform.
        x: World X coordinate.
        y: World Y coordinate.

    Returns:
        tuple[float, float]: `(column, row)`, fractional and unclamped -- a
            point outside the raster gives a negative or over-range value, which
            is the caller's to interpret.

    Examples:
        - The origin of a north-up grid maps to pixel `(0, 0)`, and one cell
          east and south maps to `(1, 1)`:
            ```python
            >>> from pyramids.dataset.engines._validate import world_to_pixel
            >>> gt = (10.0, 2.0, 0.0, 50.0, 0.0, -2.0)
            >>> world_to_pixel(gt, 10.0, 50.0)
            (0.0, 0.0)
            >>> world_to_pixel(gt, 12.0, 48.0)
            (1.0, 1.0)

            ```
    """
    inverse = gdal.InvGeoTransform(geotransform)
    # ApplyGeoTransform returns a list; a tuple matches the annotation and makes
    # the result safe to use as a dict key or compare against a literal.
    column, row = gdal.ApplyGeoTransform(inverse, x, y)
    return column, row


def window_out_of_bounds(window: object, rows: int, columns: int) -> OutOfBoundsError:
    """Build the standard out-of-bounds error for a read window.

    Returns the exception rather than raising it so a caller translating a GDAL
    failure can chain it -- `raise window_out_of_bounds(...) from exc` -- which
    a helper that raised internally could not express.

    Args:
        window: Whatever the caller was given, rendered into the message as-is.
        rows: The dataset's row count.
        columns: The dataset's column count.

    Returns:
        OutOfBoundsError: Ready to raise. The caller has already decided the
            window is outside the raster; this only words it.

    Examples:
        - The message names the window and the raster it did not fit:
            ```python
            >>> from pyramids.dataset.engines._validate import window_out_of_bounds
            >>> raise window_out_of_bounds([0, 0, 99, 99], 10, 10)
            Traceback (most recent call last):
            pyramids.base._errors.OutOfBoundsError: The window you entered ([0, 0, 99, 99]) is out of the raster bounds: (10, 10)

            ```
    """
    return OutOfBoundsError(
        f"The window you entered ({window}) is out of the raster "
        f"bounds: {rows, columns}"
    )


__all__ = [
    "resolve_band_indices",
    "validate_band_index",
    "window_out_of_bounds",
    "world_to_pixel",
]
