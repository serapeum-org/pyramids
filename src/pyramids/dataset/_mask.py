"""Per-band mask-flag value object.

Every GDAL raster band has a mask whose *nature* is described by a bitmask of
``GMF_*`` flags. :class:`MaskFlags` decodes that bitmask into four booleans so a
caller can tell *why* a band is (or is not) masked — e.g. an explicit no-data
value vs. an alpha band vs. a fully-valid band.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskFlags:
    """The decoded GDAL mask flags of a raster band.

    Args:
        all_valid: Every pixel is valid; the band has no mask (``GMF_ALL_VALID``).
        per_dataset: The mask is shared by every band of the dataset
            (``GMF_PER_DATASET``).
        alpha: The mask comes from an alpha band (``GMF_ALPHA``).
        nodata: The mask is derived from the band's no-data value
            (``GMF_NODATA``).

    Examples:
        - A fully-valid band:
            ```python
            >>> from pyramids.dataset._mask import MaskFlags
            >>> flags = MaskFlags(all_valid=True, per_dataset=False, alpha=False, nodata=False)
            >>> flags.all_valid
            True

            ```
    """

    all_valid: bool
    per_dataset: bool
    alpha: bool
    nodata: bool
