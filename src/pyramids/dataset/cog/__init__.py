"""COG — Cloud Optimized GeoTIFF read/write/validate support.

The `pyramids.dataset.cog` subpackage holds the raster-only COG
implementation: option serialization, a GDAL-driver write wrapper, and
a validation helper. User-facing methods such as
:meth:`pyramids.dataset.engines.COG.to_cog` and
:meth:`pyramids.dataset.engines.COG.validate_cog` live in
:mod:`pyramids.dataset.ops.cog` and delegate here.
"""

from __future__ import annotations

from pyramids.dataset.cog.options import (
    COG_DRIVER_OPTIONS,
    CreationOptions,
    merge_options,
    to_gdal_options,
    validate_blocksize,
    validate_option_keys,
)
from pyramids.dataset.cog.validate import ValidationReport, validate
from pyramids.dataset.cog.write import translate_to_cog

__all__ = [
    "COG_DRIVER_OPTIONS",
    "CreationOptions",
    "ValidationReport",
    "merge_options",
    "to_gdal_options",
    "translate_to_cog",
    "validate",
    "validate_blocksize",
    "validate_option_keys",
]
