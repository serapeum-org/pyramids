"""COG — Cloud Optimized GeoTIFF read/write/validate support.

The `pyramids.dataset.cog` subpackage holds the raster-only COG
implementation: option serialization, a GDAL-driver write wrapper, a
validation helper, and the high-level :func:`write_cog` facade. The
user-facing methods :meth:`pyramids.dataset.engines.cog.COG.to_cog` and
:meth:`pyramids.dataset.engines.cog.COG.validate_cog` live in
:mod:`pyramids.dataset.engines.cog` and delegate here.
"""

from __future__ import annotations

from pyramids.dataset.cog.facade import PYRAMIDS_COG_DEFAULTS, write_cog
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
    "PYRAMIDS_COG_DEFAULTS",
    "ValidationReport",
    "merge_options",
    "to_gdal_options",
    "translate_to_cog",
    "validate",
    "validate_blocksize",
    "validate_option_keys",
    "write_cog",
]
