"""Custom Errors."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _PyramidsError(Exception):
    """Base class for all pyramids exceptions.

    Logs the error message at DEBUG level on construction for traceability,
    even when the exception is caught and handled. DEBUG is hidden by default
    and only appears when verbose logging is enabled.
    """

    def __init__(self, message: str):
        super().__init__(message)
        logger.debug(f"{type(self).__name__}: {message}")


class ReadOnlyError(_PyramidsError):
    """ReadOnlyError."""


class DatasetNotFoundError(_PyramidsError):
    """DatasetNotFoundError."""


class NoDataValueError(_PyramidsError):
    """NoDataValueError."""


class AlignmentError(_PyramidsError):
    """Alignment Error."""


class DriverNotExistError(_PyramidsError):
    """Driver-Not-exist Error."""


class FileFormatNotSupportedError(_PyramidsError):
    """File Format Not Supported."""


class OptionalPackageDoesNotExist(_PyramidsError, ImportError):
    """Optional Package does not exist.

    Inherits from both `_PyramidsError` (for pyramids-branded handling)
    and `ImportError` (so `except ImportError` callers — including
    standard-library and third-party code — still catch it).
    """


class FailedToSaveError(_PyramidsError):
    """Failed to save error."""


class OutOfBoundsError(_PyramidsError):
    """Out-of-bounds error."""


class FeatureError(_PyramidsError):
    """Base class for errors raised from :mod:`pyramids.feature`.

    Use to catch any vector-side failure at once::

        try:
            fc.rasterize(...)
        except FeatureError:
            ...
    """


class InvalidGeometryError(FeatureError, ValueError):
    """A geometry is empty, malformed, or has the wrong type.

    Raised e.g. when :func:`pyramids.feature.geometry.get_coords` is
    handed a `MultiPolygon`.

    Multi-inherits from :class:`ValueError` so `except ValueError:`
    handlers keep working.
    """


class CRSError(FeatureError, ValueError):
    """CRS is missing, ambiguous, or cannot be resolved.

    Raised e.g. when :func:`pyramids.base.crs.get_epsg_from_prj`
    receives an empty projection string, or when a rasterize
    template's CRS disagrees with the vector's.

    Multi-inherits from :class:`ValueError` so `except ValueError:`
    handlers keep working.
    """


class VectorDriverError(FeatureError, RuntimeError):
    """A vector-driver-level failure.

    Raised when an internal OGR operation reports failure —
    unknown driver, `VectorTranslate` returning `None`, layer
    not found, creation option rejected.

    Multi-inherits from :class:`RuntimeError` so `except
    RuntimeError:` handlers keep working.
    """


class GeometryWarning(UserWarning):
    """Pyramids-emitted warning about geometry validity / degeneracy.

    emitted by :meth:`pyramids.feature.FeatureCollection.with_centroid`
    and other geometry-handling methods when an input is degenerate
    (empty geometry, NaN coordinates, zero-area ring) and the method
    recovers via a documented fallback rather than raising.

    Users can suppress just this category without silencing every
    pyramids / geopandas / shapely `UserWarning`::

        import warnings
        from pyramids.base._errors import GeometryWarning
        warnings.filterwarnings("ignore", category=GeometryWarning)
    """
