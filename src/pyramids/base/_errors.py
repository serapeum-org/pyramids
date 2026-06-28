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


class StacError(_PyramidsError):
    """Base class for errors raised from pyramids' STAC handling.

    Use to catch any STAC-side failure at once::

        try:
            load_asset(item, "B04")
        except StacError:
            ...
    """


class StacAssetError(StacError, KeyError):
    """A STAC asset is missing from an item, or carries no usable `href`.

    Raised by the duck-typed STAC accessors (:mod:`pyramids.stac._item`) when a
    requested asset key is absent or the asset has no href.

    Multi-inherits from :class:`KeyError` so existing `except KeyError:`
    handlers (and callers written before this class existed) keep working.
    """


class UnsupportedAssetError(StacError, ValueError):
    """No pyramids reader matches a STAC asset's media type / extension.

    Raised by :func:`pyramids.stac._loader._engine_for` when neither the
    asset's media type nor its href extension maps to a supported reader
    (GeoTIFF/COG, JPEG2000, NetCDF, GRIB, Zarr).

    Multi-inherits from :class:`ValueError` so existing `except ValueError:`
    handlers keep working.
    """


class WCSError(_PyramidsError):
    """A failure talking to an OGC Web Coverage Service (WCS).

    Raised by :meth:`pyramids.dataset.Dataset.from_wcs` (implementation in
    :mod:`pyramids.dataset._wcs`) when the server cannot be opened, the
    coverage cannot be fetched, or the server answers a ``GetCoverage`` with an
    ``<ows:ExceptionReport>`` / ``<ServiceExceptionReport>`` body instead of
    raster bytes. WCS servers commonly return such errors as **HTTP 200 +
    ``application/xml``**, so this is raised even when the transport succeeded.

    A *missing* coverage (one not advertised by ``GetCapabilities``) raises a
    plain :class:`ValueError` instead, mirroring how the rest of pyramids
    reports a bad argument as opposed to a service failure.
    """


class WFSError(_PyramidsError):
    """A failure talking to an OGC Web Feature Service (WFS).

    Raised by :meth:`pyramids.feature.FeatureCollection.from_wfs` (implementation
    in :mod:`pyramids.feature._wfs`) when the server cannot be reached, the
    feature type cannot be fetched, or the server answers with an
    ``<ows:ExceptionReport>`` / ``<ServiceExceptionReport>`` body instead of
    features. WFS servers commonly return such errors as **HTTP 200 +
    ``application/xml``**, so this is raised even when the transport succeeded.

    A *missing* feature type (one not advertised by ``GetCapabilities``) raises a
    plain :class:`ValueError` instead, mirroring how the rest of pyramids reports
    a bad argument as opposed to a service failure. This is the vector sibling of
    :class:`WCSError`.
    """


class OGCAPIError(_PyramidsError):
    """A failure talking to an **OGC API** service (Features, and later Coverages).

    Raised by :meth:`pyramids.feature.FeatureCollection.from_ogc_api_features`
    (implementation in :mod:`pyramids.feature._oapif`) when the service landing
    page / ``/collections`` document cannot be reached, returns a non-JSON or
    error body, or the items request fails. OGC API – Features is the modern
    REST/JSON successor to WFS, so this is the OGC-API-era sibling of
    :class:`WFSError`; the name is intentionally protocol-family-wide so the
    future OGC API – Coverages reader can share it.

    A *missing* collection (one not advertised by ``/collections``) raises a plain
    :class:`ValueError` instead, mirroring how the rest of pyramids reports a bad
    argument as opposed to a service failure.
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
