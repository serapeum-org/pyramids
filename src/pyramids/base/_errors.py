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


class GeolocationArrayError(_PyramidsError, ValueError):
    """The dataset cannot be warped from geolocation arrays.

    Raised by `Dataset.geolocate` / `Dataset.geolocation` when the dataset has no
    GDAL ``GEOLOCATION`` metadata domain, or the domain is missing the required
    ``X_DATASET`` / ``Y_DATASET`` coordinate arrays. Subclasses `ValueError` so an
    ``except ValueError`` still catches it.
    """


class OverviewTargetError(_PyramidsError, ValueError):
    """The dataset cannot hold the overview levels the call was asked to write.

    Raised by `Dataset.create_overviews`, `Dataset.recreate_overviews` and
    `Dataset.to_zarr(..., overview_factors=[...])` — which builds its pyramid through
    `create_overviews` — for conditions that no argument can fix:

    - a **plain VRT whose description is not a path**. A plain VRT owns no pixel storage,
      so GDAL can only write the levels to an external `.ovr` sidecar named after that
      description — with nothing usable to name it after, the levels would be stranded.
    - **levels a VRT computes rather than stores**, from `recreate_overviews` only: a
      warped VRT's `VRTWarpedRasterBand`s, or the levels a plain VRT inherits from the
      source it wraps. Neither is writable in any access mode, so neither can be
      regenerated in place. `create_overviews` still gives such a handle levels of its
      own.
    - **a stored level GDAL refuses on a handle already open for writing**, from
      `recreate_overviews` only: the levels are reached through a source GDAL opens
      read-only, so the access mode is not the blocker and there is no reopen to advise.

    Subclasses `ValueError`, which those methods already raise for bad arguments, so
    existing `except ValueError` handlers keep working. Catch this instead to tell the two
    apart: a bad argument is worth retrying with different arguments, whereas this one is a
    property of the dataset and needs a different target — `to_file(path)` first, or the
    raster that owns the levels::

        try:
            view.create_overviews(overview_levels=levels)
        except OverviewTargetError:
            view.to_file(path)
            view.close()
            saved = Dataset.read_file(path, read_only=False)
            saved.create_overviews(overview_levels=levels)
    """


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


class CoverageError(_PyramidsError):
    """A coverage's native CRS cannot be resolved.

    Raised by the protocol-neutral CRS resolver in
    :mod:`pyramids.base._coverage` when a coverage carries no spatial reference
    (the service advertises a CRS absent from the PROJ database) and no
    ``coverage_crs`` shim was supplied. It is the shared, protocol-agnostic error
    that each coverage reader re-wraps into its own branded error —
    :class:`WCSError` for :meth:`pyramids.dataset.Dataset.from_wcs`,
    :class:`OGCAPIError` for
    :meth:`pyramids.dataset.Dataset.from_ogc_coverages` — so neither reader has to
    import the other's internals.
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

    On a genuine HTTP-error status (a ``GetCapabilities`` / ``GetCoverage`` that
    returns 4xx/5xx), ``status_code`` carries the status and ``response_body`` the
    server's decoded body (the empty string when the error had no body) so a caller
    can branch on them programmatically. Both are ``None`` for failures that have no
    HTTP response (transport errors, XML ``ExceptionReport`` bodies returned as HTTP
    200, driver-level failures).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


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
    """A failure talking to an **OGC API** service (Features or Coverages).

    Raised by both OGC API readers:

    * :meth:`pyramids.feature.FeatureCollection.from_ogc_features` (implementation
      in :mod:`pyramids.feature._oapif`) — when the service landing page /
      ``/collections`` document cannot be reached, returns a non-JSON or error
      body, or the items request fails.
    * :meth:`pyramids.dataset.Dataset.from_ogc_coverages` (implementation in
      :mod:`pyramids.dataset._ogc_coverages`) — when the ``OGCAPI`` driver is
      unavailable, the coverage cannot be opened, or the windowed read returns an
      error / non-raster body.

    OGC API – Features and OGC API – Coverages are the modern REST/JSON successors
    to WFS and WCS, so this is the OGC-API-era sibling of :class:`WFSError` /
    :class:`WCSError`. The name is kept protocol-family-wide (``OGCAPIError``
    rather than a Features-only name) so every OGC API reader can reuse it.

    A *missing* collection / coverage (one not advertised by ``/collections``)
    raises a plain :class:`ValueError` instead, mirroring how the rest of pyramids
    reports a bad argument as opposed to a service failure.
    """


class VectorTileServerError(_PyramidsError):
    """A failure talking to an ArcGIS **VectorTileServer** endpoint.

    Raised by :meth:`pyramids.feature.FeatureCollection.from_vectortileserver`
    (implementation in :mod:`pyramids.feature._read`) when the service metadata
    (``<url>?f=json``) cannot be reached, returns a non-JSON body or one that does
    not describe a VectorTileServer (no ``tileInfo``), or a tile request fails at
    the transport level. A VectorTileServer serves Mapbox Vector Tiles (MVT) plus
    tiling-scheme metadata over REST, so this is the tile-based sibling of
    :class:`WFSError` (feature queries) — the reader that backs it is the vector-tile
    analogue of ``from_featureserver``.

    A *bad argument* (an unsupported tiling CRS, ``max_tiles < 1``, a ``zoom`` the
    service does not advertise, or a missing ``bbox`` with no service extent) raises
    a plain :class:`ValueError` instead, mirroring how the rest of pyramids reports a
    bad argument as opposed to a service failure.
    """


class WMSError(_PyramidsError):
    """A failure talking to an OGC Web Map Service (WMS) or Web Map Tile Service (WMTS).

    Raised by :meth:`pyramids.dataset.Dataset.from_wms` and
    :meth:`pyramids.dataset.Dataset.from_wmts` (implementation in
    :mod:`pyramids.dataset._wms`) when the server cannot be opened, the layer
    cannot be rendered / tiled, or the response is not a raster. WMS/WMTS are the
    OGC *map* services — they return a server-rendered image, so this is the
    imagery sibling of the coverage-data :class:`WCSError`.

    The name is kept family-wide (``WMSError`` rather than a separate
    ``WMTSError``) so both readers share one error, mirroring how
    :class:`OGCAPIError` covers OGC API – Features and – Coverages together.

    A *missing* WMTS layer (one not advertised by the capabilities document)
    raises a plain :class:`ValueError` instead, mirroring how the rest of pyramids
    reports a bad argument as opposed to a service failure.
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


class ContainerRasterWarning(UserWarning):
    """Pyramids-emitted warning that an opened raster is a subdataset container.

    Emitted by :meth:`pyramids.dataset.Dataset.read_file` when the path opens to a
    *container* — a raster with no bands of its own whose payload is a set of nested
    subdatasets (a NetCDF/HDF/Zarr store, a GRIB file, a WMS/WMTS endpoint, a
    Sentinel-1/-2 product). Without it, such an open returns a silent 0-band
    ``Dataset`` that fails much later somewhere unrelated; the warning names the
    subdatasets and points at :attr:`pyramids.dataset.Dataset.subdatasets` /
    :meth:`pyramids.dataset.Dataset.open_subdataset`.

    Users who open containers on purpose can silence just this category, or pass
    ``warn_on_container=False`` to :meth:`~pyramids.dataset.Dataset.read_file`::

        import warnings
        from pyramids.errors import ContainerRasterWarning
        warnings.filterwarnings("ignore", category=ContainerRasterWarning)
    """


class DtypeNarrowingWarning(UserWarning):
    """Pyramids-emitted warning that a write will not preserve the band dtype.

    Emitted by :meth:`pyramids.dataset.Dataset.to_file` when the driver it
    resolves for the destination — from the path extension, or from an explicit
    `driver=` — cannot store the source dtype. GDAL's `CreateCopy`
    substitutes the nearest type it does support rather than refusing — a
    float32 DEM written to `.png` becomes 8-bit `Byte`, so values are
    clipped and every fractional part is lost — and reports it only as a GDAL
    `RuntimeWarning`, which a caller filtering on their own warning categories
    never sees.

    Capability is measured, not read off the driver's advertised
    `DMD_CREATIONDATATYPES`: that list is not exhaustive -- this build's GTiff
    omits `Int64` and stores it faithfully -- and trusting it warned about
    `int64` written to `.tif`, the commonest write in the library and a
    lossless one. Every band is checked, so a mixed-dtype dataset whose
    narrowing band is not the first is still reported.

    Two paths sit outside the check by construction: an `.asc` destination
    never reaches a GDAL driver (the ascii writer emits full precision through
    `str()`), and `driver="COG"` delegates to
    :meth:`pyramids.dataset.Dataset.to_cog` before the check runs.

    Writing an 8-bit image to PNG or JPEG is a legitimate thing to do, so this
    warns rather than raising. Silence it once the conversion is deliberate::

        import warnings
        from pyramids.errors import DtypeNarrowingWarning
        warnings.filterwarnings("ignore", category=DtypeNarrowingWarning)
    """
