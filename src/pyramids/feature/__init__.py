"""Feature subpackage.

Submodules:

- :mod:`pyramids.feature.collection` — the :class:`FeatureCollection`
  class (a :class:`geopandas.GeoDataFrame` subclass).
- :mod:`pyramids.feature.geometry` — shape factories and coordinate
  extractors.
- :mod:`pyramids.feature._ogr` — private OGR bridge (internal only).

CRS / EPSG / reprojection helpers live in :mod:`pyramids.base.crs`.
"""

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas
from pyramids.feature.collection import FeatureCollection

_LAZY_FC_INSTALL_HINT = (
    "LazyFeatureCollection requires the optional 'dask-geopandas' "
    "dependency. Install with: pip install 'pyramids-gis[parquet-lazy]'"
)

# LazyFeatureCollection is only available when the [parquet-lazy] extra
# is installed. Two design goals:
#
# 1. Eager import on `from pyramids.feature import LazyFeatureCollection`
# should give the user the real class (when available) or a clear
# ImportError with install instructions — NOT a silent `None` sentinel
# that later breaks `isinstance` with a confusing `TypeError`.
# 2. `import pyramids.feature` on a minimal install must not fail; the
# :class:`FeatureCollection` path has to keep working.
#
# The PEP-562 `__getattr__` hook satisfies both: attribute lookups for
# `LazyFeatureCollection` try the real import, and raise a branded
# :class:`ImportError` when dask-geopandas is absent.
# `hasattr(module, "LazyFeatureCollection")` returns False on minimal installs (because
# `__getattr__` raised), so library authors can guard with a clean
# `hasattr` check instead of an `isinstance` against `None`.
try:
    import_dask_geopandas(_LAZY_FC_INSTALL_HINT)
except OptionalPackageDoesNotExist:  # pragma: no cover - minimal install path
    _HAS_DASK_GEOPANDAS = False
else:
    _HAS_DASK_GEOPANDAS = True


def __getattr__(name: str) -> object:
    """PEP-562 hook — raise a clean ImportError for missing optional classes.

    Only `LazyFeatureCollection` is handled here. Any other unknown
    attribute falls through to the default :class:`AttributeError`.
    """
    if name == "LazyFeatureCollection":
        if _HAS_DASK_GEOPANDAS:
            from pyramids.feature._lazy_collection import LazyFeatureCollection

            return LazyFeatureCollection
        raise ImportError(_LAZY_FC_INSTALL_HINT)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def has_lazy_backend() -> bool:
    """Return True when the `[parquet-lazy]` extra is installed.

    public feature-detection helper that avoids the
    `try/except ImportError` dance for consumers that only need to
    know "can this session construct a :class:`LazyFeatureCollection`?"
    Cheap — just reads the module-level flag set at import time.

    Returns:
        bool: True iff :mod:`dask_geopandas` imported successfully
        when :mod:`pyramids.feature` was first loaded.

    Examples:
        - Branch on backend availability without importing the class:
            ```python
            >>> from pyramids.feature import has_lazy_backend
            >>> isinstance(has_lazy_backend(), bool)
            True

            ```
    """
    return _HAS_DASK_GEOPANDAS


def is_lazy_fc(obj: object) -> bool:
    """Return True if `obj` is a :class:`LazyFeatureCollection` instance.

    public dispatch helper for library code that consumes either
    eager or lazy FeatureCollections. Safe to call on minimal installs —
    returns False when :mod:`dask_geopandas` is absent (no object could
    be a LazyFC in that case). Equivalent of the eager-vs-lazy
    :func:`pyramids.base.protocols.is_lazy` helper that already exists
    for arrays.

    Args:
        obj: Any object.

    Returns:
        bool: True iff `obj` is a `LazyFeatureCollection`, False
        otherwise (including the "class not available" case).

    Examples:
        - An eager FeatureCollection is NOT a LazyFC:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature import FeatureCollection, is_lazy_fc
            >>> fc = FeatureCollection(gpd.GeoDataFrame(
            ...     {"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
            ... ))
            >>> is_lazy_fc(fc)
            False

            ```
        - Plain objects return False:
            ```python
            >>> from pyramids.feature import is_lazy_fc
            >>> is_lazy_fc("not a frame")
            False
            >>> is_lazy_fc(None)
            False

            ```
    """
    if not _HAS_DASK_GEOPANDAS:
        return False
    from pyramids.feature._lazy_collection import LazyFeatureCollection

    return isinstance(obj, LazyFeatureCollection)


__all__ = [
    "FeatureCollection",
    "LazyFeatureCollection",
    "has_lazy_backend",
    "is_lazy_fc",
]
