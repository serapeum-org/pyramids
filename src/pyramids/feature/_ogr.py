"""Private OGR bridge for :mod:`pyramids.feature`.

This module is **not part of the public API**. It exists solely so that
`FeatureCollection` methods that internally need a GDAL/OGR object
(for example :func:`osgeo.gdal.Rasterize` or :func:`osgeo.gdal.Warp` with
`cutlineDSName`) can obtain one without leaking `ogr.DataSource` or
`gdal.Dataset` into the package's public surface.

All helpers in this module are context-managed where possible so that the
backing `/vsimem/` file and the OGR handle are deterministically
released when the `with` block exits.

Do not import this module from user code; its signatures are unstable.
"""

from __future__ import annotations

import io
import random
import time
from contextlib import contextmanager
from typing import Iterator

import geopandas as gpd
from geopandas import GeoDataFrame
from osgeo import gdal, ogr

from pyramids.base._errors import VectorDriverError


def _new_vsimem_path() -> str:
    """Return a fresh unique `/vsimem/` path for a GeoJSON serialization.

    The suffix is `<time_ns>_<rand>` — shorter than a UUID4 and
    still collision-proof within a single process run: `time.time_ns`
    has nanosecond resolution and the random integer adds 20 bits of
    tie-breaking for the same-nanosecond case.

    Returns:
        str: A `/vsimem/<time>_<rand>.geojson` path.

    Examples:
        - Check the shape of a freshly generated path:
            ```python
            >>> from pyramids.feature._ogr import _new_vsimem_path
            >>> path = _new_vsimem_path()
            >>> path.startswith("/vsimem/")
            True
            >>> path.endswith(".geojson")
            True

            ```
        - Two successive calls return distinct paths so concurrent
          conversions cannot clobber each other:
            ```python
            >>> from pyramids.feature._ogr import _new_vsimem_path
            >>> p1 = _new_vsimem_path()
            >>> p2 = _new_vsimem_path()
            >>> p1 != p2
            True

            ```
    """
    return f"/vsimem/{time.time_ns()}_{random.randint(0, 999_999)}.geojson"


@contextmanager
def as_datasource(
    gdf: GeoDataFrame, *, gdal_dataset: bool = False
) -> Iterator[ogr.DataSource | gdal.Dataset]:
    """Yield a short-lived OGR `DataSource` for a GeoDataFrame.

    The DataSource (or `gdal.Dataset` with OGR contents when
    `gdal_dataset=True`) is backed by an in-memory `/vsimem/` GeoJSON
    file that is unlinked when the context exits. The caller must use the
    yielded object only inside the `with` block — storing it past the
    block is a use-after-free.

    Args:
        gdf (GeoDataFrame):
            The GeoDataFrame to expose as an OGR DataSource. Any subclass
            of :class:`geopandas.GeoDataFrame` is accepted — in particular
            `pyramids.feature.FeatureCollection` works unchanged since
            it IS a `GeoDataFrame`.
        gdal_dataset (bool):
            When `True`, yield a `gdal.Dataset` (opened via
            :func:`gdal.OpenEx`) instead of an `ogr.DataSource`. This
            is the form required by :func:`gdal.Rasterize` when the
            vector argument is a Python GDAL object rather than a path.

    Yields:
        ogr.DataSource | gdal.Dataset: A short-lived handle to the vector
        data. Do not store past the `with` block.

    Raises:
        VectorDriverError: If :func:`gdal.OpenEx` / :func:`ogr.Open`
            returns `None` after the in-memory GeoJSON was written —
            usually the GeoDataFrame has malformed geometry or an
            unsupported CRS. The message includes the `/vsimem/`
            path for debugging.

    Notes:
        Cleanup is exception-safe. If `gdf.to_json` or
        :func:`gdal.FileFromMemBuffer` raises before the in-memory file
        is written, the `finally` block does **not** call
        :func:`gdal.Unlink` on a non-existent path (tracked via an
        internal `file_written` flag). When the user's `with` body
        raises, the path is still unlinked exactly once.

    Examples:
        - Open a GeoDataFrame as an OGR DataSource inside a `with`
          block, and confirm the yielded handle exposes one layer with
          the feature count the GDF had:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature._ogr import as_datasource
            >>> gdf = gpd.GeoDataFrame(
            ...     {"v": [1, 2]},
            ...     geometry=[Point(0, 0), Point(1, 1)],
            ...     crs="EPSG:4326",
            ... )
            >>> with as_datasource(gdf) as ds:
            ...     layer = ds.GetLayer(0)
            ...     n_features = layer.GetFeatureCount()
            >>> n_features
            2

            ```
        - Request a `gdal.Dataset` instead of an `ogr.DataSource`
          (the form :func:`gdal.Rasterize` expects when its vector
          argument is a Python GDAL object rather than a file path):
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from osgeo import gdal
            >>> from pyramids.feature._ogr import as_datasource
            >>> gdf = gpd.GeoDataFrame(
            ...     {"v": [10]}, geometry=[Point(0, 0)], crs="EPSG:4326"
            ... )
            >>> with as_datasource(gdf, gdal_dataset=True) as ds:
            ...     kind = isinstance(ds, gdal.Dataset)
            >>> kind
            True

            ```
    """
    mem_path = _new_vsimem_path()
    # We must write into osgeo.gdal's own /vsimem/ — geopandas' default
    # pyogrio engine uses its own bundled GDAL with a separate VFS, so a
    # `gdf.to_file("/vsimem/…")` would write to *that* engine's memory
    # store and `osgeo.gdal` would never see it. Round-tripping through
    # the GeoJSON serialization + `gdal.FileFromMemBuffer` guarantees
    # the file lands in the GDAL VFS we can open.
    # track whether the vsimem file was actually written so the
    # finally block only unlinks when there is something to unlink.
    # If `gdf.to_json()` or `FileFromMemBuffer` raises, no file
    # exists on /vsimem/ and `gdal.Unlink` would log a spurious
    # warning about a missing path.
    geojson_bytes = gdf.to_json().encode("utf-8")
    file_written = False
    try:
        gdal.FileFromMemBuffer(mem_path, geojson_bytes)
        file_written = True
        ds: ogr.DataSource | gdal.Dataset | None = (
            gdal.OpenEx(mem_path) if gdal_dataset else ogr.Open(mem_path)
        )
        # GDAL signals a failure to parse the in-memory GeoJSON by
        # returning `None` rather than raising. Convert that to an
        # explicit :class:`VectorDriverError` so callers see a typed
        # failure instead of cryptic `AttributeError: 'NoneType'`
        # deeper in the stack.
        if ds is None:
            raise VectorDriverError(
                f"GDAL/OGR could not open the in-memory GeoJSON at "
                f"{mem_path!r}. The GeoDataFrame may have malformed "
                f"geometry or an unsupported CRS."
            )
        try:
            yield ds
        finally:
            ds = None
    finally:
        if file_written:
            gdal.Unlink(mem_path)


@contextmanager
def as_vsimem_path(gdf: GeoDataFrame) -> Iterator[str]:
    """Yield a `/vsimem/` path to a GeoJSON serialization of `gdf`.

    Useful where a GDAL API needs a *path string* (e.g. the
    `cutlineDSName` option of :func:`gdal.Warp`) rather than a Python
    GDAL object. The path is unlinked on exit.

    Args:
        gdf (GeoDataFrame):
            The GeoDataFrame to serialize.

    Yields:
        str: A `/vsimem/<uuid>.geojson` path valid only inside the
        `with` block.

    Examples:
        - Confirm the yielded path has the expected shape and that the
          backing GeoJSON file exists for the duration of the `with`
          block (opened via :func:`osgeo.ogr.Open`):
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from osgeo import ogr
            >>> from pyramids.feature._ogr import as_vsimem_path
            >>> gdf = gpd.GeoDataFrame(
            ...     {"id": [7]}, geometry=[Point(0, 0)], crs="EPSG:4326"
            ... )
            >>> with as_vsimem_path(gdf) as path:
            ...     prefix_ok = path.startswith("/vsimem/")
            ...     ds = ogr.Open(path)
            ...     n = ds.GetLayer(0).GetFeatureCount()
            ...     ds = None
            >>> prefix_ok, n
            (True, 1)

            ```
    """
    mem_path = _new_vsimem_path()
    # See the note in `as_datasource` for why we use
    # `gdal.FileFromMemBuffer` instead of `gdf.to_file`.
    geojson_bytes = gdf.to_json().encode("utf-8")
    gdal.FileFromMemBuffer(mem_path, geojson_bytes)
    try:
        yield mem_path
    finally:
        gdal.Unlink(mem_path)


def datasource_to_gdf(ds: ogr.DataSource | gdal.Dataset) -> GeoDataFrame:
    """Materialize an OGR `DataSource` into a `GeoDataFrame`.

    Used by internal operations (for example :func:`gdal.Polygonize`) that
    begin by allocating an OGR DataSource and need to hand a
    `GeoDataFrame` back to the public layer. The conversion goes via a
    `/vsimem/` GeoJSON round-trip using :func:`gdal.VectorTranslate`.

    D-M4: the round-trip stays entirely in osgeo.gdal's `/vsimem/`
    VFS — no filesystem temp file is created. The serialised GeoJSON
    bytes are read back via :func:`gdal.VSIFOpenL` /
    :func:`gdal.VSIFReadL` and parsed from an in-memory
    :class:`io.BytesIO` via :func:`geopandas.read_file` (pyogrio
    accepts buffer inputs). Cleanup of the `/vsimem/` path is gated
    on the write having actually succeeded, so a failed
    `VectorTranslate` raises :class:`VectorDriverError` cleanly
    without `gdal.Unlink` masking the original error.

    Args:
        ds (ogr.DataSource | gdal.Dataset):
            The source DataSource to materialize. Not consumed; the
            caller retains ownership.

    Returns:
        GeoDataFrame: A plain `GeoDataFrame` (never a
        `FeatureCollection`) containing the layer's features. Callers
        that want a `FeatureCollection` should wrap the result:
        `FeatureCollection(datasource_to_gdf(ds))`.

    Raises:
        VectorDriverError: If :func:`gdal.VectorTranslate` fails to
            write the intermediate GeoJSON, or if the subsequent
            :func:`gdal.VSIFOpenL` cannot open the `/vsimem/` path.
            Multi-inherits from `RuntimeError` so existing
            `except RuntimeError` handlers keep working.

    Examples:
        - Round-trip a GeoDataFrame through the OGR bridge: first open
          it as an in-memory OGR `DataSource` via :func:`as_datasource`,
          then materialize it back to a `GeoDataFrame` via
          :func:`datasource_to_gdf`. Attribute and row counts survive:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature._ogr import (
            ...     as_datasource,
            ...     datasource_to_gdf,
            ... )
            >>> gdf = gpd.GeoDataFrame(
            ...     {"score": [10, 20, 30]},
            ...     geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            ...     crs="EPSG:4326",
            ... )
            >>> with as_datasource(gdf) as ds:
            ...     back = datasource_to_gdf(ds)
            >>> len(back)
            3
            >>> sorted(back["score"].tolist())
            [10, 20, 30]

            ```
    """
    # D-M4: the previous implementation wrote to a filesystem temp
    # file because pyogrio's bundled GDAL reads from its own VFS, not
    # osgeo.gdal's /vsimem/. Per-call filesystem I/O is a real cost
    # on heavy polygonize workloads. Instead: VectorTranslate into
    # osgeo /vsimem/, read the bytes back out via GDAL's own VSIFile*
    # APIs, and hand a `BytesIO` to `geopandas.read_file` — which
    # pyogrio accepts and parses from memory.
    mem_path = _new_vsimem_path()
    file_written = False
    try:
        result = gdal.VectorTranslate(mem_path, ds, format="GeoJSON")
        if result is None:
            raise VectorDriverError(
                "gdal.VectorTranslate failed to materialize the DataSource "
                "to GeoJSON."
            )
        file_written = True
        # Drop the translation handle before reading the /vsimem/ file
        # so GDAL flushes buffered output.
        result = None
        vsi_file = gdal.VSIFOpenL(mem_path, "rb")
        if vsi_file is None:
            raise VectorDriverError(
                f"GDAL could not open the in-memory GeoJSON at "
                f"{mem_path!r} for reading."
            )
        try:
            gdal.VSIFSeekL(vsi_file, 0, 2)  # SEEK_END
            size = gdal.VSIFTellL(vsi_file)
            gdal.VSIFSeekL(vsi_file, 0, 0)
            # gdal.VSIFReadL on GDAL >=3 already returns a Python
            # bytes object; wrapping in `bytes(...)` forced a
            # defensive O(size) copy that doubled peak memory on
            # polygonize outputs. Use the buffer as-is.
            data = gdal.VSIFReadL(1, size, vsi_file)
        finally:
            gdal.VSIFCloseL(vsi_file)
        gdf = gpd.read_file(io.BytesIO(data))
    finally:
        # Under gdal.UseExceptions(), Unlink on a non-existent path
        # raises RuntimeError and would mask whatever exception we
        # raised above. Gate the cleanup on the write succeeding.
        if file_written:
            gdal.Unlink(mem_path)
    return gdf
