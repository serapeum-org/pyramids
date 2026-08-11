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
import itertools
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager

import geopandas as gpd
from geopandas import GeoDataFrame
from osgeo import gdal, ogr
from pyproj.exceptions import CRSError as _PyprojCRSError

from pyramids.base._errors import VectorDriverError
from pyramids.base.crs import crs_from_user_input

# Process-wide monotonic counter guaranteeing `/vsimem/` path uniqueness;
# see the note in `pyramids._io`. `next()` on an itertools.count is atomic
# under the GIL.
_VSIMEM_COUNTER = itertools.count()


def _new_vsimem_path() -> str:
    """Return a fresh unique `/vsimem/` path for a GeoJSON serialization.

    The suffix is `<time_ns>_<counter>` — shorter than a UUID4 and
    collision-proof within a single process run: the strictly increasing
    counter guarantees uniqueness even when `time.time_ns` repeats within
    a clock tick.

    Returns:
        str: A `/vsimem/<time>_<counter>.geojson` path.

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
    return f"/vsimem/{time.time_ns()}_{next(_VSIMEM_COUNTER)}.geojson"


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


def _source_layer_wkt(ds: ogr.DataSource | gdal.Dataset) -> str:
    """WKT of the datasource's first layer, or ``""`` when it declares none.

    Args:
        ds: The datasource being materialized.

    Returns:
        str: The layer's spatial reference as WKT, or ``""``.
    """
    wkt = ""
    try:
        layer = ds.GetLayer(0)
        srs = None if layer is None else layer.GetSpatialRef()
        wkt = "" if srs is None else srs.ExportToWkt()
    except (RuntimeError, AttributeError):
        wkt = ""
    return wkt


_GEOJSON_HEADER_SCAN = 4096
"""Bytes of a GeoJSON document searched for the top-level ``crs`` member.

GDAL writes `type`, `name` then `crs` before the first feature, so the member is
always within the first few hundred bytes. Bounding the scan keeps the cost independent
of the document's size — the whole point of not parsing it.
"""


def _strip_geojson_crs(data: bytes) -> bytes | None:
    """Drop the top-level ``crs`` member from GDAL's GeoJSON without parsing it.

    Decoding and re-encoding the document to remove one key costs a full parse, a full
    re-serialisation and several times the document in peak memory — on a polygonize
    output that is the largest allocation in the pipeline, and the read path a few
    lines above deliberately avoids even *one* extra copy of the same buffer.

    The member sits in the header, so it can be excised by locating it and copying the
    two surrounding slices. The scan tracks quoted strings so a brace inside a CRS name
    cannot unbalance it, and gives up (returning ``None``) on anything it does not
    recognise rather than guessing at a malformed document.

    Args:
        data: The GeoJSON bytes GDAL wrote.

    Returns:
        bytes | None: The document without its ``crs`` member, or ``None`` when the
        member is absent or not laid out as expected.
    """
    head = bytes(data[:_GEOJSON_HEADER_SCAN])
    key = head.find(b'"crs"')
    if key == -1:
        return None
    colon = head.find(b":", key + 5)
    if colon == -1:
        return None
    start_of_value = colon + 1
    while start_of_value < len(head) and head[start_of_value : start_of_value + 1].isspace():
        start_of_value += 1
    if head[start_of_value : start_of_value + 1] != b"{":
        return None

    depth, in_string, escaped, end = 0, False, False, -1
    for index in range(start_of_value, len(head)):
        char = head[index : index + 1]
        if in_string:
            if escaped:
                escaped = False
            elif char == b"\\":
                escaped = True
            elif char == b'"':
                in_string = False
            continue
        if char == b'"':
            in_string = True
        elif char == b"{":
            depth += 1
        elif char == b"}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end == -1:
        return None

    # Absorb one separating comma so what remains is still valid JSON: the one after
    # the member, or -- if `crs` was last -- the one before it.
    cut_start, cut_end = key, end
    while cut_end < len(head) and head[cut_end : cut_end + 1].isspace():
        cut_end += 1
    if head[cut_end : cut_end + 1] == b",":
        cut_end += 1
    else:
        preceding = cut_start - 1
        while preceding >= 0 and head[preceding : preceding + 1].isspace():
            preceding -= 1
        if preceding >= 0 and head[preceding : preceding + 1] == b",":
            cut_start = preceding
    return bytes(data[:cut_start]) + bytes(data[cut_end:])


def _read_geojson_bytes(data: bytes, ds: ogr.DataSource | gdal.Dataset) -> GeoDataFrame:
    """Parse GDAL's GeoJSON output, surviving a CRS pyproj cannot look up.

    GDAL names the CRS in the GeoJSON as an authority URN
    (``urn:ogc:def:crs:EPSG::10857``), and geopandas resolves that URN through pyproj.
    When the code lives in GDAL's PROJ database but not pyproj's, the read raises
    before any geometry is returned — so `to_polygons` and `footprint` failed on
    exactly the rasters issue #943 is about, even though the CRS is fine and GDAL
    just wrote it.

    The normal path is untouched: parse the bytes as they are. Only when that fails
    on the CRS is the ``crs`` member dropped and the geometry re-read, with the
    authoritative WKT taken straight from the source layer and attached afterwards.
    That fallback costs a JSON round trip, which is why it is not the default.

    Args:
        data: The GeoJSON bytes GDAL wrote.
        ds: The source datasource, consulted only for its layer's WKT on the
            fallback path.

    Returns:
        GeoDataFrame: The parsed features, carrying the source CRS.
    """
    try:
        gdf = gpd.read_file(io.BytesIO(data))
    except _PyprojCRSError:
        wkt = _source_layer_wkt(ds)
        if not wkt:
            # Nothing authoritative to substitute, so the original failure stands.
            raise
        stripped = _strip_geojson_crs(data)
        if stripped is None:
            # The member is not where GDAL puts it. Fall back to a full parse, which
            # is slow but cannot be confused by an unexpected layout.
            document = json.loads(data)
            document.pop("crs", None)
            stripped = json.dumps(document).encode("utf-8")
        gdf = gpd.read_file(io.BytesIO(stripped))
        gdf = gdf.set_crs(crs_from_user_input(wkt), allow_override=True)
    return gdf


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
                "gdal.VectorTranslate failed to materialize the DataSource to GeoJSON."
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
        gdf = _read_geojson_bytes(data, ds)
    finally:
        # Under gdal.UseExceptions(), Unlink on a non-existent path
        # raises RuntimeError and would mask whatever exception we
        # raised above. Gate the cleanup on the write succeeding.
        if file_written:
            gdal.Unlink(mem_path)
    return gdf
