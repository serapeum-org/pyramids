"""Shared read-filter assembly for the OGC vector readers (WFS and OGC API – Features).

`pyramids.feature._wfs` and `pyramids.feature._oapif` are sibling factories that
both drive a GDAL OGR HTTP driver behind `FeatureCollection`. The read-filter
assembly (bbox / attribute filter / feature cap) is identical between them, so it
lives here once instead of being copied into each reader. The GDAL HTTP config
(auth + timeout) is shared more widely with the raster OGC readers and lives in
:mod:`pyramids.base._ogc_api`.

The antimeridian split lives here too, and it is a **network** split, not a local
one. The `bbox` these readers take is not applied to a datasource pyramids holds:
pyogrio turns it into `OGR_L_SetSpatialFilterRect`, and the `WFS` / `OAPIF`
drivers turn *that* into a request parameter — a read filtered by
``(170, -10, 180, 10)`` goes out as
``GET /collections/<id>/items?limit=1000&bbox=170,-10,180,10``. Three
consequences settle the design:

* A wrapping ``west > east`` bbox therefore costs **two requests**, one per seam
  half, whose results are concatenated here. There is no cheaper local option to
  reach for; nothing is fetched that a caller did not ask for.
* A multipart filter cannot be pushed into one request. Handing OGR a
  two-part `MultiPolygon` through `mask=` sends the *envelope* —
  ``bbox=-180,-10,180,10``, the whole globe in longitude — and filters exactly
  on the client afterwards. The answer is right and the download is the entire
  world, which is the opposite of what a bbox is for.
* A wrapping rect handed straight to OGR is silently **inverted**:
  ``SetSpatialFilterRect(170, -10, -170, 10)`` normalises its envelope and goes
  out as ``bbox=-170,-10,170,10`` — the complement of what was asked for, with
  no error. That is why :func:`read_kwargs` may record a wrapping bbox but
  :func:`read_ogc_layer` must split it before it reaches the driver.
"""

from __future__ import annotations

from itertools import repeat
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
from osgeo import gdal

from pyramids.base._coverage import seam_halves, validate_bbox
from pyramids.base._ogc_api import gdal_http_config, not_advertised

if TYPE_CHECKING:
    from pyramids.feature.collection import FeatureCollection


def require_advertised(
    name: str, advertised: frozenset[str], *, noun: str, endpoint: str
) -> None:
    """Raise ``ValueError`` if ``name`` is absent from a non-empty advertised set.

    Shared advertised-name pre-check for the OGC vector readers (WFS feature types,
    OGC API collections). An empty ``advertised`` means the discovery document did
    not enumerate names, so the check is skipped and the driver read is left to
    fail if the name is truly unknown.

    The message itself comes from :func:`pyramids.base._ogc_api.not_advertised`,
    which the raster readers raise for the same mistake. The two were written
    out separately -- the same sentence, the same sort, the same ten-name
    preview and the same ellipsis, with a preview cap declared in each -- so
    trimming the preview in one left the other listing ten. This decides only
    *whether* to refuse; what the refusal says is settled in one place.

    Args:
        name: The requested feature type / collection identifier.
        advertised: The set the discovery document advertised.
        noun: Singular label for the identifier (``"feature type"`` / ``"collection"``);
            the message pluralises it with a trailing ``s``.
        endpoint: The service endpoint, for the message.

    Raises:
        ValueError: ``advertised`` is non-empty and does not contain ``name``.
    """
    if advertised and name not in advertised:
        raise not_advertised(noun, name, endpoint, advertised)


def read_ogc_layer(
    fc_cls: type[FeatureCollection],
    connection: str,
    layer: str,
    *,
    read_kwargs: dict[str, Any],
    auth: tuple[str, str] | None,
    timeout: float,
    error_cls: type[Exception],
    read_fail_prefix: str,
    output_crs: str | None,
) -> FeatureCollection:
    """Read a layer through the GDAL OGR HTTP driver and wrap it as a FeatureCollection.

    The tail shared verbatim by :func:`pyramids.feature._wfs.from_wfs` and
    :func:`pyramids.feature._oapif.from_ogc_features`: install the GDAL HTTP config
    (auth + timeout + retries), read the ``connection`` with the assembled filters,
    normalise any read failure to ``error_cls``, wrap the frame, and optionally
    reproject. Only the connection string, discovery step, error class and
    failure-message wording differ between the two readers.

    A ``west > east`` bbox in ``read_kwargs`` is split at the 180 degree meridian
    by :func:`pyramids.base._coverage.seam_halves` and read as two requests, whose
    features are unioned by :func:`merge_seam_halves`. This is a **behaviour
    change** relative to ``origin/main``, where such a bbox was refused before any
    request went out. A non-wrapping bbox still makes exactly one request with the
    exact filters it was given, and a failure on either half raises the same
    branded message a single failed read raises.

    The split is at ±180, so it reads the bbox as lon/lat degrees — the convention
    the OGC ``bbox`` parameter declares (CRS84 by default for OGC API – Features).
    A service whose default CRS is projected has no seam at ±180, and a wrapping
    bbox against one is a caller mistake this cannot detect.

    Args:
        fc_cls: The ``FeatureCollection`` class to construct.
        connection: The GDAL OGR connection string (``WFS:…`` / ``OAPIF:…``).
        layer: The layer / collection to read.
        read_kwargs: The assembled pyogrio/GDAL read filters (see :func:`read_kwargs`).
        auth: Optional ``(user, password)`` for HTTP Basic auth.
        timeout: Request timeout in seconds.
        error_cls: Exception type to raise on read failure / missing CRS.
        read_fail_prefix: Message prefix for a read failure (kept per-reader so the
            existing wording — ``"WFS GetFeature failed for"`` / ``"OGC API items
            request failed for"`` — is preserved).
        output_crs: Optional CRS to reproject to; ``None`` leaves the server CRS.

    Returns:
        FeatureCollection: The read features, reprojected to ``output_crs`` if given.

    Raises:
        Exception: ``error_cls`` — the read failed, or ``output_crs`` was requested
            but the result carries no CRS.
    """
    config = gdal_http_config(auth, timeout)
    bbox = read_kwargs.get("bbox")
    # One window when the bbox does not wrap (or there is none), two when it does.
    # `seam_halves` hands a non-wrapping box back untouched, so the common path
    # stays a single request with the filters exactly as they were assembled.
    windows: list[tuple[float, float, float, float] | None] = (
        [None] if bbox is None else list(seam_halves(bbox))
    )
    frames: list[gpd.GeoDataFrame] = []
    with gdal.config_options(config):
        for window in windows:
            per_window = dict(read_kwargs)
            if window is not None:
                per_window["bbox"] = window
            try:
                frames.append(gpd.read_file(connection, layer=layer, **per_window))
            except Exception as exc:  # noqa: BLE001 — normalise any read failure to error_cls
                raise error_cls(f"{read_fail_prefix} {layer!r}: {exc}") from exc
    gdf = (
        frames[0]
        if len(frames) == 1
        else merge_seam_halves(frames, read_kwargs.get("rows"))
    )
    fc = fc_cls(gdf)
    if output_crs is not None:
        if fc.crs is None:
            raise error_cls(
                f"cannot reproject {layer!r} to {output_crs!r}: the OGC service "
                "returned features without a CRS"
            )
        fc = fc.to_crs(output_crs)  # to_crs preserves the FeatureCollection subclass
    return fc


def read_kwargs(
    bbox: tuple[float, float, float, float] | None,
    where: str | None,
    max_features: int | None,
) -> dict[str, Any]:
    """Assemble the pyogrio / GDAL read filters (bbox, attribute filter, count).

    The bbox is validated by :func:`pyramids.base._coverage.validate_bbox`, the
    same call the raster OGC readers make, rather than by a second copy of the
    arity / finiteness / ordering block. The two copies were written out
    separately with verbatim identical messages, which is why they were kept in
    step by a test asserting the two sentences matched rather than by there
    being one sentence.

    ``allow_antimeridian=True`` is the one deliberate difference from what the
    old copy did, and it is a **behaviour change** relative to ``origin/main``:
    a ``west > east`` bbox used to be refused as inverted, and is now read as a
    box crossing the 180 degree seam, exactly as :meth:`Dataset.crop` reads it.
    The ordering rule on the Y axis is untouched — there is no seam in latitude,
    so ``miny >= maxy`` is still an error — and so is ``minx == maxx``, a
    zero-width box either way.

    Note:
        The returned dict is **not** splat-able into :func:`geopandas.read_file`
        when the bbox wraps: ``kwargs["bbox"]`` then still carries the caller's
        ``west > east`` box, and OGR would normalise that envelope into its
        complement (see this module's docstring). :func:`read_ogc_layer` owns the
        split, and both readers reach the driver only through it.

    Args:
        bbox: Optional ``(minx, miny, maxx, maxy)``, possibly wrapping.
        where: Optional OGR SQL attribute filter.
        max_features: Optional cap on the features returned.

    Returns:
        dict[str, Any]: The read filters, keyed as pyogrio names them
            (``bbox`` / ``where`` / ``rows``).

    Raises:
        ValueError: ``bbox`` is not a 4-tuple, holds a non-finite corner or is
            inverted (``minx >= maxx`` other than a wrap, or ``miny >= maxy``),
            or ``max_features`` is less than 1.

    Examples:
        - An ordinary box becomes a single ``bbox`` filter:
            ```python
            >>> from pyramids.feature._ogc import read_kwargs
            >>> read_kwargs((1.0, 2.0, 3.0, 4.0), None, None)
            {'bbox': (1.0, 2.0, 3.0, 4.0)}

            ```
        - A box crossing the seam is recorded as the caller wrote it; the split
          happens in :func:`read_ogc_layer`, once, against the driver:
            ```python
            >>> from pyramids.feature._ogc import read_kwargs
            >>> read_kwargs((170.0, -10.0, -170.0, 10.0), None, None)
            {'bbox': (170.0, -10.0, -170.0, 10.0)}

            ```
        - An inverted latitude range stays an error — latitude has no seam:
            ```python
            >>> from pyramids.feature._ogc import read_kwargs
            >>> read_kwargs((1.0, 4.0, 3.0, 2.0), None, None)
            Traceback (most recent call last):
            ValueError: bbox must have minx < maxx and miny < maxy, got (1.0, 4.0, 3.0, 2.0)

            ```
    """
    kwargs: dict[str, Any] = {}
    if bbox is not None:
        kwargs["bbox"] = validate_bbox(bbox, allow_antimeridian=True)
    if where is not None:
        kwargs["where"] = where
    if max_features is not None:
        # 0 is rejected: pyogrio reads rows=0 as "no limit" (returns everything), so a
        # 0 cap would silently fetch the whole layer. Require >= 1 or None.
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1 or None, got {max_features}")
        kwargs["rows"] = max_features
    return kwargs


def content_keys(frame: gpd.GeoDataFrame) -> pd.Series:
    """A hashable per-row identity: the geometry's WKB plus every attribute's repr.

    Used only to spot the same feature coming back from both seam requests. The
    geometry goes in as WKB so two rows match only when their coordinates match
    exactly; the attributes go in as `repr` so a GeoJSON property that arrived as
    a list or a dict — unhashable, and common enough in an OGC API response — does
    not turn a de-duplication into a ``TypeError``.

    Two rows the source genuinely holds twice, identical in geometry and in every
    attribute, are indistinguishable to this and collapse to one. That is the
    accepted cost of not trusting an FID: a WFS/OAPIF FID is assigned per request,
    so the same integer means different features in the two halves, and keying on
    it would drop real features rather than duplicate ones.

    Args:
        frame: A frame with an active geometry column.

    Returns:
        pandas.Series: One hashable key per row, aligned to `frame`'s index.

    Examples:
        - Two rows differing only in geometry get different keys:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature._ogc import content_keys
            >>> frame = gpd.GeoDataFrame(
            ...     {"name": ["a", "a"]}, geometry=[Point(0, 0), Point(1, 1)]
            ... )
            >>> content_keys(frame).duplicated().tolist()
            [False, False]

            ```
        - An unhashable attribute value does not break the key:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature._ogc import content_keys
            >>> frame = gpd.GeoDataFrame(
            ...     {"tags": [["x"], ["x"]]}, geometry=[Point(0, 0), Point(0, 0)]
            ... )
            >>> content_keys(frame).duplicated().tolist()
            [False, True]

            ```
    """
    attributes = frame.drop(columns=[frame.geometry.name])
    # `itertuples` on a frame with no columns yields nothing at all rather than
    # one empty tuple per row, which would leave the keys shorter than the index.
    attribute_rows = (
        attributes.itertuples(index=False, name=None)
        if attributes.shape[1]
        else repeat((), len(frame))
    )
    keys = [
        (blob, tuple(repr(value) for value in values))
        for blob, values in zip(frame.geometry.to_wkb(), attribute_rows)
    ]
    return pd.Series(keys, index=frame.index, dtype=object)


def merge_seam_halves(
    frames: list[gpd.GeoDataFrame], cap: int | None
) -> gpd.GeoDataFrame:
    """Union the two seam halves' features into one frame.

    Concatenating two feature sets is the whole of the "stitch" a vector read
    needs — there is no grid to re-assemble, as there is for
    :meth:`Dataset.crop`. What it does need is the two things a plain concat
    gets wrong:

    * **Duplicates.** The halves are disjoint boxes, but a feature that actually
      straddles the seam intersects both and comes back from both requests. It is
      dropped down to one occurrence by :func:`content_keys`. Note that the
      de-duplication is over the whole concatenation, not over matched pairs, so a
      row the *source* holds twice within a single half collapses too -- and only
      when the bbox wraps, since a single-request read is passed through untouched.
      Distinguishing the two would mean tagging each row with the half it came from
      and dropping only cross-half matches; the simpler rule is kept because a
      genuinely duplicated feature is far rarer than a seam-straddling one, and
      because a caller who wants exact source multiplicity should not be asking
      one request to span the antimeridian.
    * **The index.** Each request's frame is indexed from 0, so a concat that
      kept them would hand back a frame with every label twice. The result is
      re-indexed ``0..n-1``, which is what a single-box read returns anyway.

    The ``max_features`` cap is applied to the *union*, not per half: each half is
    read with the same cap (each may legitimately hold all of them), and the
    concatenated result is then truncated, so the promise "at most this many
    features" holds across the seam. West-half features therefore fill the cap
    first when both halves are full.

    Args:
        frames: The per-half frames, west half first.
        cap: The ``max_features`` cap, or ``None`` for no cap.

    Returns:
        geopandas.GeoDataFrame: The de-duplicated union, re-indexed from 0.

    Examples:
        - A feature returned by both halves appears once:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature._ogc import merge_seam_halves
            >>> east = gpd.GeoDataFrame(
            ...     {"name": ["a", "seam"]},
            ...     geometry=[Point(172.0, 0.0), Point(180.0, 0.0)],
            ...     crs="EPSG:4326",
            ... )
            >>> west = gpd.GeoDataFrame(
            ...     {"name": ["seam", "b"]},
            ...     geometry=[Point(180.0, 0.0), Point(-172.0, 0.0)],
            ...     crs="EPSG:4326",
            ... )
            >>> merged = merge_seam_halves([east, west], None)
            >>> list(merged["name"])
            ['a', 'seam', 'b']
            >>> list(merged.index)
            [0, 1, 2]

            ```
        - The cap bounds the union rather than each half:
            ```python
            >>> from pyramids.feature._ogc import merge_seam_halves
            >>> len(merge_seam_halves([east, west], 2))
            2

            ```
    """
    combined = pd.concat(frames, ignore_index=True)
    if len(combined):
        combined = combined[~content_keys(combined).duplicated()]
    if cap is not None:
        combined = combined.iloc[:cap]
    return combined.reset_index(drop=True)
