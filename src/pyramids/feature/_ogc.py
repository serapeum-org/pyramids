"""Shared read-filter assembly for the OGC vector readers (WFS and OGC API – Features).

`pyramids.feature._wfs` and `pyramids.feature._oapif` are sibling factories that
both drive a GDAL OGR HTTP driver behind `FeatureCollection`. The read-filter
assembly (bbox / attribute filter / feature cap) is identical between them, so it
lives here once instead of being copied into each reader. The GDAL HTTP config
(auth + timeout) is shared more widely with the raster OGC readers and lives in
:mod:`pyramids.base._ogc_api`.
"""

from __future__ import annotations

from typing import Any


def read_kwargs(
    bbox: tuple[float, float, float, float] | None,
    where: str | None,
    max_features: int | None,
) -> dict[str, Any]:
    """Assemble the pyogrio / GDAL read filters (bbox, attribute filter, count).

    Raises:
        ValueError: ``bbox`` is not a 4-tuple or is inverted, or ``max_features``
            is less than 1.
    """
    kwargs: dict[str, Any] = {}
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
        minx, miny, maxx, maxy = (float(v) for v in bbox)
        if minx >= maxx or miny >= maxy:
            raise ValueError(
                f"bbox must have minx < maxx and miny < maxy, got {bbox!r}"
            )
        kwargs["bbox"] = (minx, miny, maxx, maxy)
    if where is not None:
        kwargs["where"] = where
    if max_features is not None:
        # 0 is rejected: pyogrio reads rows=0 as "no limit" (returns everything), so a
        # 0 cap would silently fetch the whole layer. Require >= 1 or None.
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1 or None, got {max_features}")
        kwargs["rows"] = max_features
    return kwargs
