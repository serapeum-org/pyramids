"""Build a lazy GDAL VRT mosaic over a STAC asset across items (PB-5).

stac-vrt builds a GDAL VRT that mosaics one asset across many STAC items, so
GDAL reads the sources on demand via `/vsicurl/` — no eager download, no dask.
This is the most pyramids-native STAC feature: pyramids already wraps GDAL VRTs
in :func:`pyramids.dataset.merge.merge_rasters`.

:func:`build_vrt_from_stac` resolves (and signs, via :func:`resolved_href`) one
asset's href on every item, rewrites each to its `/vsicurl/` form, and hands the
list to :func:`gdal.BuildVRT`. The result is wrapped as a lazy
:class:`~pyramids.dataset.Dataset` over an in-memory `.vrt` whose sources are
read only when pixels are requested.

Completeness note: GDAL treats a source it cannot use — an unreadable href, a
band count or CRS that disagrees with the first source — as a *warning*, drops
it, and builds the mosaic from what is left. That turns an expired signed URL
into a silently incomplete mosaic whose missing tiles read as nodata, so
:func:`build_vrt_from_stac` raises when any requested source was skipped
(``strict=True``, the default); pass ``strict=False`` for best-effort behaviour
with a warning instead.

Signer note: a URL-signing signer (e.g. :class:`PlanetaryComputerSigner`) is
fully supported — the SAS token rides each source href, so reads need no extra
config. An env-credentialed signer (Requester-Pays, bearer) authenticates the
VRT *build*, but the returned ``Dataset``'s later pixel reads happen outside that
config; wrap those reads in the matching ``CloudConfig`` / ``RequesterPays``
block (the same limitation as :func:`pyramids.stac.load_asset`), or prefer a
URL-signing signer.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable
from typing import Any

from osgeo import gdal

from pyramids.base._artifacts import register_vsimem
from pyramids.base.remote import _to_vsi, cloud_config_from_env
from pyramids.dataset import Dataset
from pyramids.stac._loader import resolved_href

_DROPPED_PREVIEW = 5


def _dropped_sources(requested: list[str], retained: Iterable[str]) -> list[str]:
    """Return the requested sources GDAL left out of the built VRT.

    :meth:`gdal.Dataset.GetFileList` on a freshly built VRT reports exactly the
    sources it kept, so anything requested but absent from that list was skipped
    during the build (GDAL logs a warning and carries on).

    Args:
        requested: The VSI paths handed to :func:`gdal.BuildVRT`, in order.
        retained: The source paths the built VRT actually references.

    Returns:
        The requested paths missing from `retained`, in the requested order.

    Examples:
        - A source GDAL skipped is reported back:
            ```python
            >>> from pyramids.stac._vrt import _dropped_sources
            >>> _dropped_sources(["a.tif", "b.tif"], ["a.tif"])
            ['b.tif']

            ```
        - Nothing is reported when every source was kept:
            ```python
            >>> _dropped_sources(["a.tif"], ["a.tif", "/vsimem/x.vrt"])
            []

            ```
    """
    kept = set(retained)
    return [path for path in requested if path not in kept]


def _check_dropped_sources(
    dropped: list[str], total: int, asset: str, strict: bool
) -> None:
    """Raise (or warn) when `gdal.BuildVRT` skipped part of the requested mosaic.

    Args:
        dropped: The requested sources missing from the built VRT.
        total: How many sources were requested.
        asset: The asset key being mosaicked (for the message).
        strict: Raise :class:`RuntimeError` when `True`, warn when `False`.

    Raises:
        RuntimeError: `strict` is `True` and at least one source was skipped.

    Warns:
        UserWarning: `strict` is `False` and at least one source was skipped.

    Examples:
        - A complete build passes silently:
            ```python
            >>> from pyramids.stac._vrt import _check_dropped_sources
            >>> _check_dropped_sources([], 3, "B04", strict=True)

            ```
        - A skipped source fails the build under the default strictness:
            ```python
            >>> _check_dropped_sources(["b.tif"], 3, "B04", True)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            RuntimeError: gdal.BuildVRT skipped 1 of 3 source(s) for asset 'B04', ...

            ```
        - The same skip only warns when the caller opted out of strictness:
            ```python
            >>> import warnings
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     _check_dropped_sources(["b.tif"], 3, "B04", False)
            >>> str(caught[0].message)[:40]
            'gdal.BuildVRT skipped 1 of 3 source(s) f'

            ```
    """
    if dropped:
        preview = ", ".join(dropped[:_DROPPED_PREVIEW])
        if len(dropped) > _DROPPED_PREVIEW:
            preview += f", ... (+{len(dropped) - _DROPPED_PREVIEW} more)"
        message = (
            f"gdal.BuildVRT skipped {len(dropped)} of {total} source(s) for asset "
            f"{asset!r}, so the mosaic is incomplete and the missing footprint "
            f"reads as nodata: {preview}. A source is skipped when it is "
            "unreadable (a 404 / expired signed URL) or its band count or CRS "
            "disagrees with the first source."
        )
        if strict:
            raise RuntimeError(
                f"{message} Pass strict=False to build the partial mosaic anyway."
            )
        warnings.warn(message, UserWarning, stacklevel=3)


def build_vrt_from_stac(
    items: Any,
    asset: str,
    *,
    signer: Any = None,
    separate: bool = False,
    strict: bool = True,
) -> Dataset:
    """Mosaic one STAC asset across items into a lazy VRT-backed `Dataset`.

    Args:
        items: Iterable of STAC Items (pystac objects, raw JSON dicts, or any
            duck-typed equivalent — same contract as
            :meth:`pyramids.dataset.DatasetCollection.from_stac`).
        asset: The asset key to mosaic (e.g. `"visual"`, `"B04"`).
        signer: Optional signer (e.g. a :class:`pyramids.stac.signers.Signer`).
            Its `sign_href` rewrites every source href and its `gdal_env()` is
            installed while the VRT is built. See the module note on read-time
            credentials for env-based signers.
        separate: When `False` (default) the assets are mosaicked spatially
            (overlapping/tiling sources compose into one image — the stac-vrt
            model). When `True`, each source becomes a separate band (a
            band-stack VRT), which requires the sources to share a grid.
        strict: When `True` (default), raise if GDAL skipped any requested
            source, so a partially-built mosaic is never mistaken for a
            complete one. When `False`, warn and return the partial mosaic.

    Returns:
        Dataset: A lazy `Dataset` over an in-memory `.vrt`; GDAL reads the
        underlying sources on demand (`/vsicurl/` range requests for remote
        hrefs).

    Raises:
        ValueError: When `items` yields no items.
        RuntimeError: When `gdal.BuildVRT` fails outright (every source
            unreadable), or — with `strict=True` — when it silently skipped
            some of them (unreadable href, mismatched band count or CRS).

    Warns:
        UserWarning: With `strict=False`, when some sources were skipped.

    Examples:
        - Mosaic the `visual` asset of several items into one lazy Dataset
          (requires network for remote hrefs):
            ```python
            >>> from pyramids.stac import build_vrt_from_stac  # doctest: +SKIP
            >>> ds = build_vrt_from_stac(items, asset="visual")  # doctest: +SKIP
            >>> arr = ds.read_array()  # GDAL pulls source pixels lazily  # doctest: +SKIP

            ```
        - Accept a partial mosaic when some hrefs are known to be missing:
            ```python
            >>> ds = build_vrt_from_stac(items, "visual", strict=False)  # doctest: +SKIP

            ```

    See Also:
        - :func:`pyramids.stac.load_asset`: open a *single* asset instead of
          mosaicking one asset across items.
        - :meth:`pyramids.dataset.DatasetCollection.from_stac`: stack the same
          asset across items along a **time** axis rather than mosaicking it
          into one image.
        - :func:`pyramids.dataset.merge.merge_rasters`: the eager, file-writing
          counterpart to this lazy VRT.
    """
    item_list = list(items)
    if not item_list:
        raise ValueError("build_vrt_from_stac received no items.")

    gdal_env = signer.gdal_env() if signer is not None else None
    vsi_paths = [
        _to_vsi(resolved_href(item, asset, signer=signer)) for item in item_list
    ]
    vrt_path = f"/vsimem/pyramids_stac_{uuid.uuid4().hex}.vrt"

    with cloud_config_from_env(gdal_env):
        vrt_ds = gdal.BuildVRT(
            vrt_path, vsi_paths, options=gdal.BuildVRTOptions(separate=separate)
        )
        if vrt_ds is None:
            raise RuntimeError(
                f"gdal.BuildVRT returned None for asset {asset!r} over "
                f"{len(vsi_paths)} item(s); check that every source is a "
                "readable raster with a consistent band count and CRS."
            )
        # GDAL lists exactly the sources it kept, so the difference against what
        # was requested is what it silently skipped.
        dropped = _dropped_sources(vsi_paths, vrt_ds.GetFileList() or ())
        vrt_ds.FlushCache()
        vrt_ds = None
        # Track the in-memory VRT *before* anything else can raise, so a failure
        # below cannot orphan it beyond the process-exit sweep (M1).
        register_vsimem(vrt_path)
        try:
            _check_dropped_sources(dropped, len(vsi_paths), asset, strict)
            dataset = Dataset.read_file(vrt_path)
        except Exception:
            # Nothing references the VRT on this path — reclaim it now rather
            # than leaving it in /vsimem until interpreter shutdown.
            gdal.Unlink(vrt_path)
            raise
    return dataset
