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
from typing import Any

from osgeo import gdal

from pyramids.base.remote import _to_vsi, cloud_config_from_env
from pyramids.dataset import Dataset
from pyramids.stac._loader import resolved_href


def build_vrt_from_stac(
    items: Any,
    asset: str,
    *,
    signer: Any = None,
    separate: bool = False,
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

    Returns:
        Dataset: A lazy `Dataset` over an in-memory `.vrt`; GDAL reads the
        underlying sources on demand (`/vsicurl/` range requests for remote
        hrefs).

    Raises:
        ValueError: When `items` yields no items.
        RuntimeError: When `gdal.BuildVRT` fails (e.g. sources with
            inconsistent band counts, or unreadable paths).

    Examples:
        - Mosaic the `visual` asset of several items into one lazy Dataset
          (requires network for remote hrefs):
            ```python
            >>> from pyramids.stac import build_vrt_from_stac  # doctest: +SKIP
            >>> ds = build_vrt_from_stac(items, asset="visual")  # doctest: +SKIP
            >>> arr = ds.read_array()  # GDAL pulls source pixels lazily  # doctest: +SKIP

            ```
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
        vrt_ds.FlushCache()
        vrt_ds = None
        dataset = Dataset.read_file(vrt_path)
    return dataset
