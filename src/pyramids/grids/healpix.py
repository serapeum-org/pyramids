"""Adapt a HEALPix field to a regular :class:`~pyramids.dataset.Dataset` (deferred).

HEALPix (Hierarchical Equal Area isoLatitude Pixelization) stores values per pixel,
indexed by ``nside``. Converting pixel indices to centre latitude/longitude requires
``healpy``, which is **not yet an approved pyramids dependency**. Once approved it will
be gated behind a ``pyramids-gis[healpix]`` optional extra and this adapter will reuse
:func:`pyramids.dataset.ops.interpolate.grid_points` exactly like
:func:`pyramids.grids.octahedral.from_octahedral`.

Until then :func:`from_healpix` raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.dataset.dataset import Dataset


def from_healpix(
    values: np.ndarray,
    *,
    nside: int | None = None,
    nest: bool = False,
    cell_size: float,
    method: str = "nearest",
    epsg: int = 4326,
) -> Dataset:
    """Regrid a HEALPix field onto a regular-grid :class:`Dataset` (not yet implemented).

    Args:
        values: 1-D array of per-pixel HEALPix values (length ``npix``).
        nside: HEALPix ``nside`` resolution parameter. Derived from ``len(values)``
            when omitted.
        nest: ``True`` for NESTED pixel ordering, ``False`` for RING.
        cell_size: Output pixel size in the target CRS units.
        method: Interpolation algorithm passed through to ``grid_points``.
        epsg: Output EPSG code.

    Returns:
        A single-band :class:`~pyramids.dataset.Dataset` of the interpolated surface.

    Raises:
        NotImplementedError: Always — HEALPix support is pending approval of the
            ``healpy`` dependency. Use :func:`pyramids.grids.from_orca` or
            :func:`pyramids.grids.from_octahedral` for the supported grid types.

    Examples:
        - The adapter is deferred and surfaces a guiding message:
            ```python
            >>> import numpy as np
            >>> from pyramids.grids import from_healpix
            >>> try:
            ...     from_healpix(np.zeros(12), cell_size=1.0)
            ... except NotImplementedError as exc:
            ...     print("healpy" in str(exc))
            True

            ```

    See Also:
        - :func:`pyramids.grids.from_octahedral`: the supported point-based adapter
          this function will mirror once ``healpy`` is approved.
    """
    raise NotImplementedError(
        "from_healpix is not implemented yet: it requires the 'healpy' package, "
        "which is pending approval as an optional 'pyramids-gis[healpix]' extra. "
        "Use pyramids.grids.from_orca or pyramids.grids.from_octahedral for the "
        "currently supported grid types."
    )
