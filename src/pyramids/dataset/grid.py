"""The :class:`Grid` target-grid specification for collection constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Grid:
    """Target output grid for a reprojected / aligned collection.

    Describes the grid every timestep is warped onto — currently by
    :meth:`~pyramids.dataset.DatasetCollection.from_stac`. It has three states:

    * **empty** (every field left at its default) — no reprojection; each
      timestep keeps its native grid.
    * **template** (``like``) — match an existing
      :class:`~pyramids.dataset.Dataset`'s CRS + geotransform + shape exactly.
    * **explicit** (``crs`` + ``resolution`` + ``bounds``, all three) — build a
      grid from those values, with the extent snapped by ``anchor``.

    ``like`` and the ``crs`` / ``resolution`` / ``bounds`` trio are mutually
    exclusive, and the trio is all-or-nothing (there is no inference of a
    missing member). These invariants are enforced at construction, so an
    invalid combination raises here rather than deep inside a constructor.

    Args:
        like: An existing :class:`~pyramids.dataset.Dataset` whose grid to copy,
            or ``None``.
        crs: Target CRS (EPSG int like ``32633``, or a CRS string) for an
            explicit grid.
        resolution: Target pixel size, in the target CRS's units.
        bounds: Target ``(minx, miny, maxx, maxy)`` extent, expressed in ``crs``
            (not lon/lat).
        anchor: Grid-snap rule for the explicit grid. Only ``"edge"`` is
            supported today (pixel edges snap to multiples of ``resolution`` so
            independently built grids co-register).

    Raises:
        ValueError: ``like`` is combined with any of ``crs`` / ``resolution`` /
            ``bounds``; the explicit trio is given only partially; or ``anchor``
            is not ``"edge"``.

    Examples:
        - An explicit target grid:

          ```python
          >>> from pyramids.dataset import Grid
          >>> grid = Grid(crs=32633, resolution=10, bounds=(0, 0, 1000, 1000))
          >>> grid.is_empty
          False

          ```
        - An empty grid means "keep the native grid":

          ```python
          >>> from pyramids.dataset import Grid
          >>> Grid().is_empty
          True

          ```
        - Mixing the two modes is rejected:

          ```python
          >>> from pyramids.dataset import Grid
          >>> Grid(crs=32633, resolution=10)
          Traceback (most recent call last):
              ...
          ValueError: Grid: crs, resolution, and bounds must all be given together (or use like=).

          ```
    """

    like: Any = None
    crs: int | str | None = None
    resolution: float | None = None
    bounds: tuple[float, float, float, float] | None = None
    anchor: str = "edge"

    def __post_init__(self) -> None:
        """Validate the mutually-exclusive modes and the anchor."""
        given = [v is not None for v in (self.crs, self.resolution, self.bounds)]
        if self.like is not None and any(given):
            raise ValueError(
                "Grid: like= is mutually exclusive with crs/resolution/bounds."
            )
        if self.like is None and any(given) and not all(given):
            raise ValueError(
                "Grid: crs, resolution, and bounds must all be given together "
                "(or use like=)."
            )
        if self.anchor != "edge":
            raise ValueError(f"Grid: anchor must be 'edge', got {self.anchor!r}.")

    @property
    def is_empty(self) -> bool:
        """Whether no target grid is specified (the native grid is kept)."""
        return (
            self.like is None
            and self.crs is None
            and self.resolution is None
            and self.bounds is None
        )
