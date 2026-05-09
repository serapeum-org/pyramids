"""Operator-shaped reprojection: build the plan once, apply it to many datasets.

:class:`Reprojector` (and its :class:`Aligner` subclass) is
an xesmf-shaped alternative to calling
:meth:`Dataset.to_crs` once per dataset. Construction is small and
cheap: a frozen :class:`ReprojectPlan` dataclass captures the
`(target_epsg, method, maintain_alignment)` tuple. Application
delegates to the existing :meth:`Dataset.to_crs` /
:meth:`Dataset.align` implementations so the GDAL Warp path stays
unchanged.

The win is reuse: when you have a `DatasetCollection` of 365 daily
rasters that all need the same reprojection, build one
`Reprojector` and call it 365 times rather than pay the overhead
of argument parsing + spec building on every call. Supports
`compute=False` so the whole reproject can be deferred into a
dask graph via :func:`dask.delayed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyramids.dataset import Dataset


_LAZY_IMPORT_ERROR = (
    "Lazy reprojection (compute=False) requires the optional 'dask' "
    "dependency. Install it with: pip install 'pyramids-gis[lazy]'"
)


@dataclass(frozen=True)
class ReprojectPlan:
    """Immutable, picklable reprojection specification.

    Held on a :class:`Reprojector` instance and reused across every
    `__call__`. The plan itself performs no I/O — all GDAL work
    happens inside :meth:`Reprojector.__call__`.

    Attributes:
        target_epsg: Destination EPSG code.
        method: Resampling method; passed through to
            :meth:`Dataset.to_crs` / :meth:`Dataset.align`.
        maintain_alignment: Preserve source rows/columns across the
            reproject (only meaningful for :class:`Reprojector`; the
            :class:`Aligner` subclass forces geobox alignment via a
            reference dataset and ignores this flag).
    """

    target_epsg: int
    method: str = "nearest neighbor"
    maintain_alignment: bool = False


class Reprojector:
    """Plan-once, apply-many reprojection operator.

    Args:
        target_epsg: Destination EPSG code.
        method: Resampling method name.
        maintain_alignment: See :class:`ReprojectPlan`.

    Examples:
        - Reproject a dataset via the operator:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops.reproject import Reprojector
            >>> arr = np.zeros((2, 2), dtype=np.float32)
            >>> src = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            ... )
            >>> op = Reprojector(target_epsg=3857)
            >>> out = op(src)
            >>> out.epsg
            3857

            ```
    """

    def __init__(
        self,
        target_epsg: int,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
    ) -> None:
        self._plan = ReprojectPlan(
            target_epsg=int(target_epsg),
            method=method,
            maintain_alignment=bool(maintain_alignment),
        )

    @property
    def plan(self) -> ReprojectPlan:
        """Return the immutable :class:`ReprojectPlan`."""
        return self._plan

    def __call__(self, ds: Dataset, *, compute: bool = True) -> Any:
        """Apply the plan to `ds`.

        Args:
            ds: Source :class:`~pyramids.dataset.Dataset`.
            compute: `True` (default) runs the reprojection eagerly
                and returns a new :class:`Dataset`. `False` wraps
                the eager call in :func:`dask.delayed` and returns a
                :class:`dask.delayed.Delayed` object.

        Returns:
            Dataset or dask.delayed.Delayed.
        """
        if compute:
            result = _apply_plan(self._plan, ds)
        else:
            result = _deferred(self._plan, ds)
        return result


class Aligner(Reprojector):
    """Reproject + resample to match a reference :class:`Dataset` geobox.

    Args:
        reference: The reference :class:`~pyramids.dataset.Dataset`
            defining the output cell size / extent / CRS.
        method: Resampling method.

    Examples:
        - Align a source to a reference geobox:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops.reproject import Aligner
            >>> ref = Dataset.create_from_array(
            ...     np.zeros((4, 4), dtype=np.float32),
            ...     top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
            ... )
            >>> src = Dataset.create_from_array(
            ...     np.zeros((8, 8), dtype=np.float32),
            ...     top_left_corner=(0.0, 0.0), cell_size=0.5, epsg=4326,
            ... )
            >>> aligner = Aligner(ref)
            >>> out = aligner(src)
            >>> (out.rows, out.columns)
            (4, 4)

            ```
    """

    def __init__(self, reference: Dataset, method: str = "nearest neighbor") -> None:
        super().__init__(
            target_epsg=int(reference.epsg),
            method=method,
            maintain_alignment=False,
        )
        self._reference = reference

    def __call__(self, ds: Dataset, *, compute: bool = True) -> Any:
        """Align `ds` to the reference geobox.

        Args:
            ds: Source :class:`~pyramids.dataset.Dataset`.
            compute: See :meth:`Reprojector.__call__`.

        Returns:
            Dataset or dask.delayed.Delayed.
        """
        if compute:
            result = ds.align(self._reference)
        else:
            result = _deferred_align(self._reference, self._plan.method, ds)
        return result


def _apply_plan(plan: ReprojectPlan, ds: Dataset) -> Dataset:
    """Run a :class:`ReprojectPlan` against a dataset eagerly."""
    return ds.to_crs(
        plan.target_epsg,
        method=plan.method,
        maintain_alignment=plan.maintain_alignment,
    )


def _deferred(plan: ReprojectPlan, ds: Dataset) -> Any:
    """Wrap :func:`_apply_plan` in :func:`dask.delayed`."""
    try:
        import dask
    except ImportError as exc:
        raise ImportError(_LAZY_IMPORT_ERROR) from exc
    return dask.delayed(_apply_plan)(plan, ds)


def _deferred_align(reference: Dataset, method: str, ds: Dataset) -> Any:
    """Wrap :meth:`Dataset.align` in :func:`dask.delayed`."""
    try:
        import dask
    except ImportError as exc:
        raise ImportError(_LAZY_IMPORT_ERROR) from exc
    return dask.delayed(_align_sync)(reference, method, ds)


def _align_sync(reference: Dataset, method: str, ds: Dataset) -> Dataset:
    """Synchronous align body — module-level for pickleability.

    `method` is accepted for API parity with :class:`Reprojector` but
    :meth:`Dataset.align` currently fixes the resampling to nearest
    neighbor (see :func:`osgeo.gdal.Warp`'s default). Argument kept so
    that if :meth:`Dataset.align` grows a `method=` kwarg in future
    the operator contract does not need to change.
    """
    del method  # reserved for future align(method=...) support
    return ds.align(reference)
