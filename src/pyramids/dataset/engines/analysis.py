"""Analysis engine.

Owns the Analysis family of operations on a Dataset. Accessed as
``ds.analysis``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.analysis.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from hpc.indexing import get_indices2, get_pixels2
from osgeo import gdal
from pandas import DataFrame

from pyramids.base._domain import (
    fits_dtype,
    free_no_data,
    is_nan_sentinel,
    is_stored_no_data,
    occurs_in,
)
from pyramids.base._errors import (
    AlignmentError,
    NoDataCollisionWarning,
    OutOfBoundsError,
    ReadOnlyError,
)
from pyramids.base._utils import (
    _is_identity_packing,
    apply_unpack,
    gdal_to_numpy_dtype,
    numpy_to_gdal_dtype,
    require_cleopatra,
)
from pyramids.dataset._mask import MaskFlags
from pyramids.dataset._plot_helpers import (
    ModeSpec,
    RenderRequest,
    RgbSpec,
    render_array,
)
from pyramids.dataset.abstract_dataset import RasterBase
from pyramids.dataset.window import Window
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from cleopatra.basemap.geo import Basemap
    from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pyramids.dataset.dataset import Dataset

from pyramids.base.crs import crs_spec
from pyramids.base.georeference import GeoReference
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines._validate import (
    resolve_band_indices,
    validate_band_index,
)

# A windowed point-sample read is worth it while the points' bounding box stays
# under this many pixels, or within this multiple of the point count -- beyond
# that the pixels read but discarded cost more than the per-point GDAL calls.
_POINT_WINDOW_MIN_PIXELS = 4096
_POINT_WINDOW_MAX_WASTE = 16
# The waste ratio alone bounds nothing absolute: ten million clustered points
# would authorise a 160-megapixel read (~1.3 GB at float64). This caps the block
# at roughly 64 MB of float64 regardless of how many points asked for it; past
# it, the per-point reads are the cheaper failure mode.
_POINT_WINDOW_MAX_PIXELS = 8_000_000


# Module-level logger: the dtype probe in `apply` swallows whatever an awkward
# callable raises and falls back to the source type, so the reason has to surface
# somewhere.
logger = logging.getLogger(__name__)


class _DeriveNoData:
    """Singleton marking `combine`'s "work the sentinel out for me" default.

    A distinct object because `None` is already meaningful for `no_data_value`
    -- it asks for a result with no sentinel at all. A named class rather than
    a bare `object()` so the rendered signature reads
    `no_data_value: Any = <derive>` instead of a memory address that changes on
    every docs build.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Render as `<derive>` in signatures, `help()` and the API docs.

        Returns:
            str: The stable placeholder shown wherever the default is printed.
        """
        return "<derive>"


_DERIVE_NO_DATA = _DeriveNoData()


@dataclass(frozen=True)
class _PointWindow:
    """One windowed read covering a batch of in-bounds sample points."""

    x_off: int
    y_off: int
    x_size: int
    y_size: int
    strip_rows: int
    in_rows: np.ndarray
    in_cols: np.ndarray


def _point_sample_fill(gdal_band: gdal.Band) -> tuple[Any, np.dtype]:
    """Value and dtype for cells a point sample does not reach.

    An integer band with no no-data value has no in-range sentinel to spare, so
    the output is promoted to float and the gaps filled with `NaN`.

    Args:
        gdal_band: The band being sampled.

    Returns:
        tuple[Any, numpy.dtype]: The fill value and the output dtype.
    """
    no_data_value = gdal_band.GetNoDataValue()
    band_dtype = np.dtype(gdal_to_numpy_dtype(gdal_band.DataType))
    if no_data_value is None:
        out_dtype = (
            band_dtype
            if np.issubdtype(band_dtype, np.floating)
            else np.dtype("float64")
        )
        return np.nan, out_dtype
    return no_data_value, band_dtype


_KEEP_NO_DATA = object()
"""`astype`'s default for `no_data_value`: keep the raster's own sentinel."""


def _regapped(values: np.ndarray, domain: np.ndarray, sentinel: Any) -> np.ndarray:
    """`values` with every cell outside `domain` set back to its band's sentinel.

    The members that transform cell values leave the gaps out and put them back here, so a
    gap is never clipped, rounded or cast into a measurement. Writing in place, rather than
    through `np.where`, keeps the band's type: a `numpy.float64` sentinel would otherwise
    promote a `float32` band.

    Args:
        values: The transformed cells, 2-D for one band or `(bands, rows, cols)`.
        domain: True where a cell holds data, shaped like `values`.
        sentinel: The value that marks a gap — one for every band, or a list holding each
            band's own, which is what a multi-sensor stack carries. `None`, or a list entry
            of `None`, leaves that band's gaps as the operation left them.

    Returns:
        np.ndarray: The cells, gaps re-marked.
    """
    marks = list(sentinel) if isinstance(sentinel, (list, tuple)) else [sentinel]
    out = values
    if not domain.all() and any(one is not None for one in marks):
        out = np.array(values, copy=True)
        planes = (
            [(out, domain, marks[0])]
            if out.ndim == 2
            else [
                (out[index], domain[index], _mark_for(marks, index))
                for index in range(out.shape[0])
            ]
        )
        for plane, kept, mark in planes:
            if mark is not None:
                plane[~kept] = mark
    return out


def _mark_for(marks: list, index: int) -> Any:
    """The sentinel for one band of a stack.

    Args:
        marks: The sentinels, one per band — or fewer, when the caller passed one for a
            stack, which every band then shares.
        index: The band's index.

    Returns:
        Any: That band's sentinel, or the first when the list is shorter than the stack.
    """
    return marks[index] if index < len(marks) else marks[0]


def _same_gap(one: Any, other: Any) -> bool:
    """Whether two bands declare the same gap marker.

    Compared by value, not by `repr`: GDAL hands a sentinel back as `numpy.float64` while a
    caller passes a Python `float`, and `-9999.0` is the same declaration either way. Two
    NaN markers agree as well, though NaN equals nothing.

    Args:
        one: One band's sentinel, or `None`.
        other: Another band's sentinel, or `None`.

    Returns:
        bool: `True` when they declare the same thing.
    """
    if one is None or other is None:
        same = one is None and other is None
    else:
        left, right = float(one), float(other)
        same = left == right or (math.isnan(left) and math.isnan(right))
    return same


def _declared_gaps(sentinels: Sequence[Any]) -> Any:
    """What a result should declare as its gaps, given each band's own sentinel.

    One value when every band declares the same thing — the ordinary single-band and
    uniform-stack cases, where a scalar keeps the metadata simplest — and the per-band list
    otherwise. Collapsing a mixed stack onto the first band's sentinel rewrote the others'
    declarations, and a band holding another band's sentinel as a real value then read as
    missing.

    Args:
        sentinels: One sentinel per band, `None` where a band declares none.

    Returns:
        Any: The scalar the bands agree on, or the list of per-band sentinels.
    """
    marks = list(sentinels)
    agreed = all(_same_gap(one, marks[0]) for one in marks)
    return marks[0] if agreed and marks else marks


def _gap_marker(
    mark: Any, target: np.dtype, domain: np.ndarray, band: int, bands: int
) -> Any:
    """What to write into one band's gap cells before the cast fills the rest.

    Args:
        mark: The band's declared sentinel, or `None`.
        target: The type being cast to.
        domain: True where a cell holds data.
        band: The band's index.
        bands: How many bands there are.

    Returns:
        Any: The value to fill that band's plane with.

    Raises:
        ValueError: The band has gaps, declares no sentinel, and `target` is an integer
            type, which has no NaN to leave them as.
    """
    marker = mark
    if marker is None:
        plane = domain if bands == 1 or domain.ndim == 2 else domain[band]
        if not plane.all():
            if target.kind != "f":
                raise ValueError(
                    f"astype({target.name!r}) would leave {int((~plane).sum())} gap cells "
                    f"of band {band + 1} unmarked: it declares no no-data value and "
                    f"{target.name} has no NaN, so every gap would read as an ordinary "
                    f"number. Pass no_data_value= with one {target.name} holds, or fill "
                    f"the gaps first."
                )
            marker = np.nan
        else:
            marker = 0
    return marker


def _refuse_a_sentinel_the_data_holds(
    cast: np.ndarray,
    domain: np.ndarray,
    marks: Sequence[Any],
    target: np.dtype,
    bands: int,
) -> None:
    """Refuse a cast in which a cell holding data lands on its band's sentinel.

    The mirror of "a gap stays a gap": data has to stay data. A sentinel a real cell
    already holds — `255` after `clip(0, 255)`, or `-9999` after truncating `-9999.4` —
    would read as missing from then on.

    Args:
        cast: The cast cells.
        domain: True where a cell holds data.
        marks: Each band's sentinel, `None` where a band declares none.
        target: The type cast to.
        bands: How many bands there are.

    Raises:
        ValueError: A band's data holds its own sentinel.
    """
    for band in range(bands):
        mark = marks[band]
        if mark is None or not np.isfinite(float(mark)):
            continue
        plane = cast if cast.ndim == 2 else cast[band]
        kept = domain if domain.ndim == 2 else domain[band]
        collisions = int(np.count_nonzero(plane[kept] == target.type(mark)))
        if collisions:
            raise ValueError(
                f"astype({target.name!r}) would mark {collisions} cell(s) of band "
                f"{band + 1} that hold data as missing: they already hold {float(mark)} "
                f"once cast, and that is the value the result declares as its gap. Bound "
                f"or shift them first (clip), or pass no_data_value= with one the data "
                f"never takes."
            )


def _holds(target: np.dtype, value: Any) -> bool:
    """Whether `target` can carry `value` as a sentinel at all.

    A float target takes any value inside its range: the sentinel is **snapped** to the
    type on the way in (`-9999.9` into `float32` is declared and written as
    `-9999.900390625`), and `is_stored_no_data` recognises a gap with the slack a sentinel
    picks up passing through storage, so a value that merely loses precision marks exactly
    the cells it should. What must not happen is a snapped sentinel landing on a cell that
    holds data — `1e-50` into `float32` snaps to `0.0` — and that is
    :func:`_refuse_a_sentinel_the_data_holds`'s question, asked of the cast cells rather
    than guessed from the type.

    An integer target is stricter, because no snapping keeps a gap a gap there: a fraction
    or an out-of-range value wraps into an ordinary number.

    Args:
        target: The band type.
        value: The candidate sentinel.

    Returns:
        bool: `True` for a float type and any value inside its range, NaN and the
        infinities included; for an integer type, only a whole number inside its range.
    """
    number = float(value)
    if np.issubdtype(target, np.floating):
        fits = not np.isfinite(number) or abs(number) <= float(np.finfo(target).max)
    else:
        limits = np.iinfo(target)
        fits = (
            bool(np.isfinite(number))
            and number.is_integer()
            and limits.min <= number <= limits.max
        )
    return fits


def _mask_dtype(band: np.dtype, fill: Any) -> np.dtype:
    """The type a masked result needs: the band's own, unless the fill will not fit it.

    `np.where` answers a different type depending on how the fill is spelled, and neither
    answer is the band's own. A Python `0.0` octuples a `uint8` raster — `np.where` gives
    `float64` for a value a byte holds — while under NEP 50 it leaves a `float32` raster
    alone; and the `numpy.float64` GDAL hands back as a declared no-data value is strongly
    typed, so it doubles that same `float32` raster. A band keeps its own type when it can
    hold what is written into it:

    - a float band holds any real value at its own precision, NaN and the infinities
      included — every float width has those — so it widens only for a *finite* magnitude
      no value of that width can represent: a `float16` band and `70000.0`;
    - an integer band holds an integral sentinel in range — GDAL hands those back as
      floats (`255.0`), which is why the value is checked rather than its Python type.

    A fractional fill into an integer band, or NaN where no integer could mean "missing",
    genuinely needs a wider type and gets one.

    `np.result_type` is asked only for those cases, and never as the *first* question:
    the fill arrives as a `numpy.float64` — that is how GDAL hands back a declared no-data
    value — and a numpy scalar is strongly typed, so `np.result_type(float32, it)` answers
    `float64` for a number `float32` holds exactly. The band's own type is decided by what
    it can hold, not by numpy's promotion.

    Args:
        band: The source band's dtype.
        fill: What an unselected cell will hold.

    Returns:
        numpy.dtype: The result's dtype.

    Examples:
        - A float band keeps its width for anything it can represent, the infinities
          included:

          ```python
          >>> import numpy as np
          >>> from pyramids.dataset.engines.analysis import _mask_dtype
          >>> [str(_mask_dtype(np.dtype("float32"), one)) for one in (-9999.0, 2.5, np.inf)]
          ['float32', 'float32', 'float32']

          ```
        - And widens for a finite magnitude no value of that width can hold:

          ```python
          >>> import numpy as np
          >>> from pyramids.dataset.engines.analysis import _mask_dtype
          >>> str(_mask_dtype(np.dtype("float16"), 70000.0))
          'float64'

          ```
        - An integer band keeps its type for an integral sentinel in range, and widens
          for a fractional fill or for NaN:

          ```python
          >>> import numpy as np
          >>> from pyramids.dataset.engines.analysis import _mask_dtype
          >>> [str(_mask_dtype(np.dtype("uint8"), one)) for one in (255.0, np.nan)]
          ['uint8', 'float64']
          >>> str(_mask_dtype(np.dtype("int16"), 0.5))
          'float64'

          ```
    """
    filler = np.asarray(fill)
    chosen = np.result_type(band, filler)
    if np.issubdtype(band, np.floating):
        holds = bool(np.isnan(filler).all()) or bool(np.isinf(filler).all())
        if holds or bool(np.abs(filler) <= np.finfo(band).max):
            chosen = np.dtype(band)
    elif np.issubdtype(band, np.integer) and not np.isnan(filler).any():
        limits = np.iinfo(band)
        value = float(filler)
        if value.is_integer() and limits.min <= value <= limits.max:
            chosen = np.dtype(band)
    return chosen


class Analysis(_Engine["Dataset"]):
    """Mixin providing analysis, statistics, and data extraction operations for Dataset."""

    def stats(
        self,
        band: int | None = None,
        mask: GeoDataFrame | None = None,
        *,
        approx_ok: bool = True,
    ) -> DataFrame:
        """Get statistics of a band [Min, max, mean, std].

        **In physical units.** On a band declaring CF packing the four numbers come
        back as `read_array` would answer, not as GDAL stores them: `min`, `max` and
        `mean` take `raw * scale + offset` and `std` takes `|scale|`, since an
        additive offset moves a distribution without widening it. GDAL is still asked
        for the figures — no pixel is read — and the transform is applied to its
        answer, which is exact because CF packing is affine. Before #1124 these
        disagreed with `read_array` by the packing factor.

        One consequence worth knowing: the `.aux.xml` sidecar noted below caches the
        **stored** figures, so what is on disk will not match what this returns.

        Args:
            band (int, optional):
                Band index. If None, the statistics of all bands will be returned.
            mask (Polygon GeoDataFrame or Dataset, optional):
                GeodataFrame with a geometry of polygon type.
            approx_ok (bool, optional):
                Allow GDAL to answer from overviews or a subsample rather than
                reading every pixel. Default `True`, which is fast but can return
                values that differ from the exact ones -- pass `False` when the
                figures must be exact.

        Returns:
            DataFrame:
                DataFrame with the stats of each band, the dataframe has the following
                `float64` columns [min, max, mean, std], in physical units, and the index
                of the dataframe is the band names.

                ```text

                                   min         max        mean       std
                    Band_1  270.369720  270.762299  270.551361  0.154270
                    Band_2  269.611938  269.744751  269.673645  0.043788
                    Band_3  273.641479  274.168823  273.953979  0.198447
                    Band_4  273.991516  274.540344  274.310669  0.205754
                ```

        Raises:
            ValueError: The `mask` does not overlap the dataset, or `band` is
                outside the band range.
            RuntimeError: GDAL could not compute statistics for a band -- most
                often a band with no valid pixels at all.

        Notes:
            - The value of the stats will be stored in an xml file by the name of the raster file with the extension of
              .aux.xml.
            - The content of the file will be like the following:

              ```xml

                  <PAMDataset>
                    <PAMRasterBand band="1">
                      <Description>Band_1</Description>
                      <Metadata>
                        <MDI key="RepresentationType">ATHEMATIC</MDI>
                        <MDI key="STATISTICS_MAXIMUM">88</MDI>
                        <MDI key="STATISTICS_MEAN">7.9662921348315</MDI>
                        <MDI key="STATISTICS_MINIMUM">0</MDI>
                        <MDI key="STATISTICS_STDDEV">18.294377743948</MDI>
                        <MDI key="STATISTICS_VALID_PERCENT">48.9</MDI>
                      </Metadata>
                    </PAMRasterBand>
                  </PAMDataset>

              ```

        Examples:
            - Get the statistics of all bands in the dataset:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 10, 10)
              >>> geotransform = (0, 0.05, 0, 0, 0, -0.05)
              >>> dataset = Dataset.from_array(arr, geo_ref=GeoReference(geo=geotransform, epsg=4326))
              >>> print(dataset.stats()) # doctest: +SKIP
                           min       max      mean       std
              Band_1  0.006443  0.942943  0.468935  0.266634
              Band_2  0.020377  0.978130  0.477189  0.306864
              Band_3  0.019652  0.992184  0.537215  0.286502
              Band_4  0.011955  0.984313  0.503616  0.295852
              >>> print(dataset.stats(band=1))  # doctest: +SKIP
                           min      max      mean       std
              Band_2  0.020377  0.97813  0.477189  0.306864

              ```

            - Get the statistics of all the bands using a mask polygon.

              - Create the polygon using shapely polygon, and use the xmin, ymin, xmax, ymax = [0.1, -0.2,
                0.2 -0.1] to cover the 4 cells.
              ```python
              >>> from shapely.geometry import Polygon
              >>> import geopandas as gpd
              >>> mask = gpd.GeoDataFrame(geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])],crs=4326)
              >>> print(dataset.stats(mask=mask))  # doctest: +SKIP
                           min       max      mean       std
              Band_1  0.193441  0.702108  0.541478  0.202932
              Band_2  0.281281  0.932573  0.665602  0.239410
              Band_3  0.031395  0.982235  0.493086  0.377608
              Band_4  0.079562  0.930965  0.591025  0.341578

              ```

            - On a CF-packed band the figures come back in physical units, the same ones
              `read_array` answers in, and the no-data cell is left out:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> packed = Dataset.from_array(
              ...     np.array([[100, 200], [300, -9999]], dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
              ...     no_data_value=-9999,
              ... )
              >>> packed.scale = [0.01]
              >>> packed.offset = [1.5]
              >>> packed.stats(band=0, approx_ok=False).loc["Band_1"].round(6).tolist()
              [2.5, 4.5, 3.5, 0.816497]
              >>> float(packed.read_array(masked=True).max())
              4.5

              ```

        """
        # Ahead of the band_names lookup below, which would otherwise surface a
        # bare IndexError where every other band-taking entry point raises a
        # ValueError naming the range.
        validate_band_index(band, self._ds.band_count)
        dst: Dataset | None = None
        if mask is not None:
            dst = self._ds.crop(mask, touch=True)

        if band is None:
            df = pd.DataFrame(
                index=self._ds.band_names,
                columns=["min", "max", "mean", "std"],
                # `float64`, not `float32`: these are physical values now, and a band
                # packed at 1e-5 around an offset of 273.15 -- an ordinary way to store
                # temperature -- has more significant digits than `float32` carries.
                # Rounding them here would throw away exactly the resolution the
                # packing existed to preserve.
                dtype=np.float64,
            )
            for i in range(self._ds.band_count):
                if mask is not None and dst is not None:
                    df.iloc[i, :] = dst.analysis._get_stats(i, approx_ok=approx_ok)
                else:
                    df.iloc[i, :] = self._get_stats(i, approx_ok=approx_ok)
        else:
            df = pd.DataFrame(
                index=[self._ds.band_names[band]],
                columns=["min", "max", "mean", "std"],
                # `float64`, not `float32`: these are physical values now, and a band
                # packed at 1e-5 around an offset of 273.15 -- an ordinary way to store
                # temperature -- has more significant digits than `float32` carries.
                # Rounding them here would throw away exactly the resolution the
                # packing existed to preserve.
                dtype=np.float64,
            )
            if mask is not None and dst is not None:
                df.iloc[0, :] = dst.analysis._get_stats(band, approx_ok=approx_ok)
            else:
                df.iloc[0, :] = self._get_stats(band, approx_ok=approx_ok)

        return df

    def _get_stats(
        self, band: int | None = None, *, approx_ok: bool = True
    ) -> list[float]:
        """Return summary statistics for one band.

        Reads GDAL band statistics, computing them on the fly when the cached values are
        absent or empty.

        Args:
            band (int | None):
                Zero-based band index. Defaults to the first band (0) when None.
            approx_ok (bool):
                Let GDAL answer from overviews or a subsample rather than
                scanning every cell. Default `True`, the historical behaviour.
                Ignored by the recovery path below, which is always exact.

        Returns:
            list[float]: The ``[minimum, maximum, mean, standard_deviation]`` values.
        """
        band_index = band if band is not None else 0
        band_i = self._ds._iloc(band_index)
        try:
            # First argument is GDAL's `approx_ok`: True lets it answer from
            # overviews or a subsample. It was hard-coded, so `stats()` could
            # silently return approximated figures with no way to ask for exact
            # ones; it is now the caller's choice, defaulting to the old
            # behaviour.
            vals = band_i.GetStatistics(approx_ok, True)
        except RuntimeError:
            # when the GetStatistics gives an error "RuntimeError: Failed to compute statistics, no valid pixels
            # found in sampling."
            vals = [0]

        if sum(vals) == 0:
            warnings.warn(
                f"Band {band} has no statistics, and the statistics are going to be calculate"
            )
            # Deliberately exact, even when the caller asked for approximate.
            # This branch is only reached because the approximate route already
            # failed or returned nothing usable, so repeating it either raises
            # the same error or answers from the same overviews -- on a sparse
            # band those give min == max and a zero deviation where the full
            # scan gives the real spread. The full scan is the recovery.
            vals = band_i.ComputeStatistics(False)

        return self._unpack_stats(
            list(vals), *self._ds._effective_packing(band if band is not None else 0)
        )

    @staticmethod
    def _unpack_stats(values: list[float], scale: Any, offset: Any) -> list[float]:
        """Put `[min, max, mean, std]` into physical units when the band is packed.

        `stats` never reads a pixel -- it asks GDAL, which answers in the stored units. So
        a packed band reported raw counts while `read_array` returned physical values, and
        the two disagreed by the packing factor (#1124).

        No re-read is needed to fix that. CF packing is affine, `real = raw * scale +
        offset`, so the location statistics shift and scale while the spread only scales:
        `min`, `max` and `mean` take the full transform, `std` takes `|scale|` alone,
        because an additive offset moves a distribution without widening it. A negative
        scale would swap min and max, so they are reordered rather than left crossed.

        The pair is supplied by the owning dataset (`_effective_packing`) rather than
        read off the GDAL band here. Reading the band directly made `stats` and
        `read_array` resolve the packing from different places, and on a `NetCDF`
        variable carrying `_scale` in Python over an unpacked band they answered 19.0
        and 48.0 for the same maximum.

        Args:
            values: `[min, max, mean, std]` in stored units.
            scale: The band's `scale_factor`, or `None`.
            offset: The band's `add_offset`, or `None`.

        Returns:
            list[float]: The same four numbers in physical units, or unchanged when the
                band is not packed.
        """
        if _is_identity_packing(scale, offset) or len(values) != 4:
            return values
        factor = 1.0 if scale is None else float(scale)
        shift = 0.0 if offset is None else float(offset)
        low, high, mean, std = (float(v) for v in values)
        low, high = sorted((low * factor + shift, high * factor + shift))
        return [low, high, mean * factor + shift, std * abs(factor)]

    def _require_band(self, band: int) -> None:
        """Refuse a band index the dataset does not have.

        Indexing the no-data sentinel tuple with an out-of-range band answers
        `IndexError: tuple index out of range`, which names neither the band
        nor the dataset. A negative index is worse: it selects a real band from
        the other end and answers for the wrong one, unless something further
        down happens to catch it.

        Args:
            band: The band index the caller asked for.

        Raises:
            ValueError: `band` is negative or beyond the last band.
        """
        if not 0 <= band < self._ds.band_count:
            raise ValueError(
                f"band {band} is out of range for a {self._ds.band_count}-band dataset."
            )

    def count_domain_cells(self, band: int = 0) -> int:
        """Count the cells inside the domain -- every cell the no-data sentinel does not mark.

        The band is streamed in row strips, so a very large or `/vsicurl` raster is
        never read whole. Which cell is a gap is judged against the band's **stored**
        values, where `no_data_value` lives, so a CF-packed band (`scale_factor` /
        `add_offset`) is counted the same as an unpacked one: its gaps hold the stored
        sentinel, not the physical number a default read shows there. A band that
        declares no sentinel has no gaps, and every cell counts.

        Args:
            band (int):
                Zero-based band index. Default is 0.

        Returns:
            int:
                Number of cells the band holds data in.

        Raises:
            ValueError: `band` is out of range for the dataset.

        Examples:
            - One of four cells is a gap:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ...     no_data_value=-9999.0,
                ... )
                >>> ds.count_domain_cells()
                3

                ```
            - A packed band's gap is found in its stored counts, although a default read
              shows it as `-99.99`:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> packed = Dataset.from_array(
                ...     np.array([[100, 200], [300, -9999]], dtype="int16"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ...     no_data_value=-9999,
                ... )
                >>> packed.scale = [0.01]
                >>> packed.count_domain_cells()
                3

                ```
            - A band index the dataset does not have is refused:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.ones((2, 2), dtype="float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> ds.count_domain_cells(band=3)
                Traceback (most recent call last):
                    ...
                ValueError: band 3 is out of range for a 1-band dataset.

                ```

        See Also:
            domain_area: The same cells weighed by their ground area.
        """
        self._require_band(band)
        no_data_value = self._ds.no_data_value[band]

        # Count the no-data cells directly rather than counting the *non-zero* values
        # among them. `count_nonzero(arr[mask])` asks "how many no-data cells hold a
        # non-zero value", which equals the no-data count only while the sentinel
        # happens to be non-zero; with `no_data_value == 0` it is always 0, so nothing
        # was subtracted and every cell counted as domain.
        def _count(acc: int, strip: np.ndarray, _window: list[int]) -> int:
            return acc + int(is_stored_no_data(strip, no_data_value).sum())

        # Stream the count in row strips so a very large or /vsicurl source is never
        # read whole (#967). A summed count is order-independent, so the tiled total
        # is byte-identical to the whole-band count.
        # Stored counts: the fold matches the stored sentinel, which a physical strip
        # never contains on a packed band -- every gap would count as a cell.
        no_data_count = self._ds.io.stream_reduce(_count, 0, band=band, unpack=False)
        domain_count = self._ds.rows * self._ds.columns - no_data_count
        return int(domain_count)

    def domain_area(self, band: int = 0, unit: str = "m2") -> float:
        """Ground area of the cells inside the domain -- the weighted count.

        :meth:`count_domain_cells` weighs every cell the same, which on a
        geographic grid is wrong by the ratio of the latitudes involved: a
        1-degree cell just below 80 degrees north covers 2 272 km2 and one at the
        equator 12 308 km2, so counting them alike overstates a polar domain
        by roughly four times. This asks the same question in ground units.

        The cells are the same ones `count_domain_cells` counts -- whatever the
        band's no-data sentinel does not mark, judged against the band's stored
        values, so a CF-packed band's gaps are left out like any other -- and the
        two compose rather than introducing a second idea of what "inside" means.

        Args:
            band: Band index. Default is 0.
            unit: `m2` (default), `km2` or `ha`. Case and surrounding
                whitespace are ignored, so `KM2` and `" km2 "` also work.

        Returns:
            float: The summed area of the band's valid cells.

        Raises:
            CRSError: The raster has no CRS. A `ValueError` subclass, so an
                `except ValueError` still catches it.
            ValueError: `band` is out of range for the dataset -- validated
                before the areas are asked for, so a bad band on a raster that
                also has no CRS is still reported as a bad band -- `unit` is
                not recognised, or the raster is geographic and rotated. See
                :meth:`Cell.cell_area` for the rest of the CRS and geotransform
                conditions it defers to.

        Examples:
            - A global 1-degree grid with no gaps covers the whole ellipsoid,
              which is the check that the weighting is right:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> geo_ref = GeoReference(top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326)
                >>> grid = Dataset.from_array(np.ones((180, 360), "float32"), geo_ref=geo_ref)
                >>> round(grid.domain_area(unit="km2") / 1e6, 3)
                510.066

                ```
            - Masking everything below 60 degrees north leaves the polar cap,
              where an unweighted count would be nearly four times out:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> geo_ref = GeoReference(top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326)
                >>> values = np.ones((180, 360), "float32")
                >>> values[30:, :] = -9999.0
                >>> cap = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
                >>> round(cap.domain_area(unit="km2") / 1e6, 3)
                34.416

                ```
            - The same cap counted rather than weighed, which is the error this
              method removes -- nearly fourfold at that latitude:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> geo_ref = GeoReference(top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326)
                >>> values = np.ones((180, 360), "float32")
                >>> values[30:, :] = -9999.0
                >>> cap = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
                >>> nominal = float(cap.cell_area(unit="km2")[90, 0])
                >>> round(cap.count_domain_cells() * nominal / cap.domain_area(unit="km2"), 1)
                3.9

                ```

        See Also:
            count_domain_cells: The unweighted count this refines.
            Cell.cell_area: The per-cell areas this sums.
        """
        # Checked before `cell_area`, so a bad band does not pay for the row
        # integration first, nor report a CRS problem when the raster has both.
        self._require_band(band)
        areas = self._ds.cell.cell_area(unit=unit)
        # One weight per row. Every cell in a row shares an area, so the fold
        # needs the column only to count -- see `_sum`.
        per_row = areas[:, 0]
        no_data_value = self._ds.no_data_value[band]

        def _sum(acc: float, strip: np.ndarray, window: list[int]) -> float:
            # `window` is [xoff, yoff, xsize, ysize]. Only the row offset and
            # height are read, which ties this to `stream_reduce`'s contract of
            # full-width strips: a tiled window would count columns it was not
            # given the weights for. That is a coupling, not the
            # shape-independence an earlier comment here claimed.
            #
            # Counting per row and then taking one dot product costs `ysize`
            # numbers. Multiplying the strip by its weights instead would
            # allocate a dense float64 the size of the strip -- 205 MB per
            # strip on a 100 000-column band, eight times what the unweighted
            # `count_domain_cells` needs -- purely to scale by a value that is
            # constant along each row.
            yoff, ysize = window[1], window[3]
            inside = ~is_stored_no_data(strip, no_data_value)
            counts = np.count_nonzero(inside, axis=1)
            return acc + float(counts @ per_row[yoff : yoff + ysize])

        # Streamed in row strips for the same reason `count_domain_cells` is:
        # a very large or `/vsicurl` band is never held whole, and a sum is
        # order-independent so the tiled total matches the whole-band one.
        # Stored counts, for the same reason as `count_domain_cells`: "inside" is
        # decided against the stored sentinel.
        return float(self._ds.io.stream_reduce(_sum, 0.0, band=band, unpack=False))

    def apply(
        self,
        func,
        band: int = 0,
        inplace: bool = False,
        *,
        elementwise: bool = False,
    ) -> Dataset | None:
        """Apply a function to all domain cells.

        - apply method executes a mathematical operation on the raster array.
        - The function is applied to all domain cells at once using vectorized NumPy operations.

        **`func` works in physical units and computes new values.** On a CF-packed band
        (`scale_factor` / `add_offset`) the domain values are unpacked before `func` sees
        them, exactly as `read_array` returns them. Which cells are domain is still judged
        against the stored counts, because `no_data_value` is a stored value, so the
        sentinel never reaches `func`. Having computed new values, the result has spent the
        packing: it declares none (`scale == [1.0]`), holds `func`'s output as written, and
        fills its no-data cells with the source's `no_data_value`. Its dtype is wide
        enough for both the values `func` is given and what it returns, probed from a
        single domain value, rather than the source's stored type, so a float-valued
        function on an integer or packed band is not truncated on the way back.

        Args:
            func (function):
                Defined function taking one input: the band's domain values as a
                flat array (one tile's, under `elementwise=True`), in physical units.
                A callable that only accepts scalars still works — it is lifted with
                `np.vectorize` — but the whole array is what it is offered
                first, not one cell at a time.
            band (int):
                Zero-based index of the band to transform. Default is `0`. The result
                has this one band only.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created.
                Default is False. In place, the dataset's packing is spent along with its values: it
                declares none afterwards, and a `NetCDF` variable drops its own `_scale` / `_offset`
                too, so the next read does not apply the recipe to the computed values a second time.
            elementwise (bool):
                Opt-in streaming mode. When `True`, `func` is applied one tile at
                a time instead of to the whole band at once, so a very large or
                `/vsicurl` source is never materialised whole. Only pass `True`
                when `func` is a genuine **per-pixel** map (e.g. `np.abs`,
                `lambda v: v * 2 + 1`): the tiled result is then byte-identical to
                the default whole-array pass. A `func` that depends on the whole
                array -- a min/max normalisation, a rank, any global reduction --
                would give a different result tiled, so it must keep the default
                `False`. Default `False` (whole-array, unchanged behaviour).

        Returns:
            Dataset | None:
                A new single-band Dataset with the function applied, or `None` when
                `inplace=True` -- the `Dataset.apply` facade substitutes the real
                `self` in that case (this collaborator only holds a `weakref.proxy`
                back-reference, so it cannot satisfy an `is` identity check itself).

        Raises:
            TypeError: `func` is not callable.

        Examples:
            - Create a dataset from an array filled with values between -1 and 1:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.uniform(-1, 1, size=(5, 5))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> print(dataset.read_array()) # doctest: +SKIP
              [[ 0.94997539 -0.80083622 -0.30948769 -0.77439961 -0.83836424]
               [-0.36810158 -0.23979251  0.88051216 -0.46882913  0.64511056]
               [ 0.50585374 -0.46905902  0.67856589  0.2779605   0.05589759]
               [ 0.63382852 -0.49259597  0.18471423 -0.49308984 -0.52840286]
               [-0.34076174 -0.53073014 -0.18485789 -0.40033474 -0.38962938]]

              ```

            - Apply the absolute function to the dataset:

              ```python
              >>> abs_dataset = dataset.apply(np.abs)
              >>> print(abs_dataset.read_array()) # doctest: +SKIP
              [[0.94997539 0.80083622 0.30948769 0.77439961 0.83836424]
               [0.36810158 0.23979251 0.88051216 0.46882913 0.64511056]
               [0.50585374 0.46905902 0.67856589 0.2779605  0.05589759]
               [0.63382852 0.49259597 0.18471423 0.49308984 0.52840286]
               [0.34076174 0.53073014 0.18485789 0.40033474 0.38962938]]

              ```

            - On a CF-packed band `func` sees physical values, the no-data cell is skipped,
              and the result declares no packing of its own:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> packed = Dataset.from_array(
              ...     np.array([[100, 200], [300, -9999]], dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
              ...     no_data_value=-9999,
              ... )
              >>> packed.scale = [0.01]
              >>> doubled = packed.apply(lambda values: values * 2)
              >>> doubled.read_array().tolist()
              [[2.0, 4.0], [6.0, -9999.0]]
              >>> doubled.scale, float(doubled.no_data_value[0])
              ([1.0], -9999.0)

              ```

            - A float-valued function on an integer band widens the result instead of
              truncating it, on the tiled path as well:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> counts = Dataset.from_array(
              ...     np.array([[1, 2], [3, 4]], dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
              ... )
              >>> halved = counts.apply(lambda values: values / 2, elementwise=True)
              >>> halved.dtype, halved.read_array().tolist()
              (['float64'], [[0.5, 1.0], [1.5, 2.0]])

              ```

            - In place, a packed band's recipe is spent with its values, so the next
              read does not scale the computed numbers again:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> packed = Dataset.from_array(
              ...     np.array([[100, 200]], dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
              ... )
              >>> packed.scale = [0.01]
              >>> _ = packed.apply(lambda values: values * 2, inplace=True)
              >>> packed.read_array().tolist(), packed.scale
              ([[2.0, 4.0]], [1.0])

              ```

        See Also:
            Analysis.combine: The two-raster counterpart, for a difference,
                ratio or any other binary operation.
        """
        if not callable(func):
            raise TypeError("The second argument should be a function")

        no_data_value = self._ds.no_data_value[band]
        # What the output buffer is pre-filled with. It only ever survives in cells the
        # domain excludes, so it is the sentinel when the band declares one. A band that
        # declares none excludes no cell -- every value is overwritten -- and then any
        # value the dtype can hold will do; `None` cannot be one, which is what made
        # `np.full(shape, None, dtype=int16)` raise on a plain integer GeoTIFF.
        buffer_fill = 0 if no_data_value is None else no_data_value

        if elementwise:
            # The tiled path writes into the destination as it goes, so the dtype has
            # to be settled before the first tile. Probe it from the function itself
            # rather than assuming the source's: a float-valued function on an integer
            # raster used to be written back as integers and silently truncated
            # (#1124). A probe that raises falls back to the old behaviour, so an
            # awkward callable is no worse off than before.
            result_dtype = self._elementwise_result_dtype(func, band)
            dst_obj = self._ds.__class__._build_dataset(
                self._ds.columns,
                self._ds.rows,
                1,
                numpy_to_gdal_dtype(result_dtype),
                self._ds.geotransform,
                self._ds.crs,
                no_data_value,
            )
            self._apply_elementwise_tiled(
                func, band, buffer_fill, dst_obj, result_dtype
            )
        else:
            # `band=` as a keyword, never positional: NetCDF.read_array puts
            # `variable` first, so read_array(band) mis-binds on a variable view.
            src_array, domain_mask = self._domain_read(band)
            new_array = np.full(
                (self._ds.rows, self._ds.columns),
                buffer_fill,
                # The domain goes to the probe on this arm too: cell [0, 0] is often
                # the sentinel, and a func that refuses it fell back to the source
                # dtype and truncated -- the tiled arm had this fixed, this one not.
                dtype=self._storable_dtype(func, src_array, domain_mask),
            )
            self._apply_func_to_domain(
                func, src_array, new_array, no_data_value, domain_mask
            )
            dtype = numpy_to_gdal_dtype(new_array.dtype)
            dst_obj = self._ds.__class__._build_dataset(
                self._ds.columns,
                self._ds.rows,
                1,
                dtype,
                self._ds.geotransform,
                self._ds.crs,
                no_data_value,
            )
            dst_obj.raster.GetRasterBand(1).WriteArray(new_array)

        if inplace:
            self._ds._update_inplace(dst_obj.raster)
            # The values are physical now, so the recipe is spent (rule 2).
            self._ds._spend_packing()
            return None
        return dst_obj

    def _domain_read(self, band: int, **read_kwargs) -> tuple[Any, Any]:
        """The band's physical values, with the domain mask judged in stored units.

        The two halves of the CF contract meet here. `read_array` answers in physical
        units since #1124, but `no_data_value` is — per CF, per GDAL, and because it is
        what gets written back — a **stored** value. Comparing the two directly matches
        nothing on a packed raster: a band whose sentinel is `-9999` reads back
        `-98.49`, so every no-data cell is silently promoted to a real measurement in
        whatever reduction, extraction or render asked the question.

        So the mask is built where the sentinel lives, against the stored counts, and
        the values are unpacked afterwards. The two are the same set of cells — the
        packing is affine and injective — but only this order is exact, and only this
        order keeps working when the sentinel is `NaN` on a float band.

        Args:
            band: Zero-based band index.
            **read_kwargs: Forwarded to `read_array` (`window=`, and so on).

        Returns:
            tuple: `(values, domain_mask)` — the physical array, and `True` wherever a
                cell holds a real measurement.
        """
        raw = np.asarray(self._ds.read_array(band=band, unpack=False, **read_kwargs))
        domain_mask = ~is_stored_no_data(raw, self._ds.no_data_value[band])
        values = apply_unpack(raw, *self._ds._effective_packing(band))
        return values, domain_mask

    def _physical_no_data(self, band: int) -> Any:
        """The band's sentinel as it appears in a default (physical) read.

        `no_data_value` is a **stored** value — CF puts `_FillValue` in the packed
        datatype, GDAL reports it that way, and it is what gets written back. Most
        callers want a *mask*, and those build it in stored units through
        `_domain_read`. A few instead need the sentinel as a *value*, because they
        hand it to something that will do its own comparison against the physical
        array — cleopatra's `exclude_value` (`plot`, the collection and NetCDF
        animations), `get_pixels2` / `get_indices2`'s exclude lists (`extract`,
        `overlay`), the feature table `to_feature_collection` drops rows from, an
        ASCII header, the terrain-RGB encoder, the focal kernels' gap search, or the
        no-data a materialised Zarr store declares. Those need the same number the
        array actually holds.

        The pair applied is `Dataset._effective_packing(band)`, the same one the read
        uses, so the two cannot disagree.

        Args:
            band: Zero-based band index.

        Returns:
            The sentinel in physical units as a `float`, or the sentinel unchanged when
            the band declares no usable packing (and `None` stays `None`).

        Examples:
            - The stored `-9999` of a band packed at `scale=0.5` reads as `-4999.5`:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> packed = Dataset.from_array(
                ...     np.array([[100, -9999]], dtype="int16"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ...     no_data_value=-9999,
                ... )
                >>> packed.scale = [0.5]
                >>> packed.analysis._physical_no_data(0), float(packed.read_array()[0, 1])
                (-4999.5, -4999.5)

                ```
            - An unpacked band's sentinel is returned as declared:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.array([[1.0, -9999.0]]),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ...     no_data_value=-9999.0,
                ... )
                >>> float(ds.analysis._physical_no_data(0))
                -9999.0

                ```
        """
        sentinel = self._ds.no_data_value[band]
        scale, offset = self._ds._effective_packing(band)
        if sentinel is None or _is_identity_packing(scale, offset):
            return sentinel
        return float(
            apply_unpack(np.asarray([sentinel], dtype="float64"), scale, offset)[0]
        )

    def _elementwise_result_dtype(self, func, band: int) -> np.dtype:
        """The dtype `func`'s result needs, probed from a one-element call.

        The tiled path commits to a destination type before it has produced a single
        value, so the type has to be predicted. Applying the function to one domain value
        answers it exactly for the vectorised callables `apply` documents, and anything
        that cannot survive that call keeps the source's type -- the behaviour before
        #1124, so no worse.

        Both the destination band **and** the per-tile output buffer are built from
        this one answer. Building only the band from it left the buffer at the source
        tile's type, which rounded a float result before it ever reached the wider
        band: `apply(lambda a: a * 0.01, elementwise=True)` on the issue's own numbers
        still returned `[14.0, 12.0, 10.0]` while the whole-array arm returned
        `[14.35, 12.72, 10.0]`.

        Only the **first tile** is read, and the probe value is taken from its
        *domain*. Reading the whole band here defeated the mode this is for -- whose
        stated purpose is that a very large or `/vsicurl` source is never materialised
        whole -- and on a packed source that full read comes back `float64`, eight
        bytes a pixel of exactly the array `elementwise=True` exists to avoid. Taking
        the domain rather than cell `[0, 0]` matters because that cell is often the
        no-data sentinel, and a `func` that raises on it would fall back to the source
        dtype and quietly restore the truncation.

        Args:
            func: The callable `apply` was given.
            band: The band index being read.

        Returns:
            np.dtype: A dtype wide enough for the result, which GDAL can store.
        """
        resolved = np.dtype(self._ds.numpy_dtype[band])
        try:
            window = next(iter(self._ds.io._tile_offsets()))
            tile, domain = self._domain_read(band, window=list(window))
            resolved = self._storable_dtype(func, np.asarray(tile), domain)
        except Exception:
            logger.debug("could not probe the result dtype for apply", exc_info=True)
        return resolved

    @staticmethod
    def _storable_dtype(func, source: np.ndarray, domain=None) -> np.dtype:
        """The narrowest dtype that holds `func`'s result and that GDAL can store.

        Probed by calling `func` on one domain value, the way `apply` will call it
        (`_probe_call`: the array first, then lifted through `np.vectorize`). Writing the
        result back at the source's type truncated a float-valued function on an integer
        raster (#1124), so the result's own type wins -- but only when GDAL has a matching
        type. A callable that yields `object`, or one that fails on the sample even when
        lifted, keeps the source's type: the behaviour before #1124, and better than
        refusing to run.

        Args:
            func: The callable `apply` was given.
            source: The array being read, supplying the probe value and the fallback type.
            domain: Which cells hold measurements, when the caller knows. The probe is
                taken from one of those rather than from cell `[0, 0]`, which is often
                the no-data sentinel -- a `func` that raises on the sentinel would fall
                back to the source dtype and restore the truncation this exists to fix.

        Returns:
            np.dtype: A dtype `numpy_to_gdal_dtype` accepts.
        """
        resolved = source.dtype
        flat = np.asarray(source).reshape(-1)
        if domain is not None:
            candidates = flat[np.asarray(domain).reshape(-1)]
            if candidates.size:
                flat = candidates
        try:
            probe = Analysis._probe_call(func, flat[:1])
            promoted = np.result_type(source.dtype, probe.dtype)
            numpy_to_gdal_dtype(promoted)
        except Exception:
            logger.debug("keeping the source dtype for apply", exc_info=True)
        else:
            resolved = promoted
        return resolved

    @staticmethod
    def _probe_call(func, sample: np.ndarray) -> np.ndarray:
        """Call `func` on a one-element sample the way `apply` itself will call it.

        `apply` hands a vectorised callable the array and lifts a scalar-only one --
        `math.sqrt`, `math.log` -- through `np.vectorize` when the array call raises. The
        probe has to follow the same path, or it predicts a type for a call that never
        happens: `math.sqrt` refused the array, the probe fell back to the source dtype,
        and the lifted call then wrote `sqrt(2)` into an `int16` buffer as `1`.

        Args:
            func: The callable `apply` was given.
            sample: A one-element array taken from the domain.

        Returns:
            np.ndarray: The result, whose dtype is the prediction.

        Raises:
            Exception: Whatever `func` raises when lifted through `np.vectorize` as
                well, and anything other than `TypeError` / `ValueError` from the
                array call. `_storable_dtype` catches it and keeps the source dtype.

        Examples:
            - A scalar-only function refuses the array and is lifted, as `apply` lifts
              it, so the probe predicts the float result it will really return:
                ```python
                >>> import math
                >>> import numpy as np
                >>> from pyramids.dataset.engines.analysis import Analysis
                >>> Analysis._probe_call(math.sqrt, np.array([2], dtype="int16")).dtype
                dtype('float64')

                ```
            - A vectorised function is called on the array directly:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset.engines.analysis import Analysis
                >>> probe = Analysis._probe_call(lambda v: v * 2, np.array([2], dtype="int16"))
                >>> probe.tolist(), probe.dtype
                ([4], dtype('int16'))

                ```
        """
        try:
            result = np.asarray(func(sample))
        except (TypeError, ValueError):
            result = np.asarray(np.vectorize(func)(sample))
        return result

    @staticmethod
    def _apply_func_to_domain(
        func, src_array, out_array, no_data_value, domain_mask=None
    ) -> None:
        """Apply `func` to the domain (non-no-data) cells of `src_array` into `out_array`.

        Args:
            func: The per-domain-values callable to apply.
            src_array: The source array supplying the domain values.
            out_array: The pre-filled output array written in place.
            no_data_value: The value marking cells to exclude from the domain. Used
                only when `domain_mask` is not supplied.
            domain_mask: The domain, when the caller has already resolved it — which a
                caller reading a packed band must, since `src_array` is then physical
                while `no_data_value` is stored and the two never match.
        """
        if domain_mask is None:
            domain_mask = ~is_stored_no_data(src_array, no_data_value)
        domain_values = src_array[domain_mask]
        # An empty domain (an all-no-data tile, common when streaming) needs no
        # write -- out_array is already the no-data fill -- and short-circuiting
        # here avoids `np.vectorize(func)` raising "cannot call 'vectorize' on
        # size 0 inputs" on a fully-masked tile, keeping the tiled path
        # byte-identical to the whole-array pass (#969).
        if domain_values.size == 0:
            return
        try:
            out_array[domain_mask] = func(domain_values)
        except (ValueError, TypeError):
            out_array[domain_mask] = np.vectorize(func)(domain_values)

    def _apply_elementwise_tiled(
        self, func, band, no_data_value, dst_obj, result_dtype
    ) -> None:
        """Apply an elementwise `func` over one band tile by tile, out of core.

        Reads the band a square window at a time, applies `func` to that tile's
        domain values, and writes the block straight into `dst_obj`, so the full
        band is never materialised. For a per-pixel `func` the result is
        byte-identical to the whole-array path (#969).

        Args:
            func: The per-pixel callable to apply to each tile's domain values.
            band: Zero-based index of the source band to transform.
            no_data_value: What the tile buffer is pre-filled with, and so what the
                cells the domain excludes keep -- the band's sentinel, or a storable
                placeholder when it declares none (and so excludes nothing).
            dst_obj: The single-band destination Dataset written in place.
            result_dtype: The dtype the destination band was built at. The buffer has
                to match it, not the source tile's: allocating at the tile's type
                rounded a float result inside the buffer before it ever reached the
                wider band, so widening the band alone fixed nothing here.
        """
        dst_band = dst_obj.raster.GetRasterBand(1)
        for xoff, yoff, xsize, ysize in self._ds.io._tile_offsets():
            tile, domain_mask = self._domain_read(
                band, window=[xoff, yoff, xsize, ysize]
            )
            new_tile = np.full(tile.shape, no_data_value, dtype=result_dtype)
            self._apply_func_to_domain(func, tile, new_tile, no_data_value, domain_mask)
            dst_band.WriteArray(new_tile, xoff, yoff)

    def combine(
        self,
        other: Dataset,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        band: int | None = None,
        no_data_value: Any = _DERIVE_NO_DATA,
    ) -> Dataset:
        """Combine this dataset with a second one cell by cell, keeping the grid.

        The binary counterpart of :meth:`apply`: `func` receives the two
        rasters' matching cells and the result is wrapped back into a
        :class:`~pyramids.dataset.Dataset` carrying this dataset's geotransform
        and CRS — so a difference, a ratio or a per-cell maximum never leaves
        the `Dataset` and the georeferencing cannot be rebuilt wrongly on the
        way out.

        The operands must already share a grid. `combine` does **not** resample:
        :meth:`Dataset.align <pyramids.dataset.Dataset.align>` is the explicit
        step for that, and applying it implicitly here would silently resample
        data inside what reads as pure arithmetic.

        It is a whole-array operation: both operands are read in full and peak
        memory runs to several times one band, with no tiled or lazy path of its
        own. For rasters near the memory limit, reach for
        :meth:`apply(elementwise=True) <Analysis.apply>`, which streams a single
        raster tile by tile, or `read_array(chunks=)` and dask.

        `func` works in physical units. A CF-packed operand (`scale_factor` /
        `add_offset`) is unpacked with its own packing before `func` sees it, so a
        packed raster and an unpacked one combine in the same units, while each
        operand's no-data cells are still found against its **stored** sentinel.
        The result holds computed values and declares no packing.

        A `NetCDF` variable view combines like any other raster, and the result
        is built with the **left** operand's class -- the same rule
        :meth:`apply` follows for its one operand -- so a `Variable` on the left
        comes back as a `Variable` wrapping a plain in-memory raster, while the
        reverse pairing gives a plain `Dataset`. Write it with `.nc`, or read the array out and
        wrap it with :meth:`Dataset.from_array`; a `.tif` destination is refused
        by the NetCDF writer.

        Args:
            other (Dataset):
                The second operand. Must occupy this dataset's grid and CRS
                (:meth:`Spatial.same_grid <pyramids.dataset.engines.Spatial.same_grid>`).
                Passing **this same dataset** is allowed and reads it twice, as
                two operands always are, so `func` receives two independent arrays
                and may mutate either. A one-operand transform that wants this
                method's contract — every band, a dtype from the computed values,
                a derived sentinel — should ask for :meth:`_fold`, the same
                machinery over a single read and the route `ds * 2` takes. `_fold`
                is private because it hands `func` the *same* array as both
                arguments: a callable that writes into its arguments would see the
                other change under it.
            func (Callable):
                Callable taking the two operands' cell values — as flat arrays,
                the same contract :meth:`apply` uses — and returning one array
                of the same length.
            band (int, optional):
                Zero-based band to combine, producing a single-band result. The
                default `None` combines every band, which then requires both
                rasters to carry the same band count. Note this differs from
                :meth:`apply`, which defaults to band `0`.
            no_data_value (Any, optional):
                Sentinel for the result, one value across every band. Left
                unset it is derived from the result's dtype, and for an integer
                result *against the values* `func` computed, so no in-range
                number is claimed as a gap by the arithmetic that produced it:

                * floating result — `NaN`, always. A `NaN` is not a measurement,
                  so a cell `func` computes as `NaN` (`0/0` in a normalised
                  difference, `log` of a negative) **is** a gap and the result
                  says so. That is the one case where a derived sentinel marks
                  a computed value, and it is deliberate;
                * predicate's Byte result — `255`, free beside `0` and `1`;
                * integer result that masked nothing — no sentinel at all;
                * integer result that masked something — the first of the
                  operands' own sentinels, the package default, then the dtype's
                  extremes that both fits and occurs nowhere in the result --
                  and, for the narrow integer widths, the rest of the range
                  after those.

                Pass an explicit value to choose it — it is honoured, with a
                :class:`~pyramids.errors.NoDataCollisionWarning` if the result
                holds it — or `None` for a result with no sentinel, which also
                switches off the domain masking so every cell is handed to
                `func`, including the ones the inputs marked as no-data.

        Returns:
            Dataset:
                A new in-memory dataset on this dataset's grid, holding the
                combined values. Cells that are no-data in *either* operand are
                no-data in the result.

        Raises:
            TypeError: `other` is not a Dataset, or `func` is not callable.
            AlignmentError: The two rasters do not share a grid/CRS.
            ValueError: `band` is `None` and the band counts differ; `band` is
                out of range for either operand (raised by the read, which names
                the band but not which side); an explicit `no_data_value=` does
                not fit the result dtype, or no sentinel is free to mark what
                the operands masked out; `func` returned a different number of
                values than it was given, or a dtype GDAL has no band type for.

        Warns:
            NoDataCollisionWarning: An explicit `no_data_value=` is also a value
                `func` computed, so those cells read back as gaps. Only an
                explicit sentinel can collide — a derived one is chosen against
                the computed values precisely so it cannot.

        Examples:
            - Two aligned rasters, differenced without leaving the Dataset:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
              >>> surface = Dataset.from_array(np.full((20, 20), 30.0, "float32"), geo_ref=geo_ref)
              >>> bare = Dataset.from_array(np.full((20, 20), 22.0, "float32"), geo_ref=geo_ref)
              >>> canopy = surface.combine(bare, lambda a, b: a - b)
              >>> canopy.epsg, canopy.geotransform == surface.geotransform
              (4326, True)
              >>> float(np.asarray(canopy.read_array()).mean())
              8.0

              ```

            - The arithmetic operators are the same call:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
              >>> nir = Dataset.from_array(np.full((4, 4), 0.6, "float32"), geo_ref=geo_ref)
              >>> red = Dataset.from_array(np.full((4, 4), 0.2, "float32"), geo_ref=geo_ref)
              >>> ndvi = (nir - red) / (nir + red)
              >>> round(float(np.asarray(ndvi.read_array()).mean()), 4)
              0.5
              >>> ndvi.shape
              (1, 4, 4)

              ```

            - A *scalar* operator is the same machinery over one read, through
              :meth:`_fold`, and agrees with the two-operand spelling:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
              >>> ds = Dataset.from_array(np.full((4, 4), 5.0, "float32"), geo_ref=geo_ref)
              >>> folded = ds.combine(ds, lambda values, _ignored: values * 2)
              >>> float(np.asarray(folded.read_array()).mean())
              10.0
              >>> float(np.asarray((ds * 2).read_array()).mean())
              10.0

              ```

            - A cell that is no-data on either side stays no-data, and a float
              result declares `NaN`:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
              >>> values = np.full((3, 3), 10.0, "float32")
              >>> values[0, 0] = -9999.0
              >>> masked = Dataset.from_array(values, geo_ref=geo_ref)
              >>> other = Dataset.from_array(np.full((3, 3), 4.0, "float32"), geo_ref=geo_ref)
              >>> result = np.asarray((masked - other).read_array())
              >>> bool(np.isnan(result[0, 0])), float(result[1, 1])
              (True, 6.0)
              >>> float((masked - other).no_data_value[0])
              nan

              ```

            - A raster on a different grid is refused rather than resampled;
              `align` is the explicit step that makes it combinable:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> fine = Dataset.from_array(
              ...     np.full((8, 8), 10.0, "float32"),
              ...     geo_ref=GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326),
              ... )
              >>> coarse = Dataset.from_array(
              ...     np.full((4, 4), 4.0, "float32"),
              ...     geo_ref=GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.5, epsg=4326),
              ... )
              >>> fine - coarse
              Traceback (most recent call last):
                  ...
              pyramids.base._errors.AlignmentError: the two rasters do not share a grid/CRS, ...
              >>> float(np.asarray((fine - coarse.align(fine)).read_array()).mean())
              6.0

              ```

        See Also:
            Analysis.apply: The one-raster counterpart — same domain-values
                contract, one operand.
            Spatial.same_grid: Whether two rasters can be combined without
                resampling.
            Spatial.align: Puts a mismatched raster onto this one's grid, which
                is the explicit step `combine` refuses to take implicitly.
        """
        return self._combine(
            other, func, band=band, no_data_value=no_data_value, folded=False
        )

    def _fold(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        band: int | None = None,
        no_data_value: Any = _DERIVE_NO_DATA,
    ) -> Dataset:
        """Run a **one**-operand callable through :meth:`combine`'s machinery.

        The scalar operators are this: `ds * 2` folds its constant into a callable and
        wants everything `combine` gives a raster operand — every band, a dtype from the
        computed values, a derived sentinel — with only one array to read. Routing it
        through `combine` with this dataset as its own second operand would read,
        CF-unpack and mask that array twice, so the second read is skipped here rather
        than inferred from the operands being equal.

        Args:
            func: A callable taking one flat array of domain values and returning one
                array of the same length.
            band: Zero-based band, or `None` for every band.
            no_data_value: The result's sentinel; the default derives one, and `None`
                turns masking off.

        Returns:
            Dataset: A new raster on this one's grid, carrying `func`'s values.

        Raises:
            TypeError: `func` is not callable.
            ValueError: `band` is out of range; an explicit `no_data_value=` does not
                fit the result dtype, or no sentinel is free to mark what was masked
                out; `func` returned an array of the wrong shape, or a dtype GDAL has
                no band type for.

        Warns:
            NoDataCollisionWarning: An explicit `no_data_value=` is also a value `func`
                computed. The arithmetic operators never pass one, so this is out of
                reach on the route they take.

        Examples:
            - Scale a band, the operation `ds * 2` performs:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
              >>> ds = Dataset.from_array(np.full((4, 4), 5.0, "float32"), geo_ref=geo_ref)
              >>> float(np.asarray(ds.analysis._fold(lambda v: v * 2).read_array()).mean())
              10.0

              ```

            - Every band is folded, not just the first, and a floating result declares
              `NaN` — the contract :meth:`combine` gives, over one read:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
              >>> stack = Dataset.from_array(
              ...     np.stack([np.full((4, 4), 1.0, "float32"), np.full((4, 4), 2.0, "float32")]),
              ...     geo_ref=geo_ref,
              ... )
              >>> shifted = stack.analysis._fold(lambda v: v + 10)
              >>> shifted.band_count
              2
              >>> [float(band.mean()) for band in np.asarray(shifted.read_array())]
              [11.0, 12.0]
              >>> shifted.no_data_value
              (nan, nan)

              ```

            - An integer result that masked nothing declares no sentinel at all, which
              is why an identity such as `ds * 1` short-circuits to `copy()` rather than
              folding — folding it would drop the band's declared no-data value:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
              >>> ints = Dataset.from_array(np.full((4, 4), 3, "int16"), geo_ref=geo_ref)
              >>> float(ints.no_data_value[0])
              -9999.0
              >>> doubled = ints.analysis._fold(lambda v: v * 2)
              >>> str(np.asarray(doubled.read_array()).dtype), doubled.no_data_value
              ('int16', (None,))
              >>> (ints * 1).no_data_value == ints.no_data_value
              True

              ```

        See Also:
            Analysis.combine: The two-operand form this shares its machinery with.
            Analysis.apply: The single-band, sentinel-preserving alternative.
            pyramids.dataset.dataset._is_identity: The short-circuit that keeps a
                no-op off this path.
        """
        return self._combine(
            self._ds,
            lambda values, _ignored: func(values),
            band=band,
            no_data_value=no_data_value,
            folded=True,
        )

    def _combine(
        self,
        other: Dataset,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        band: int | None,
        no_data_value: Any,
        folded: bool,
    ) -> Dataset:
        """Shared body of :meth:`combine` and :meth:`_fold`.

        After the refusal checks and before either operand is read, it asks the left operand's
        `_combine_layout_source` hook about the band layouts, and hands the answer, untouched, to
        that operand's `_label_combined` once the result is built. A plain `Dataset` checks
        nothing, answers `None` and labels nothing. A `NetCDF` refuses band dimensions that do not
        pair up, answers with the operand whose layout describes the result (itself when it has
        band dimensions, else `other` when that has them, else `None`), the dimensions whose
        coordinates disagree and the partner to fill missing labels from, and labels the result
        from those, so `combine`, the operators and `_fold` all keep the layout. A folded call
        hands the hook `None` as the other operand, since its one layout has nothing to be
        compared with or filled from.

        Args:
            other: The second operand. Ignored as a *source* when `folded` is set — it
                is this dataset, and reading it again would only cost.
            func: The binary callable.
            band: Zero-based band, or `None` for every band.
            no_data_value: The result's sentinel, or the derive sentinel.
            folded: Whether the caller has declared both operands to be this dataset.
                Declared, not inferred: an identity test cannot be spelled here (the
                engine holds a `weakref.proxy`, which no `is` can match), and inferring
                it from `==` would rest on no raster class ever defining an elementwise
                `__eq__` — which the four other comparisons make a plausible next
                request.

        Returns:
            Dataset: The combined raster, built with the **left** operand's class and
            carrying this dataset's geotransform, CRS, metadata and band names, plus the band
            dimensions `_label_combined` puts on a `NetCDF` result.

        Raises:
            TypeError: `other` is not a raster, or `func` is not callable.
            AlignmentError: The operands do not share a grid/CRS.
            ValueError: The band counts differ; a `NetCDF` left operand's band dimensions do
                not pair up with `other`'s (raised by `_combine_layout_source`); `band` is out
                of range; an explicit `no_data_value` does not fit the result dtype, or no
                candidate sentinel is both storable and absent from the result; `func` returned
                the wrong shape, or a dtype GDAL has no band type for.

        Warns:
            NoDataCollisionWarning: An explicit `no_data_value` occurs among the values
                `func` computed. The `stacklevel` below is counted for the
                `Dataset.combine` facade, the documented entry point.
        """
        if not isinstance(other, RasterBase):
            raise TypeError(f"`other` must be a Dataset, got {type(other).__name__}")
        self._check_combinable(other, func, band)
        # A fold's second operand is this dataset, as the engine's proxy no identity test can
        # match, so the hook is told there is no other layout rather than handed one to compare.
        layout_source = self._ds._combine_layout_source(None if folded else other, band)

        left, left_sentinels, left_domain = self._operand_arrays(self._ds, band)
        right_sentinels: list[Any]
        if folded:
            # One array, offered to `func` as both arguments and to the sentinel
            # derivation once — its candidates are already in `left_sentinels`.
            right, right_sentinels, right_domain = left, [], left_domain
        else:
            right, right_sentinels, right_domain = self._operand_arrays(other, band)
        masked = no_data_value is not None
        domain: np.typing.NDArray | None
        if not masked:
            # `None`, not an all-True mask: the unmasked path exists because the
            # caller asked for no masking, so allocating a full boolean array and
            # fancy-indexing through it twice is pure overhead.
            domain = None
        elif right_domain is left_domain:
            domain = left_domain
        else:
            domain = left_domain & right_domain

        values, boolean = self._computed_values(func, left, right, domain)
        sentinel = (
            self._resolve_combined_no_data(
                no_data_value,
                values.dtype,
                # Every band of both operands, left first: the result carries
                # one sentinel, and any of them that fits the dtype and does not
                # occur in the result can mark what either operand masked out.
                [*left_sentinels, *right_sentinels],
                values,
                excluded=values.size != left.size,
                boolean=boolean,
            )
            if masked
            else None
        )
        out = self._place_values(values, left.shape, domain, sentinel)

        combined = self._ds.__class__._build_dataset(
            self._ds.columns,
            self._ds.rows,
            1 if out.ndim == 2 else out.shape[0],
            numpy_to_gdal_dtype(out),
            self._ds.geotransform,
            self._ds.crs,
            sentinel,
            array=out,
        )
        # Dataset-level tags travel for the same reason band names do: a result
        # that has forgotten which sensor or scene it came from is harder to use
        # than the arrays it was built from. Band-level scale, offset and units
        # are deliberately not carried -- `func` can change what the numbers
        # mean, and a ratio of two scaled bands is not on either input's scale.
        # Assigned through, not `dict(...)`: `Dataset.meta_data` is a plain dict
        # but `NetCDF.meta_data` is a `NetCDFMetadata`, which is not iterable --
        # coercing it turned every operator on a NetCDF variable into a
        # TypeError. Both setters accept their own type.
        combined.meta_data = self._ds.meta_data
        # Band identity is half the reason to keep the operation inside the
        # Dataset: an NDVI or change-detection stack whose bands come back as
        # `Band_1`, `Band_2` has lost what told the caller which is which.
        combined.band_names = (
            [self._ds.band_names[band]]
            if band is not None
            else list(self._ds.band_names)
        )
        self._ds._label_combined(combined, layout_source)
        return combined

    def _check_combinable(self, other: Dataset, func: Any, band: int | None) -> None:
        """Refuse a pair :meth:`combine` cannot run, before reading any pixels.

        Args:
            other: The second operand.
            func: The binary callable.
            band: The selected band, or `None` for every band.

        Raises:
            TypeError: `func` is not callable.
            AlignmentError: The rasters do not share a grid/CRS.
            ValueError: `band` is `None` and the band counts differ.
        """
        if not callable(func):
            raise TypeError(f"`func` must be callable, got {type(func).__name__}")
        if not self._ds.spatial.same_grid(other):
            raise AlignmentError(
                "the two rasters do not share a grid/CRS, so they cannot be "
                "combined cell by cell; align them first "
                "(`other = other.align(self)`) and combine the result"
            )
        # Before either read: the message is already written in terms of
        # `band_count`, which is a header field, and combining a 12-band scene
        # with a 1-band mask should not pull gigabytes off disk to refuse on a
        # comparison that needed no pixels. `cli.py` orders its own grid check
        # the same way.
        if band is None and self._ds.band_count != other.band_count:
            raise ValueError(
                f"the operands carry a different number of bands "
                f"({self._ds.band_count} and {other.band_count}); pass `band=` to "
                "combine one band from each"
            )

    @classmethod
    def _computed_values(
        cls,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        left: np.ndarray,
        right: np.ndarray,
        domain: np.ndarray | None,
    ) -> tuple[np.typing.NDArray, bool]:
        """Run `func` over the cells `domain` selects and check what came back.

        Args:
            func: The binary callable to apply.
            left: The left operand's array.
            right: The right operand's array, shaped like `left`.
            domain: Boolean mask of the cells to combine, or `None` for all.

        Returns:
            tuple: The computed values as a flat array, and whether `func`
            returned booleans — which GDAL has no band type for, so they are
            promoted to Byte here and given a `255` sentinel later.

        Raises:
            ValueError: `func` returned an array of the wrong shape.
        """
        expected = left.size if domain is None else int(domain.sum())
        left_values = left.ravel() if domain is None else left[domain]
        # `is`, so a folded call (one dataset, both operands) selects once. `left[domain]`
        # allocates a full copy of the domain values, which is the peak-memory half of
        # reading one raster instead of two.
        if right is left:
            right_values = left_values
        elif domain is None:
            right_values = right.ravel()
        else:
            right_values = right[domain]
        values = np.asarray(cls._combine_domain(func, left_values, right_values))
        if values.shape != (expected,):
            # Shape, not just size: a `func` returning a column vector has the
            # right count and the wrong rank, and used to reach the masked
            # assignment and fail there with a numpy message naming neither
            # `func` nor the contract. Returning `(n, 1)` is an ordinary mistake
            # -- it is what a scikit-learn-style predictor does by default.
            raise ValueError(
                f"`func` returned an array of shape {values.shape} for "
                f"{expected} cells; it must return one flat array as long as "
                "the arguments it was given"
            )
        boolean = values.dtype == np.bool_
        return (values.astype("uint8") if boolean else values), boolean

    @staticmethod
    def _place_values(
        values: np.ndarray,
        shape: tuple[int, ...],
        domain: np.ndarray | None,
        sentinel: Any,
    ) -> np.typing.NDArray:
        """Lay the computed values back out on the raster's grid.

        Args:
            values: The flat computed values.
            shape: The shape the result must take.
            domain: The mask `values` was computed over, or `None` for all cells.
            sentinel: The no-data value filling the cells `domain` excluded.

        Returns:
            np.ndarray: The result array, shaped like the operands.
        """
        out: np.typing.NDArray
        if domain is None:
            out = values.reshape(shape)
        else:
            out = np.full(shape, 0 if sentinel is None else sentinel, values.dtype)
            out[domain] = values
        return out

    @staticmethod
    def _operand_arrays(
        ds: Dataset, band: int | None
    ) -> tuple[np.typing.NDArray, list[Any], np.typing.NDArray]:
        """Read one operand for :meth:`combine`, with its sentinels and its domain.

        The domain comes back with the values because the two cannot be derived from
        each other after the fact: the array is physical and the sentinels are stored,
        so the mask has to be taken while the counts are still in hand. Returning them
        separately let the caller compare `-98.49` against `-9999` and find no no-data
        at all on a packed raster.

        Args:
            ds: The dataset to read.
            band: Zero-based band to read, or `None` for every band.

        Returns:
            tuple: The physical array — 2-D for a single band, `(bands, rows, cols)`
            otherwise — the per-band no-data sentinels aligned to its bands, and the
            boolean domain mask shaped like the array.
        """
        # `band=` as a keyword, never positional: NetCDF.read_array puts
        # `variable` first, so read_array(band) mis-binds on a variable view.
        raw = np.asarray(ds.read_array(band=band, unpack=False))
        sentinels = (
            [ds.no_data_value[band]] if band is not None else list(ds.no_data_value)
        )
        domain = Analysis._domain_mask(raw, sentinels)
        indices = [band] if band is not None else range(ds.band_count)
        packing = [ds._effective_packing(index) for index in indices]
        if raw.ndim == 2:
            array = apply_unpack(raw, *packing[0])
        else:
            array = np.stack(
                [apply_unpack(raw[i], *packing[i]) for i in range(raw.shape[0])]
            )
        return np.asarray(array), sentinels, domain

    @staticmethod
    def _domain_mask(array: np.ndarray, sentinels: Sequence[Any]) -> np.typing.NDArray:
        """Boolean mask, True where a cell holds data rather than its band's sentinel.

        Args:
            array: A 2-D band or a 3-D `(bands, rows, cols)` stack.
            sentinels: One no-data sentinel per band of `array`.

        Returns:
            np.ndarray: A boolean array shaped like `array`.
        """
        if array.ndim == 2:
            mask = ~is_stored_no_data(array, sentinels[0])
        else:
            mask = np.stack(
                [
                    ~is_stored_no_data(array[index], sentinels[index])
                    for index in range(array.shape[0])
                ]
            )
        return mask

    @staticmethod
    def _combine_domain(
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Apply a binary `func` to two aligned flat arrays of domain values.

        Mirrors :meth:`_apply_func_to_domain`, including its empty-domain guard:
        a vectorized callable is used as given, a scalar-only one is lifted with
        `np.vectorize` rather than rejected, and neither is asked to run over an
        empty domain.

        Note the cost of that lift. The fallback is reached through a blanket
        `except`, so a `ValueError` or `TypeError` raised *inside* a vectorized
        `func` -- a bad cast, a shape mistake, a bug in the caller's own code --
        is retried one cell at a time instead of surfacing. On a full-domain
        raster that is one Python call per pixel, so the caller sees a long
        stall rather than their exception.

        Args:
            func: The binary callable to apply.
            left: Domain values of the left operand.
            right: Domain values of the right operand, aligned to `left`.

        Returns:
            np.ndarray: The combined values, aligned to `left`.
        """
        try:
            values = np.asarray(func(left, right))
        except (ValueError, TypeError):
            if left.size == 0:
                # Two fully-masked operands leave nothing to compute, and
                # `np.vectorize` refuses size-0 inputs outright ("unless
                # `otypes` is set") -- so the scalar-callable path died on a
                # raster the ufunc path handled. An all-no-data pair is not
                # exotic: it is the normal state of an ocean tile in a land
                # product. #969 closed the same hole in `apply`.
                #
                # `result_type`, not a hard-coded `float64`: this arm is only
                # reached for a scalar-only `func`, and a fixed dtype here would
                # make the same expression over the same all-masked inputs come
                # back `float64`/NaN spelled one way and `int32`/-9999 spelled
                # the other -- half a mosaic in each band type.
                values = np.empty(0, dtype=np.result_type(left.dtype, right.dtype))
            else:
                values = np.asarray(np.vectorize(func)(left, right))
        return values

    @classmethod
    def _resolve_combined_no_data(
        cls,
        requested: Any,
        dtype: np.dtype,
        candidates: Sequence[Any],
        values: np.ndarray,
        *,
        excluded: bool,
        boolean: bool,
    ) -> Any:
        """Pick the sentinel :meth:`combine` stamps on its result.

        A sentinel is a real value of the band's dtype, so for an in-range
        candidate the one question that matters is whether `func` also computed
        it. Inheriting an operand's sentinel blind is how a whole result silently
        becomes no-data: `uint8` `200 + 55` lands exactly on `255`, the sentinel
        `from_array` gives every `uint8` band, and `int32` `0 - 9999` lands on
        the package default. So an integer sentinel is chosen *against the
        computed values*, and only when there is something to mark. The rule is
        therefore per dtype, not per gap:

        * an explicit `requested` is honoured, but the caller warns when it
          collides;
        * a floating result takes `NaN`. Unlike an integer sentinel this is not
          checked against the values, and does not need to be: `NaN` is not a
          measurement, so a cell `func` computed as `NaN` genuinely has no value
          and the result marking it as a gap is the right answer, not a
          collision. It is stamped whether or not anything was masked, since it
          can never turn a real number into a gap;
        * a predicate's Byte result takes `255`, free beside `0` and `1`, for the
          same reason;
        * an integer result that masked nothing declares no sentinel at all --
          there is no gap to mark, and every in-range value would be a lie;
        * otherwise the first candidate that fits the dtype and occurs nowhere
          in the result, searched through the operands' own sentinels, then the
          package default, then the dtype's extremes (max before min for an
          unsigned dtype, whose min is the very usable `0`). For the narrow
          integer widths the search continues into the rest of the range rather
          than refusing there, walking inward from the preferred extreme, so an
          `int8` result holding every candidate is answered as long as any of
          its 256 values is unused. `int32` and wider still refuse once the
          candidates are gone.

        Args:
            requested: The caller's `no_data_value`, or `_DERIVE_NO_DATA` when
                it was left unset.
            dtype: The dtype `func` produced.
            candidates: The operands' sentinels, in preference order.
            values: The values `func` computed, used to detect a collision.
            excluded: Whether the domain mask actually dropped any cell.
            boolean: Whether `func` returned a boolean, now stored as Byte.

        Returns:
            Any: The sentinel to write into the result's bands, or `None` for a
            result that declares none.

        Raises:
            ValueError: `requested` cannot be stored in `dtype`, or no candidate
                is both storable and absent from the result.
        """
        if requested is not _DERIVE_NO_DATA:
            sentinel = cls._requested_no_data(requested, dtype)
            if occurs_in(values, sentinel):
                warnings.warn(
                    f"the requested no-data value {sentinel!r} is also a value "
                    "`func` computed, so those cells read back as gaps; pass a "
                    "`no_data_value=` the result cannot hold",
                    NoDataCollisionWarning,
                    # 5 frames out is the caller of `Dataset.combine`: warn ->
                    # _resolve_combined_no_data -> Analysis._combine ->
                    # Analysis.combine -> Dataset.combine -> user. Reached as
                    # `ds.analysis.combine` the facade frame is absent, so the
                    # same count lands one frame too far -- on whatever called
                    # the caller. A single constant cannot serve both entry
                    # points; the facade is the documented one, so it is the one
                    # that is right. `_fold`'s callers never reach here: the
                    # operators always let it derive its sentinel, and only an
                    # explicit one can collide.
                    stacklevel=5,
                )
        elif np.issubdtype(dtype, np.floating):
            sentinel = np.nan
        elif boolean:
            sentinel = 255
        elif not excluded:
            sentinel = None
        else:
            sentinel = free_no_data(dtype, candidates, values)
            if sentinel is None:
                raise ValueError(
                    f"the {np.dtype(dtype).name} result of `func` leaves no "
                    "free value to mark the cells its operands masked out -- "
                    "every candidate sentinel occurs in the result; pass "
                    "`no_data_value=None` to combine every cell unmasked, or "
                    "have `func` return a wider dtype"
                )
        return sentinel

    @staticmethod
    def _requested_no_data(requested: Any, dtype: np.dtype) -> Any:
        """Validate a caller-supplied sentinel against the result dtype.

        Args:
            requested: The caller's `no_data_value`.
            dtype: The dtype `func` produced.

        Returns:
            Any: `requested`, unchanged.

        Raises:
            ValueError: `requested` cannot be stored in `dtype`.
        """
        if isinstance(requested, (bool, np.bool_)):
            # `True` fits every numeric dtype as `1`, so it would silently become
            # a `1.0` sentinel. The operators already refuse a bool as the
            # additive identity; refusing it here keeps one rule.
            raise ValueError(
                f"the no-data value {requested!r} is a bool; pass the number you "
                "mean, or `None` for a result with no sentinel"
            )
        if not fits_dtype(requested, dtype):
            raise ValueError(
                f"the no-data value {requested!r} cannot be stored in the "
                f"{np.dtype(dtype).name} result of `func`; pass an explicit "
                "`no_data_value=` that fits it, or `no_data_value=None` for a "
                "result with no sentinel"
            )
        return requested

    def fill(
        self, value: float | int, inplace: bool = False, path: str | Path | None = None
    ) -> Dataset | None:
        """Fill the domain cells with a certain value.

        Every cell the no-data sentinel does not mark takes `value`; the gaps keep the
        sentinel. The raster is streamed tile by tile through `IO.stream_transform`, so a
        very large or `/vsicurl` source is never read whole, and the gaps are judged
        against the stored counts, where the sentinel lives.

        `value` is a physical value. On a CF-packed band (`scale_factor` / `add_offset`)
        the result is a computed raster: `float64`, declaring no packing, holding `value`
        as given and the source's stored `no_data_value` at the gaps. An unpacked source
        keeps its own dtype.

        Args:
            value (float | int):
                Numeric value to fill, in physical units.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created. Default is False.
                In place, a packed dataset's recipe is spent too (see :meth:`apply`).
            path (str | Path, optional):
                Output `.tif` path for a disk-backed result. `None` (default) keeps it in memory.

        Returns:
            Dataset | None:
                A new Dataset with cells filled, or `None` when
                `inplace=True` -- see :meth:`apply` for why.

        Examples:
            - Create a Dataset with 1 band, 5 rows, 5 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 5, size=(5, 5))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> print(dataset.read_array()) # doctest: +SKIP
              [[1 1 3 1 2]
               [2 2 2 1 2]
               [2 2 3 1 3]
               [3 4 3 3 4]
               [4 4 2 1 1]]
              >>> new_dataset = dataset.fill(10)
              >>> print(new_dataset.read_array())
              [[10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]]

              ```

            - On a CF-packed band the fill is a physical value, and the gap keeps the
              sentinel:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> packed = Dataset.from_array(
              ...     np.array([[100, 200], [300, -9999]], dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
              ...     no_data_value=-9999,
              ... )
              >>> packed.scale = [0.01]
              >>> filled = packed.fill(7.5)
              >>> filled.read_array().tolist(), filled.dtype, filled.scale
              ([[7.5, 7.5], [7.5, -9999.0]], ['float64'], [1.0])

              ```
        """
        no_data_value = self._ds.no_data_value[0]

        def _fill_tile(tile: np.ndarray) -> np.ndarray:
            # The band-wide predicate, not a tolerance of this method's own:
            # `fill` decides from it which cells are the domain, so a cell it
            # calls no-data here and the histogram calls data is the same
            # disagreement, reached through the writer instead of a reader.
            # On a packed band the tile is physical and holds no stored sentinel, so
            # this finds no gap and fills every cell; `stream_transform` then puts the
            # sentinel back at every gap, found in the stored counts, which is what
            # keeps them.
            tile[~is_stored_no_data(tile, no_data_value)] = value
            return tile

        # Stream the fill tile-by-tile so a very large or /vsicurl source is never
        # read whole (#967). The domain mask is per-pixel, so tiling is byte-identical.
        dst = self._ds.io.stream_transform(_fill_tile, path=path)
        if inplace:
            self._ds._update_inplace(dst.raster)
            # The fill wrote physical values, so the recipe is spent (rule 2).
            self._ds._spend_packing()
            return None
        return dst

    def where(
        self,
        cond: Any,
        other: Any = _DERIVE_NO_DATA,
        *,
        drop: bool = False,
    ) -> Dataset:
        """Keep the cells a condition selects and mask the rest.

        The counterpart of :meth:`fill`, which writes to the cells that *are* data: `where`
        decides which cells stay data at all. A cell the condition selects keeps its value
        exactly; every other cell takes `other`, which defaults to the raster's own no-data
        value, so the common call returns the same raster with the unwanted cells gone.

        A condition cell that is itself no-data reads as **false**. That is what makes
        `raster.where(raster > 5)` behave: a comparison declares `255` for a cell it could
        not judge — a gap in the operand — and keeping such a cell would keep one the
        comparison never approved. It is also xarray's answer, whose comparison is false at
        a NaN.

        Args:
            cond: What to keep. A boolean (or `0` / `1`) array broadcastable to this
                raster's cells; a raster on the same grid, which a comparison such as
                `raster > 5` produces and whose own no-data cells read as false; or a
                callable handed this raster's physical values and returning either.
            other: What an unselected cell holds. Left out, it is the raster's declared
                no-data value, or NaN when it declares none. An explicit `None` is NaN
                whatever the raster declares — the two are not the same argument. A
                number writes that number instead.

                A *selected* cell that was already a gap stays a gap, marked the way the
                result marks its gaps: `where(cond, 0.0)` is not a `fillna`, and the
                result declares the sentinel and still holds it there. A NaN `other`
                makes the result declare NaN, and those kept gaps are NaN too, so the
                source's sentinel appears nowhere in it.
            drop: Trim the result to what the condition selected a cell in, discarding
                the rows, the columns **and** the bands it was false across — xarray
                drops labels in every dimension, and a cube's empty steps go with its
                empty rows. Read off the condition, as xarray reads it: `other` does not
                save a row, and a cell that was already missing is kept if the condition
                selected it. The grid is unchanged — the origin moves to the first
                surviving cell and the cell size stays as it was. Off by default, which
                keeps every row, column and band.

                A variable carrying **two or more** band dimensions is trimmed spatially
                only: its bands are the flattened product of those dimensions, and the
                surviving set is not a rectangle of that product in general.

        Returns:
            Dataset: A new raster on this one's grid and CRS — trimmed to what survived
            when `drop` is set — carrying this raster's band names and metadata, as every
            combined result does.

        Raises:
            AlignmentError: A raster condition is on another grid. `where` does not
                resample; :meth:`Dataset.align <pyramids.dataset.Dataset.align>` is the
                explicit step for that.
            TypeError: `other` is neither a number nor `None` — a boolean included, since
                writing `True` into a band is never what was meant.
            ValueError: An array condition's shape does not broadcast onto this raster's
                cells; `drop` was asked for and the condition selected no cells at all,
                which leaves no raster to build; or this is a `NetCDF` container, which has
                no raster of its own — call it on one of its variables.

        Examples:
            - Keep the cells above a threshold, masking the rest:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> values = np.array([[1.0, 2.0], [3.0, 4.0]])
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> raster.where(raster > 2).read_array().tolist()
              [[-9999.0, -9999.0], [3.0, 4.0]]

              ```
            - Write a value into the cells that were not selected:

              ```python
              >>> raster.where(values > 2, 0.0).read_array().tolist()
              [[0.0, 0.0], [3.0, 4.0]]

              ```
            - Trim to what survived:

              ```python
              >>> kept = raster.where(values > 2, drop=True)
              >>> (kept.rows, kept.columns)
              (1, 2)

              ```
            - A NaN `other` marks the gaps the condition kept the result's way, so the
              source's sentinel is nowhere in it:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> gapped = Dataset.from_array(
              ...     np.array([[1.0, -9999.0], [3.0, 4.0]]), geo_ref=geo_ref,
              ...     no_data_value=-9999.0,
              ... )
              >>> masked = gapped.where([[True, True], [True, False]], np.nan)
              >>> masked.read_array().tolist()
              [[1.0, nan], [3.0, nan]]
              >>> masked.isnull().read_array().tolist()
              [[0, 1], [0, 1]]

              ```
            - `drop` cuts the empty bands of a stack as well as its empty rows and
              columns:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> stack = Dataset.from_array(
              ...     np.arange(8.0).reshape(2, 2, 2), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> trimmed = stack.where(stack > 4, drop=True)
              >>> (trimmed.band_count, trimmed.rows, trimmed.columns)
              (1, 2, 2)

              ```
        """
        layout_source = self._where_layout_source(cond)
        values, sentinels, domain = self._operand_arrays(self._ds, None)
        selected = self._where_condition(cond, values, domain)
        result = self._where_result(values, sentinels, domain, selected, other)
        if drop:
            result = self._where_trimmed(result, selected)
        # A raster condition may carry a layout of its own, which the hook has already
        # reconciled with this one's; `_identified` labelled the result from the receiver
        # alone, so the reconciled answer replaces it.
        self._ds._label_combined(result, layout_source)
        return result

    def equals(self, other: Any) -> bool:
        """Whether two rasters hold the same values on the same grid.

        What :meth:`Dataset.same_grid <pyramids.dataset.Dataset.same_grid>` does not answer:
        that says the two *could* be combined cell by cell, this says they actually agree.
        A gap equals a gap — comparing the sentinels as ordinary numbers would call two
        rasters different for marking the same missing cell with a different value, and
        would call a NaN unequal to itself.

        Attributes are ignored, as xarray ignores them here; :meth:`identical` is the one
        that reads them.

        The cheap invariants are checked first — the band count, the grid, and a NetCDF's
        band dimensions and their coordinates — so two rasters that cannot possibly agree
        are refused without reading a cell of either.

        Args:
            other: The raster to compare with. Anything that is not one answers `False`
                rather than raising, so a heterogeneous list can be filtered with it
                without a type check first. It is a method, not `==`: `Dataset` overloads
                the ordering comparisons to build masks and leaves `==` as Python's
                identity, so `a == b` is `False` for two equal rasters.

        Returns:
            bool: `True` when every cell agrees and every gap lines up.

        Raises:
            ValueError: This is a `NetCDF` container, which has no raster of its own —
                call it on one of its variables. Only the *receiver* is refused: a
                variable answers `False` for a container passed as `other`.

        Examples:
            - A raster equals its own copy, and stops equalling it after one cell changes:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> values = np.array([[1.0, 2.0], [3.0, 4.0]])
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> raster.equals(raster.copy())
              True
              >>> raster.equals(Dataset.from_array(values * 2, geo_ref=geo_ref))
              False

              ```
        """
        return self._compares_equal(other, attributes=False)

    def identical(self, other: Any) -> bool:
        """Whether two rasters are equal **and** carry the same attributes.

        :meth:`equals` with the metadata read too: the dataset-level tags and the band
        names. Two rasters holding identical numbers but describing different things are
        equal and not identical, which is the distinction xarray draws.

        Neither method reads the declared no-data value or the band type, also as xarray
        has neither concept: two rasters marking the same missing cells with `-9999.0` and
        with `-1.0` are identical, and so are a `float64` raster and its `float32` copy.
        Compare `no_data_value` and `dtype` yourself when those matter.

        Args:
            other: The raster to compare with; anything else answers `False`.

        Returns:
            bool: `True` when `equals` holds and the attributes match as well.

        Raises:
            ValueError: This is a `NetCDF` container, which has no raster of its own —
                call it on one of its variables.

        Examples:
            - The same numbers under a different description:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> values = np.array([[1.0, 2.0], [3.0, 4.0]])
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> relabelled = raster.copy()
              >>> relabelled.band_names = ["reflectance"]
              >>> raster.equals(relabelled), raster.identical(relabelled)
              (True, False)

              ```
        """
        return self._compares_equal(other, attributes=True)

    def _compares_equal(self, other: Any, *, attributes: bool) -> bool:
        """The shared body of :meth:`equals` and :meth:`identical`.

        Args:
            other: The raster to compare with.
            attributes: Whether the metadata and band names are read too.

        Returns:
            bool: The verdict.
        """
        self._refuse_a_container("identical" if attributes else "equals")
        verdict = isinstance(other, RasterBase) and self._invariants_match(other)
        if verdict:
            raster = cast("Dataset", other)
            verdict = self._values_match(raster)
            if verdict and attributes:
                verdict = self._attributes_match(raster)
        return verdict

    def _invariants_match(self, other: Any) -> bool:
        """Whether the header fields agree, before a cell of either raster is read.

        Args:
            other: The raster to compare with.

        Returns:
            bool: `True` when the shape, the grid and the band layout all agree.
        """
        same = (
            self._ds.rows == other.rows
            and self._ds.columns == other.columns
            and self._ds.band_count == other.band_count
            and self._ds.spatial.same_grid(other)
        )
        if same:
            # Duck-typed: a NetCDF variable carries band dimensions and a plain raster does
            # not, and two rasters of the same shape whose steps are stamped differently are
            # not the same cube.
            same = tuple(getattr(self._ds, "_band_dim_names", ())) == tuple(
                getattr(other, "_band_dim_names", ())
            ) and getattr(self._ds, "_band_dim_values_map", {}) == getattr(
                other, "_band_dim_values_map", {}
            )
        return same

    def _values_match(self, other: Dataset) -> bool:
        """Whether every cell agrees, a gap counting as equal to a gap.

        A NaN sitting inside the domain — a raster that declares a numeric sentinel and
        holds a NaN anyway, which is what `where(cond, np.nan)` produces — counts as equal
        to the same NaN on the other side, so a raster equals its own copy. Without that,
        `np.array_equal` would answer `False` for a raster compared with itself.

        Args:
            other: The raster to compare with.

        Returns:
            bool: `True` when the gaps line up and the values agree everywhere else.
        """
        mine, _, my_domain = self._operand_arrays(self._ds, None)
        theirs, _, their_domain = self._operand_arrays(other, None)
        aligned = bool(np.array_equal(my_domain, their_domain))
        return aligned and bool(
            np.array_equal(
                np.where(my_domain, mine, 0.0),
                np.where(their_domain, theirs, 0.0),
                equal_nan=True,
            )
        )

    def _attributes_match(self, other: Dataset) -> bool:
        """Whether the tags and band names agree.

        Args:
            other: The raster to compare with.

        Returns:
            bool: `True` when both match.
        """
        return self._attribute_tags(self._ds) == self._attribute_tags(other) and list(
            self._ds.band_names
        ) == list(other.band_names)

    @staticmethod
    def _attribute_tags(ds: Dataset) -> dict:
        """The tags :meth:`identical` compares, whichever shape the raster keeps them in.

        A plain raster keeps them in `meta_data`, a `dict` of GDAL items. A `NetCDF`
        variable's `meta_data` is a `NetCDFMetadata` — a structured snapshot of the whole
        *store*, not of this variable — and `dict()` on it raises, which is what made
        `identical` fail on every variable read from a file. Its own tags are `attrs`,
        the xarray-shaped mapping, and those are what xarray compares here.

        Reading the snapshot would be wrong even if it were a mapping: every variable in
        a file answers the same one, so it would say nothing about the variable.

        On a **classic** container `attrs` falls back to GDAL's whole prefixed metadata
        dictionary (`NC_GLOBAL#Conventions`, `temperature#units`, the synthetic
        `NETCDF_DIM_*` entries), which is the same 19-key blob for every variable in the
        file. The tags therefore cannot separate two classic variables on their own — the
        values, the grid and the band layout do, and those are compared first.

        Args:
            ds: The raster to read.

        Returns:
            dict: The tags, empty when the raster carries none.
        """
        tags = getattr(ds, "meta_data", None)
        if not isinstance(tags, Mapping):
            tags = getattr(ds, "attrs", None)
        return dict(tags) if isinstance(tags, Mapping) else {}

    def fillna(self, value: float | int) -> Dataset:
        """Give every gap a value, so the raster has no missing cells left.

        The inverse of :meth:`fill`, which writes to the cells that already hold data and
        leaves the gaps alone. `fillna` writes only to the gaps. The names are one letter
        apart and the behaviours are opposite, so check which one you meant.

        The band type is :meth:`where`'s judgement about the same fill: the two make one
        decision, so a `float32` raster stays `float32` for `2.5` and for either infinity,
        and both widen to `float64` for `1e300`. The widening happens *before* the fill
        goes in — writing `1e300` straight into a `float32` band stores `inf` behind a
        `RuntimeWarning`.

        Args:
            value: What each gap takes, in physical units.

        Returns:
            Dataset: A new raster on this one's grid, holding `value` wherever this one held
            its no-data value. It declares the same no-data value, which now marks nothing —
            as a filled raster's does — in the band's own type, widened only when that type
            cannot hold `value`.

        Raises:
            ValueError: This is a `NetCDF` container, which has no raster of its own —
                call it on one of its variables.

        Examples:
            - Fill the gaps with zero:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> values = np.array([[1.0, -9999.0], [3.0, 4.0]])
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> raster.fillna(0.0).read_array().tolist()
              [[1.0, 0.0], [3.0, 4.0]]

              ```
            - A `float32` band keeps its width for a value it can hold, and widens for one
              it cannot — the same answer `where` gives for that fill:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> narrow = Dataset.from_array(
              ...     np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32"),
              ...     geo_ref=geo_ref, no_data_value=-9999.0,
              ... )
              >>> narrow.fillna(2.5).dtype, narrow.where(narrow.notnull(), 2.5).dtype
              (['float32'], ['float32'])
              >>> narrow.fillna(1e300).dtype, narrow.where(narrow.notnull(), 1e300).dtype
              (['float64'], ['float64'])

              ```
        """
        self._refuse_a_container("fillna")
        values, sentinels, domain = self._operand_arrays(self._ds, None)
        declared = next((one for one in sentinels if one is not None), None)
        # The same judgement `where` makes about the same fill, and made before the fill
        # goes in: `np.where` on a float32 band with `1e300` keeps float32 and overflows
        # to inf with only a RuntimeWarning, so the band is widened first when it cannot
        # hold what is being written into it.
        dtype = _mask_dtype(values.dtype, value)
        out = np.asarray(np.where(domain, values.astype(dtype, copy=False), value))
        return self._identified(self._rebuilt(out, declared))

    def isnull(self) -> Dataset:
        """Flag the gaps: `1` where a cell is missing, `0` where it holds data.

        The flags come back in the `uint8` a comparison returns, so the result reads as a
        condition for :meth:`where`. Mind which way round: `where` keeps what the condition
        *selects*, and a selected cell that was already a gap stays one, so it is the
        complement that closes the gaps — `raster.where(raster.notnull(), 0.0)` zeroes them,
        while `raster.where(raster.isnull(), 0.0)` zeroes the cells that hold data and
        leaves every gap where it was. xarray's `where` answers exactly the same both ways;
        it differs only in flagging with a boolean array, which GDAL has no band type for.

        Returns:
            Dataset: A `uint8` raster on this one's grid, `1` at each gap. It declares no
            no-data value: every cell is either missing or not, so there is nothing a flag
            could fail to judge.

        Raises:
            ValueError: This is a `NetCDF` container, which has no raster of its own —
                call it on one of its variables.

        Examples:
            - Flag the one gap:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> values = np.array([[1.0, -9999.0], [3.0, 4.0]])
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> raster.isnull().read_array().tolist()
              [[0, 1], [0, 0]]

              ```
            - Which way round the flags read as a `where` condition:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> values = np.array([[1.0, -9999.0], [3.0, 4.0]])
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> raster.where(raster.notnull(), 0.0).read_array().tolist()
              [[1.0, 0.0], [3.0, 4.0]]
              >>> raster.where(raster.isnull(), 0.0).read_array().tolist()
              [[0.0, -9999.0], [0.0, 0.0]]

              ```
        """
        return self._null_flags(missing=True)

    def notnull(self) -> Dataset:
        """Flag the data: `1` where a cell holds a value, `0` where it is missing.

        The complement of :meth:`isnull`, in the same `uint8` flags, so it reads as a
        condition for :meth:`where` — and it is this one, not `isnull`, that a `where`
        closing the gaps takes: `raster.where(raster.notnull(), 0.0)` keeps every cell that
        holds data and writes `0.0` into the gaps, which is what :meth:`fillna` does.
        Selecting on it alone, `raster.where(raster.notnull())`, is a no-op.

        Returns:
            Dataset: A `uint8` raster on this one's grid, `1` at each cell that holds data,
            declaring no no-data value.

        Raises:
            ValueError: This is a `NetCDF` container, which has no raster of its own —
                call it on one of its variables.

        Examples:
            - Flag the cells that hold data:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> values = np.array([[1.0, -9999.0], [3.0, 4.0]])
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> raster.notnull().read_array().tolist()
              [[1, 0], [1, 1]]

              ```
            - Selecting on the flags changes nothing, and filling through them matches
              `fillna`:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
              >>> values = np.array([[1.0, -9999.0], [3.0, 4.0]])
              >>> raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)
              >>> raster.equals(raster.where(raster.notnull()))
              True
              >>> raster.where(raster.notnull(), 0.0).equals(raster.fillna(0.0))
              True

              ```
        """
        return self._null_flags(missing=False)

    def _null_flags(self, *, missing: bool) -> Dataset:
        """The `uint8` gap flags, one way round or the other.

        Args:
            missing: `True` to flag the gaps (`isnull`), `False` to flag the data
                (`notnull`).

        Returns:
            Dataset: The flags, declaring no no-data value.
        """
        self._refuse_a_container("isnull" if missing else "notnull")
        _, _, domain = self._operand_arrays(self._ds, None)
        flags = np.asarray(
            (~domain if missing else domain).astype("uint8"), dtype="uint8"
        )
        return self._identified(self._rebuilt(flags, None))

    def _where_layout_source(self, cond: Any) -> Any:
        """Check a raster condition's grid and band layout, and say what labels the result.

        A raster condition goes through the same two checks an operator's right operand
        does — the grid, so nothing is silently resampled, and the band layout, through the
        hook that lets a `NetCDF` refuse band dimensions that do not pair up. Anything else
        is an array and has neither.

        Args:
            cond: The condition as the caller gave it.

        Returns:
            Any: What `_label_combined` should label the result from, or `None`.
        """
        self._refuse_a_container("where")
        if isinstance(cond, RasterBase):
            raster = cast("Dataset", cond)
            self._check_combinable(raster, np.logical_and, None)
            return self._ds._combine_layout_source(raster, None)
        # An array or a callable brings no layout of its own, which is the shape a fold
        # has: one operand, nothing to compare it with or fill labels from. Asking the
        # hook that way answers this raster's own layout, where passing `None` through
        # would leave `NetCDF._label_combined` unpacking it.
        return self._ds._combine_layout_source(None, None)

    def clip(self, min: Any = None, max: Any = None) -> Dataset:
        """Bound the values to `[min, max]`; a gap stays a gap.

        The gaps are left out of the clipping and re-marked afterwards. Clipping the stored
        array instead would lift a `-9999.0` gap to the lower bound and turn a missing cell
        into a measurement.

        Args:
            min: The lower bound, or `None` for none. Named as xarray names it.
            max: The upper bound, or `None` for none.

        Returns:
            Dataset: A raster on this one's grid, carrying its band names and metadata. The
            band keeps its type when both bounds fit it, and widens when one does not — the
            same judgement :meth:`where` and :meth:`fillna` make. A CF-packed band is the
            exception: the bounds apply to the physical values, so the result holds those
            (`float64`) with the packing dropped.

        Raises:
            ValueError: Neither bound is given, `min` is above `max` — numpy would quietly
                set every cell to `max` there, which is never what was meant — or a bound
                is NaN, which numpy compares false against everything, leaving every cell
                NaN and unflagged.
            TypeError: A bound is not a real number.

        Examples:
            - Clamp to `[2, 6]`, the gap untouched:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[1.0, -9999.0, 5.0, 9.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.clip(2.0, 6.0).read_array().tolist()
              [[2.0, -9999.0, 5.0, 6.0]]

              ```
            - One bound is enough:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[1.0, 5.0, 9.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.clip(max=6.0).read_array().tolist()
              [[1.0, 5.0, 6.0]]

              ```
            - Crossed bounds are refused, where numpy would set every cell to `max`:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[1.0, 5.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.clip(6.0, 2.0)
              Traceback (most recent call last):
                  ...
              ValueError: clip() needs min at or below max, got min=6.0 above max=2.0.

              ```

        See Also:
            Dataset.astype: Clip first, to keep the values inside the target type's range.
            Dataset.where: Mask the out-of-range cells instead of bounding them.
        """
        self._refuse_a_container("clip")
        lower, upper = min, max
        if lower is None and upper is None:
            raise ValueError("clip() needs at least one of min and max.")
        for bound in (lower, upper):
            if bound is not None and (
                not isinstance(bound, Real) or isinstance(bound, (bool, np.bool_))
            ):
                raise TypeError(f"clip() needs a number for a bound; got {bound!r}.")
            if bound is not None and np.isnan(float(bound)):
                raise ValueError(
                    "clip() cannot bound anything with NaN: every cell would come back "
                    "NaN, and a raster that declares another sentinel does not read those "
                    "as missing. Drop the bound, or use where() to mask."
                )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"clip() needs min at or below max, got min={lower!r} above max={upper!r}."
            )
        values, sentinels, domain = self._operand_arrays(self._ds, None)
        declared = _declared_gaps(sentinels)
        dtype = values.dtype
        for bound in (lower, upper):
            if bound is not None:
                dtype = np.promote_types(dtype, _mask_dtype(values.dtype, bound))
        # Cast back after clipping, not only before it: a bound that arrives as a numpy
        # scalar — every percentile, mean or quantile does — promotes the result under NEP
        # 50, so a float32 band came back float64 and a uint8 band int64. `where` casts its
        # own output for the same reason.
        clipped = np.clip(values.astype(dtype, copy=False), lower, upper).astype(
            dtype, copy=False
        )
        return self._identified(
            self._rebuilt(_regapped(clipped, domain, declared), declared)
        )

    def round(self, decimals: int = 0) -> Dataset:
        """Round the values to `decimals` places; a gap stays a gap.

        The gaps are left out and re-marked afterwards: a sentinel with a fraction, such as
        `-9999.5`, rounds to `-10000.0`, which no longer matches what was declared.

        Args:
            decimals: How many decimal places to keep. `0` (default) rounds to whole
                numbers; a negative count rounds to tens, hundreds and so on, as numpy
                does. Halves round to even, also as numpy does. Rounding an **integer**
                band to tens can leave its range — numpy wraps `uint8` 255 to 4 — so the
                result widens instead, and only when it must.

        Returns:
            Dataset: A raster on this one's grid, in the band's own type — except for a
            CF-packed band, whose physical values are what is rounded, so the result holds
            those (`float64`) with the packing dropped, as an xarray decoded array does.

        Raises:
            TypeError: `decimals` is not an integer.

        Examples:
            - Round to one decimal place:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[0.333, 1.667]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.round(1).read_array().tolist()
              [[0.3, 1.7]]

              ```
            - Halves round to even, as numpy's do:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[0.5, 1.5, 2.5]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.round().read_array().tolist()
              [[0.0, 2.0, 2.0]]

              ```
            - A negative count rounds to tens:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[14.0, 26.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.round(-1).read_array().tolist()
              [[10.0, 30.0]]

              ```

        See Also:
            Dataset.astype: Change the band type once the values are whole.
        """
        self._refuse_a_container("round")
        if isinstance(decimals, bool) or not isinstance(decimals, (int, np.integer)):
            raise TypeError(
                f"round() needs an integer number of decimals, got {decimals!r}."
            )
        values, sentinels, domain = self._operand_arrays(self._ds, None)
        declared = _declared_gaps(sentinels)
        rounded = np.round(values, int(decimals))
        if values.dtype.kind in "iu" and int(decimals) < 0:
            # Rounding an integer band to tens can leave its own type: numpy wraps there,
            # so `uint8` 255 rounds to 4 and `int8` 127 to -126 — silent corruption of the
            # kind `clip` widens to avoid. The result widens the same way, and only when
            # it must.
            wide = np.round(values.astype("float64"), int(decimals))
            limits = np.iinfo(values.dtype)
            inside = wide[domain] if domain.any() else wide
            fits = bool(((inside >= limits.min) & (inside <= limits.max)).all())
            rounded = wide.astype(values.dtype, copy=False) if fits else wide
        return self._identified(
            self._rebuilt(_regapped(rounded, domain, declared), declared)
        )

    def astype(self, dtype: Any, *, no_data_value: Any = _KEEP_NO_DATA) -> Dataset:
        """Change the band type; a gap stays a gap, re-marked for the new type.

        The cells that hold data are cast as numpy casts them: a float truncates towards
        zero into an integer type, and a value outside the target's range is **not**
        refused — numpy leaves an out-of-range float cast undefined (on x86 `300.0` into
        `uint8` comes out `44`). :meth:`clip` first when that matters. The gaps are not
        cast at all: they are re-marked with the target's sentinel, so a missing cell
        stays missing. A raster that declares no sentinel but holds NaN is gap-holding
        too — those cells are written back as NaN, which only a float target has, so an
        integer cast that would leave them unmarked is refused.

        Args:
            dtype: The target type — anything `numpy.dtype` accepts that GDAL can store as a
                real number: signed or unsigned integers, or floats.
            no_data_value: The sentinel the result declares. Left out, it is the raster's
                own, snapped to the new type — `-9999.9` into `float32` is declared as
                `-9999.900390625`, the value its gap cells then hold. Pass one when the new
                type cannot carry the raster's own at all, or `None` for a result that
                declares none — which a gap-holding raster allows only into a float type.

        Returns:
            Dataset: A raster on this one's grid, in `dtype`.

        Raises:
            TypeError: `dtype` is not a real numeric type — a boolean, a complex number or a
                string has no GDAL band type here.
            ValueError: The sentinel is outside `dtype`'s range, or is a fraction where
                `dtype` is an integer type — `-9999` into `uint8` would wrap to `241`, a
                real value, and the gaps would become data; NaN has no integer at all. Or a
                cell holding data lands on the sentinel once cast. Or the raster holds gaps,
                the result would declare no sentinel, and `dtype` is an integer type, which
                has no NaN to leave them as.

        Examples:
            - Floats cast to `int32`, the gap still a gap:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[1.7, -9999.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> cast = raster.astype("int32")
              >>> cast.read_array().tolist(), cast.dtype
              ([[1, -9999]], ['int32'])

              ```
            - A sentinel the new type cannot hold is refused, unless a new one is named:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[1.0, -9999.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> cast = raster.astype("uint8", no_data_value=255)
              >>> cast.read_array().tolist(), float(cast.no_data_value[0])
              ([[1, 255]], 255.0)

              ```

        See Also:
            Dataset.clip: Bound the values first, so none falls outside the new type.
            Dataset.round: Round first where the cast would otherwise truncate.
        """
        self._refuse_a_container("astype")
        target = np.dtype(dtype)
        if target.kind not in "iuf":
            raise TypeError(
                f"astype() needs a real numeric type GDAL can store — a signed or unsigned "
                f"integer, or a float — and {target.name} is not one."
            )
        values, sentinels, domain = self._operand_arrays(self._ds, None)
        bands = 1 if values.ndim == 2 else values.shape[0]
        if no_data_value is _KEEP_NO_DATA:
            marks = [sentinels[i] if i < len(sentinels) else None for i in range(bands)]
        else:
            marks = [no_data_value] * bands
        for mark in marks:
            if mark is not None and not _holds(target, mark):
                raise ValueError(
                    f"astype({target.name!r}) cannot mark a gap with {float(mark)}, which "
                    f"{target.name} does not hold — cast, it would become an ordinary value "
                    f"and every gap would read as data. Pass no_data_value= with one it "
                    f"does hold."
                )
        # Snapped to the target before anything is written or declared, so the value the
        # result declares is exactly the value its gap cells hold. A float sentinel loses
        # precision here (`-9999.9` into float32 becomes `-9999.900390625`); what it must
        # not do is land on a cell holding data, which is checked after the cast.
        marks = [None if mark is None else float(target.type(mark)) for mark in marks]
        markers = [
            _gap_marker(mark, target, domain, band, bands)
            for band, mark in enumerate(marks)
        ]
        # Filled, never `np.empty`: the cells outside the domain are not cast, so an
        # uninitialised buffer would ship whatever the allocator held as data.
        cast = np.empty(values.shape, dtype=target)
        for band in range(bands):
            plane = cast if values.ndim == 2 else cast[band]
            plane[...] = markers[band]
        cast[domain] = values[domain].astype(target)
        _refuse_a_sentinel_the_data_holds(cast, domain, marks, target, bands)
        return self._identified(self._rebuilt(cast, _declared_gaps(marks)))

    def isin(self, test_elements: Any) -> Dataset:
        """Flag the cells whose value is one of `test_elements`: `1` if so, `0` if not.

        The flags come back as `uint8`, like :meth:`isnull`'s — GDAL has no boolean band —
        and read as a condition for :meth:`where`. A gap is in no set, not even when the set
        holds the gap's own sentinel: it is missing, not that value, as xarray flags a NaN
        `False` whatever it is asked for.

        Args:
            test_elements: One value, or a sequence of them — a list, tuple, set or array.

        Returns:
            Dataset: A `uint8` raster on this one's grid, declaring no no-data value.

        Examples:
            - Flag the cells holding 2 or 5:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[1.0, 2.0, 5.0, -9999.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.isin([2.0, 5.0]).read_array().tolist()
              [[0, 1, 1, 0]]

              ```
            - The flags read as a condition for `where`:

              ```python
              >>> import numpy as np
              >>> from pyramids.base.georeference import GeoReference
              >>> from pyramids.dataset import Dataset
              >>> geo_ref = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)
              >>> raster = Dataset.from_array(
              ...     np.array([[1.0, 2.0, 5.0]]), geo_ref=geo_ref, no_data_value=-9999.0
              ... )
              >>> raster.where(raster.isin([2.0, 5.0])).read_array().tolist()
              [[-9999.0, 2.0, 5.0]]

              ```

        See Also:
            Dataset.where: Keep the cells the flags select.
            Dataset.isnull: The same `uint8` flags, for the gaps.
        """
        self._refuse_a_container("isin")
        values, _, domain = self._operand_arrays(self._ds, None)
        # A `set` survives `np.asarray` as a 0-d object array holding the set itself, so
        # every comparison is False and the flags come back all zero — which reads as a
        # `where` condition that blanks the raster. Spelled as a list, it compares.
        wanted = (
            list(test_elements)
            if isinstance(test_elements, (set, frozenset))
            else test_elements
        )
        flags = np.isin(values, np.asarray(wanted)) & domain
        return self._identified(self._rebuilt(flags.astype("uint8"), None))

    def _refuse_a_container(self, caller: str) -> None:
        """Refuse a `NetCDF` container by name, since it has no raster of its own.

        A container's raster is a placeholder — its variables hold the cells — so every
        member here works on a variable. Without this the caller meets whichever internal
        guard it reaches first, and the generic one names `read_array`, which the caller
        never called.

        Args:
            caller: The member named in the refusal.

        Raises:
            ValueError: The receiver is a container.
        """
        variables = getattr(self._ds, "variable_names", None)
        if variables and not getattr(self._ds, "_band_dim_names", ()):
            raise ValueError(
                f"{caller}() works on a raster, and a container has none of its own — "
                f"its variables do. Call it on one of them: "
                f"`nc.get_variable({variables[0]!r}).{caller}(...)`."
            )

    def _where_condition(
        self, cond: Any, values: np.typing.NDArray, domain: np.typing.NDArray
    ) -> np.typing.NDArray:
        """The condition as a boolean mask shaped like this raster's cells.

        Args:
            cond: A callable, a raster, or an array.
            values: This raster's physical values, for a callable condition.
            domain: This raster's valid-cell mask, unused here but kept for symmetry with
                the other operand's, which a raster condition contributes.

        Returns:
            numpy.ndarray: True wherever a cell is selected.

        Raises:
            ValueError: An array condition does not broadcast onto the values.
        """
        if callable(cond) and not isinstance(cond, RasterBase):
            cond = cond(values)
        if isinstance(cond, RasterBase):
            # Its own gaps are cells it could not judge, so they select nothing.
            other_values, _, other_domain = self._operand_arrays(
                cast("Dataset", cond), None
            )
            flags = np.asarray(other_values) != 0
            mask = np.asarray(flags & other_domain)
        else:
            mask = np.asarray(cond) != 0
        # A one-step variable reads back as `(rows, cols)` while its cube layout is
        # `(1, rows, cols)`, and `broadcast_to` cannot drop a leading axis — so a
        # condition built from `_materialize_variable_array` was refused on a one-step
        # cube and accepted on a two-step one. A leading singleton carries no information,
        # so it is dropped rather than made to depend on the cube's length.
        while np.ndim(mask) == values.ndim + 1 and np.shape(mask)[0] == 1:
            mask = np.asarray(mask)[0]
        try:
            resolved = np.broadcast_to(mask, values.shape)
        except ValueError:
            raise ValueError(
                f"where() needs a condition that covers this raster's cells: its shape is "
                f"{np.shape(mask)}, which does not broadcast onto {values.shape}."
            ) from None
        return np.asarray(resolved)

    def _where_result(
        self,
        values: np.typing.NDArray,
        sentinels: list[Any],
        domain: np.typing.NDArray,
        selected: np.typing.NDArray,
        other: Any,
    ) -> Dataset:
        """Build the masked raster: selected cells keep their value, the rest take `other`.

        Args:
            values: This raster's physical values.
            sentinels: Its per-band no-data values.
            domain: True where a cell holds data rather than its sentinel.
            selected: True where the condition selected a cell.
            other: What an unselected cell holds, or the derive sentinel.

        Returns:
            Dataset: The result, on this raster's grid. It declares the fill that went into
            the unselected cells, which is NaN whenever `other` resolved to NaN. A cell the
            condition selected that was already a gap stays one, marked the way the result
            marks its gaps rather than the way the source did.

        Raises:
            TypeError: `other` is neither a number nor `None`. Booleans are refused with
                the rest: `np.where` would happily write `True` into the band as `1`.
        """
        declared = next((one for one in sentinels if one is not None), None)
        fill = declared if other is _DERIVE_NO_DATA else other
        if fill is None:
            fill = np.nan
        # `Real` alone: every real numpy scalar registers as one, while `np.complex128`
        # is an `np.number` and slipped through to produce a complex band under a real
        # no-data value. A bool is a `Real` equal to 1, and is refused as a caller's bug.
        elif not isinstance(fill, Real) or isinstance(fill, (bool, np.bool_)):
            raise TypeError(
                f"where() needs a number for `other`, or None for NaN; got {other!r}."
            )
        # The gaps of the result are wherever `fill` went, so a NaN fill makes the result
        # declare NaN: keeping a numeric sentinel would leave a raster declaring a value
        # it does not hold, unable to find its own missing cells.
        fills_with_nan = bool(np.isnan(np.asarray(fill, dtype="float64")).all())
        # A selected cell that was already a gap stays one, marked the way the *result*
        # marks its gaps — not the way the source did. Writing the source's sentinel back
        # while declaring NaN would reclassify every such cell as a measurement, and the
        # number that leaked was the raw `-9999.0`.
        gap = np.nan if fills_with_nan or declared is None else declared
        # `_mask_dtype` of the band and the fill, not whatever `np.where` promotes to: a
        # Python float octuples a `uint8` band and a `numpy.float64` sentinel doubles a
        # `float32` one, including on the `where(notnull())` that the docstring calls a
        # no-op. A fill the band cannot hold still widens it.
        dtype = _mask_dtype(values.dtype, fill)
        filler = np.asarray(fill)
        kept = np.where(domain, values, gap)
        out = np.asarray(np.where(selected, kept, filler)).astype(dtype, copy=False)
        if fills_with_nan:
            declared = np.nan
        out = np.asarray(out)
        return self._identified(self._rebuilt(out, declared))

    def _rebuilt(self, values: np.typing.NDArray, sentinel: Any) -> Dataset:
        """A raster of `values` on this one's grid, declaring `sentinel`.

        Args:
            values: The cells, 2-D for one band or `(bands, rows, cols)`.
            sentinel: The no-data value to declare, or `None` for none.

        Returns:
            Dataset: The raster, before its identity is put back on.
        """
        return self._ds.__class__._build_dataset(
            self._ds.columns,
            self._ds.rows,
            1 if values.ndim == 2 else values.shape[0],
            numpy_to_gdal_dtype(values),
            self._ds.geotransform,
            self._ds.crs,
            sentinel,
            array=values,
        )

    def _identified(self, result: Dataset) -> Dataset:
        """Put this raster's identity on a result built from it, cell for cell.

        The same assignments `combine` makes, and for the same reason: a masked, filled
        or flagged raster is still the same band of the same scene, and one that comes
        back as `Band_1` with no tags has lost what told the caller which band it is.

        On a `NetCDF` variable the band **dimensions** are identity too — without them
        `sel` refuses the result — so the labelling hook runs as well. The layout taken is
        the receiver's own: the shape a fold has, one operand with nothing to compare it
        against. `where(drop=True)` is the one caller that shortens a dimension, and it
        runs after this one — :meth:`_restamped` cuts the sizes and the stamps the
        receiver's length left behind.

        The name the variable answers to travels too, as it does through every member
        along a dimension. Without it a masked variable came back as the placeholder
        `variable`, so `to_dataframe()` renamed its column and `to_dataframe(variables=…)`
        refused the variable's own name. `_parent_nc` deliberately does **not** travel:
        the result no longer holds the store's values, and pointing it back at that store
        would let a reader recover coordinates for cells that are no longer there.

        Args:
            result: The freshly built raster.

        Returns:
            Dataset: `result`.
        """
        result.meta_data = self._ds.meta_data
        result.band_names = list(self._ds.band_names)
        # Duck-typed: only a NetCDF variable carries one, and a plain raster has no name
        # to lose.
        name = getattr(self._ds, "_source_var_name", None)
        if name is not None:
            result._source_var_name = name  # type: ignore[attr-defined]
            # The name is for labelling; it must not read as store identity. `copy()`
            # clears `_source_var_name` for exactly that reason, and a raster rebuilt
            # here is in the same position — its cells are its own — so the flag says so
            # and the lazy read refuses it in words instead of failing inside GDAL.
            result._rebuilt_in_memory = True  # type: ignore[attr-defined]
        self._ds._label_combined(result, self._ds._combine_layout_source(None, None))
        return result

    def _where_trimmed(self, result: Dataset, selected: np.typing.NDArray) -> Dataset:
        """Trim `result` to the smallest rectangle the condition selected a cell in.

        Read off the **condition**, not off the result's gaps, which is what xarray does:
        a row the condition was false across is dropped whether or not `other` wrote a
        number into it, and a cell that was already missing is kept if the condition
        selected it. Reading the result instead made `other` cancel the trim entirely and
        made an all-true condition trim a raster's pre-existing edge gaps away.

        Taken as an index slice rather than as a `crop(bbox=...)`: a bbox crop drops the
        rows and columns that are no-data from edge to edge as well, which is the second
        half of that same divergence — `crop` trimming gaps the condition had kept.

        The bands the condition is false across go too, which for a cube means its empty
        steps: xarray drops labels in *every* dimension, not only the two spatial ones.
        A variable carrying **two or more** band dimensions is the exception — its bands
        are the flattened product of them, and the surviving set is not a rectangle of
        that product in general — so those keep every band and only the grid is trimmed.

        Args:
            result: The masked raster, on the full grid.
            selected: True wherever the condition selected a cell, shaped like the cells.

        Returns:
            Dataset: The trimmed raster, on the same grid, its origin at the first cell
            that survived.

        Raises:
            ValueError: The condition selected nothing, so there is no rectangle to keep.
        """
        flat = selected if selected.ndim == 2 else np.any(selected, axis=0)
        rows = np.flatnonzero(np.any(flat, axis=1))
        columns = np.flatnonzero(np.any(flat, axis=0))
        if rows.size == 0 or columns.size == 0:
            raise ValueError(
                "where(drop=True) kept no cells, and a raster of no cells cannot be built. "
                "Check the condition, or leave `drop` off to keep the grid."
            )
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(columns[0]), int(columns[-1]) + 1
        cells = np.asarray(result.read_array(unpack=False))
        bands = self._surviving_bands(selected, cells)
        block = np.ascontiguousarray(
            cells[top:bottom, left:right]
            if cells.ndim == 2
            else cells[np.ix_(bands, range(top, bottom), range(left, right))]
        )
        geo = result.geotransform
        # The two skews carry a rotated grid's corner across as well, so this is the same
        # arithmetic for a north-up, a south-up and a rotated geotransform.
        shifted = (
            geo[0] + left * geo[1] + top * geo[2],
            geo[1],
            geo[2],
            geo[3] + left * geo[4] + top * geo[5],
            geo[4],
            geo[5],
        )
        trimmed = result.__class__._build_dataset(
            right - left,
            bottom - top,
            1 if block.ndim == 2 else block.shape[0],
            numpy_to_gdal_dtype(block),
            shifted,
            result.crs,
            result.no_data_value[0],
            array=block,
        )
        return self._restamped(self._identified(trimmed), bands)

    def _surviving_bands(
        self, selected: np.typing.NDArray, cells: np.typing.NDArray
    ) -> np.typing.NDArray:
        """The band indices the condition selected a cell in, or all of them.

        Every band survives on a 2-D raster, and on a variable whose bands are the
        flattened product of two or more dimensions — there the kept set is not a
        rectangle of that product in general, so the trim stays spatial.

        An all-false condition also keeps every band, rather than answering the empty set
        that would leave no raster to build. Nothing reaches that fallback through
        :meth:`where`, whose trim refuses an empty selection before this is asked.

        Args:
            selected: True wherever the condition selected a cell.
            cells: The result's stored values, for its band count.

        Returns:
            numpy.ndarray: The band indices to keep, in order.
        """
        count = 1 if cells.ndim == 2 else cells.shape[0]
        keep = np.arange(count)
        if selected.ndim == 3 and len(getattr(self._ds, "_band_dim_names", ())) <= 1:
            survivors = np.flatnonzero(np.any(selected, axis=(1, 2)))
            if survivors.size:
                keep = survivors
        return keep

    def _restamped(self, trimmed: Dataset, bands: np.typing.NDArray) -> Dataset:
        """Cut the one band dimension's coordinates to the bands that survived.

        `_identified` labels the result from the receiver, whose dimension is as long as
        it was before the trim, so the sizes and stamps are corrected here. A receiver
        with no band dimension, or one that kept every band, needs nothing.

        Args:
            trimmed: The trimmed raster, already labelled.
            bands: The band indices that survived.

        Returns:
            Dataset: `trimmed`.
        """
        names = list(getattr(self._ds, "_band_dim_names", ()))
        if len(names) == 1 and len(bands) != self._ds.band_count:
            dim = names[0]
            trimmed._band_dim_sizes = (len(bands),)  # type: ignore[attr-defined]
            stamps = dict(getattr(self._ds, "_band_dim_values_map", {}))
            coords = stamps.get(dim)
            if coords is not None:
                stamps[dim] = [coords[int(one)] for one in bands]
            trimmed._band_dim_values_map = stamps  # type: ignore[attr-defined]
            # The legacy `(name, values)` pair is a view of the canonical fields, and
            # one staticmethod owns that derivation — including the staleness guard for
            # a band-shrinking operation, which this is.
            trimmed._band_dim_name, trimmed._band_dim_values = (  # type: ignore[attr-defined]
                trimmed._derive_primary_band_view(  # type: ignore[attr-defined]
                    tuple(trimmed._band_dim_names),
                    trimmed._band_dim_values_map,
                    tuple(trimmed._band_dim_sizes),
                    trimmed.band_count,
                )
            )
        return trimmed

    def _extract_streamed(
        self, band: int | None, exclude_list: list
    ) -> np.typing.NDArray:
        """Stream the maskless `extract` in full-width row strips (see `extract`).

        `get_pixels2` selects from band 0 in row-major order within each strip;
        full-width top-to-bottom strips keep that order across the raster, so
        concatenating the strips reproduces the whole-array selection exactly (#967).

        Args:
            band (int, optional):
                Band to read, or `None` for all bands.
            exclude_list (list):
                Values to exclude (no-data, and `exclude_value` when given).

        Returns:
            np.ndarray:
                The extracted values, byte-identical to the eager whole-array pass.
        """

        def _collect(
            acc: list[np.ndarray], strip: np.ndarray, _window: list[int]
        ) -> list[np.ndarray]:
            acc.append(get_pixels2(strip, exclude_list))
            return acc

        parts = [
            part
            for part in self._ds.io.stream_reduce(_collect, [], band=band)
            if part.size
        ]
        multiband = band is None and self._ds.band_count > 1
        if parts:
            return np.concatenate(parts, axis=1 if multiband else 0)
        return np.asarray([])

    def extract(
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        mask: FeatureCollection | GeoDataFrame | None = None,
    ) -> np.typing.NDArray:
        """Extract.

        - Extract method gets all the values in a raster, and excludes the values in the exclude_value parameter.
        - If the mask parameter is given, the raster will be clipped to the extent of the given mask and the
          values within the mask are extracted.
        - Values come back in physical units, as `read_array` returns them: a CF-packed band
          (`scale_factor` / `add_offset`) is unpacked, and `exclude_value` is compared against the
          unpacked values. Without a `mask` the no-data cells are left out, matched against the
          sentinel expressed in those same physical units -- the selected band's, or band 0's
          when `band` is `None`.

        Args:
            band (int, optional):
                Band index. Default is None.
            exclude_value (Numeric, optional):
                Values to exclude from extracted values, in physical units. If the dataset is multi-band, the
                values in `exclude_value` will be filtered out from the first band only.
            mask (FeatureCollection | GeoDataFrame, optional):
                Vector data containing point geometries at which to extract the values. Default is None.

        Returns:
            np.ndarray:
                The extracted values from each band in the dataset will be in one row in the returned array.

        Raises:
            ValueError: `mask` holds geometries other than single points.

        Examples:
            - Extract all values from the dataset:

              - First, create a dataset with 2 bands, 4 rows and 4 columns:

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.random.randint(1, 5, size=(2, 4, 4))
                >>> top_left_corner = (0, 0)
                >>> cell_size = 0.05
                >>> dataset = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
                ... )
                >>> (dataset.band_count, dataset.rows, dataset.columns)
                (2, 4, 4)
                >>> dataset.band_names
                ['Band_1', 'Band_2']
                >>> print(dataset.read_array()) # doctest: +SKIP
                [[[1 3 3 4]
                  [1 4 2 4]
                  [2 4 2 1]
                  [1 3 2 3]]
                 [[3 2 1 3]
                  [4 3 2 2]
                  [2 2 3 4]
                  [1 4 1 4]]]

                ```

              - Now, extract the values in the dataset:

                ```python
                >>> values = dataset.extract()
                >>> print(values) # doctest: +SKIP
                [[1 3 3 4 1 4 2 4 2 4 2 1 1 3 2 3]
                 [3 2 1 3 4 3 2 2 2 2 3 4 1 4 1 4]]

                ```

              - Extract all the values except 2:

                ```python
                >>> values = dataset.extract(exclude_value=2)
                >>> print(values) # doctest: +SKIP

                ```

            - Extract values at the location of the given point geometries:

              ```python
              >>> import geopandas as gpd
              >>> from shapely.geometry import Point

              ```

              - Create the points using shapely and GeoPandas to cover the 4 cells with xmin, ymin, xmax, ymax = [0.1, -0.2, 0.2, -0.1]:

                ```python
                >>> points = gpd.GeoDataFrame(geometry=[Point(0.1, -0.1), Point(0.1, -0.2), Point(0.2, -0.2), Point(0.2, -0.1)],crs=4326)
                >>> values = dataset.extract(mask=points)
                >>> print(values) # doctest: +SKIP
                [[4 3 3 4]
                 [3 4 4 2]]

                ```
        """
        # The physical sentinel, because `get_pixels2` compares it against values
        # `read_array` produced. The stored `-9999` matches nothing in an array that
        # holds `-98.49`, so every no-data cell was extracted as a measurement.
        physical_sentinel = self._physical_no_data(band if band is not None else 0)
        no_data_value = physical_sentinel if physical_sentinel is not None else np.nan
        if mask is None:
            exclude_list = (
                [no_data_value, exclude_value]
                if exclude_value is not None
                else [no_data_value]
            )
            values = self._extract_streamed(band, exclude_list)
        else:
            arr = self._ds.read_array(band=band)
            geom_types = set(getattr(mask, "geom_type", []))
            # map(str, ...) — missing geometries yield float nan, which is not
            # orderable against the str type names.
            if geom_types - {"Point"}:
                raise ValueError(
                    "extract(mask=...) expects Point geometries — one value is read "
                    f"per point; got {sorted(map(str, geom_types))}. For polygon "
                    "zones use Dataset.zonal_stats(); to clip a raster use "
                    "Dataset.crop(); explode MultiPoint masks into single points "
                    "first."
                )
            indices = self._ds.map_to_array_coordinates(mask)
            if arr.ndim > 2:
                values = arr[:, indices[:, 0], indices[:, 1]]
            else:
                values = arr[indices[:, 0], indices[:, 1]]

        return np.asarray(values)

    def _points_to_xy(
        self, points: FeatureCollection | GeoDataFrame | DataFrame
    ) -> np.typing.NDArray:
        """Extract an ``(N, 2)`` float array of ``(x, y)`` coordinates from points.

        Args:
            points: A point :class:`~pyramids.feature.FeatureCollection` /
                :class:`~geopandas.GeoDataFrame`, or a :class:`~pandas.DataFrame`
                carrying ``x`` and ``y`` columns.

        Returns:
            np.ndarray: Coordinates with shape ``(N, 2)`` as ``float``.

        Raises:
            ValueError: A ``DataFrame`` lacking ``x``/``y`` columns.
            TypeError: ``points`` is not a supported type.
        """
        if isinstance(points, FeatureCollection):
            verts = points.with_coordinates()
            return cast(
                np.typing.NDArray, verts.loc[:, ["x", "y"]].to_numpy(dtype=float)
            )
        if isinstance(points, GeoDataFrame):
            verts = FeatureCollection(points).with_coordinates()
            return cast(
                np.typing.NDArray, verts.loc[:, ["x", "y"]].to_numpy(dtype=float)
            )
        if isinstance(points, DataFrame):
            if not all(col in points.columns for col in ("x", "y")):
                raise ValueError(
                    "If the input is a DataFrame, it must have 'x' and 'y' columns."
                )
            return cast(
                np.typing.NDArray, points.loc[:, ["x", "y"]].to_numpy(dtype=float)
            )
        raise TypeError(
            "points must be a FeatureCollection, GeoDataFrame, or DataFrame with "
            f"x/y columns - given {type(points)}."
        )

    def sample(
        self,
        points: FeatureCollection | GeoDataFrame | DataFrame,
        *,
        bands: int | list[int] | None = None,
        masked: bool = False,
        on_out_of_bounds: str = "nodata",
    ) -> np.typing.NDArray:
        """Sample band values at point coordinates.

        The memory- and out-of-bounds-safe counterpart to
        :meth:`extract` with a point mask. Each point is mapped to its
        containing pixel with a **vectorised inverse geotransform** (``O(1)`` per
        point) and read with a **1x1 windowed read** — so a handful of points on
        a multi-gigabyte raster touches only those pixels, never the whole array.
        Points falling outside the raster are handled explicitly instead of being
        silently snapped to the nearest edge cell.

        Args:
            points (FeatureCollection | GeoDataFrame | DataFrame):
                Point locations to sample. A ``FeatureCollection`` /
                ``GeoDataFrame`` with point geometry, or a ``DataFrame`` with
                ``x`` and ``y`` columns. Coordinates must already be in the
                raster's CRS (no reprojection is performed).
            bands (int | list[int] | None):
                Which band(s) to sample, zero-based. ``None`` (default) samples
                every band and returns a ``(n_bands, n_points)`` array; a single
                ``int`` returns a 1-D ``(n_points,)`` array; a list returns a
                ``(len(bands), n_points)`` array in the requested order.
            masked (bool):
                When ``True`` return a :class:`numpy.ma.MaskedArray` with
                out-of-bounds points masked. Defaults to ``False``.
            on_out_of_bounds (str):
                How to treat points outside the raster extent:

                - ``"nodata"`` (default): fill with the band's no-data value
                  (``NaN`` when the band has none).
                - ``"raise"``: raise :class:`OutOfBoundsError`.
                - ``"snap"``: clamp to the nearest edge pixel (the legacy
                  :meth:`extract` behaviour).

        Returns:
            np.ndarray:
                Sampled values, ordered to match ``points``. Shape is
                ``(n_points,)`` for a single ``int`` band, otherwise
                ``(n_bands, n_points)``. A :class:`numpy.ma.MaskedArray` when
                ``masked=True``.

        Raises:
            ValueError: ``on_out_of_bounds`` is not one of the allowed values, or
                ``bands`` references a band outside the raster.
            OutOfBoundsError: ``on_out_of_bounds="raise"`` and a point lies
                outside the raster extent.
            TypeError: ``points`` is not a supported type.

        Examples:
            - Sample a 2-band raster at three points and read the per-band values:
                ```python
                >>> import numpy as np
                >>> from geopandas import GeoDataFrame
                >>> from shapely.geometry import Point
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.arange(2 * 5 * 5, dtype="float32").reshape(2, 5, 5)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
                ... )
                >>> pts = GeoDataFrame(
                ...     geometry=[Point(0.5, 4.5), Point(2.5, 2.5)], crs=4326
                ... )
                >>> ds.sample(pts).tolist()
                [[0.0, 12.0], [25.0, 37.0]]

                ```
            - Sample a single band and get a flat array of values:
                ```python
                >>> import numpy as np
                >>> from geopandas import GeoDataFrame
                >>> from shapely.geometry import Point
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(25, dtype="float32").reshape(1, 5, 5)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
                ... )
                >>> pts = GeoDataFrame(geometry=[Point(0.5, 4.5), Point(4.5, 0.5)], crs=4326)
                >>> ds.sample(pts, bands=0).tolist()
                [0.0, 24.0]

                ```
            - Points outside the extent become no-data instead of snapping:
                ```python
                >>> import numpy as np
                >>> from geopandas import GeoDataFrame
                >>> from shapely.geometry import Point
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(25, dtype="float32").reshape(1, 5, 5)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
                ... )
                >>> pts = GeoDataFrame(geometry=[Point(2.5, 2.5), Point(100, 100)], crs=4326)
                >>> ds.sample(pts, bands=0).tolist()
                [12.0, -9999.0]

                ```
        """
        if on_out_of_bounds not in ("nodata", "raise", "snap"):
            raise ValueError(
                "on_out_of_bounds must be one of 'nodata', 'raise', 'snap'; got "
                f"{on_out_of_bounds!r}."
            )

        band_list, squeeze = self._resolve_sample_bands(bands, self._ds.band_count)

        xy = self._points_to_xy(points)
        n_points = xy.shape[0]

        x0, dx, rxy, y0, ryx, dy = self._ds.geotransform
        det = dx * dy - rxy * ryx
        delta_x = xy[:, 0] - x0
        delta_y = xy[:, 1] - y0
        col = np.floor((dy * delta_x - rxy * delta_y) / det).astype(int)
        row = np.floor((-ryx * delta_x + dx * delta_y) / det).astype(int)

        n_rows, n_cols = self._ds.rows, self._ds.columns
        out_of_bounds = (row < 0) | (row >= n_rows) | (col < 0) | (col >= n_cols)
        if on_out_of_bounds == "raise" and out_of_bounds.any():
            raise OutOfBoundsError(
                f"{int(out_of_bounds.sum())} of {n_points} points fall outside the "
                "raster extent."
            )
        if on_out_of_bounds == "snap":
            row = np.clip(row, 0, n_rows - 1)
            col = np.clip(col, 0, n_cols - 1)
            out_of_bounds = np.zeros(n_points, dtype=bool)

        in_bounds_idx = np.flatnonzero(~out_of_bounds)
        rows_out = self._read_point_samples(
            band_list, col, row, in_bounds_idx, n_points
        )

        stacked = np.vstack(rows_out) if rows_out else np.empty((0, n_points))
        result: np.ndarray = stacked[0] if squeeze else stacked
        if masked:
            mask = (
                out_of_bounds
                if squeeze
                else np.broadcast_to(out_of_bounds, result.shape)
            )
            result = np.ma.masked_array(result, mask=np.array(mask))
        return result

    @staticmethod
    def _resolve_sample_bands(
        bands: int | list[int] | None, band_count: int
    ) -> tuple[list[int], bool]:
        """Resolve the band list and squeeze flag for :meth:`sample`.

        Args:
            bands: ``None`` (all bands), a single ``int``, or a list of indices.
            band_count: Number of bands in the dataset.

        Returns:
            ``(band_list, squeeze)`` — the resolved zero-based band indices and
            whether a single-``int`` request should collapse the leading axis.

        Raises:
            ValueError: A requested band is outside ``[0, band_count)``.
        """
        return resolve_band_indices(bands, band_count)

    def _read_point_samples(
        self,
        band_list: list[int],
        col: np.ndarray,
        row: np.ndarray,
        in_bounds_idx: np.ndarray,
        n_points: int,
    ) -> list[np.ndarray]:
        """Sample the in-bounds points from each band in ``band_list``.

        Two strategies, chosen per call from how tightly the points cluster:

        * **A windowed read** over their bounding box, then array indexing —
          one GDAL call per band instead of one per point.
        * **A 1x1 read per point**, kept for sparse or widely scattered points,
          where the bounding box would pull in far more pixels than were asked
          for.

        The switch compares the bounding box area against the point count, so a
        handful of scattered points never drags in a near-full-raster read while
        a dense batch stops paying per-point GDAL overhead.

        A window that clears that test but is large in absolute terms is read in
        horizontal strips of at most ``_POINT_WINDOW_MAX_PIXELS`` rather than in
        one block, so the peak allocation is bounded without falling back to
        per-point reads. That fallback would be the wrong answer here by
        construction: a batch big enough to trip an absolute ceiling is a dense
        one, and dense is exactly the case per-point reads are slowest for.

        Out-of-bounds points keep the fill value — the band's no-data value, or
        ``NaN`` when it has none (which promotes an integer band to float).

        Returns:
            One ``(n_points,)`` array per band, in ``band_list`` order.
        """
        plan = self._plan_point_window(col, row, in_bounds_idx)
        rows_out: list[np.ndarray] = []
        for b in band_list:
            gdal_band = self._ds.raster.GetRasterBand(b + 1)
            fill, out_dtype = _point_sample_fill(gdal_band)
            band_values = np.full(n_points, fill, dtype=out_dtype)
            if plan is None:
                self._sample_per_point(gdal_band, band_values, col, row, in_bounds_idx)
            else:
                self._sample_windowed(gdal_band, band_values, in_bounds_idx, plan)
            rows_out.append(band_values)
        return rows_out

    @staticmethod
    def _plan_point_window(
        col: np.ndarray, row: np.ndarray, in_bounds_idx: np.ndarray
    ) -> _PointWindow | None:
        """Decide whether one windowed read beats a read per point.

        Args:
            col: Fractional column of every requested point.
            row: Fractional row of every requested point.
            in_bounds_idx: Indices of the points that fall inside the raster.

        Returns:
            _PointWindow | None: The window to read, or `None` when the points
                are too sparse for one to pay off.
        """
        n_in_bounds = int(len(in_bounds_idx))
        if n_in_bounds <= 1:
            return None
        in_cols = col[in_bounds_idx].astype(int)
        in_rows = row[in_bounds_idx].astype(int)
        x_off, y_off = int(in_cols.min()), int(in_rows.min())
        x_size = int(in_cols.max()) - x_off + 1
        y_size = int(in_rows.max()) - y_off + 1
        # Worth reading as a window only while the box stays a small multiple of
        # the points themselves; past that the wasted pixels cost more than the
        # per-point calls they would save.
        if x_size * y_size > max(
            _POINT_WINDOW_MIN_PIXELS, n_in_bounds * _POINT_WINDOW_MAX_WASTE
        ):
            return None
        # Rows per read, so a box that clears the ratio test but is large in
        # absolute terms is still bounded. One strip covers the whole box in the
        # common case, which is a single read exactly as before.
        strip_rows = max(1, _POINT_WINDOW_MAX_PIXELS // max(x_size, 1))
        return _PointWindow(
            x_off=x_off,
            y_off=y_off,
            x_size=x_size,
            y_size=y_size,
            strip_rows=strip_rows,
            in_rows=in_rows,
            in_cols=in_cols,
        )

    @staticmethod
    def _sample_windowed(
        gdal_band: gdal.Band,
        band_values: np.ndarray,
        in_bounds_idx: np.ndarray,
        plan: _PointWindow,
    ) -> None:
        """Fill `band_values` from strip reads over the planned window.

        Args:
            gdal_band: The band to read.
            band_values: Output array, modified in place.
            in_bounds_idx: Indices of the points that fall inside the raster.
            plan: The window and strip height to read.
        """
        for strip_start in range(0, plan.y_size, plan.strip_rows):
            strip_height = min(plan.strip_rows, plan.y_size - strip_start)
            block = np.asarray(
                gdal_band.ReadAsArray(
                    plan.x_off, plan.y_off + strip_start, plan.x_size, strip_height
                )
            )
            local_rows = plan.in_rows - plan.y_off - strip_start
            in_strip = (local_rows >= 0) & (local_rows < strip_height)
            if in_strip.any():
                band_values[in_bounds_idx[in_strip]] = block[
                    local_rows[in_strip], plan.in_cols[in_strip] - plan.x_off
                ]

    @staticmethod
    def _sample_per_point(
        gdal_band: gdal.Band,
        band_values: np.ndarray,
        col: np.ndarray,
        row: np.ndarray,
        in_bounds_idx: np.ndarray,
    ) -> None:
        """Fill `band_values` with one 1x1 read per point.

        Args:
            gdal_band: The band to read.
            band_values: Output array, modified in place.
            col: Fractional column of every requested point.
            row: Fractional row of every requested point.
            in_bounds_idx: Indices of the points that fall inside the raster.
        """
        for i in in_bounds_idx:
            window = gdal_band.ReadAsArray(int(col[i]), int(row[i]), 1, 1)
            band_values[i] = window[0, 0]

    def sieve(
        self,
        threshold: int,
        *,
        band: int = 0,
        connectedness: int = 4,
        mask: Dataset | None = None,
    ) -> Dataset:
        """Remove small pixel clumps with ``gdal.SieveFilter``.

        Raster polygons — connected groups of identical-value pixels — smaller
        than ``threshold`` pixels are dissolved into their largest neighbour.
        This is the standard clean-up for "salt-and-pepper" speckle in
        classification rasters. Implemented natively via GDAL; returns a new
        single-band :class:`~pyramids.dataset.Dataset`.

        Args:
            threshold (int):
                Minimum polygon size to keep, in pixels. Clumps with fewer
                pixels are merged away. Must be ``>= 1``.
            band (int):
                Zero-based index of the band to sieve. Defaults to ``0``.
            connectedness (int):
                Pixel connectivity used to define a clump: ``4`` (edge-adjacent,
                the default) or ``8`` (edge- and diagonal-adjacent).
            mask (Dataset | None):
                Optional single-band mask. Pixels where the mask is zero are
                excluded from sieving. ``None`` (default) uses the source band's
                no-data mask.

        Returns:
            Dataset:
                A new single-band dataset with small clumps removed, sharing the
                source geotransform, CRS, and no-data value.

        Raises:
            ValueError: ``threshold < 1``, ``connectedness`` is not 4 or 8, or
                ``band`` is out of range.

        Examples:
            - Remove an isolated speckle pixel from a classified raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.ones((6, 6), dtype="int32")
                >>> arr[0:3, 0:3] = 2      # a 9-pixel clump (kept)
                >>> arr[5, 5] = 2          # a lone pixel (removed)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 6), cell_size=1.0, epsg=4326),
                ... )
                >>> cleaned = ds.sieve(threshold=4).read_array()
                >>> int(cleaned[5, 5])     # merged into the background
                1
                >>> int(cleaned[0, 0])     # large clump survives
                2

                ```
            - 8-connectivity joins diagonal neighbours that 4-connectivity keeps
              separate:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.ones((5, 5), dtype="int32")
                >>> arr[1, 1] = 2
                >>> arr[2, 2] = 2          # touches (1,1) only diagonally
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
                ... )
                >>> int(ds.sieve(threshold=2, connectedness=8).read_array()[1, 1])
                2

                ```
        """
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}.")
        if connectedness not in (4, 8):
            raise ValueError(f"connectedness must be 4 or 8, got {connectedness}.")
        validate_band_index(band, self._ds.band_count)

        # Seed the sieve target with GDAL's block-based copy of the one band
        # (geotransform, CRS, dtype, and no-data carried across in the C layer)
        # instead of a full-band ``ReadAsArray`` -> ``WriteArray`` NumPy round
        # trip, so the whole band is never materialised as a NumPy array (#969).
        # gdal.Translate also carries the band's color table / RAT / scale-offset
        # onto the result (the old bare-MEM seed dropped them); the sieved pixels
        # are unchanged either way, so this only preserves more metadata.
        out_ds = gdal.Translate("", self._ds.raster, format="MEM", bandList=[band + 1])
        dst_band = out_ds.GetRasterBand(1)

        mask_band = mask.raster.GetRasterBand(1) if mask is not None else None
        gdal.SieveFilter(dst_band, mask_band, dst_band, threshold, connectedness)
        dst_band.FlushCache()
        return self._ds.__class__(out_ds, access="write")

    def proximity(
        self,
        *,
        band: int = 0,
        target_values: list[int] | None = None,
        distance_units: str = "GEO",
        max_distance: float | None = None,
        nodata: float | None = None,
    ) -> Dataset:
        """Compute per-pixel distance to the nearest target pixel (``gdal.ComputeProximity``).

        The GDAL-native equivalent of ``gdal_proximity``: every output pixel
        holds the Euclidean distance to the closest "target" pixel in the source
        band. Targets are the pixels whose value is in ``target_values`` (or any
        non-zero pixel when ``target_values`` is ``None``). Useful for
        distance-to-coast, distance-to-river, buffer analyses, etc.

        Args:
            band (int):
                Zero-based index of the source band. Defaults to ``0``.
            target_values (list[int] | None):
                Pixel values that count as targets. ``None`` (default) treats
                every non-zero pixel as a target.
            distance_units (str):
                ``"GEO"`` (default) measures distance in the CRS's georeferenced
                units; ``"PIXEL"`` measures it in pixels.
            max_distance (float | None):
                Stop searching beyond this distance. Pixels farther than this get
                ``nodata`` when given, otherwise ``max_distance``. ``None``
                (default) searches the whole raster.
            nodata (float | None):
                Value written to the output band's no-data slot and used to fill
                pixels beyond ``max_distance``. ``None`` (default) sets no
                no-data value.

        Returns:
            Dataset:
                A new single-band ``Float32`` dataset of distances, sharing the
                source geotransform and CRS.

        Raises:
            ValueError: ``distance_units`` is not ``"GEO"``/``"PIXEL"``,
                ``band`` is out of range, or ``max_distance`` is negative.

        Examples:
            - Distance (in pixels) from every cell to a single target pixel:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.zeros((5, 5), dtype="int32")
                >>> arr[2, 2] = 1
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
                ... )
                >>> dist = ds.proximity(distance_units="PIXEL").read_array()
                >>> float(dist[2, 2])      # the target itself
                0.0
                >>> float(dist[2, 0])      # two cells to the left
                2.0

                ```
            - GEO units scale distances by the cell size:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.zeros((5, 5), dtype="int32")
                >>> arr[2, 2] = 1
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 10), cell_size=2.0, epsg=4326),
                ... )
                >>> dist = ds.proximity(distance_units="GEO").read_array()
                >>> float(dist[2, 0])      # two cells x 2.0 units
                4.0

                ```
        """
        if distance_units not in ("GEO", "PIXEL"):
            raise ValueError(
                f"distance_units must be 'GEO' or 'PIXEL', got {distance_units!r}."
            )
        validate_band_index(band, self._ds.band_count)
        if max_distance is not None and max_distance < 0:
            raise ValueError(f"max_distance must be >= 0, got {max_distance}.")

        src_band = self._ds.raster.GetRasterBand(band + 1)
        out_ds = gdal.GetDriverByName("MEM").Create(
            "", self._ds.columns, self._ds.rows, 1, gdal.GDT_Float32
        )
        out_ds.SetGeoTransform(self._ds.geotransform)
        out_ds.SetProjection(self._ds.crs)
        prox_band = out_ds.GetRasterBand(1)

        options = [f"DISTUNITS={distance_units}"]
        if target_values is not None:
            options.append("VALUES=" + ",".join(str(v) for v in target_values))
        if max_distance is not None:
            options.append(f"MAXDIST={max_distance}")
        if nodata is not None:
            options.append(f"NODATA={nodata}")
            prox_band.SetNoDataValue(float(nodata))

        gdal.ComputeProximity(src_band, prox_band, options=options)
        prox_band.FlushCache()
        return self._ds.__class__(out_ds, access="write")

    def overlay(
        self,
        classes_map,
        band: int = 0,
        exclude_value: float | int | None = None,
    ) -> dict[float, list[float]]:
        """Overlay.

        Overlay method extracts all the values in the dataset for each class in the given class map.

        Both rasters are read as `read_array` returns them, so a CF-packed band (`scale_factor` /
        `add_offset`) contributes physical values, and the no-data cells are left out by matching the
        sentinel expressed in those same physical units.

        Args:
            classes_map (Dataset):
                Dataset object for the raster that has classes you want to overlay with the raster.
            band (int):
                If the raster is multi-band, choose the band you want to overlay with the classes map. Default is 0.
            exclude_value (Numeric, optional):
                Values you want to exclude from extracted values, in physical units. Default is None.

        Returns:
            Dict:
                Dictionary with class values as keys (from the class map), and for each key a list of all the intersected
                values in the base map.

        Raises:
            AlignmentError: `classes_map` is not aligned with this dataset.

        Examples:
            - Build a small value raster and an aligned class raster in memory:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> values = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
              >>> dataset = Dataset.from_array(
              ...     values,
              ...     geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
              ... )
              >>> class_map = np.array([[1, 1], [2, 2]], dtype="int32")
              >>> classes = Dataset.from_array(
              ...     class_map,
              ...     geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
              ... )

              ```

            - Overlay the value raster with the class raster. The result maps each
              class to the list of values that fall inside it:

              ```python
              >>> overlaid = dataset.overlay(classes)
              >>> sorted(int(key) for key in overlaid)
              [1, 2]

              ```

            - Use a class key to read the values that overlay that class:

              ```python
              >>> [float(value) for value in sorted(overlaid[1], key=float)]
              [10.0, 20.0]
              >>> [float(value) for value in sorted(overlaid[2], key=float)]
              [30.0, 40.0]

              ```
        """
        if not self._ds.spatial._check_alignment(classes_map):
            raise AlignmentError(
                "The class Dataset is not aligned with the current raster, please use the method "
                "'align' to align both rasters."
            )
        # The physical sentinel, because `get_indices2` compares it against strips
        # `read_array` produced; the stored one matches nothing on a packed band.
        physical_sentinel = self._physical_no_data(band if band is not None else 0)
        no_data_value = physical_sentinel if physical_sentinel is not None else np.nan
        mask = (
            [no_data_value, exclude_value]
            if exclude_value is not None
            else [no_data_value]
        )

        def _group(
            acc: dict[Any, list[Any]], strip: np.ndarray, window: list[int]
        ) -> dict[Any, list[Any]]:
            # Read the aligned class strip over the same window; the rasters are
            # aligned (checked above), so the windows index the same cells.
            classes = classes_map.read_array(window=window)
            for ind_i in get_indices2(strip, mask):
                key = classes[ind_i[0], ind_i[1]]
                if key not in acc:
                    acc[key] = []
                acc[key].append(strip[ind_i[0], ind_i[1]])
            return acc

        # Stream base + class rasters in row strips so neither is read whole (#967).
        # Full-width top-to-bottom strips keep row-major order, so each class's value
        # list is byte-identical to the whole-array pass.
        values: dict[Any, list[Any]] = self._ds.io.stream_reduce(_group, {}, band=band)
        return values

    def get_mask(self, band: int = 0) -> np.typing.NDArray:
        """Get the mask array.

        Args:
            band (int):
                Band index. Default is 0.

        Returns:
            np.ndarray:
                Array of the mask. 0 value for cells out of the domain, and 255 for cells in the domain.
        """
        arr = np.asarray(self._ds._iloc(band).GetMaskBand().ReadAsArray())
        return arr

    def mask_flags(self, band: int = 0) -> MaskFlags:
        """Decode the GDAL mask flags of ``band`` into a :class:`MaskFlags`.

        Tells you *why* a band is masked (or not): a fully-valid band, a shared
        per-dataset mask, an alpha-band mask, or a no-data-derived mask.

        Args:
            band: Band index. Default 0.

        Returns:
            MaskFlags: the four decoded boolean flags.

        Examples:
            - A band with a no-data value reports ``nodata``:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.ones((4, 4), "float32"),
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0),
                ... )
                >>> ds.mask_flags().nodata
                True

                ```
        """
        flags = self._ds._iloc(band).GetMaskFlags()
        return MaskFlags(
            all_valid=bool(flags & gdal.GMF_ALL_VALID),
            per_dataset=bool(flags & gdal.GMF_PER_DATASET),
            alpha=bool(flags & gdal.GMF_ALPHA),
            nodata=bool(flags & gdal.GMF_NODATA),
        )

    def read_masks(
        self,
        band: int | None = None,
        *,
        window: Window | None = None,
    ) -> np.typing.NDArray:
        """Read per-band mask arrays (``0`` invalid, ``255`` valid).

        The companion to :meth:`Dataset.read_array(masked=True) <read_array>`:
        instead of applying the mask, it returns the mask itself, so you can
        inspect *which* pixels are masked.

        Args:
            band: Band index. ``None`` (default) returns every band's mask
                stacked as ``(band_count, rows, cols)``; an index returns a
                single ``(rows, cols)`` mask.
            window: Optional :class:`Window` to read only a sub-block.

        Returns:
            numpy.ndarray: the mask array(s); ``0`` marks out-of-domain pixels
            and ``255`` marks valid pixels.

        Examples:
            - The mask of a no-data raster is ``0`` exactly at the no-data cells:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.array([[1.0, -9999.0, 3.0, 4.0]] * 4, dtype="float32")
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0),
                ... )
                >>> mask = ds.read_masks(0)
                >>> mask.shape
                (4, 4)
                >>> bool((mask[:, 1] == 0).all())
                True

                ```
        """
        if window is None:
            read_args: tuple = ()
        else:
            clamped = window.crop(self._ds.rows, self._ds.columns)
            if clamped is None:
                raise OutOfBoundsError(
                    f"window {window} lies entirely outside the raster "
                    f"({self._ds.rows}x{self._ds.columns})."
                )
            read_args = clamped.to_read_args()
        bands = [band] if band is not None else range(self._ds.band_count)
        masks = [
            np.asarray(self._ds._iloc(index).GetMaskBand().ReadAsArray(*read_args))
            for index in bands
        ]
        result = masks[0] if band is not None else np.stack(masks)
        return result

    def create_mask_band(self, *, per_dataset: bool = True) -> None:
        """Create a mask band on the dataset.

        Args:
            per_dataset: ``True`` (default) creates a single mask shared by every
                band (``GMF_PER_DATASET``); ``False`` creates a per-band mask.

        Raises:
            ReadOnlyError: The dataset is opened read-only.

        Examples:
            - After creating a per-dataset mask, the flags report it:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> import tempfile, os
                >>> path = os.path.join(tempfile.mkdtemp(), "m.tif")
                >>> Dataset.from_array(
                ...     np.ones((4, 4), "float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0),
                ... ).to_file(path)
                >>> ds = Dataset.read_file(path, read_only=False)
                >>> ds.create_mask_band()
                >>> ds.mask_flags().per_dataset
                True

                ```
        """
        if self._ds.access == "read_only":
            raise ReadOnlyError(
                "The Dataset is opened read-only. Please read the dataset using "
                "read_only=False to create a mask band."
            )
        self._ds.raster.CreateMaskBand(gdal.GMF_PER_DATASET if per_dataset else 0)

    def _warn_if_nodata_absent(self, arr: np.ndarray, no_data_val: Any) -> None:
        """Warn when the band's nodata value does not actually appear in the data."""
        # One predicate for both spellings of the sentinel. The `else` arm used
        # `np.isclose(arr, no_data_val)`, which is always False for a float NaN
        # sentinel -- so a raster whose nodata is NaN warned that its nodata
        # was absent even when every cell was nodata.
        if not is_stored_no_data(arr, no_data_val).any():
            self._ds.logger.warning(
                "the nodata value stored in the raster does not exist in the raster "
                "so either the raster extent is all full of data, or the no_data_value stored in the raster is"
                " not correct"
            )

    @staticmethod
    def _apply_exclude_values(
        arr: np.ndarray, exclude_values: list[Any], no_data_val: Any
    ) -> np.ndarray:
        """Set cells matching any exclude value to nodata, promoting to float if needed."""
        for val in exclude_values:
            try:
                # None nodata on an int array raises (None reads as float); promote.
                arr[np.isclose(arr, val)] = no_data_val
            except TypeError:
                arr = arr.astype(np.float32)
                arr[np.isclose(arr, val)] = no_data_val
        return arr

    def footprint(
        self,
        band: int = 0,
        exclude_values: list[Any] | None = None,
        *,
        max_samples: int | None = None,
    ) -> GeoDataFrame | None:
        """Extract the real coverage of the values in a certain band.

        The coverage is a mask, so it is decided where the sentinel lives: against the
        band's **stored** values, never the physical ones. A CF-packed band
        (`scale_factor` / `add_offset`) is therefore footprinted exactly like an
        unpacked one -- its gaps hold the stored sentinel, not the number a default
        `read_array` shows there -- and the values themselves are never used.

        Args:
            band (int):
                Band index. Default is 0.
            exclude_values (List[Any] | None):
                Values treated as uncovered in addition to the band's no-data sentinel,
                e.g. `[0]` for the dry cells of a flood-depth raster. They are compared
                against the band's stored values, the same units as `no_data_value`, so
                on a packed band give them as stored counts.

                - Example of exclude_values usage:

                  ```python
                  >>> exclude_values = [0]

                  ```

                - This parameter is introduced particularly in the case of rasters that has the no_data_value stored in
                  the `no_data_value` property does not match the value stored in the band, so this option can correct
                  this behavior.
            max_samples (int, optional):
                Opt-in cap on how many pixels of the band are read to build the
                coverage mask. When set and the band has more than
                ``max_samples`` cells, GDAL reads a nearest-neighbour
                **decimated** grid (~``max_samples`` cells) instead of the full
                band, so a very large raster is footprinted without materialising
                it whole. The extracted polygon is then **approximate** -- traced
                on the coarser grid, so its edges and area are coarser than the
                exact footprint. ``None`` (default) reads every pixel, so the
                footprint is exact.

        Returns:
            GeoDataFrame | None:
                - geodataframe containing the polygon representing the extent of the raster. the extent column should
                  contain a value of 2 only.
                - if the dataset had separate polygons, each polygon will be in a separate row.
                - `None` (with a logged warning) when no cell of the band is covered.

        Raises:
            ValueError: `max_samples` is not `None` and is less than 1.

        Examples:
            - Build a raster whose non-flooded cells are ``0`` and whose flooded cells
              carry a positive depth. Excluding the zero cells extracts the flood extent
              as one polygon per connected region:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.zeros((4, 4), dtype="float32")
              >>> arr[1:3, 1:3] = 5.0    # a 2x2 block of flooded cells
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
              ... )

              ```

            - Extract the footprint of the flooded cells by excluding the zero-depth
              cells. Covered cells are flagged with the value ``2``:

              ```python
              >>> extent = dataset.footprint(band=0, exclude_values=[0])
              >>> extent.shape
              (1, 2)
              >>> list(extent.columns)
              ['Band_1', 'geometry']
              >>> float(extent["Band_1"].iloc[0])
              2.0
              >>> float(extent.geometry.iloc[0].area)
              4.0
              >>> extent.plot()  # doctest: +SKIP
              <Axes: >

              ```

            - A CF-packed band's gap is found in its stored counts, so it stays out of
              the footprint:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> packed = Dataset.from_array(
              ...     np.array([[-9999, 100], [200, 300]], dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
              ...     no_data_value=-9999,
              ... )
              >>> packed.scale = [0.01]
              >>> float(packed.footprint().geometry.iloc[0].area)
              3.0

              ```
        """
        # Stored counts: a footprint is a coverage mask, decided against the stored
        # sentinel, and the values themselves are never used. A physical read made
        # every gap on a packed band look covered.
        arr = self._read_decimated(band, max_samples, unpack=False)
        no_data_val = self._ds.no_data_value[band]
        # A decimated read spans the same extent with fewer, larger cells, so the
        # mask's geotransform must scale its pixel size (and rotation terms) to
        # the coarser grid; the origin is unchanged. Full-resolution reads leave
        # the geotransform untouched.
        geotransform = tuple(
            self._ds.transform.rescaled_to((self._ds.rows, self._ds.columns), arr.shape)
        )

        self._warn_if_nodata_absent(arr, no_data_val)
        if exclude_values:
            arr = self._apply_exclude_values(arr, exclude_values, no_data_val)

        # Build the coverage mask: covered cells -> 2, nodata cells -> 0.
        valid = ~is_stored_no_data(arr, no_data_val)
        if not valid.any():
            self._ds.logger.warning("the raster is full of no_data_value")
            return None
        # _band_to_polygon polygonises the mask using the band as its own Polygonize
        # mask, which drops mask==0 cells, so only the covered (2) cells are collected
        # for any source nodata value. float32 keeps the mask lightweight.
        arr = np.where(valid, 2, 0).astype(np.float32)
        # The scratch mask must be a plain raster Dataset that exposes GetRasterBand for
        # polygonisation. self._ds.from_array would build a bandless NetCDF
        # container for a variable view, so call the base Dataset classmethod explicitly.
        # Local import breaks the engines <-> Dataset import cycle.
        from pyramids.dataset.dataset import Dataset

        new_dataset = Dataset.from_array(
            arr,
            no_data_value=0,
            geo_ref=GeoReference(
                geo=cast(
                    "tuple[float, float, float, float, float, float]", geotransform
                ),
                epsg=crs_spec(self._ds.epsg, self._ds.crs),
            ),
        )
        # The mask is always single-band (the one extracted band flagged as 2 / nodata),
        # so polygonise its first band regardless of the source band index.
        gdf = new_dataset.to_polygons(band=0)
        names = self._ds.band_names
        col_name = names[band] if band < len(names) else f"Band_{band + 1}"
        gdf.rename(columns={"Band_1": col_name}, inplace=True)

        return gdf

    @staticmethod
    def normalize(array: np.ndarray) -> np.typing.NDArray:
        """Normalize numpy arrays into scale 0.0-1.0.

        Args:
            array (np.ndarray): Numpy array to normalize.

        Returns:
            np.ndarray: Normalized array.
        """
        array_min = array.min()
        array_max = array.max()
        val = (array - array_min) / (array_max - array_min)
        return np.asarray(val)

    @staticmethod
    def _rescale(
        array: np.ndarray, min_value: float, max_value: float
    ) -> np.typing.NDArray:
        val = (array - min_value) / (max_value - min_value)
        return val

    def get_histogram(
        self,
        band: int = 0,
        bins: int = 6,
        min_value: float | None = None,
        max_value: float | None = None,
        include_out_of_range: bool = False,
        approx_ok: bool = False,
    ) -> tuple[list, list[tuple[Any, Any]]]:
        """Get the histogram of a band from GDAL, with its bucket edges in physical units.

        GDAL buckets the band without handing any pixel to Python, and it answers in the
        **stored** units. On a CF-packed band (`scale_factor` / `add_offset`) the edges
        are therefore converted to physical units, `real = stored * scale + offset`, so
        they agree with `stats`, `read_array` and the array-based `plot_histogram`. A
        caller's `min_value` / `max_value` are taken as physical values too, and converted
        to stored units before GDAL sees them; a negative `scale_factor` reverses them, so
        the window is re-sorted first. The counts are unit-free and come back unchanged.

        Args:
            band (int, optional):
                Zero-based band index. Default is 0.
            bins (int, optional):
                Number of bins. Default is 6.
            min_value (float, optional):
                Low end of the bucketed range, in physical units. Default is None, the
                band's own minimum.
            max_value (float, optional):
                High end of the bucketed range, in physical units. Default is None, the
                band's own maximum. GDAL leaves a value equal to it out of the last bucket
                unless `include_out_of_range=True`.
            include_out_of_range (bool, optional):
                If True, add out-of-range values into the first and last buckets. Default is False.
            approx_ok (bool, optional):
                If True, compute an approximate histogram by using subsampling or overviews. Default is False.

        Returns:
            tuple[list, list[tuple[Any, Any]]]:
                The count in each bucket, and each bucket's `(low, high)` edges in
                physical units, one pair per bucket and in ascending order.

        Hint:
            - The value of the histogram will be stored in an xml file by the name of the raster file with the extension
                of .aux.xml. On a packed band its `HistMin` / `HistMax` are the **stored** window, so they will not
                match the edges this returns.

            - The content of the file will be like the following:
              ```xml

                  <PAMDataset>
                    <PAMRasterBand band="1">
                      <Description>Band_1</Description>
                      <Histograms>
                        <HistItem>
                          <HistMin>0</HistMin>
                          <HistMax>88</HistMax>
                          <BucketCount>6</BucketCount>
                          <IncludeOutOfRange>0</IncludeOutOfRange>
                          <Approximate>0</Approximate>
                          <HistCounts>75|6|0|4|2|1</HistCounts>
                        </HistItem>
                      </Histograms>
                    </PAMRasterBand>
                  </PAMDataset>

              ```

        Examples:
            - Create `Dataset` consists of 4 bands, 10 rows, 10 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.default_rng(1337).integers(1, 12, size=(10, 10))
              >>> print(arr)
              [[ 7 10  8  3  6 11  5 11  4 10]
               [ 6  2  1  3  2  4  4  6  7 10]
               [ 3  1  6  3 10  9  2  5  4  1]
               [ 9  5  6  4  3  1  1 10  6  1]
               [ 5 10 11  6 10  1  1  9  4  9]
               [ 6  8  7  1  8  7 11 11  9  9]
               [ 4  3  5  1  1 11  4  9  6 11]
               [ 7  9  9  2  8  2  4  3  5  7]
               [11  8  1  9  5  5  4  4  7 10]
               [ 6  2 10  3  8  4  1  9  3  6]]
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```

            - Now, let's get the histogram of the first band using the `get_histogram` method with the default
                parameters:
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0)
                >>> print(hist)
                [19, 21, 8, 18, 17, 9]
                >>> print([(round(low, 2), round(high, 2)) for low, high in ranges])
                [(1.0, 2.67), (2.67, 4.33), (4.33, 6.0), (6.0, 7.67), (7.67, 9.33), (9.33, 11.0)]

                ```
            - we can also exclude values from the histogram by using the `min_value` and `max_value`. The bucket
                edges then span the requested `[min_value, max_value]` window rather than the band's own range:
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0, min_value=5, max_value=10)
                >>> print(hist)
                [8, 11, 7, 6, 11, 0]
                >>> print([(round(low, 2), round(high, 2)) for low, high in ranges])
                [(5.0, 5.83), (5.83, 6.67), (6.67, 7.5), (7.5, 8.33), (8.33, 9.17), (9.17, 10.0)]

                ```
            - For datasets with big dimensions, computing the histogram can take some time; approximating the computation
                of the histogram can save a lot of computation time. When using the parameter `approx_ok` with a `True`
                value the histogram will be calculated from resampling the band or from the overviews if they exist.
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0, approx_ok=True)
                >>> print(hist)
                [19, 21, 8, 18, 17, 9]
                >>> print([(round(low, 2), round(high, 2)) for low, high in ranges])
                [(1.0, 2.67), (2.67, 4.33), (4.33, 6.0), (6.0, 7.67), (7.67, 9.33), (9.33, 11.0)]

                ```
            - As you see for small datasets, the approximation of the histogram will be the same as without approximation.
            - On a CF-packed band the window is asked for, and the edges come back, in physical units:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> packed = Dataset.from_array(
                ...     np.array([[100, 200], [300, 400]], dtype="int16"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> packed.scale = [0.01]
                >>> hist, ranges = packed.get_histogram(band=0, bins=4, min_value=1.0, max_value=5.0)
                >>> hist
                [1, 1, 1, 1]
                >>> [(round(low, 2), round(high, 2)) for low, high in ranges]
                [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]

                ```

        """
        band_obj = self._ds._iloc(band)
        # `ComputeRasterMinMax` and `GetHistogram` are GDAL metadata calls, so they
        # speak stored units -- exactly like `GetStatistics`, which `stats` already
        # transforms. The bucket *edges* are values, so they get the same treatment,
        # and a caller's `min_value` / `max_value` are physical (they would have come
        # from `stats` or a read) and are converted the other way before GDAL sees
        # them. Counts are unit-free and need nothing. Without this the array-based
        # `plot_histogram`, which reads through `read_array`, and this one disagreed
        # about the same band.
        scale, offset = self._ds._effective_packing(band)
        packed = not _is_identity_packing(scale, offset)
        factor = 1.0 if scale is None or not packed else float(scale)
        shift = 0.0 if offset is None or not packed else float(offset)

        def _to_stored(value: float) -> float:
            return (value - shift) / factor

        def _to_physical(value: float) -> float:
            return value * factor + shift

        stored_min, stored_max = band_obj.ComputeRasterMinMax()
        stored_low = stored_min if min_value is None else _to_stored(min_value)
        stored_high = stored_max if max_value is None else _to_stored(max_value)
        # A negative `scale_factor` -- legal in CF, and how a geostationary scan
        # angle is stored -- reverses the order when converting, so the window is
        # re-sorted before GDAL is asked to bucket over it.
        stored_low, stored_high = sorted((stored_low, stored_high))

        bin_width = (stored_high - stored_low) / bins
        # Anchor the edges at the low end of the window the buckets were actually
        # computed over, not at the raster minimum. When a caller narrowed the
        # range the two differ, so the returned edges described buckets that
        # `GetHistogram` never filled.
        edges = [
            sorted(
                (
                    _to_physical(stored_low + i * bin_width),
                    _to_physical(stored_low + (i + 1) * bin_width),
                )
            )
            for i in range(bins)
        ]
        ranges = [(low, high) for low, high in edges]

        hist = band_obj.GetHistogram(
            min=stored_low,
            max=stored_high,
            buckets=bins,
            include_out_of_range=include_out_of_range,
            approx_ok=approx_ok,
        )
        return hist, ranges

    def _read_decimated(
        self, band: int, max_samples: int | None, *, unpack: bool = True
    ) -> np.ndarray:
        """Read a band whole, or a nearest-neighbour decimated version of it.

        When `max_samples` is set and the band has more cells than that, GDAL
        reads a coarser grid of roughly `max_samples` cells (decimated in the C
        layer) so the whole band is never materialised; otherwise the full band
        is read. Nearest-neighbour keeps the samples real pixel values.

        Args:
            band: Zero-based band index to read.
            max_samples: Approximate pixel budget, or `None` for an exact read.
            unpack: Physical values (the default) or, with `False`, the stored counts
                -- which a caller that judges cells against `no_data_value` needs, since
                the sentinel is a stored value.

        Returns:
            np.ndarray: The band array, full-resolution or decimated.

        Raises:
            ValueError: `max_samples` is not `None` and is less than 1.
        """
        if max_samples is not None and max_samples < 1:
            raise ValueError(
                f"max_samples must be a positive integer or None, got {max_samples}."
            )
        rows = self._ds.rows
        cols = self._ds.columns
        total = rows * cols
        if max_samples is None or total <= max_samples:
            return cast(np.ndarray, self._ds.read_array(band=band, unpack=unpack))
        factor = (total / max_samples) ** 0.5
        out_rows = max(1, round(rows / factor))
        out_cols = max(1, round(cols / factor))
        return cast(
            np.ndarray,
            self._ds.read_array(
                band=band,
                out_shape=(out_rows, out_cols),
                resampling="nearest",
                unpack=unpack,
            ),
        )

    def plot_histogram(
        self,
        band: int = 0,
        bins: int = 15,
        exclude_value: Any | None = None,
        ax: Axes | None = None,
        *,
        max_samples: int | None = None,
        **kwargs: Any,
    ):
        """Plot the value distribution of a band as a histogram.

        Backed by cleopatra's
        :class:`~cleopatra.glyphs.stats.histogram_glyph.HistogramGlyph`. The band is
        read into memory, the band's no-data cells, `exclude_value`
        (and any `NaN` for floating-point bands) are dropped, and only the
        remaining valid samples reach the glyph. Requires the `[viz]` extra.

        The histogram is drawn over physical values, as `read_array` returns them: a
        CF-packed band (`scale_factor` / `add_offset`) is unpacked. Its no-data cells
        are found first, against the stored values where the sentinel lives, and only
        then unpacked -- so a packed band's gaps are dropped rather than landing in the
        lowest bucket as the physical number a default read shows there.

        Args:
            band (int, optional):
                Band index to read. Default is `0`.
            bins (int, optional):
                Number of histogram bins. Default is `15`.
            exclude_value (Any, optional):
                An extra value to drop from the samples, in addition to the
                band's no-data cells and `NaN`. Compared against the physical
                values. Default is `None`.
            ax (matplotlib.axes.Axes, optional):
                Draw the histogram into these axes instead of creating them, so it can
                sit in a caller-owned layout. An axes already carries its figure, so
                `ax` on its own is sufficient and there is no separate `fig`
                parameter here. A new figure/axes is created when left unset. Default is
                `None`.
            max_samples (int, optional):
                Opt-in cap on how many pixels are read. When set and the band
                has more than `max_samples` cells, GDAL reads a
                nearest-neighbour **decimated** version (~`max_samples` cells)
                instead of the full band, so a very large raster is histogrammed
                without materialising it whole. The distribution is then
                **approximate** -- a subsample of the pixels, the usual
                expectation for a large raster. `None` (default) reads every
                pixel, so the histogram is exact.
            **kwargs:
                Style options forwarded to the `HistogramGlyph`
                constructor, filtered via
                :meth:`HistogramGlyph.filter_kwargs` so only accepted keys
                are passed.

        Returns:
            tuple:
                `(fig, ax, hist)` from
                :meth:`HistogramGlyph.histogram` — the
                :class:`matplotlib.figure.Figure`, the
                :class:`matplotlib.axes.Axes`, and the histogram `dict`.

        Raises:
            ValueError: If the band has no valid samples left after masking
                the no-data cells, `exclude_value`, and `NaN`, or `max_samples`
                is less than 1.

        Examples:
            - Plot the distribution of a band and reuse the matplotlib
              handles (tagged `+SKIP` — needs the `[viz]` extra):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.arange(100, dtype="float32").reshape(10, 10)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> fig, ax, hist = ds.plot_histogram(band=0, bins=8)  # doctest: +SKIP
                >>> _ = ax.set_title("band 0 distribution")  # doctest: +SKIP

                ```
            - Drop a sentinel value before binning:

                ```python
                >>> arr = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]], dtype="float32")
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> fig, ax, hist = ds.plot_histogram(band=0, exclude_value=99.0)  # doctest: +SKIP

                ```
        """
        require_cleopatra()
        from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph

        # The mask is judged against the stored counts, where the sentinel lives, and
        # the histogram is then drawn over the physical values -- the same order
        # `_domain_read` uses. Masking a physical read against the stored sentinel
        # matched nothing, so every gap landed in the lowest bucket.
        stored = self._read_decimated(band, max_samples, unpack=False).flatten()
        no_data_value = self._ds.no_data_value[band]
        in_domain = ~is_stored_no_data(stored, no_data_value)
        arr = np.asarray(apply_unpack(stored, *self._ds._effective_packing(band)))
        mask = np.ones(arr.shape, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            mask &= ~np.isnan(arr)
        # `is_stored_no_data` rather than a branch on the sentinel's spelling
        # plus an exact `!=`: it answers for a NaN sentinel and a concrete one
        # alike, so the two-branch form collapses -- and every other reader of
        # this band asks the same question through the same predicate, so the
        # histogram cannot count a different set of pixels from the warning
        # printed beside it. Its tolerance is the band dtype's, not a constant:
        # a fixed `rtol=1e-5` masked everything within 0.1 of a -9999 sentinel
        # and within 20 000 of a 2e9 one, so the bars quietly lost real cells.
        mask &= in_domain
        if exclude_value is not None:
            mask &= arr != exclude_value
        values = arr[mask]
        if values.size == 0:
            raise ValueError(
                f"Band {band} has no valid samples to histogram after masking "
                "no-data / exclude_value / NaN."
            )
        glyph = HistogramGlyph(values, ax=ax, **HistogramGlyph.filter_kwargs(kwargs))
        result = glyph.histogram(bins=bins)
        return result

    def to_image(
        self,
        band: int = 0,
        cmap: str = "viridis",
        exclude_value: Any | None = None,
    ):
        """Export a band as a colour-mapped RGB image.

        Reads the band, masks the no-data value (and an optional
        `exclude_value`), applies a matplotlib colormap via cleopatra's
        :meth:`ArrayGlyph.apply_colormap`, and returns the result as a
        :class:`PIL.Image.Image`. Masked / no-data pixels are rendered with
        the colormap's "bad" fill colour. Requires the `[viz]` extra.

        The colours are mapped over physical values, as `read_array` returns
        them: a CF-packed band (`scale_factor` / `add_offset`) is unpacked
        first. Its no-data cells are still found against the stored sentinel,
        so they are masked either way.

        Args:
            band (int, optional):
                Band index to export. Default is `0`.
            cmap (str, optional):
                Matplotlib colormap name. Default is `"viridis"`.
            exclude_value (Any, optional):
                An extra value to mask out, in addition to the band's
                no-data value, in physical units. Default is `None`.

        Returns:
            PIL.Image.Image:
                An RGB image of the colour-mapped band, the same width and
                height as the raster band.

        Raises:
            ValueError: If the band has no valid (non-nodata) pixels left
                after masking the no-data value, `exclude_value`, and
                `NaN` — there is then nothing to colour-map.

        Examples:
            - Export a band as a viridis thumbnail, inspect its size, and
              save it to disk (tagged `+SKIP` — needs the `[viz]` extra):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.arange(48, dtype="float32").reshape(6, 8)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> img = ds.to_image(band=0, cmap="viridis")  # doctest: +SKIP
                >>> img.size  # (width, height) == (columns, rows)  # doctest: +SKIP
                (8, 6)
                >>> img.save("band0.png")  # doctest: +SKIP

                ```
        """
        require_cleopatra()
        from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

        arr, domain = self._domain_read(band)
        # The *physical* sentinel: `arr` is what `read_array` returns, and cleopatra
        # does its own comparison against it, so the stored `-9999` would mask
        # nothing on a packed band. The mask below comes from `_domain_read`, which
        # judged it in stored units where the sentinel actually lives.
        no_data_value = self._physical_no_data(band)
        # The list cleopatra masks with, which is not the same question as the
        # one below. A NaN sentinel has no value to compare against, so it is
        # left out and the NaN branch covers it.
        exclude: list = []
        if not is_nan_sentinel(no_data_value):
            exclude.append(no_data_value)
        if exclude_value is not None:
            exclude.append(exclude_value)
        valid = np.ones(arr.shape, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            valid &= ~np.isnan(arr)
        # The same predicate `plot_histogram` above and
        # `_warn_if_nodata_absent` and `footprint` below ask of the same band.
        # These are two renderings of one raster, so a cell the histogram drops
        # and the image draws is a disagreement a reader can see -- and this
        # was the last site still asking in its own words, an exact `!=` over
        # `exclude`. `exclude_value` stays an exact match: it is a value the
        # caller named, not a sentinel the band declares. What this mask decides
        # is whether the band has anything to draw at all -- which cells come
        # out as the colormap's "bad" fill is cleopatra's own comparison against
        # the `exclude` list below, on its own tolerance.
        valid &= domain
        if exclude_value is not None:
            valid &= arr != exclude_value
        if not valid.any():
            raise ValueError(
                f"Band {band} has no valid (non-nodata) pixels to render to "
                "an image after masking no-data / exclude_value / NaN."
            )
        glyph = ArrayGlyph(arr, exclude_value=exclude if exclude else np.nan)
        image = glyph.to_image(glyph.apply_colormap(cmap))
        return image

    def plot_vector_field(
        self,
        u_band: int = 0,
        v_band: int = 1,
        kind: str = "quiver",
        ax: Axes | None = None,
        **kwargs: Any,
    ):
        """Plot two bands as a 2-component vector field.

        Reads ``u_band`` and ``v_band`` as the vector components over the
        dataset's cell-centre coordinate grid (built from the geotransform)
        and renders them via cleopatra's
        :class:`~cleopatra.glyphs.gridded.vector_glyph.VectorGlyph` as arrows, wind barbs,
        or streamlines, coloured by vector magnitude. Requires the ``[viz]``
        extra.

        The grid is taken from the dataset's 1-D ``x``/``y`` cell-centre
        arrays, so an **axis-aligned (unrotated)** geotransform is assumed —
        the rotation terms are ignored, as elsewhere in pyramids' extent-based
        plotting. Orientation is handled, though: ``v`` is treated as the
        northward (``+y``) component, and because ``streamplot`` requires
        strictly-increasing coordinates, a descending ``x``/``y`` (e.g. a
        north-up raster's ``y``) is flipped to ascending with the data
        rows/cols mirrored to match — a pure relabelling, so each vector keeps
        its true location for every ``kind``, while an already-ascending
        (south-up) axis is left as-is.

        Args:
            u_band (int, optional):
                Band index of the x-component (``u``). Default is ``0``.
            v_band (int, optional):
                Band index of the y-component (``v``). Default is ``1``.
            kind (str, optional):
                Render kind: ``"quiver"`` (default), ``"barbs"``, or
                ``"streamplot"``.
            ax (matplotlib.axes.Axes, optional):
                Draw the vector field into these axes instead of creating them, which is
                what lets it be composed onto a shared map (pair it with
                ``add_colorbar=False``). Any layers already on the axes — e.g. a scalar
                :meth:`plot` drawn first — are **preserved**, and the arrows are drawn on
                top rather than clearing them. Because the host is preserved, calling
                ``plot_vector_field`` again on the same ``ax`` **adds** another field on
                top rather than replacing the previous one; start from a fresh axes to
                redraw. An axes already carries its figure, so ``ax`` on its own is
                sufficient and there is no separate ``fig`` parameter here. A new
                figure/axes is created when left unset. Default is ``None``.
            **kwargs:
                Style options forwarded to the ``VectorGlyph`` constructor,
                filtered via :meth:`VectorGlyph.filter_kwargs` (e.g.
                ``density``, ``scale``, ``cmap``, ``add_colorbar``, ``thin``).
                ``thin=n`` draws every nth grid point so a large ``quiver`` /
                ``barbs`` grid is not one arrow per cell; it applies to
                ``quiver`` / ``barbs`` only — ``streamplot`` ignores it (with a
                warning), use ``density`` there. Arrows are coloured by vector
                magnitude through ``cmap``. For a single **solid** colour pass
                ``color=`` a matplotlib colour (e.g. ``color="black"``): it is
                turned into a one-colour colormap, so the whole field (arrows,
                barbs, or streamlines) renders in that colour, and the
                otherwise-meaningless magnitude colorbar is suppressed by default
                (equivalent to ``cmap=matplotlib.colors.ListedColormap(["black"])``
                with ``add_colorbar=False``). ``color=`` and ``cmap=`` are
                mutually exclusive. (Unlike :meth:`plot`'s ``color=``, which is a
                magnitude ``ColorScaling``, here ``color=`` is a solid matplotlib
                colour.) Pass ``add_colorbar=False`` when composing onto a shared
                map.

        Returns:
            tuple:
                ``(fig, ax, im)`` from :meth:`VectorGlyph.plot` — the
                :class:`matplotlib.figure.Figure`, the
                :class:`matplotlib.axes.Axes`, and the mappable coloured by
                vector magnitude.

        Raises:
            ValueError: If ``u_band`` or ``v_band`` is out of range for the
                dataset, if ``kind`` is not one of ``"quiver"``, ``"barbs"``,
                or ``"streamplot"``, if both ``color=`` and ``cmap=`` are given
                (they are mutually exclusive), or if ``color=`` is not a valid
                matplotlib colour.

        Examples:
            - Render a two-band ``(u, v)`` stack as arrows (tagged ``+SKIP``
              — needs the ``[viz]`` extra):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> rng = np.random.default_rng(0)
                >>> uv = rng.standard_normal((2, 6, 6)).astype("float32")
                >>> ds = Dataset.from_array(
                ...     uv,
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
                ... )
                >>> fig, ax, im = ds.plot_vector_field(u_band=0, v_band=1, kind="quiver")  # doctest: +SKIP

                ```
            - Draw streamlines without the magnitude colorbar (e.g. to add a
              shared one later):

                ```python
                >>> fig, ax, im = ds.plot_vector_field(kind="streamplot", add_colorbar=False)  # doctest: +SKIP

                ```
            - Compose the arrows over a scalar map on a shared axes; the scalar
              layer is preserved:

                ```python
                >>> import matplotlib.pyplot as plt  # doctest: +SKIP
                >>> fig, host = plt.subplots()  # doctest: +SKIP
                >>> ds.plot(band=0, fig=fig, ax=host)  # doctest: +SKIP
                >>> ds.plot_vector_field(u_band=0, v_band=1, ax=host, add_colorbar=False)  # doctest: +SKIP

                ```
            - Draw solid black arrows instead of colouring them by magnitude:

                ```python
                >>> fig, ax, im = ds.plot_vector_field(u_band=0, v_band=1, color="black")  # doctest: +SKIP

                ```
        """
        require_cleopatra()
        from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph

        # Local ([viz]-extra only): matplotlib ships with cleopatra, so it imports
        # once require_cleopatra() above passes; a module-level import would break a
        # bare install without the [viz] extra (matplotlib is TYPE_CHECKING-only here).
        from matplotlib.colors import ListedColormap, is_color_like

        band_count = self._ds.band_count
        for name, idx in (("u_band", u_band), ("v_band", v_band)):
            validate_band_index(
                idx,
                band_count,
                name=name,
                hint=(" plot_vector_field needs two in-range bands (u, v components)."),
            )
        # Solid colour: cleopatra colours the field by magnitude through a
        # colormap and has no scalar ``color=`` (its ``color=`` is a magnitude
        # ``ColorScaling``), so translate a matplotlib colour (``color="black"``)
        # into a one-colour colormap — the whole field (arrows, barbs, or
        # streamlines) then renders in that colour. Validated here (cheap,
        # data-independent) before the band reads below. The conflict guard keys
        # on a real colormap, not presence, so a caller's ``cmap=None`` is fine.
        color = kwargs.pop("color", None)
        if color is not None:
            if kwargs.get("cmap") is not None:
                raise ValueError(
                    "pass either color= (a solid arrow colour) or cmap=, not both"
                )
            if not is_color_like(color):
                raise ValueError(f"color= must be a matplotlib colour, got {color!r}")
            kwargs["cmap"] = ListedColormap([color])
            # A single colour has no magnitude scale, so a magnitude colorbar
            # would be misleading; default it off (an explicit add_colorbar wins).
            kwargs.setdefault("add_colorbar", False)
        u = self._ds.read_array(band=u_band)
        v = self._ds.read_array(band=v_band)
        x = self._ds.x
        y = self._ds.y
        # matplotlib's ``streamplot`` requires strictly-increasing 1-D
        # coordinates, but a north-up raster's ``y`` (and occasionally ``x``)
        # is descending. Flip the axis to ascending and mirror the data
        # rows/cols so the field stays spatially correct for every kind
        # (``quiver``/``barbs`` are direction-agnostic; ``streamplot`` is not).
        if y[0] > y[-1]:
            y = y[::-1]
            u = u[::-1, :]
            v = v[::-1, :]
        if x[0] > x[-1]:
            x = x[::-1]
            u = u[:, ::-1]
            v = v[:, ::-1]
        xx, yy = np.meshgrid(x, y)
        glyph = VectorGlyph(xx, yy, u, v, ax=ax, **VectorGlyph.filter_kwargs(kwargs))
        # A caller-supplied ``ax`` is a host to compose onto (e.g. a scalar map
        # drawn first), which is the documented reason the parameter exists. Tell
        # cleopatra (>=0.39.0) to keep the host's existing artists instead of
        # clearing the axes; when we create our own axes there is nothing to
        # preserve, so composition stays off.
        result = glyph.plot(kind=kind, compose=ax is not None)
        return result

    def plot(
        self,
        band: int,
        exclude_value: Any | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        overview: bool | None = False,
        overview_index: int | None = 0,
        percentile: int | None = None,
        basemap: bool | str | dict[str, Any] | Basemap | None = None,
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> ArrayGlyph:
        """Plot the values/overviews of a given band.

        This is the generic rendering engine. It assumes ``band`` has already been resolved
        by the caller (typically a per-class facade such as :meth:`Dataset.plot` or
        :meth:`NetCDF.plot`). It does **not** apply any band-resolution policy (no RGB
        heuristic, no `ColorInterpretation` lookup, no default-to-zero fallback) \u2014 those
        are dataset-type-specific decisions that belong on the facades.

        When the resolved band carries a GDAL colour table and the caller passes neither
        ``cmap`` nor ``color``, the raster renders through that palette: the colour table is
        turned into a colormap and handed to cleopatra with
        ``color=ColorScaling.boundary(bounds=...)`` so each pixel value shows its own colour
        (#913). An explicit ``cmap`` / ``color`` opts out.

        The plot function uses `cleopatra` as a backend to plot the raster data; for more
        information see the
        [ArrayGlyph reference](https://serapeum-org.github.io/cleopatra/latest/api/array-glyph-class/).

        The rendered values are physical, as `read_array` returns them: a CF-packed band
        (`scale_factor` / `add_offset`) is unpacked. cleopatra masks the no-data cells by
        comparing values, so the band's sentinel is handed to it in those same physical
        units (`-9999` at `scale=0.01` goes over as `-99.99`) -- otherwise a packed
        band's gaps would be drawn as data and stretch the colour scale down to them.

        Implementation note: this method is a thin caller around the
        shared :func:`pyramids.dataset._plot_helpers.render_array`
        helper. It resolves the data (``arr``), extent, exclude value,
        and curvilinear coords from the underlying ``Dataset``, then
        forwards to ``render_array(..., mode="plot", ...)`` for a
        single 2-D slice or ``mode="facet"`` when ``NetCDF.plot``
        injects a pre-built ``_facet_stack`` and ``facet_kwargs``.
        ``DatasetCollection.plot`` reuses the same helper with
        ``mode="animate"``. The shared helper owns the actual
        ``ArrayGlyph`` construction and dispatch — see the module
        docstring of :mod:`pyramids.dataset._plot_helpers` for the
        three-mode contract.

        Args:
            band (int):
                Concrete band index to render. Must be provided \u2014 the engine does not resolve
                bands.
            exclude_value (Any, optional):
                Value to exclude from the plot, in addition to the band's no-data cells. Compared
                against the physical values that are drawn. Default is None.
            rgb (List[int], optional):
                The indices of the red, green, and blue bands in the `Dataset`. the `rgb` parameter can be a list of
                three values, or a list of four values if the alpha band is also included. Only meaningful for
                Sentinel-style multi-band rasters; pass-through to cleopatra.
            surface_reflectance (int, optional):
                Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data.
            cutoff (List, optional):
                clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
                and scale them back to between 0 and 1). Default is None.
            overview (bool, optional):
                True if you want to plot the overview. Default is False.
            overview_index (int, optional):
                Index of the overview. Default is 0.
            percentile: int
                The percentile value to be used for scaling.
            basemap (bool, str, or Basemap, optional):
                Reference layer under the plot, dispatched by type. ``True`` or a tile-provider
                string (e.g. "CartoDB.Positron") draws a pyramids web-tile basemap. A
                ``pyramids.plot.Basemap(relief=..., features=...)`` draws a
                shaded-relief / coastline reference layer instead. Default is None (no basemap).
                Requires the [viz] extra (mercantile, xyzservices, Pillow). A ``Basemap`` is not
                supported on the faceted path.
            fig (matplotlib.figure.Figure, optional):
                Draw into this figure instead of creating one. Pass it alongside ``ax``;
                supplying ``fig`` on its own currently raises inside cleopatra
                (serapeum-org/cleopatra#326). Default is ``None``.
            ax (matplotlib.axes.Axes, optional):
                Draw into these axes instead of creating them. This is what lets several
                rasters share one figure — e.g. a ``plt.subplots`` grid where each panel is
                a different band or dataset — while every panel keeps the georeferenced
                extent and nodata masking this method applies. An axes already carries its
                figure, so ``ax`` on its own is sufficient. The returned glyph exposes both
                objects back as ``cleo.fig`` / ``cleo.ax``. Default is ``None``.
        kwargs:
                Colour-scale, contour, cell-value and data-style options moved onto typed
                render groups (all re-exported from ``pyramids.plot``): pass
                ``color=ColorScaling(...)`` / ``contour=Contour(...)`` /
                ``cells=CellValues(...)`` / ``data_style=DataStyle(...)``, the colour bar as
                ``colorbar=ColorBar(...)``, and point overlays as ``points=PointOverlay(...)``.
                The loose forms they replace — ``color_scale`` / ``gamma`` / ``bounds`` /
                ``midpoint`` / ``line_*`` / ``levels`` / ``display_cell_value`` / ``num_size`` /
                ``background_color_threshold`` / ``style`` / ``hillshade`` / ``point_*`` /
                ``cbar_*`` / ``ticks_spacing`` — are no longer accepted and now raise. The
                remaining still-loose kwargs pass through to cleopatra:

                - `points` (array | PointOverlay): Point overlay. A bare 3-column array
                  `(value, row, col)` draws unstyled points; pass a
                  `pyramids.plot.PointOverlay(points, color=..., size=..., ...)` to style them.
                - `cmap` (str, optional): Color map style. Default is `'coolwarm_r'`.
                - `figsize` (tuple, optional): Figure size. Default is `(8, 8)`.
                - `title` (str, optional): Title of the plot. Default is `'Total Discharge'`.
                - `title_size` (int, optional): Title size. Default is `15`.
                - `add_colorbar` (bool, optional): Whether to draw the colour bar. Default is
                  `True`; when `False` the returned glyph's `cbar` is `None`.
                - `colorbar` (bool | ColorBar, optional): Colour-bar spec
                  `pyramids.plot.ColorBar(label=..., orientation=..., ...)` — replaces the
                  removed loose `cbar_*` / `ticks_spacing` kwargs. `False` hides it, `None`
                  uses the default.
                - `full_bleed` (bool | str, optional): Chrome-free layout: drop axes/margins
                  so the array fills the figure. Default `False`.
        Returns:
            ArrayGlyph:
                A cleopatra ``ArrayGlyph`` wrapping the rendered figure. The underlying matplotlib
                primitives are exposed on the glyph \u2014 use them as the escape hatch when you need
                to further customise the plot with raw matplotlib calls:

                - ``cleo.fig`` / ``cleo.ax`` \u2014 the :class:`matplotlib.figure.Figure` and
                  :class:`matplotlib.axes.Axes`.
                - ``cleo.im`` \u2014 the colour-mapped mappable, populated for every ``kind=``
                  (imshow/pcolormesh/contour/contourf); e.g. ``cleo.im.set_clim(0, 100)``.
                - ``cleo.cbar`` \u2014 the auto-created :class:`matplotlib.colorbar.Colorbar`, or
                  ``None`` when ``add_colorbar=False`` (or for RGB renders).
                - ``cleo.apply_style(style)`` (cleopatra >= 0.25) — re-apply
                  a ``DATA_STYLES`` preset by name in place, without re-plotting.

                For the full ``ArrayGlyph`` API see the
                [ArrayGlyph reference](https://serapeum-org.github.io/cleopatra/latest/api/array-glyph-class/).
        Examples:
            - Plot a certain band:
              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> dataset.plot(band=0)  # doctest: +SKIP
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
            - plot using a power scale.
              ```python
              >>> from pyramids.plot import ColorScaling  # doctest: +SKIP
              >>> dataset.plot(band=0, color=ColorScaling.power(gamma=0.7))  # doctest: +SKIP
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
            - plot using a SymLogNorm scale.
              ```python
              >>> dataset.plot(band=0, color=ColorScaling.sym_log())  # doctest: +SKIP
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
            - plot using a BoundaryNorm scale.
              ```python
              >>> dataset.plot(band=0, color=ColorScaling.boundary(bounds=[0, 0.2, 0.4, 0.6, 0.8, 1]))  # doctest: +SKIP
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
            - plot using a midpoint scale.
              ```python
              >>> dataset.plot(band=0, color=ColorScaling.midpoint(at=0))  # doctest: +SKIP
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
        """
        # Each band's sentinel in the units the plotted array is in. `plot` reads
        # through `read_array`, so on a packed band the gap holds `-98.49`, and the
        # stored `-9999` handed to cleopatra masked nothing: the gap was drawn as
        # data and squeezed the whole valid range into the top of the colormap.
        no_data_value = [
            np.nan if value is None else value
            for value in (
                self._physical_no_data(index) for index in range(self._ds.band_count)
            )
        ]
        # `coords` is the PR-3 curvilinear kwarg; the helper handles the
        # mutually-exclusive `extent` swap. `facet_kwargs` (PR-4) is
        # forwarded by `NetCDF.plot` to switch the helper to the
        # `mode="facet"` branch; the pre-built stack arrives alongside as
        # `_facet_stack` and its spatial extent as `_extent` (the facet
        # stack is *injected*, not read from `self._ds`, so the engine
        # can't derive the extent from `self._ds.bbox` — the caller must
        # supply it). `_chunks` (PR-5) is injected by `NetCDF.plot` to
        # switch the static-plot read path to the dask-backed lazy read;
        # only the rendered slice is materialised.
        coords = kwargs.pop("coords", None)
        facet_kwargs = kwargs.pop("facet_kwargs", None)
        facet_stack = kwargs.pop("_facet_stack", None)
        injected_extent = kwargs.pop("_extent", None)
        chunks = kwargs.pop("_chunks", None)
        mode = "facet" if facet_kwargs else "plot"
        arr = self._resolve_plot_array(
            band, rgb, overview, overview_index, mode, facet_stack, chunks
        )
        exclude_value = (
            [no_data_value[band], exclude_value]
            if exclude_value is not None
            else [no_data_value[band]]
        )
        # On the self-read paths (`mode="plot"` / `_chunks`) the data and
        # the extent both come from `self._ds`. On the injected-stack path
        # (`mode="facet"`) the caller passes `_extent` so the panels are
        # placed at the stack's own spatial domain rather than implicitly
        # trusting that it matches `self._ds.bbox`.
        effective_extent = (
            injected_extent if injected_extent is not None else self._ds.bbox
        )
        # Render a paletted band through its GDAL colour table (#913): build a discrete
        # colormap from the palette and hand it to cleopatra as an explicit ``cmap`` plus a
        # boundary-norm ``color=ColorScaling.boundary(bounds=...)`` (cleopatra 0.30 moved the
        # colour scale onto the typed group). Only the single-band static path (no ``rgb``,
        # no facet) carries a palette, and an explicit ``cmap`` / ``color`` from the caller
        # wins over it.
        if (
            mode == "plot"
            and rgb is None
            and kwargs.get("cmap") is None
            and kwargs.get("color") is None
        ):
            # Read only the resolved band's colour table, not every band's — the
            # full-dataset ``color_table`` rebuilds a row per entry for all bands.
            band_color_table = self._ds.bands._get_color_table(band=band)
            if not band_color_table.empty:
                from cleopatra.styling.scaling import ColorScaling

                cmap, bounds = self._palette_colormap(band_color_table)
                kwargs["cmap"] = cmap
                kwargs["color"] = ColorScaling.boundary(bounds=bounds)
        return render_array(
            RenderRequest(
                arr=arr,
                extent=effective_extent,
                coords=coords,
                exclude_value=exclude_value,
                rgb=RgbSpec(
                    rgb=rgb,
                    surface_reflectance=surface_reflectance,
                    cutoff=cutoff,
                    percentile=percentile,
                ),
                mode=ModeSpec(mode=mode, facet_kwargs=facet_kwargs),
                ax=ax,
                fig=fig,
                basemap=basemap,
                basemap_epsg=self._ds.epsg,
            ),
            **kwargs,
        )

    def _resolve_plot_array(
        self,
        band: int,
        rgb: list[int] | None,
        overview: bool | None,
        overview_index: int | None,
        mode: str,
        facet_stack: Any,
        chunks: Any,
    ) -> Any:
        """Resolve the array to render for :meth:`plot`.

        - ``mode="facet"``: use the caller-injected ``_facet_stack``.
        - ``_chunks`` injected: lazy-read and materialise only the requested band
          (see :meth:`_read_plot_lazy_array`).
        - otherwise: eager-read the band — or the full ``(bands, rows, cols)``
          array when ``rgb`` is set so cleopatra can pick the colour channels —
          from an overview when ``overview`` is truthy.
        """
        if mode == "facet":
            arr = facet_stack
        elif chunks is not None:
            arr = self._read_plot_lazy_array(band, chunks)
        else:
            read_band = None if rgb is not None else band
            if overview:
                arr = self._ds.read_overview_array(
                    band=read_band,
                    overview_index=(
                        overview_index if overview_index is not None else 0
                    ),
                )
            else:
                arr = self._ds.read_array(band=read_band)
        return arr

    def _read_plot_lazy_array(self, band: int, chunks: Any) -> np.ndarray:
        """Lazy-read path for :meth:`plot` (``_chunks`` injected by ``NetCDF.plot``).

        Builds a dask array of the variable and materialises only the requested
        slice. ``read_array(chunks=...)`` keeps the variable's native
        ``(d0, ..., rows, cols)`` shape, so a >2-D result is reshaped to
        ``(-1, rows, cols)`` to match the eager ``read_array`` band flattening
        before ``band`` indexes it; only that band's chunks are computed.
        """
        lazy = self._ds.read_array(chunks=chunks)
        if not hasattr(lazy, "compute"):
            result = lazy if band is None else lazy[band]
        elif lazy.ndim > 2:
            lazy = lazy.reshape(-1, *lazy.shape[-2:])
            result = np.asarray(lazy[band].compute())
        else:
            result = np.asarray(lazy.compute())
        return cast(np.ndarray, result)

    @staticmethod
    def _process_color_table(color_table: DataFrame) -> DataFrame:
        require_cleopatra()
        from cleopatra.styling.colors import Colors

        # if the color_table does not contain the red, green, and blue columns, assume it has one column with
        # the color as hex and then, convert the color to rgb.
        if all(elem in color_table.columns for elem in ["red", "green", "blue"]):
            color_df = color_table.loc[:, ["values", "red", "green", "blue"]]
        elif "color" in color_table.columns:
            color = Colors(color_table["color"].tolist())
            color_rgb = color.to_rgb(normalized=False)
            color_df = DataFrame(columns=["values"])
            color_df["values"] = color_table["values"].to_list()
            color_df.loc[:, ["red", "green", "blue"]] = color_rgb
        else:
            raise ValueError(
                f"color_table must contain either red, green, blue, or color columns. given columns are: "
                f"{color_table.columns}"
            )
        if "alpha" not in color_table.columns:
            color_df.loc[:, "alpha"] = 255
        else:
            color_df.loc[:, "alpha"] = color_table["alpha"]
        return color_df

    @staticmethod
    def _palette_colormap(color_table: DataFrame) -> tuple[Any, list[float]]:
        """Build a colormap + boundary edges from a GDAL colour table.

        Normalises the ``[values, red, green, blue, alpha]`` colour table (via
        :meth:`_process_color_table`) into a matplotlib ``ListedColormap`` carrying the
        palette's exact colours, and derives the ``BoundaryNorm`` bin edges from
        cleopatra's ``category_boundaries``. The colormap is handed to cleopatra as an
        explicit ``cmap`` with ``color=ColorScaling.boundary(bounds=...)`` so a paletted
        raster renders through its own colours — pyramids only builds the mapping; cleopatra
        draws it.

        cleopatra renders with ``BoundaryNorm(bounds, ncolors=256)``, so the palette is
        turned into a **256-entry step lookup** whose slots are filled by asking that
        exact norm which slot each class maps to and placing the class's colour there.
        Each class then indexes its own exact, opaque colour regardless of the
        palette's value range. (A fixed round-trip formula mis-indexes once the
        densified entry count exceeds ~131, because the norm stretches regions to slots
        by truncation; a ``LinearSegmentedColormap`` would instead interpolate between
        stops, bleeding alpha toward GDAL's transparent ``(0, 0, 0, 0)`` gap-filler
        entries — GDAL densifies a colour table to ``0..maxvalue``.) An exact
        one-swatch-per-class rendering with a discrete legend would need a first-class
        categorical colour-table API on cleopatra's ``ArrayGlyph``; see the follow-up
        tracked for #913.

        Args:
            color_table (DataFrame):
                The band's colour table — ``values`` plus ``red``/``green``/``blue``
                (and optional ``alpha``), or a hex ``color`` column. Must be non-empty
                (the plot path only calls this once a colour table is present).

        Returns:
            tuple[matplotlib.colors.ListedColormap, list[float]]:
                The 256-entry step colormap and the ``len(values) + 1`` ascending
                boundary edges, sorted by colour-table value.
        """
        require_cleopatra()
        from cleopatra.styling.colors import category_boundaries
        from matplotlib.colors import BoundaryNorm, ListedColormap

        processed = Analysis._process_color_table(color_table).sort_values("values")
        rgba = (
            processed[["red", "green", "blue", "alpha"]].to_numpy(dtype=float) / 255.0
        )
        values = [float(v) for v in processed["values"].to_list()]
        bounds = category_boundaries(values)
        norm = BoundaryNorm(bounds, 256)
        slots = np.asarray(norm(np.asarray(values))).astype(int)
        lut = np.tile(rgba[0], (256, 1))
        lut[slots] = rgba
        cmap = ListedColormap(lut)
        return cmap, bounds
