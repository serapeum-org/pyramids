"""Unpacking CF-packed rasters on read, and not corrupting them on the way out (#1124).

A packed raster stores small integers plus a `scale_factor` / `add_offset` recipe. Reading
the integers and calling them the answer is wrong by the packing factor, quietly: the code
runs, the plot renders, and a wave height of 14.35 m reads as 1435.

These pin three things the change has to get right at once:

- packed data comes back physical, from `read_array`, `stats` and the plot path alike;
- unpacked data — nearly every raster — is untouched, same dtype and no copy. GDAL answers
  `None` for a band that was never packed, but pyramids' own `scale` / `offset` report that
  as `1.0` / `0.0` and a file may store the identity outright, so both spellings count;
- `apply` neither truncates a float result into an integer band nor silently spends the
  packing while leaving the raw values in place.
"""

from __future__ import annotations

import logging
import math
from unittest.mock import patch

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._utils import (
    _is_identity_packing,
    apply_unpack,
    carry_band_packing,
    carry_packing,
)
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.collection import DatasetCollection, _agree_on_one_sentinel
from pyramids.dataset.engines.analysis import Analysis
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

PACKED_NC = "tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc"
# `z` stores +/-100 with scale 0.01 and offset 1.5, so the physical range is 0.5 to 2.5.
PACKED_RAW_MAX = 100.0
PACKED_PHYSICAL_MAX = 2.5


def _packed_raster(values: list[list[int]], scale: float, offset: float) -> Dataset:
    """Build a small in-memory `int16` raster that declares a packing recipe.

    Args:
        values: The rows of stored counts.
        scale: The `scale_factor` to declare.
        offset: The `add_offset` to declare.

    Returns:
        Dataset: A one-band packed dataset with a trivial georeference.
    """
    dataset = _int_raster(values)
    dataset.scale = [scale]
    dataset.offset = [offset]
    return dataset


def _int_raster(values: list[list[int]]) -> Dataset:
    """Build a small in-memory `int16` raster.

    Args:
        values: The rows of stored counts.

    Returns:
        Dataset: A one-band `int16` dataset with a trivial georeference.
    """
    array = np.array(values, dtype="int16")
    return Dataset.from_array(
        array,
        geo_ref=GeoReference(
            top_left_corner=(0, array.shape[0]), cell_size=1.0, epsg=4326
        ),
        no_data_value=-9999,
    )


def _packed_stack(scales: list[float], offsets: list[float], size: int = 8) -> Dataset:
    """Build a square multi-band `int16` raster with one packing recipe per band.

    Args:
        scales: One `scale_factor` per band.
        offsets: One `add_offset` per band.
        size: Rows and columns; 8 leaves room for GDAL to build overviews.

    Returns:
        Dataset: A `(len(scales), size, size)` dataset holding 100 in every cell.
    """
    array = np.full((len(scales), size, size), 100, dtype="int16")
    dataset = Dataset.from_array(
        array,
        geo_ref=GeoReference(top_left_corner=(0, size), cell_size=1.0, epsg=4326),
    )
    dataset.scale = list(scales)
    dataset.offset = list(offsets)
    return dataset


def _bare_raster() -> Dataset:
    """A raster wrapped straight off a MEM handle, so it declares no no-data value.

    `Dataset.from_array` always stamps a sentinel, which leaves the "no sentinel at
    all" branch of `_physical_no_data` unreachable through it.

    Returns:
        Dataset: A one-band `int16` dataset whose `no_data_value` is `None`.
    """
    mem = gdal.GetDriverByName("MEM").Create("", 2, 1, 1, gdal.GDT_Int16)
    mem.SetGeoTransform((0.0, 1.0, 0.0, 1.0, 0.0, -1.0))
    return Dataset(mem, access="write")


def _refuse_the_sentinel(values):
    """Halve the values, but raise on the no-data sentinel, as a real `func` may.

    Args:
        values: The probe or domain values.

    Returns:
        The halved values.

    Raises:
        ValueError: When the sentinel is among the values.
    """
    if np.any(np.asarray(values) == -9999):
        raise ValueError("the sentinel is not a measurement")
    return values * 0.5


class _RefusingBand:
    """A band on a driver that cannot store CF packing, so both setters raise."""

    def __init__(self, scale=None, offset=None):
        self._scale = scale
        self._offset = offset

    def GetScale(self):
        """The band's `scale_factor`."""
        return self._scale

    def GetOffset(self):
        """The band's `add_offset`."""
        return self._offset

    def SetScale(self, value):
        """Refuse the write, the way a driver without the slot does."""
        raise RuntimeError("this driver stores no scale")

    def SetOffset(self, value):
        """Refuse the write, the way a driver without the slot does."""
        raise RuntimeError("this driver stores no offset")


class _RefusingRaster:
    """A raster whose every band refuses to store packing."""

    def __init__(self, band_count: int):
        self.RasterCount = band_count
        self._bands = [_RefusingBand() for _ in range(band_count)]

    def GetRasterBand(self, index: int) -> _RefusingBand:
        """The one-based band, as GDAL numbers them."""
        return self._bands[index - 1]


class TestIdentityIsFree:
    """An unpacked raster must cost nothing, since unpacking is now the default."""

    @pytest.mark.parametrize(
        "scale, offset",
        [
            (None, None),
            (1.0, 0.0),
            (1, 0),
            (np.array([1.0, 1.0]), np.array([0.0, 0.0])),
        ],
        ids=["unset", "float-identity", "int-identity", "array-identity"],
    )
    def test_an_identity_transform_returns_the_array_untouched(self, scale, offset):
        """No promotion, no copy, same values.

        Test scenario:
            The identity reaches this in more than one spelling. GDAL answers `None`
            for a band that was never packed, but `Dataset.scale` / `.offset` normalise
            that to `1.0` / `0.0`, and a file may store the identity outright. If only
            `None` counted as "nothing to do", every raster whose pair came through the
            property -- or was written that way -- would be promoted to `float64` and
            double its memory for no gain.
        """
        source = np.array([0, 1, 2], dtype="int16")
        result = apply_unpack(source, scale, offset)
        assert result.dtype == np.dtype("int16"), result.dtype
        np.testing.assert_array_equal(result, source)

    def test_a_real_packing_is_applied_as_float64(self):
        """A genuine scale/offset still transforms, and widens to `float64`."""
        result = apply_unpack(np.array([0, 1, 2], dtype="int16"), 0.1, 5.0)
        assert result.dtype == np.dtype("float64"), result.dtype
        np.testing.assert_allclose(result, [5.0, 5.1, 5.2])

    @pytest.mark.parametrize(
        "scale, offset, expected",
        [(1.0, 0.0, True), (0.01, 0.0, False), (1.0, 1.5, False), (None, None, True)],
    )
    def test_the_identity_predicate_agrees(self, scale, offset, expected):
        """`_is_identity_packing` is what decides whether the read is free."""
        assert _is_identity_packing(scale, offset) is expected

    @pytest.mark.parametrize(
        "scale", [0.0, np.nan, np.inf, -np.inf], ids=["zero", "nan", "inf", "-inf"]
    )
    def test_an_unusable_scale_is_refused_rather_than_applied(self, scale):
        """A malformed `scale_factor` leaves the counts visible instead of blanking them.

        Test scenario:
            None of these is a legal CF `scale_factor`: zero maps the whole band onto
            the offset, a non-finite one maps it onto `NaN`. While reads were raw by
            default a malformed file was harmless. Now that unpacking happens without
            being asked, honouring one would silently destroy the band -- so the pair
            is refused whole, offset included, and the stored values stay readable.
        """
        source = np.array([0, 1, 2], dtype="int16")
        assert _is_identity_packing(scale, 5.0) is True, (
            f"scale={scale} must be refused, not applied"
        )
        result = apply_unpack(source, scale, 5.0)
        assert result.dtype == np.dtype("int16"), result.dtype
        np.testing.assert_array_equal(result, source)

    def test_an_offset_alone_is_still_a_packing(self):
        """A shift with no factor transforms; only both halves idle is a no-op."""
        assert _is_identity_packing(None, 5.0) is False
        np.testing.assert_allclose(
            apply_unpack(np.array([0, 1], dtype="int16"), None, 5.0), [5.0, 6.0]
        )

    def test_an_array_pair_counts_only_when_every_element_is_the_identity(self):
        """One packed band in a per-band factor makes the whole read a real one."""
        assert _is_identity_packing(np.array([1.0, 0.5]), np.array([0.0, 0.0])) is False
        assert _is_identity_packing(np.array([1.0, 1.0]), np.array([0.0, 2.0])) is False


class TestPackedReadsArePhysical:
    """The reported symptom: every entry point answered in raw counts."""

    def test_read_array_unpacks_by_default(self):
        """A packed variable reads back in physical units without being asked.

        Test scenario:
            This is issue #1124's headline. The store holds 100; the honest answer is 2.5.
        """
        store = NetCDF.read_file(PACKED_NC)
        try:
            got = np.asarray(store.get_variable("z").read_array(), dtype="float64")
        finally:
            store.close()
        assert float(np.nanmax(got)) == pytest.approx(PACKED_PHYSICAL_MAX)

    def test_the_raw_store_is_still_reachable(self):
        """`unpack=False` is the escape hatch, and it still answers in counts."""
        store = NetCDF.read_file(PACKED_NC)
        try:
            raw = np.asarray(
                store.get_variable("z").read_array(unpack=False), dtype="float64"
            )
        finally:
            store.close()
        assert float(np.nanmax(raw)) == pytest.approx(PACKED_RAW_MAX)

    @pytest.mark.parametrize(
        "multidim", [True, False], ids=["multidim-opener", "classic-opener"]
    )
    def test_the_lazy_path_agrees_with_the_eager_one(self, multidim):
        """`chunks=` builds its array separately, so it can drift from the eager read.

        Test scenario:
            Both openers, because they put the packing in different places. Opened
            multidimensionally the variable carries `_scale` / `_offset` of its own;
            opened classically it does not, and only the driver's band declares them.
            While the eager arm consulted the band and the lazy arm consulted only
            `_scale`, the classic-opened variable read 2.5 eagerly and 100.0 through
            `chunks=` -- the same variable, in different units, by keyword.
        """
        pytest.importorskip("dask")
        store = NetCDF.read_file(PACKED_NC, open_as_multi_dimensional=multidim)
        try:
            variable = store.get_variable("z")
            eager = np.asarray(variable.read_array(), dtype="float64")
            lazy = np.asarray(variable.read_array(chunks="auto"), dtype="float64")
        finally:
            store.close()
        np.testing.assert_allclose(np.nanmax(lazy), np.nanmax(eager))
        assert float(np.nanmax(eager)) == pytest.approx(PACKED_PHYSICAL_MAX), (
            f"both paths agree, but on the raw counts: {np.nanmax(eager)}"
        )

    def test_unpacking_is_not_applied_twice(self):
        """The value is the packing applied once, not once per layer.

        Test scenario:
            A variable's in-memory raster carries the same `scale`/`offset` the MDArray
            declares, so the shared raster read already unpacks. Applying it again on the
            way out gave `100 -> 2.5 -> 1.525`, which looks plausible and is wrong.
        """
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            got = float(np.nanmax(np.asarray(variable.read_array(), dtype="float64")))
            raw = float(
                np.nanmax(
                    np.asarray(variable.read_array(unpack=False), dtype="float64")
                )
            )
            scale, offset = variable.scale[0], variable.offset[0]
        finally:
            store.close()
        assert got == pytest.approx(raw * scale + offset)


class TestStatsArePhysical:
    """`stats` never reads a pixel, so it needs the transform, not the flag."""

    def test_stats_report_physical_values(self):
        """The four numbers come back in the same units as `read_array`.

        Test scenario:
            `stats` asks GDAL, which answers in stored units, so it disagreed with
            `read_array` by the packing factor. CF packing is affine, so the statistics
            transform analytically -- no re-read.
        """
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            frame = variable.stats(approx_ok=False)
            scale, offset = variable.scale[0], variable.offset[0]
        finally:
            store.close()
        assert float(frame["max"].iloc[0]) == pytest.approx(PACKED_PHYSICAL_MAX)
        assert float(frame["min"].iloc[0]) == pytest.approx(
            -PACKED_RAW_MAX * scale + offset
        )

    def test_the_spread_scales_but_does_not_shift(self):
        """`std` takes `|scale|` only — an offset moves a distribution, it does not widen it."""
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            physical = float(variable.stats(approx_ok=False)["std"].iloc[0])
            scale = variable.scale[0]
            raw_spread = float(
                np.nanstd(
                    np.asarray(variable.read_array(unpack=False), dtype="float64")
                )
            )
        finally:
            store.close()
        assert physical == pytest.approx(raw_spread * abs(scale), rel=1e-3)

    def test_stats_and_read_array_resolve_the_packing_alike(self):
        """One resolver, or the same band gets reported in two different units.

        Test scenario:
            A `NetCDF` variable can carry its packing in Python (`_scale` / `_offset`,
            which `_preserve_netcdf_metadata` copies onto every spatial result) while its band
            declares something else. While `read_array` preferred the variable's pair
            and `stats` read the band, the two answered 48.0 and 19.0 for the same
            maximum. Both now ask the dataset.
        """
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            variable._scale, variable._offset = 2.0, 10.0
            read = float(np.nanmax(np.asarray(variable.read_array(), dtype="float64")))
            reported = float(variable.stats(approx_ok=False)["max"].iloc[0])
        finally:
            store.close()
        assert reported == pytest.approx(read), (
            f"stats says {reported}, read_array says {read}"
        )

    def test_an_unpacked_raster_keeps_its_statistics(self):
        """The transform must not disturb a raster that was never packed."""
        dataset = _int_raster([[1, 2, 3], [4, 5, 6]])
        frame = dataset.stats(approx_ok=False)
        assert float(frame["min"].iloc[0]) == pytest.approx(1.0)
        assert float(frame["max"].iloc[0]) == pytest.approx(6.0)

    def test_a_negative_scale_does_not_leave_min_above_max(self):
        """A reversing factor swaps the extremes, so they are re-sorted.

        Test scenario:
            A negative `scale_factor` is legal CF -- it is how a geostationary scan
            angle is stored. Transforming the stored min and max in place would report
            `min=2.5, max=0.5`, a frame no consumer can use.
        """
        dataset = _packed_raster([[100, -100], [0, 50]], -0.01, 1.5)
        frame = dataset.stats(approx_ok=False)
        assert float(frame["min"].iloc[0]) == pytest.approx(0.5)
        assert float(frame["max"].iloc[0]) == pytest.approx(2.5)
        assert float(frame["std"].iloc[0]) > 0, "|scale| keeps the spread positive"

    def test_a_report_that_is_not_the_four_numbers_is_left_alone(self):
        """`_unpack_stats` transforms `[min, max, mean, std]` and nothing else."""
        assert Analysis._unpack_stats([1.0, 2.0], 0.01, 1.5) == [1.0, 2.0]

    @pytest.mark.parametrize("band", [None, 0], ids=["all-bands", "one-band"])
    def test_the_report_keeps_float64_resolution(self, band):
        """Both frames are `float64`, or the packing's own precision is rounded away.

        Test scenario:
            A band packed at 1e-5 around an offset of 273.15 -- an ordinary way to
            store temperature -- has more significant digits than `float32` carries.
            Building the frame at `float32`, as it was, threw away exactly the
            resolution the packing existed to preserve.
        """
        dataset = _packed_raster([[0, 30000]], 1e-5, 273.15)
        frame = dataset.stats(band=band, approx_ok=False)
        assert frame.to_numpy().dtype == np.float64, frame.to_numpy().dtype
        assert float(frame["max"].iloc[0]) == pytest.approx(273.45, abs=1e-9), (
            f"the packed resolution was rounded away: {frame['max'].iloc[0]!r}"
        )


class TestApplyDoesNotCorrupt:
    """Two independent faults in `apply`, both silent."""

    @pytest.mark.parametrize("elementwise", [False, True], ids=["whole", "tiled"])
    def test_a_float_result_is_not_truncated_into_an_integer_band(self, elementwise):
        """The output takes the function's result type, not the source's.

        Test scenario:
            The issue's own numbers. Building the destination at the source band's type
            wrote `1435 * 0.01` back as `14`, losing everything after the point.

            Both arms, because they truncate in different places and fixing one left
            the other. The whole-array arm sizes its output array from the probe; the
            `elementwise` (tiled, out-of-core) arm sizes the destination *band* from it
            but used to allocate each tile buffer at the source tile's type, rounding
            the result before it ever reached the wider band.
        """
        dataset = _int_raster([[1435, 1272, 1000]])
        result = dataset.apply(lambda a: a * 0.01, elementwise=elementwise)
        got = np.asarray(result.read_array(), dtype="float64")
        np.testing.assert_allclose(got[0], [14.35, 12.72, 10.0])

    @pytest.mark.parametrize("elementwise", [False, True], ids=["whole", "tiled"])
    def test_integer_arithmetic_keeps_its_width(self, elementwise):
        """Promotion is driven by the result, so integer maths must not widen.

        Test scenario:
            Under NumPy 2's promotion rules `int16 * 2` is still `int16`. Widening it
            anyway would quadruple the storage of every integer `apply` for nothing.
        """
        dataset = _int_raster([[1, 2, 3]])
        result = dataset.apply(lambda a: a * 2, elementwise=elementwise)
        assert result.dtype == ["int16"], result.dtype

    def test_a_result_gdal_cannot_store_keeps_the_source_type(self):
        """An `object`-valued function falls back rather than refusing to run."""
        dataset = _int_raster([[1, 2, 3]])
        result = dataset.apply(lambda a: np.array([str(v) for v in a], dtype=object))
        assert result.dtype == ["int16"], result.dtype

    def test_the_identity_leaves_a_packed_raster_alone(self):
        """`apply(lambda a: a)` must not change what the data means.

        Test scenario:
            `apply` dropped the band's `scale`/`offset` while writing the raw values back
            unchanged, so the identity function moved a packed variable's physical value
            from 2.5 to 100.0 -- a 40x corruption through a function that does nothing.
        """
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            before = float(
                np.nanmax(np.asarray(variable.read_array(), dtype="float64"))
            )
            result = variable.apply(lambda a: a)
            after = float(np.nanmax(np.asarray(result.read_array(), dtype="float64")))
        finally:
            store.close()
        assert after == pytest.approx(before), (
            f"the identity changed the physical value: {before} -> {after}"
        )


class TestStreamingRespectsTheDestination:
    """`stream_transform` writes what its destination declares it holds."""

    def test_an_in_place_transform_does_not_destroy_the_store(self):
        """The source's counts survive a transform written back into the source.

        Test scenario:
            `out=ds` is the documented in-place form. Reading physical values and
            writing them into a band that still declares its packing overwrote the
            counts with numbers the next read scaled again -- 2.5 became 1.55 and the
            original 100 was gone. Unrecoverable, silent, and on the raster the
            caller streamed precisely because it was too big to copy.
        """
        dataset = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        dataset.io.stream_transform(lambda tile: tile * 2, out=dataset, tile_size=2)
        stored = np.asarray(dataset.read_array(unpack=False), dtype="float64")
        np.testing.assert_allclose(stored.ravel(), [200.0, -200.0, 0.0, 100.0])

    def test_a_fresh_destination_takes_physical_values(self):
        """With no `out`, the result is in the same units `read_array` answers in."""
        dataset = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        result = dataset.io.stream_transform(lambda tile: tile * 2, tile_size=2)
        got = np.asarray(result.read_array(), dtype="float64")
        np.testing.assert_allclose(got.ravel(), [5.0, 1.0, 3.0, 4.0])

    def test_the_block_mappers_agree_with_apply(self):
        """`map_blocks`, `stream_transform` and `apply` are three spellings of one thing.

        Test scenario:
            `map_blocks`'s eager arm read raw counts through `ReadAsArray` and wrote
            them into a destination declaring no packing -- neither the stored form
            nor the physical one, and disagreeing with `stream_transform`, with
            `apply`, and with its own lazy arm, which already read physically.
        """
        source = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        mapped = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        streamed = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        applied = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        del source

        results = [
            mapped.map_blocks(lambda tile: tile * 2, tile_size=2),
            streamed.io.stream_transform(lambda tile: tile * 2, tile_size=2),
            applied.apply(lambda tile: tile * 2),
        ]
        for result in results:
            np.testing.assert_allclose(
                np.asarray(result.read_array(), dtype="float64").ravel(),
                [5.0, 1.0, 3.0, 4.0],
            )

    def test_map_blocks_leaves_an_unpacked_raster_narrow(self):
        """The widening is for packed sources only; an ordinary raster keeps its type."""
        dataset = _int_raster([[1, 2], [3, 4]])
        result = dataset.map_blocks(lambda tile: tile * 2, tile_size=2)
        assert result.dtype == ["int16"], result.dtype
        np.testing.assert_array_equal(
            np.asarray(result.read_array()).ravel(), [2, 4, 6, 8]
        )

    def test_an_unpacked_raster_is_unaffected_by_the_rule(self):
        """The destination check must not disturb the ordinary case."""
        dataset = _int_raster([[1, 2], [3, 4]])
        result = dataset.io.stream_transform(lambda tile: tile * 2, tile_size=2)
        got = np.asarray(result.read_array(), dtype="float64")
        np.testing.assert_allclose(got.ravel(), [2.0, 4.0, 6.0, 8.0])

    def test_an_explicit_unpacked_destination_takes_physical_values(self):
        """The rule reads the destination, not the source, so a plain `out` gets metres.

        Test scenario:
            The packed source could be streamed into a raster that declares no packing
            of its own -- a `float64` result the caller allocated. That destination
            cannot re-scale what it is handed, so it has to receive the physical
            values, exactly as the `out=None` default does.
        """
        source = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        destination = Dataset.from_array(
            np.zeros((2, 2), dtype="float64"),
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
        )
        source.io.stream_transform(lambda tile: tile * 2, out=destination, tile_size=2)
        np.testing.assert_allclose(
            np.asarray(destination.read_array(), dtype="float64").ravel(),
            [5.0, 1.0, 3.0, 4.0],
        )

    def test_an_explicit_dtype_beats_the_packed_default(self):
        """A packed source widens the allocation to `float64` only when nothing else says.

        Test scenario:
            The widening exists so a `float64` tile is not truncated on the way into a
            band built at the source's `int16`. A caller who names a dtype has already
            made that decision, so the keyword must still win.
        """
        source = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        result = source.io.stream_transform(
            lambda tile: tile.astype("int16"), tile_size=2, dtype="int16"
        )
        assert result.dtype == ["int16"], result.dtype

    def test_map_blocks_honours_an_explicit_dtype_too(self):
        """`map_blocks` shares the rule, and the same escape from it."""
        source = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        result = source.map_blocks(lambda tile: tile * 2, tile_size=2, dtype="float32")
        assert result.dtype == ["float32"], result.dtype

    def test_map_blocks_reads_every_band_physically(self):
        """The all-bands arm of the eager loop unpacks band by band, like `read_array`.

        Test scenario:
            The single-band arm and the all-bands arm read separately, so fixing one
            leaves the other in counts. Both bands here store 100 under different
            recipes, which is exactly the case a shared factor would get wrong.
        """
        source = _packed_stack([0.01, 2.0], [1.5, -3.0], size=4)
        result = source.map_blocks(lambda tile: tile * 2, tile_size=4)
        got = np.asarray(result.read_array(), dtype="float64")
        assert got.shape == (2, 4, 4), got.shape
        assert float(got[0, 0, 0]) == pytest.approx(5.0), got[0, 0, 0]
        assert float(got[1, 0, 0]) == pytest.approx(394.0), got[1, 0, 0]

    def test_band_packing_answers_per_band_for_an_all_bands_read(self):
        """`_band_packing(None)` reports a pair per band, unset normalised to identity.

        Test scenario:
            A raster packed on only one of its bands must not be mistaken for an
            unpacked one, so the answer is a list rather than a single pair -- and the
            unset band reads as `1.0` / `0.0`, which the identity test understands,
            rather than `None`, which would not broadcast.
        """
        mixed = _packed_stack([0.01, 1.0], [1.5, 0.0], size=4)
        scales, offsets = mixed.io._band_packing(None)
        assert scales == pytest.approx([0.01, 1.0]), scales
        assert offsets == pytest.approx([1.5, 0.0]), offsets
        assert mixed.io._band_packing(0) == pytest.approx((0.01, 1.5))

    def test_band_packing_normalises_a_band_that_declares_nothing(self):
        """An in-memory band answers `None`, which the all-bands form fills in."""
        plain = _int_raster([[1, 2]])
        assert plain.io._band_packing(0) == (None, None)
        assert plain.io._band_packing(None) == ([1.0], [0.0])


class TestSentinelReconciliationKeepsTheStore:
    """Reconciling a stack's no-data values must not rewrite its counts."""

    def test_reconciling_packed_timesteps_leaves_the_counts_alone(self):
        """The observations survive; only the sentinel cells move.

        Test scenario:
            The helper is defined end to end in stored units -- the sentinels it
            compares, the dtype it fits a replacement into, and the band it writes
            back to. Reading physical values made it match no sentinel at all (so the
            reconciliation silently did nothing) and then rounded those values into
            the `int16` store, destroying the counts of every packed timestep it
            touched. It is reached from `DatasetCollection.crop`.
        """
        first = _packed_raster([[100, -100], [0, -9999]], 0.01, 1.5)
        first.no_data_value = [-9999]
        second = _packed_raster([[100, -100], [0, -32768]], 0.01, 1.5)
        second.no_data_value = [-32768]

        _agree_on_one_sentinel([first, second])

        for dataset in (first, second):
            stored = np.asarray(dataset.read_array(unpack=False), dtype="float64")
            np.testing.assert_allclose(stored.ravel()[:3], [100.0, -100.0, 0.0])
        agreed = {first.no_data_value[0], second.no_data_value[0]}
        assert len(agreed) == 1, f"the timesteps still disagree: {agreed}"


class TestGridOpsKeepThePacking:
    """A raster that only moves onto a new grid keeps the recipe for its counts."""

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda ds: ds.to_crs(3857), id="to_crs"),
            pytest.param(
                lambda ds: ds.to_crs(3857, maintain_alignment=True),
                id="to_crs-maintain-alignment",
            ),
            pytest.param(lambda ds: ds.resample(cell_size=2.0), id="resample"),
        ],
    )
    def test_a_regridded_raster_still_declares_its_packing(self, operation):
        """`scale` and `offset` survive, so the values still mean what they meant.

        Test scenario:
            All three build their destination at the source's own type and fill it
            with `gdal.ReprojectImage`, which moves stored counts -- a byte-copy. Only
            the plain `to_crs` happened to survive, because GDAL's `Warp` carries the
            band scale itself; the other two dropped it and returned raw counts, the
            hundredfold error #1124 was filed about.
        """
        dataset = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)

        result = operation(dataset)

        assert result.scale[0] == pytest.approx(0.01), (
            f"the packing was dropped: scale={result.scale}"
        )
        values = np.asarray(result.read_array(), dtype="float64")
        assert float(np.nanmax(values)) <= 2.5 + 1e-9, (
            f"values look like raw counts, not metres: max {np.nanmax(values)}"
        )

    def test_align_keeps_the_packing_too(self):
        """`align` moves pixels onto a template grid, so it is the same byte-copy."""
        template = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        dataset = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)

        result = dataset.align(template)

        assert result.scale[0] == pytest.approx(0.01), (
            f"align dropped the packing: scale={result.scale}"
        )
        np.testing.assert_allclose(
            np.asarray(result.read_array(), dtype="float64").ravel(),
            [2.5, 0.5, 1.5, 2.0],
        )


class TestTheSentinelStaysOutOfTheDomain:
    """`no_data_value` is a stored value; reads are physical. Masking must still work."""

    @staticmethod
    def _with_a_gap() -> Dataset:
        """A packed raster whose last cell is the declared sentinel."""
        dataset = _packed_raster([[100, -100], [0, -9999]], 0.01, 1.5)
        dataset.no_data_value = [-9999]
        return dataset

    def test_the_sentinel_reads_back_transformed(self):
        """The premise: a stored -9999 is not -9999 once the band is unpacked.

        Test scenario:
            This is why every sentinel comparison in the library had to be revisited.
            `no_data_value` stays the stored value -- CF puts `_FillValue` in the
            packed datatype, and it is what gets written back -- so comparing it
            against a physical read matches nothing at all.
        """
        dataset = self._with_a_gap()
        values = np.asarray(dataset.read_array(), dtype="float64")
        assert dataset.no_data_value[0] == pytest.approx(-9999.0), (
            "no_data_value must stay the stored sentinel"
        )
        assert float(values.ravel()[-1]) == pytest.approx(-98.49), (
            f"the sentinel cell should read back transformed, got {values.ravel()[-1]}"
        )

    @pytest.mark.parametrize("elementwise", [False, True], ids=["whole", "tiled"])
    def test_apply_leaves_the_sentinel_cell_alone(self, elementwise):
        """`apply` must not treat a no-data cell as a measurement.

        Test scenario:
            The mask is built against the stored counts, where the sentinel lives,
            and the values are unpacked afterwards. Comparing the physical array to
            the stored sentinel found no no-data at all, so the gap was doubled along
            with the real cells and silently became data.
        """
        result = self._with_a_gap().apply(lambda a: a * 2, elementwise=elementwise)
        got = np.asarray(result.read_array(), dtype="float64").ravel()
        np.testing.assert_allclose(got[:3], [5.0, 1.0, 3.0])
        assert got[-1] == pytest.approx(-9999.0), (
            f"the sentinel cell was transformed into data: {got[-1]}"
        )

    def test_extract_skips_the_sentinel_cell(self):
        """`extract` returns the measurements, not the gaps."""
        values = np.asarray(self._with_a_gap().extract(), dtype="float64").ravel()
        assert values.size == 3, f"expected the 3 real cells, got {values}"
        np.testing.assert_allclose(np.sort(values), [0.5, 1.5, 2.5])

    def test_combine_excludes_a_gap_in_either_operand(self):
        """A cell missing from one side is missing from the result."""
        left, right = self._with_a_gap(), self._with_a_gap()
        result = left.combine(right, lambda a, b: a + b)
        got = np.asarray(result.read_array(), dtype="float64").ravel()
        np.testing.assert_allclose(got[:3], [5.0, 1.0, 3.0])
        assert got[-1] != pytest.approx(-196.98), (
            "the two sentinels were added together as if they were data"
        )


class TestEveryReadOnOneBandAgrees:
    """Sibling read APIs must not answer the same band in different units."""

    @pytest.mark.parametrize(
        "read",
        [
            pytest.param(lambda ds: np.asarray(ds.read_array())[0, 0], id="read_array"),
            pytest.param(lambda ds: float(ds.point(0.5, 1.5, band=0)), id="point"),
            pytest.param(
                lambda ds: np.asarray(ds.read_part((0, 0, 2, 2), band=0)).ravel()[0],
                id="read_part",
            ),
            pytest.param(
                lambda ds: np.asarray(ds.preview(band=0)).ravel()[0], id="preview"
            ),
        ],
    )
    def test_the_overview_read_family_is_physical_too(self, read):
        """`point`, `read_part` and `preview` answer in `read_array`'s units.

        Test scenario:
            All three go through raw `ReadAsArray` rather than `read_array`, so they
            were left in stored counts when the default flipped: `read_array()[0, 0]`
            gave 2.5 while `point()` on the same cell gave 100. `plot(overview=True)`
            renders through this family, so one raster drew on two different colour
            scales depending on that keyword.
        """
        dataset = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        assert float(read(dataset)) == pytest.approx(2.5)

    def test_the_histogram_buckets_span_the_physical_range(self):
        """`get_histogram`'s edges are values, so they follow `stats`.

        Test scenario:
            `ComputeRasterMinMax` and `GetHistogram` are GDAL metadata calls in stored
            units, exactly like the `GetStatistics` this change already transforms. The
            edges came back as (-100, 0), (0, 100) while `stats` reported 0.5 to 2.5 --
            and `plot_histogram`, which reads through `read_array`, disagreed with both.
        """
        dataset = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        _counts, ranges = dataset.get_histogram(band=0, bins=2)
        assert ranges[0][0] == pytest.approx(0.5), f"first edge {ranges[0][0]}"
        assert ranges[-1][-1] == pytest.approx(2.5), f"last edge {ranges[-1][-1]}"

    def test_an_unpacked_raster_keeps_its_histogram(self):
        """The transform must not disturb a raster that was never packed."""
        dataset = _int_raster([[1, 2], [3, 4]])
        _counts, ranges = dataset.get_histogram(band=0, bins=2)
        assert ranges[0][0] == pytest.approx(1.0)
        assert ranges[-1][-1] == pytest.approx(4.0)

    def test_a_negative_scale_leaves_every_bucket_the_right_way_round(self):
        """A reversing factor must not produce an edge pair with the low end second.

        Test scenario:
            A negative `scale_factor` is legal CF. Converting the stored window to
            physical flips its ends, so both the window and each bucket's pair are
            re-sorted; otherwise the buckets come back as `(2.5, 1.5)` and anything
            plotting them draws negative-width bars.
        """
        dataset = _packed_raster([[100, -100], [0, 50]], -0.01, 1.5)
        counts, ranges = dataset.get_histogram(band=0, bins=2)
        assert sum(counts) > 0, f"the buckets caught nothing: {counts}"
        for low, high in ranges:
            assert low <= high, f"a bucket came back reversed: {(low, high)}"
        edges = sorted({edge for pair in ranges for edge in pair})
        assert edges[0] == pytest.approx(0.5), edges
        assert edges[-1] == pytest.approx(2.5), edges

    @pytest.mark.parametrize("scale", [0.01, -0.01], ids=["positive", "negative"])
    def test_a_caller_window_is_given_in_physical_units(self, scale):
        """`min_value` / `max_value` are values, so they arrive in the read's units.

        Test scenario:
            A caller narrowing the range got those numbers from `stats` or a read, both
            of which are physical now. They are converted to stored counts before GDAL
            buckets over them, and the edges converted back, so the window the caller
            asked for is the window described -- whichever direction the factor runs.
        """
        dataset = _packed_raster([[100, -100], [0, 50]], scale, 1.5)
        _counts, ranges = dataset.get_histogram(
            band=0, bins=2, min_value=1.0, max_value=2.0
        )
        edges = sorted({edge for pair in ranges for edge in pair})
        assert edges[0] == pytest.approx(1.0), edges
        assert edges[-1] == pytest.approx(2.0), edges

    def test_an_overview_read_is_physical(self):
        """`read_overview_array` answers in `read_array`'s units, single band and all.

        Test scenario:
            `plot(overview=True)` renders through this, so leaving it in stored counts
            drew the same raster on two different colour scales depending on the flag.
            Both branches, because the all-bands one allocates at the stored dtype and
            fills band by band, so the transform lands in a different place.
        """
        single = _packed_raster([[100] * 8] * 8, 0.01, 1.5)
        single.create_overviews()
        one = np.asarray(single.read_overview_array(band=0, overview_index=0))
        assert float(one.ravel()[0]) == pytest.approx(2.5), one.ravel()[0]

        stack = _packed_stack([0.01, 2.0], [1.5, -3.0])
        stack.create_overviews()
        every = np.asarray(stack.read_overview_array())
        assert every.shape[0] == 2, every.shape
        assert float(every[0].ravel()[0]) == pytest.approx(2.5), every[0].ravel()[0]
        assert float(every[1].ravel()[0]) == pytest.approx(197.0), every[1].ravel()[0]

    def test_an_unpacked_overview_read_keeps_its_dtype(self):
        """The transform must cost an ordinary raster nothing, here as everywhere."""
        dataset = _int_raster([[3] * 8] * 8)
        dataset.create_overviews()
        got = np.asarray(dataset.read_overview_array(band=0, overview_index=0))
        assert got.dtype == np.dtype("int16"), got.dtype
        assert int(got.ravel()[0]) == 3


class TestCarryingThePackingOntoARebuild:
    """`carry_packing` is what keeps a store-copy from losing the recipe.

    An operation that rebuilds a raster out of the counts it read has to hand the
    recipe on with them. Losing it leaves counts that nothing identifies as counts --
    the same hundredfold error as never unpacking, only now unfixable from the result.
    """

    def test_every_band_gets_its_own_pair(self):
        """The carry is per band, not one recipe stamped over the whole raster."""
        driver = gdal.GetDriverByName("MEM")
        source = driver.Create("", 2, 1, 2, gdal.GDT_Int16)
        source.GetRasterBand(1).SetScale(0.01)
        source.GetRasterBand(1).SetOffset(1.5)
        source.GetRasterBand(2).SetScale(2.0)
        source.GetRasterBand(2).SetOffset(-3.0)
        target = driver.Create("", 2, 1, 2, gdal.GDT_Int16)

        carry_packing(source, target)

        carried = [
            (target.GetRasterBand(i).GetScale(), target.GetRasterBand(i).GetOffset())
            for i in (1, 2)
        ]
        assert carried == [(0.01, 1.5), (2.0, -3.0)], carried

    def test_bands_are_matched_by_position_over_the_shorter_raster(self):
        """A rebuild that dropped a band carries what it can rather than raising."""
        driver = gdal.GetDriverByName("MEM")
        source = driver.Create("", 2, 1, 3, gdal.GDT_Int16)
        for index in (1, 2, 3):
            source.GetRasterBand(index).SetScale(0.01 * index)
        target = driver.Create("", 2, 1, 2, gdal.GDT_Int16)

        carry_packing(source, target)

        assert target.GetRasterBand(1).GetScale() == pytest.approx(0.01)
        assert target.GetRasterBand(2).GetScale() == pytest.approx(0.02)

    @pytest.mark.parametrize(
        "source, target",
        [(None, "raster"), ("raster", None), (None, None)],
        ids=["no-source", "no-target", "neither"],
    )
    def test_a_missing_raster_is_a_no_op(self, source, target):
        """A caller whose rebuild produced nothing must not have to guard the call."""
        driver = gdal.GetDriverByName("MEM")
        resolved = [
            driver.Create("", 2, 1, 1, gdal.GDT_Int16) if side == "raster" else None
            for side in (source, target)
        ]
        carry_packing(*resolved)

    def test_a_source_band_declaring_nothing_leaves_the_target_unset(self):
        """Nothing to carry means nothing written, not an identity stamped on."""
        driver = gdal.GetDriverByName("MEM")
        source = driver.Create("", 2, 1, 1, gdal.GDT_Int16)
        target = driver.Create("", 2, 1, 1, gdal.GDT_Int16)

        carried = carry_band_packing(source.GetRasterBand(1), target.GetRasterBand(1))

        assert carried is True, "an unset source is not a refusal"
        assert target.GetRasterBand(1).GetScale() is None
        assert target.GetRasterBand(1).GetOffset() is None

    def test_a_driver_that_refuses_is_reported_once_for_the_whole_raster(self, caplog):
        """Every band is tried and the report comes once, not once per band.

        Test scenario:
            Stopping at the first refusal left bands 2..N with nothing while band 1
            kept its recipe -- a partial carry, which is worse than none, because the
            result looks internally consistent and is wrong only where nobody looked.
            The values survive either way, so the loss is a debug note, not a raise.
        """
        driver = gdal.GetDriverByName("MEM")
        source = driver.Create("", 2, 1, 3, gdal.GDT_Int16)
        for index in (1, 2, 3):
            source.GetRasterBand(index).SetScale(0.01)
        target = _RefusingRaster(3)

        with caplog.at_level(logging.DEBUG, logger="pyramids.base._utils"):
            carry_packing(source, target)

        notes = [
            record for record in caplog.records if "packing" in record.getMessage()
        ]
        assert len(notes) == 1, f"expected one report, got {len(notes)}"
        assert "3 of 3" in notes[0].getMessage(), notes[0].getMessage()

    def test_carry_band_packing_answers_false_when_the_target_refuses(self):
        """The band-at-a-time form reports the refusal so a loop can count it."""
        driver = gdal.GetDriverByName("MEM")
        source = driver.Create("", 2, 1, 1, gdal.GDT_Int16)
        source.GetRasterBand(1).SetScale(0.01)

        assert carry_band_packing(source.GetRasterBand(1), _RefusingBand()) is False

    def test_carry_band_packing_survives_a_target_with_no_setters_at_all(self):
        """An `AttributeError` is the same loss as a `RuntimeError`, not a crash."""
        driver = gdal.GetDriverByName("MEM")
        source = driver.Create("", 2, 1, 1, gdal.GDT_Int16)
        source.GetRasterBand(1).SetScale(0.01)

        assert carry_band_packing(source.GetRasterBand(1), object()) is False


class TestTheSentinelAsAValue:
    """`_physical_no_data` -- the sentinel as it appears in a physical read.

    Most callers want a *mask*, and build it in stored units. A few instead need the
    sentinel as a number, because they hand it to something that does its own
    comparison against the physical array: cleopatra's `exclude_value`, `get_pixels2`'s
    exclude list, an ASCII grid's header.
    """

    def test_a_packed_sentinel_is_transformed(self):
        """The number that actually appears in the array, not the one on the band."""
        dataset = _packed_raster([[100, -9999]], 0.01, 1.5)
        dataset.no_data_value = [-9999]
        assert dataset.analysis._physical_no_data(0) == pytest.approx(-98.49)
        assert dataset.no_data_value[0] == pytest.approx(-9999.0), (
            "the declared sentinel must stay the stored one"
        )

    def test_an_unpacked_sentinel_is_returned_unchanged(self):
        """No packing, no transform -- and no float noise introduced either."""
        dataset = _int_raster([[1, -9999]])
        assert dataset.analysis._physical_no_data(0) == -9999.0

    def test_an_undeclared_sentinel_stays_none(self):
        """A band with no sentinel has nothing to convert, packed or not."""
        dataset = _bare_raster()
        dataset.scale = [0.01]
        dataset.offset = [1.5]
        assert dataset.no_data_value[0] is None, dataset.no_data_value
        assert dataset.analysis._physical_no_data(0) is None

    def test_an_ascii_grid_declares_the_fill_it_actually_holds(self, tmp_path):
        """An ASCII grid has nowhere to put the recipe, so its header names the value.

        Test scenario:
            `to_ascii` writes the physical numbers. Writing the stored `-9999` into the
            header over a grid holding `-98.49` declared a fill that occurs nowhere, so
            every reader saw the gap as a measurement.
        """
        dataset = _packed_raster([[100, -9999], [0, 50]], 0.01, 1.5)
        dataset.no_data_value = [-9999]
        path = tmp_path / "packed.asc"
        dataset.to_file(str(path))

        header = dict(
            line.split() for line in path.read_text().splitlines()[:6] if line.split()
        )
        assert float(header["NODATA_value"]) == pytest.approx(-98.49), header


class TestApplyPicksTheResultType:
    """The dtype probe behind `apply`, and its fallbacks."""

    def test_an_all_no_data_tile_still_finds_a_value_to_probe(self):
        """With no domain cell available the whole tile is used rather than nothing.

        Test scenario:
            An all-no-data tile is common when streaming a sparse raster. Probing an
            empty selection would raise inside the helper and quietly restore the
            source dtype -- the truncation the probe exists to prevent.
        """
        source = np.array([[-9999, -9999]], dtype="int16")
        domain = np.zeros_like(source, dtype=bool)
        resolved = Analysis._storable_dtype(lambda a: a * 0.5, source, domain)
        assert resolved == np.dtype("float64"), resolved

    def test_the_probe_is_taken_from_the_domain_not_from_cell_zero(self):
        """A `func` that refuses the sentinel must still get a usable probe value.

        Test scenario:
            Cell `[0, 0]` is often the sentinel. With the domain in hand the probe
            skips it and the float result widens the band; without one the call raises
            and the source dtype survives -- the contrast is the point of passing it.
        """
        source = np.array([[-9999, 4, 6]], dtype="int16")
        domain = np.array([[False, True, True]])
        with_domain = Analysis._storable_dtype(_refuse_the_sentinel, source, domain)
        without = Analysis._storable_dtype(_refuse_the_sentinel, source)
        assert with_domain == np.dtype("float64"), with_domain
        assert without == np.dtype("int16"), without

    def test_a_probe_that_cannot_run_keeps_the_source_dtype(self):
        """The tiled arm falls back rather than refusing to run.

        Test scenario:
            The probe reads the first tile, which can fail for reasons that have
            nothing to do with `func` -- a source that will not window-read, say. The
            fallback is the behaviour before #1124, so an awkward case is no worse off.
        """
        dataset = _int_raster([[1, 2, 3]])
        with patch.object(Analysis, "_domain_read", side_effect=RuntimeError("boom")):
            resolved = dataset.analysis._elementwise_result_dtype(lambda a: a * 0.5, 0)
        assert resolved == np.dtype("int16"), resolved

    def test_the_domain_is_derived_when_the_caller_supplies_none(self):
        """`_apply_func_to_domain` still knows how to find the domain itself.

        Test scenario:
            Both callers inside `apply` now resolve the mask first, because their array
            is physical while the sentinel is stored. A caller working in stored units
            throughout has no such problem and can leave the mask to the helper.
        """
        source = np.array([[1, -9999, 3]], dtype="int16")
        out = np.full(source.shape, -9999, dtype="int16")

        Analysis._apply_func_to_domain(lambda a: a * 2, source, out, -9999)

        np.testing.assert_array_equal(out, [[2, -9999, 6]])

    def test_the_read_returns_physical_values_beside_a_stored_mask(self):
        """`_domain_read` is where the two halves of the CF contract meet."""
        dataset = _packed_raster([[100, -9999], [0, 50]], 0.01, 1.5)
        dataset.no_data_value = [-9999]

        values, domain = dataset.analysis._domain_read(0)

        np.testing.assert_array_equal(domain, [[True, False], [True, True]])
        assert float(np.asarray(values).ravel()[0]) == pytest.approx(2.5)


class TestStoreCopiesKeepTheRecipe:
    """Every crop path rebuilds the store, so every one has to carry the packing."""

    def test_a_bbox_crop_keeps_the_packing(self):
        """The windowed fast path is a read of the store, so it moves counts.

        Test scenario:
            A north-up bbox crop in the source CRS skips the warp and reads the AOI
            window directly, then hands the result through the all-no-data trim, which
            rebuilds again. Both rebuilds have to declare the recipe or a small crop of
            a packed raster comes back a hundredfold off.
        """
        dataset = _packed_raster([[100, -100, 0], [50, 25, 75]], 0.01, 1.5)

        crop = dataset.crop(bbox=[0.0, 0.0, 2.0, 2.0])

        assert crop.scale[0] == pytest.approx(0.01), f"scale={crop.scale}"
        peak = float(np.nanmax(np.asarray(crop.read_array(), dtype="float64")))
        assert peak == pytest.approx(2.5), f"values look like raw counts: {peak}"

    def test_a_raster_mask_crop_keeps_the_counts_and_the_recipe(self):
        """The tiled mask-apply writes stored counts into a band that declares them."""
        dataset = _packed_raster([[100, -100, 0], [50, 25, 75]], 0.01, 1.5)
        mask = Dataset.from_array(
            np.array([[1, 1, -9999], [1, 1, 1]], dtype="int16"),
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            no_data_value=-9999,
        )

        cropped = dataset.crop(mask)

        assert cropped.scale[0] == pytest.approx(0.01), f"scale={cropped.scale}"
        stored = np.asarray(cropped.read_array(unpack=False), dtype="float64").ravel()
        np.testing.assert_allclose(stored, [100.0, -100.0, -9999.0, 50.0, 25.0, 75.0])

    def test_an_array_mask_crop_takes_the_same_route(self):
        """The eager arm -- a numpy mask is already in memory, so it is not tiled."""
        dataset = _packed_raster([[100, -100, 0], [50, 25, 75]], 0.01, 1.5)

        cropped = dataset.spatial._crop_aligned(
            np.array([[1, 1, -9999], [1, 1, 1]], dtype="int16"), mask_noval=-9999
        )

        assert cropped.scale[0] == pytest.approx(0.01), f"scale={cropped.scale}"
        stored = np.asarray(cropped.read_array(unpack=False), dtype="float64").ravel()
        np.testing.assert_allclose(stored, [100.0, -100.0, -9999.0, 50.0, 25.0, 75.0])

    def test_a_seam_crossing_crop_keeps_the_packing(self):
        """The antimeridian stitch concatenates two halves of one store.

        Test scenario:
            A `west > east` geographic bbox is the STAC convention for a crossing. Each
            half is cropped and the two are joined along the seam, which reads them
            with `unpack=False` -- so the rebuilt strip has to declare what turns those
            counts back into measurements.
        """
        world = np.full((18, 36), 100, dtype="int16")
        dataset = Dataset.from_array(
            world,
            geo_ref=GeoReference(top_left_corner=(-180, 90), cell_size=10.0, epsg=4326),
        )
        dataset.scale = [0.01]
        dataset.offset = [1.5]

        stitched = dataset.crop(bbox=(170.0, -10.0, -170.0, 10.0))

        assert stitched.scale[0] == pytest.approx(0.01), f"scale={stitched.scale}"
        peak = float(np.nanmax(np.asarray(stitched.read_array(), dtype="float64")))
        assert peak == pytest.approx(2.5), f"the stitch returned raw counts: {peak}"


class TestTheCollectionAnswersInOneUnit:
    """A packed stack must not change dtype with the number of timesteps read."""

    def test_an_empty_read_of_a_packed_collection_is_float64(self):
        """`head(0)` and `head(1)` have to agree about what a read of this cube is.

        Test scenario:
            The empty branch answered `_meta.dtype`, the *stored* `int16`, while the
            stack below it comes back `float64` now that the timesteps unpack. Anything
            that allocates from `head(0).dtype` and then fills it truncated.
        """
        step = _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        collection = DatasetCollection(step, time_length=2, datasets=[step, step])

        empty = collection.head(0)

        assert empty.dtype == np.dtype("float64"), empty.dtype
        assert collection.head(1).dtype == np.dtype("float64")
        assert empty.shape == (0, 2, 2), empty.shape

    def test_an_empty_read_of_an_unpacked_collection_keeps_the_stored_dtype(self):
        """The widening is for packed stacks only; an ordinary one stays narrow."""
        step = _int_raster([[1, 2], [3, 4]])
        collection = DatasetCollection(step, time_length=2, datasets=[step, step])

        assert collection.head(0).dtype == np.dtype("int16")
        assert collection.head(1).dtype == np.dtype("int16")


class TestTheNetCDFPackingResolver:
    """`NetCDF._effective_packing` -- one answer, from two possible homes."""

    def test_the_variables_own_pair_wins_over_its_band(self):
        """`_preserve_netcdf_metadata` copies `_scale`/`_offset` onto every result, so they are truth.

        Test scenario:
            While the eager arm consulted the band and the lazy arm consulted only the
            variable, the same variable reported two different maxima by keyword. The
            variable's own pair is the one the rest of the module maintains, so it
            wins, and the band is only the fallback.
        """
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            variable._scale, variable._offset = 2.0, 10.0
            resolved = variable._effective_packing()
            band_pair = (
                variable.raster.GetRasterBand(1).GetScale(),
                variable.raster.GetRasterBand(1).GetOffset(),
            )
        finally:
            store.close()
        assert resolved == pytest.approx((2.0, 10.0)), resolved
        assert band_pair == pytest.approx((0.01, 1.5)), (
            f"the band must still disagree, or the test proves no preference: "
            f"{band_pair}"
        )

    def test_a_classic_opened_variable_falls_back_to_its_band(self):
        """Nothing copies the MDArray's attributes onto a classically opened variable.

        Test scenario:
            Opened with `open_as_multi_dimensional=False` the variable carries no pair
            of its own, and only the driver's band declares the packing. Reading only
            `_scale` there answered "unpacked" and returned raw counts.
        """
        store = NetCDF.read_file(PACKED_NC, open_as_multi_dimensional=False)
        try:
            variable = store.get_variable("z")
            own_pair = (
                getattr(variable, "_scale", None),
                getattr(variable, "_offset", None),
            )
            resolved = variable._effective_packing()
        finally:
            store.close()
        assert own_pair == (None, None), own_pair
        assert resolved == pytest.approx((0.01, 1.5)), resolved


class TestBandWiseRebuildsCarryTheirRecipe:
    """Rebuilds whose bands do not line up by position, so they carry one at a time."""

    def test_stacking_packed_files_keeps_each_ones_factor(self, tmp_path):
        """`from_band_files` writes N single-band stores into one N-band raster.

        Test scenario:
            The array branch (taken when the inputs need aligning, or disagree on
            dtype) reads each source with `unpack=False` and writes the counts into the
            promoted stored type. Without the per-band carry the stack would hold three
            sets of counts under no recipe at all, while the `BuildVRT` branch beside it
            got the same carry free from `CreateCopy` -- two spellings, two answers.
        """
        paths = []
        for index, (scale, offset) in enumerate([(0.01, 1.5), (2.0, -3.0)]):
            source = _packed_raster([[100] * 4] * 4, scale, offset)
            path = tmp_path / f"band{index}.tif"
            source.to_file(str(path))
            paths.append(str(path))

        stacked = Dataset.from_band_files(paths, align=True)

        assert stacked.scale == pytest.approx([0.01, 2.0]), stacked.scale
        assert stacked.offset == pytest.approx([1.5, -3.0]), stacked.offset
        corner = np.asarray(stacked.read_array(), dtype="float64")[:, 0, 0]
        np.testing.assert_allclose(corner, [2.5, 197.0])

    def test_writing_a_packed_raster_into_a_container_keeps_its_recipe(self):
        """`set_variable` stores counts, so the MDArray has to declare the packing.

        Test scenario:
            GDAL keeps `scale_factor` / `add_offset` in the MDArray's own slots rather
            than in its attribute dictionary, so the attribute write cannot carry them.
            Without the explicit `SetScale` / `SetOffset` the variable comes back as
            bare counts -- the raster went in reading 2.5 and came out reading 100.
        """
        container = NetCDF.from_array(
            arr=np.zeros((2, 2), dtype="int16"),
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            variable_name="plain",
        )
        container.set_variable(
            "packed", _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
        )

        stored_back = container.get_variable("packed")

        assert stored_back._scale == pytest.approx(0.01), stored_back._scale
        np.testing.assert_allclose(
            np.asarray(stored_back.read_array(), dtype="float64").ravel(),
            [2.5, 0.5, 1.5, 2.0],
        )
        np.testing.assert_allclose(
            np.asarray(stored_back.read_array(unpack=False), dtype="float64").ravel(),
            [100.0, -100.0, 0.0, 50.0],
        )

    def test_a_store_that_refuses_the_recipe_still_takes_the_values(self):
        """A driver without the slots loses the packing, it does not lose the write.

        Test scenario:
            Not every MDArray backend can hold `scale_factor` / `add_offset`. The
            counts are intact either way, so the refusal is swallowed and the loss is
            left to the read that finds no packing -- failing the write would be worse.
        """
        container = NetCDF.from_array(
            arr=np.zeros((2, 2), dtype="int16"),
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            variable_name="plain",
        )
        with patch.object(
            gdal.MDArray, "SetScale", side_effect=RuntimeError("no slot")
        ):
            container.set_variable(
                "packed", _packed_raster([[100, -100], [0, 50]], 0.01, 1.5)
            )

        stored_back = container.get_variable("packed")

        assert stored_back.raster.GetRasterBand(1).GetScale() is None
        np.testing.assert_allclose(
            np.asarray(stored_back.read_array(unpack=False), dtype="float64").ravel(),
            [100.0, -100.0, 0.0, 50.0],
        )

    def test_a_dataset_with_no_handle_declares_no_packing(self):
        """The resolver answers rather than raising when the raster is already gone.

        Test scenario:
            `_effective_packing` is consulted from read paths, statistics and the
            streaming transforms alike, so a closed or half-built dataset must not turn
            an ordinary "nothing declared" question into an `AttributeError`.
        """
        dataset = _int_raster([[1, 2]])
        dataset._raster = None
        assert dataset._effective_packing(0) == (None, None)


class TestEveryBandIsUnpackedWithItsOwnFactor:
    """An all-bands read has one recipe per band, not one for the raster."""

    def test_combine_unpacks_each_operand_band_separately(self):
        """`_operand_arrays`' 3-D arm stacks per-band transforms, it does not broadcast.

        Test scenario:
            The two bands here are packed differently and hold different counts, so a
            single shared factor would get one of them wrong. `combine` reads with
            `unpack=False`, takes the domain against the stored sentinels, then unpacks
            band by band -- the only order in which both halves stay correct.
        """
        stack = np.stack(
            [np.full((2, 2), 100, dtype="int16"), np.full((2, 2), 50, dtype="int16")]
        )
        left = Dataset.from_array(
            stack,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            no_data_value=-9999,
        )
        left.scale, left.offset = [0.01, 2.0], [1.5, -3.0]
        right = Dataset.from_array(
            stack.copy(),
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            no_data_value=-9999,
        )
        right.scale, right.offset = [0.01, 2.0], [1.5, -3.0]

        result = left.combine(right, lambda a, b: a + b)

        got = np.asarray(result.read_array(), dtype="float64")
        assert got.shape == (2, 2, 2), got.shape
        np.testing.assert_allclose(got[:, 0, 0], [5.0, 194.0])

    def test_the_lazy_read_can_also_reach_the_stored_counts(self):
        """`unpack=False` means the same thing through `chunks=` as it does eagerly."""
        pytest.importorskip("dask")
        store = NetCDF.read_file(PACKED_NC)
        try:
            lazy = store.get_variable("z").read_array(chunks="auto", unpack=False)
            raw = np.asarray(lazy.compute(), dtype="float64")
        finally:
            store.close()
        assert float(np.nanmax(raw)) == pytest.approx(PACKED_RAW_MAX)


class TestPythonHeldPackingReachesEveryReader:
    """A `NetCDF` variable whose packing lives only in `_scale` / `_offset`."""

    @pytest.mark.parametrize(
        "read",
        [
            pytest.param(
                lambda v, x, y: np.asarray(v.read_array(), dtype="float64").ravel()[0],
                id="read_array",
            ),
            pytest.param(lambda v, x, y: float(v.point(x, y, band=0)), id="point"),
            pytest.param(
                lambda v, x, y: np.asarray(
                    v.read_part((0.0, 0.0, 4.0, 3.0), band=0), dtype="float64"
                ).ravel()[0],
                id="read_part",
            ),
            pytest.param(
                lambda v, x, y: np.asarray(v.preview(band=0), dtype="float64").ravel()[
                    0
                ],
                id="preview",
            ),
        ],
    )
    def test_every_reader_applies_the_variables_own_pair(self, read):
        """The band declares nothing, so every reader has to consult the variable.

        Test scenario:
            `sel()` builds its result from raw `ReadAsArray` counts over a band that
            declares no packing, carrying the recipe only on `_scale` / `_offset`. The
            shared unpack step read the band directly, so `point`, `read_part` and
            `preview` answered 1200 where `read_array` answered 13.5 on the same cell.
        """
        counts = np.arange(2 * 3 * 4, dtype="int16").reshape(2, 3, 4) * 100
        container = NetCDF.from_array(
            counts,
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
            variable_name="t",
        )
        variable = container.get_variable("t")
        variable._scale, variable._offset = 0.01, 1.5
        selected = variable.sel(
            **{variable._band_dim_name: variable._band_dim_values[1]}
        )
        assert selected.raster.GetRasterBand(1).GetScale() is None, (
            "the fixture must hold its packing only in Python to test this"
        )
        gt = selected.geotransform
        x, y = gt[0] + gt[1] * 0.5, gt[3] + gt[5] * 0.5
        assert read(selected, x, y) == pytest.approx(1200 * 0.01 + 1.5)


class TestAnInPlaceComputeSpendsThePacking:
    """`apply(inplace=True)` / `fill(inplace=True)` on a packed `NetCDF` variable."""

    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            pytest.param(lambda v: v.apply(lambda a: a, inplace=True), 2.5, id="apply"),
            pytest.param(lambda v: v.fill(7.0, inplace=True), 7.0, id="fill"),
        ],
    )
    def test_the_values_are_not_unpacked_a_second_time(self, operation, expected):
        """The in-place result reads back once-unpacked.

        Test scenario:
            `_update_inplace` preserves `_scale` / `_offset` -- right for `set_crs`, which
            does not touch values -- and the resolver prefers them. So after an in-place
            compute had written physical values, the next read applied the recipe
            again: the identity `apply` turned 2.5 into 1.525.
        """
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            operation(variable)
            got = float(np.nanmax(np.asarray(variable.read_array(), dtype="float64")))
            spent = (variable._scale, variable._offset)
        finally:
            store.close()
        assert got == pytest.approx(expected), f"read back {got}, expected {expected}"
        assert spent == (None, None), f"the recipe was left attached: {spent}"


class TestSetVariableRoundTrip:
    """`sel` -> `set_variable` must write back a value that reads the same."""

    def test_a_selection_written_back_keeps_its_meaning(self):
        """The recipe travels with the counts `set_variable` stores.

        Test scenario:
            `sel()` holds its recipe only in `_scale` / `_offset`; its band declares
            none. `set_variable` took the recipe off band 1, so it wrote the counts back
            bare and the round trip its own docstring describes turned 13.5 into 1200.
        """
        counts = np.arange(2 * 3 * 4, dtype="int16").reshape(2, 3, 4) * 100
        container = NetCDF.from_array(
            counts,
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
            variable_name="t",
        )
        variable = container.get_variable("t")
        variable._scale, variable._offset = 0.01, 1.5
        selected = variable.sel(
            **{variable._band_dim_name: variable._band_dim_values[1]}
        )
        expected = float(np.asarray(selected.read_array(), dtype="float64").ravel()[0])

        container.set_variable("copy", selected)

        written = container.get_variable("copy")
        got = float(np.asarray(written.read_array(), dtype="float64").ravel()[0])
        assert got == pytest.approx(expected), f"wrote {expected}, read back {got}"


def _halve_positive(values):
    """Halve an array, refusing any negative value -- as a real sentinel-shy func might."""
    values = np.asarray(values)
    if np.any(values < 0):
        raise ValueError("negative input")
    return values / 2


class TestApplyPredictsTheTypeTheCallProduces:
    """The dtype probe must take the same path the real call takes."""

    @pytest.mark.parametrize("elementwise", [False, True], ids=["whole", "tiled"])
    def test_a_scalar_only_callable_is_not_truncated(self, elementwise):
        """`math.sqrt` refuses an array, so `apply` lifts it -- and the probe must too.

        Test scenario:
            The probe called `func` on an array; `math.sqrt` raised, the probe fell back
            to the source dtype, and the lifted call then wrote `sqrt(2)` into an
            `int16` buffer as `1`.
        """
        dataset = _int_raster([[4, 2], [9, 3]])
        result = dataset.apply(math.sqrt, elementwise=elementwise)
        np.testing.assert_allclose(
            np.asarray(result.read_array(), dtype="float64").ravel(),
            [2.0, math.sqrt(2), 3.0, math.sqrt(3)],
        )

    @pytest.mark.parametrize("elementwise", [False, True], ids=["whole", "tiled"])
    def test_a_func_that_refuses_the_sentinel_still_widens(self, elementwise):
        """The probe samples the domain on both arms, never the sentinel at `[0, 0]`."""
        dataset = _int_raster([[-9999, 3], [5, 7]])
        result = dataset.apply(_halve_positive, elementwise=elementwise)
        got = np.asarray(result.read_array(), dtype="float64").ravel()
        np.testing.assert_allclose(got[1:], [1.5, 2.5, 3.5])


class TestEachBandIsJudgedOnItsOwn:
    """Per-band packing must stay per band, in every path that reads more than one."""

    @staticmethod
    def _two_bands(scales: list[float]) -> Dataset:
        """A two-band `int16` raster, each band with its own factor and one gap."""
        counts = np.array(
            [[[-9999, 100], [200, 300]], [[-9999, 100], [200, 300]]], dtype="int16"
        )
        dataset = Dataset.from_array(
            counts,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            no_data_value=[-9999, -9999],
        )
        dataset.scale = scales
        dataset.offset = [0.0, 0.0]
        return dataset

    def test_extract_uses_the_packing_of_the_band_it_reads(self):
        """`extract(band=1)` finds band 1's gap with band 1's recipe, not band 0's.

        Test scenario:
            The sentinel was converted with band 0's factor whatever band was asked
            for, so on a raster packed at 0.01 / 0.1 `extract(band=1)` kept the gap as
            a value of -999.9.
        """
        values = np.asarray(
            self._two_bands([0.01, 0.1]).extract(band=1), dtype="float64"
        ).ravel()
        np.testing.assert_allclose(np.sort(values), [10.0, 20.0, 30.0])

    def test_one_malformed_band_does_not_switch_off_the_others(self):
        """A zero factor on one band leaves that band alone and unpacks the rest.

        Test scenario:
            The usability guard judged the whole per-band array at once, so a single
            band with `scale=0` disabled unpacking for every band in an all-bands read
            -- while the same bands read one at a time were unpacked.
        """
        dataset = self._two_bands([0.01, 0.0])
        together = np.asarray(dataset.read_array(), dtype="float64")
        alone = float(np.asarray(dataset.read_array(band=0), dtype="float64")[0, 1])
        assert together[0, 0, 1] == pytest.approx(alone), (
            f"band 0 read {together[0, 0, 1]} with its sibling, {alone} on its own"
        )
        assert together[1, 0, 1] == pytest.approx(100.0), (
            "the malformed band should be left in stored counts"
        )


class TestACollectionAnswersInOneUnit:
    """`values` and the lazy `data` cube read the same stack; they must agree."""

    def test_a_temporal_reduction_matches_the_eager_stack(self, tmp_path):
        """`col.mean()` is the lazy spelling of `col.values.mean(0)`.

        Test scenario:
            `values` reads through `read_array` and went physical; `data`, and every
            temporal reduction built on it, reads raw blocks and stayed in stored
            counts. On two packed timesteps `col.mean()` answered 150 where
            `col.values.mean(0)` answered 3.0. Each file carries its own recipe.
        """
        pytest.importorskip("dask")
        for index, counts in enumerate(
            ([[100, 200], [300, 400]], [[200, 300], [400, 500]])
        ):
            step = _packed_raster(counts, 0.01, 1.5)
            step.to_file(str(tmp_path / f"t{index}.tif"))
        collection = DatasetCollection.from_files(str(tmp_path), glob="*.tif")

        eager = np.asarray(collection.values, dtype="float64").mean(axis=0)
        reduced = collection.mean()
        lazy = np.asarray(
            reduced.read_array() if hasattr(reduced, "read_array") else reduced,
            dtype="float64",
        )

        np.testing.assert_allclose(lazy.reshape(eager.shape), eager)


class TestAZarrStoreDescribesWhatItHolds:
    """`to_zarr` materialises a packed raster; its metadata has to say so."""

    def test_the_gap_survives_the_round_trip(self, tmp_path):
        """The written sentinel is the one the written array actually holds.

        Test scenario:
            `to_zarr` writes the physical values `read_array` returns, and pyramids'
            Zarr metadata has no channel for a recipe. It recorded the stored `-9999`
            and the stored `int16` over a `float64` array whose gap held `-98.49`, so
            reading the store back treated the gap as a measurement.
        """
        pytest.importorskip("zarr")
        pytest.importorskip("dask")
        source = _packed_raster([[-9999, 100], [200, 300]], 0.01, 1.5)
        source.no_data_value = [-9999]
        source.to_file(str(tmp_path / "source.tif"))
        Dataset.read_file(str(tmp_path / "source.tif")).to_zarr(
            str(tmp_path / "out.zarr")
        )

        restored = Dataset.from_zarr(str(tmp_path / "out.zarr"))
        values = restored.read_array(masked=True)

        assert bool(np.ma.getmaskarray(values).ravel()[0]), "the gap was read as data"
        np.testing.assert_allclose(np.asarray(values).ravel()[1:], [2.5, 3.5, 4.5])
