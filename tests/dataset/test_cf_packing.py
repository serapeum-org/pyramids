"""Unpacking CF-packed rasters on read, and not corrupting them on the way out (#1124).

A packed raster stores small integers plus a `scale_factor` / `add_offset` recipe. Reading
the integers and calling them the answer is wrong by the packing factor, quietly: the code
runs, the plot renders, and a wave height of 14.35 m reads as 1435.

These pin three things the change has to get right at once:

- packed data comes back physical, from `read_array`, `stats` and the plot path alike;
- unpacked data — nearly every raster — is untouched, same dtype and no copy, because GDAL
  reports `scale=1.0, offset=0.0` rather than `None` for a band that was never packed;
- `apply` neither truncates a float result into an integer band nor silently spends the
  packing while leaving the raw values in place.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._utils import _is_identity_packing, apply_unpack
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.collection import _agree_on_one_sentinel
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
            GDAL answers `1.0` / `0.0` for a band that was never packed, not `None`. If
            only `None` counted as "nothing to do", flipping the default would promote
            every ordinary raster to `float64` and double its memory for no gain.
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

    def test_the_lazy_path_agrees_with_the_eager_one(self):
        """`chunks=` builds its array separately, so it can drift from the eager read."""
        pytest.importorskip("dask")
        store = NetCDF.read_file(PACKED_NC)
        try:
            variable = store.get_variable("z")
            eager = np.asarray(variable.read_array(), dtype="float64")
            lazy = np.asarray(variable.read_array(chunks="auto"), dtype="float64")
        finally:
            store.close()
        np.testing.assert_allclose(np.nanmax(lazy), np.nanmax(eager))

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

    def test_an_unpacked_raster_keeps_its_statistics(self):
        """The transform must not disturb a raster that was never packed."""
        dataset = _int_raster([[1, 2, 3], [4, 5, 6]])
        frame = dataset.stats(approx_ok=False)
        assert float(frame["min"].iloc[0]) == pytest.approx(1.0)
        assert float(frame["max"].iloc[0]) == pytest.approx(6.0)


class TestApplyDoesNotCorrupt:
    """Two independent faults in `apply`, both silent."""

    def test_a_float_result_is_not_truncated_into_an_integer_band(self):
        """The output takes the function's result type, not the source's.

        Test scenario:
            The issue's own numbers. Building the destination at the source band's type
            wrote `1435 * 0.01` back as `14`, losing everything after the point.
        """
        dataset = _int_raster([[1435, 1272, 1000]])
        result = dataset.apply(lambda a: a * 0.01)
        got = np.asarray(result.read_array(), dtype="float64")
        np.testing.assert_allclose(got[0], [14.35, 12.72, 10.0])

    def test_integer_arithmetic_keeps_its_width(self):
        """Promotion is driven by the result, so integer maths must not widen.

        Test scenario:
            Under NumPy 2's promotion rules `int16 * 2` is still `int16`. Widening it
            anyway would quadruple the storage of every integer `apply` for nothing.
        """
        dataset = _int_raster([[1, 2, 3]])
        result = dataset.apply(lambda a: a * 2)
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

    def test_an_unpacked_raster_is_unaffected_by_the_rule(self):
        """The destination check must not disturb the ordinary case."""
        dataset = _int_raster([[1, 2], [3, 4]])
        result = dataset.io.stream_transform(lambda tile: tile * 2, tile_size=2)
        got = np.asarray(result.read_array(), dtype="float64")
        np.testing.assert_allclose(got.ravel(), [2.0, 4.0, 6.0, 8.0])


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
