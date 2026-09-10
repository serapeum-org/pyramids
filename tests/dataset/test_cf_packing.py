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
            which `_wrap_like` copies onto every spatial result) while its band
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
