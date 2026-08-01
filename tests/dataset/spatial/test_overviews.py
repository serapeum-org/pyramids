"""Tests for the Dataset class overview methods."""

import contextlib
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def _overviewed_raster(tmp_path, name: str = "ovr_src.tif") -> str:
    """Write a georeferenced 3-band raster and build two overview levels on it.

    Band ``i`` is filled with the constant ``i + 1`` so a caller can tell the bands
    apart after decimation, and the cell size (0.5) halves per level.
    """
    path = str(tmp_path / name)
    raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 3, gdal.GDT_Float32)
    raster.SetGeoTransform((10.0, 0.5, 0.0, 80.0, 0.0, -0.5))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    raster.SetProjection(srs.ExportToWkt())
    for index in range(3):
        band = raster.GetRasterBand(index + 1)
        band.SetNoDataValue(-9999.0)
        band.WriteArray(np.full((64, 64), float(index + 1), dtype="float32"))
    raster = None

    handle = gdal.Open(path, gdal.GA_Update)
    handle.BuildOverviews("AVERAGE", [2, 4])
    handle = None
    return path


def _mixed_overview_vrt(tmp_path) -> str:
    """Build a 2-band VRT reporting ``overview_count == [1, 0]``.

    Band 1 sources a raster carrying an external `.ovr` and declares it through a
    per-band `<Overview>`; band 2 sources a raster without one. Both bands must point at
    *different* files — sharing one source makes GDAL expose its overviews on both bands
    and the counts come back `[1, 1]`.
    """
    driver = gdal.GetDriverByName("GTiff")
    sources = []
    for name in ("with_ovr.tif", "without_ovr.tif"):
        path = str(tmp_path / name)
        raster = driver.Create(path, 8, 8, 1, gdal.GDT_Float32)
        raster.GetRasterBand(1).WriteArray(np.arange(64, dtype="float32").reshape(8, 8))
        raster = None
        sources.append(path)
    with_ovr, without_ovr = sources
    handle = gdal.Open(with_ovr)
    handle.BuildOverviews("NEAREST", [2])
    handle = None

    vrt = tmp_path / "mixed.vrt"
    vrt.write_text(
        f'<VRTDataset rasterXSize="8" rasterYSize="8">'
        f'<VRTRasterBand dataType="Float32" band="1">'
        f'<SimpleSource><SourceFilename relativeToVRT="0">{with_ovr}</SourceFilename>'
        f"<SourceBand>1</SourceBand></SimpleSource>"
        f'<Overview><SourceFilename relativeToVRT="0">{with_ovr}.ovr</SourceFilename>'
        f"<SourceBand>1</SourceBand></Overview></VRTRasterBand>"
        f'<VRTRasterBand dataType="Float32" band="2">'
        f'<SimpleSource><SourceFilename relativeToVRT="0">{without_ovr}</SourceFilename>'
        f"<SourceBand>1</SourceBand></SimpleSource></VRTRasterBand></VRTDataset>"
    )
    return str(vrt)


def test_create_overviews(era5_image: gdal.Dataset, clean_overview_after_test):
    dataset = Dataset(era5_image)
    dataset.create_overviews()
    assert dataset.raster.GetRasterBand(1).GetOverviewCount() == 2
    # test the overview_number property
    assert dataset.overview_count == [2] * dataset.band_count
    assert Path(f"{dataset.file_name}.ovr").exists()


def test_get_overview(era5_image: gdal.Dataset, clean_overview_after_test):
    dataset = Dataset(era5_image)
    band = 0
    overview_index = 0
    dataset.create_overviews()
    ovr = dataset.get_overview(band, overview_index)
    assert isinstance(ovr, gdal.Band)

    with pytest.raises(ValueError):
        dataset.get_overview(band, 5)


class TestReCreateOverviews:
    def test_recreate_overviews_internal(
        self,
        era5_image_internal_overviews_read_only_false: Dataset,
        clean_overview_after_test,
    ):
        """Test recreating overviews for a dataset with internal overviews"""
        dataset = Dataset(era5_image_internal_overviews_read_only_false)
        dataset.recreate_overviews(resampling_method="average")

    def test_recreate_overviews_external(
        self,
        era5_image: gdal.Dataset,
        clean_overview_after_test,
    ):
        """Test recreating overviews for a dataset with external overviews"""
        dataset = Dataset(era5_image)
        dataset.create_overviews(overview_levels=[2])
        dataset.recreate_overviews(resampling_method="average")


class TestRecreateOverviewsContract:
    """`recreate_overviews` signals explicitly instead of silently doing nothing (#863).

    The regeneration loop iterates `overview_count` times, so a dataset with no
    overviews used to return having rebuilt nothing and raised nothing — and the
    read-only guard only fired incidentally, when GDAL happened to refuse a rewrite.
    """

    def test_no_overviews_on_writable_dataset_warns(self, era5_raster_path, tmp_path):
        """A writable dataset with no overviews warns rather than silently no-oping.

        Test scenario:
            Copy a raster that has no overviews, open it writable and call
            ``recreate_overviews`` — expected: a `UserWarning` pointing at
            ``create_overviews`` instead of a silent return.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "no_ovr.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            assert not any(dataset.overview_count), "fixture must start with none"
            with pytest.warns(UserWarning, match=r"call create_overviews\(\) first"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_no_overviews_on_read_only_dataset_warns(self, era5_raster_path, tmp_path):
        """A read-only dataset with no overviews warns about the real cause.

        Test scenario:
            The pre-fix silent path — read-only *and* nothing to regenerate, so GDAL
            never threw. Expected: the same "nothing to regenerate" warning as the
            writable case, not a `ReadOnlyError` — the blocker is the empty count, not
            the access mode, and reopening writable would only produce this warning
            anyway.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "ro_no_ovr.tif")
        dataset = Dataset.read_file(str(work), read_only=True)
        try:
            with pytest.warns(UserWarning, match=r"call create_overviews\(\) first"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_internal_overviews_on_read_only_dataset_raises(
        self, era5_internal_overviews_path, tmp_path
    ):
        """Internal overviews cannot be rewritten through a read-only handle.

        Test scenario:
            A raster carrying internal overviews opened read-only — expected:
            `ReadOnlyError` (the pre-existing guarantee, preserved by the fix).
        """
        work = shutil.copy(era5_internal_overviews_path, tmp_path / "ro_internal.tif")
        dataset = Dataset.read_file(str(work), read_only=True)
        try:
            assert any(dataset.overview_count), "fixture must carry internal overviews"
            with pytest.raises(ReadOnlyError, match="opened read-only"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    @pytest.mark.parametrize(
        "gdal_message, expected_error, expected_text",
        [
            ("Failed to write overview block: disk full", RuntimeError, "disk full"),
            (
                "Attempt to write to a read only dataset",
                ReadOnlyError,
                "read_only=False",
            ),
            (
                "/mnt/read-only-archive/dem.tif, band 1: No space left on device",
                RuntimeError,
                "No space left on device",
            ),
        ],
        ids=[
            "unrelated-failure-propagates",
            "read-only-wording-is-translated",
            "read-only-in-the-path-is-not-a-refusal",
        ],
    )
    def test_gdal_failures_are_classified_by_message(
        self,
        era5_raster_path,
        tmp_path,
        monkeypatch,
        gdal_message,
        expected_error,
        expected_text,
    ):
        """Only a genuine write refusal becomes `ReadOnlyError`; other failures propagate.

        Test scenario:
            `gdal.RegenerateOverview` is patched to raise a `RuntimeError` while a
            writable dataset that does have overviews is regenerated, so no CPL error
            number is set and the phrase fallback decides. Expected: a disk-full failure
            surfaces unchanged as `RuntimeError` (the pre-fix code relabelled *every*
            `RuntimeError`); GDAL's spaced "read only dataset" wording is translated; and
            a failure whose *path* merely contains "read-only" stays a `RuntimeError`,
            since GDAL prefixes messages with the dataset path and a bare substring test
            would misdiagnose it. The real `CPLE_NoWriteAccess` route is exercised in
            ``test_internal_overviews_on_read_only_dataset_raises``.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "gdal_failure.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            dataset.create_overviews(overview_levels=[2])

            def raise_runtime_error(*args, **kwargs):
                raise RuntimeError(gdal_message)

            monkeypatch.setattr(gdal, "RegenerateOverview", raise_runtime_error)
            with pytest.raises(expected_error) as excinfo:
                dataset.recreate_overviews()
            assert type(excinfo.value) is expected_error, (
                f"expected exactly {expected_error.__name__}, got {type(excinfo.value).__name__}"
            )
            assert expected_text in str(excinfo.value), (
                f"expected {expected_text!r} in the message, got: {excinfo.value}"
            )
            if expected_error is ReadOnlyError:
                assert isinstance(excinfo.value.__cause__, RuntimeError), (
                    "the original GDAL error must stay chained as __cause__"
                )
        finally:
            dataset.close()

    def test_failing_status_without_an_exception_raises(
        self, era5_raster_path, tmp_path, monkeypatch
    ):
        """A non-`CE_None` return is caught even when GDAL is not raising exceptions.

        Test scenario:
            `gdal.UseExceptions()` is process-global, so a caller that turned it off
            leaves `RegenerateOverview` *returning* `CE_Failure` instead of raising.
            Patch it to do exactly that on a writable dataset that does have overviews
            — expected: a `RuntimeError` naming the band and level, since the
            try/except alone would let this through as the silent no-op the method
            exists to remove.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "status_failure.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            dataset.create_overviews(overview_levels=[2])
            monkeypatch.setattr(
                gdal, "RegenerateOverview", lambda *args, **kwargs: gdal.CE_Failure
            )
            with pytest.raises(RuntimeError) as excinfo:
                dataset.recreate_overviews()
            assert type(excinfo.value) is RuntimeError, (
                f"a failing status must not be relabelled, got {type(excinfo.value).__name__}"
            )
            assert "overview 0 of band 0" in str(excinfo.value), (
                f"the error must name the band and level, got: {excinfo.value}"
            )
        finally:
            dataset.close()

    def test_propagated_failure_is_noted_with_the_band_and_level(
        self, era5_raster_path, tmp_path, monkeypatch
    ):
        """A propagated GDAL failure carries a note saying where regeneration stopped.

        Test scenario:
            An unrelated `RuntimeError` is raised from `gdal.RegenerateOverview` and
            re-raised unchanged — expected: `__notes__` names the band and level and
            says earlier bands may already have been rewritten, because the loop
            rewrites in place and leaves the dataset half-regenerated.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "noted_failure.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            dataset.create_overviews(overview_levels=[2])

            def raise_runtime_error(*args, **kwargs):
                raise RuntimeError("Failed to write overview block: disk full")

            monkeypatch.setattr(gdal, "RegenerateOverview", raise_runtime_error)
            with pytest.raises(RuntimeError) as excinfo:
                dataset.recreate_overviews()
            notes = getattr(excinfo.value, "__notes__", [])
            assert any("overview 0 of band 0" in note for note in notes), (
                f"the note must name the band and level, got {notes}"
            )
            assert any("already have been rewritten" in note for note in notes), (
                f"the note must flag the partial rewrite, got {notes}"
            )
        finally:
            dataset.close()

    def test_mixed_counts_still_regenerate_the_populated_bands(
        self, tmp_path, monkeypatch
    ):
        """After warning about the empty bands, the populated ones are regenerated.

        Test scenario:
            Record every `gdal.RegenerateOverview` call on the mixed `[1, 0]` VRT —
            expected: exactly one call, for band 0's single level, proving the mixed
            path warns *and* carries on rather than returning early. The recorder also
            keeps the call off the VRT's read-only `.ovr`, which is why the sibling
            warning test has to suppress that failure.
        """
        dataset = Dataset.read_file(_mixed_overview_vrt(tmp_path), read_only=True)
        try:
            calls = []
            monkeypatch.setattr(
                gdal,
                "RegenerateOverview",
                lambda band, ovr, method: calls.append(method) or gdal.CE_None,
            )
            with pytest.warns(UserWarning, match=r"call create_overviews\(\) first"):
                dataset.recreate_overviews(resampling_method="average")
            assert calls == ["average"], (
                f"only band 0's single overview should regenerate, got {calls}"
            )
        finally:
            dataset.close()

    def test_dataset_without_bands_warns_about_the_bands(self):
        """A band-less dataset is told it has no bands, not to call `create_overviews`.

        Test scenario:
            A zero-band in-memory raster, so `overview_count == []` — expected: the
            "no bands" warning. `bands_without` is empty too, so the all-bands-empty
            branch would also match; the band-less check has to come first or the
            warning would point at `create_overviews`, which cannot help here.
        """
        dataset = Dataset(gdal.GetDriverByName("MEM").Create("", 4, 4, 0))
        try:
            assert dataset.overview_count == [], (
                f"fixture must have no bands, got {dataset.overview_count}"
            )
            with pytest.warns(UserWarning, match="no bands"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_mixed_band_counts_warn_naming_the_skipped_bands(self, tmp_path):
        """Bands without overviews are named instead of being silently skipped.

        Test scenario:
            A VRT whose band 1 sources a raster carrying an external `.ovr` and whose
            band 2 sources one without — `overview_count == [1, 0]`. Expected: a warning
            naming band 1 (0-based) as skipped, since `range(0)` would otherwise
            regenerate nothing and report nothing, leaving that band silently stale.
            Regenerating band 0 then fails because the VRT opens its `.ovr` source
            read-only; that is beside the point here, so it is suppressed — the warning
            is emitted before the loop either way.
        """
        dataset = Dataset.read_file(_mixed_overview_vrt(tmp_path), read_only=True)
        try:
            assert dataset.overview_count == [1, 0], (
                f"fixture must produce mixed counts, got {dataset.overview_count}"
            )
            with pytest.warns(
                UserWarning, match=r"Bands 1 \(0-based\) have no overviews"
            ):
                with contextlib.suppress(ReadOnlyError):
                    dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_in_memory_dataset_without_overviews_warns(self):
        """An in-memory dataset gets the same no-overviews warning as an on-disk one.

        Test scenario:
            A `create_from_array` raster with no path (MEM driver, `access == "write"`)
            — expected: the no-overviews warning, so the edit-in-memory workflow reports
            the empty count the same way and is never blocked by an access-mode error.
        """
        dataset = Dataset.create_from_array(
            np.ones((16, 16), dtype=np.float32),
            top_left_corner=(0.0, 16.0),
            cell_size=1.0,
            epsg=4326,
        )
        try:
            with pytest.warns(UserWarning, match=r"call create_overviews\(\) first"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_warning_points_at_the_caller_not_the_facade(
        self, era5_raster_path, tmp_path
    ):
        """The warning is attributed to the calling line, not `Dataset.recreate_overviews`.

        Test scenario:
            `Dataset.recreate_overviews` is a facade over the engine, so the warning
            needs `stacklevel=3` to skip it — expected: the recorded warning names this
            test file. At `stacklevel=2` it named `dataset.py`, which also collapsed
            every call site in a user loop onto one dedupe key.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "blame.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            with pytest.warns(UserWarning) as recorded:
                dataset.recreate_overviews()
            assert Path(recorded[0].filename).name == Path(__file__).name, (
                f"warning blamed {recorded[0].filename}, expected this test file"
            )
        finally:
            dataset.close()

    def test_warning_points_at_the_caller_from_the_engine_entry_point(
        self, era5_raster_path, tmp_path
    ):
        """Calling the engine directly is blamed on the caller too, one frame shallower.

        Test scenario:
            `ds.io.recreate_overviews()` reaches the warning through one frame fewer
            than the `Dataset` facade, so a `stacklevel` hard-coded for the facade
            overshoots into pytest's own frame or, at module level, `<sys>:0` —
            expected: the recorded warning still names this test file, proving the
            frame walk adapts to both entry points.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "blame_engine.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            with pytest.warns(UserWarning) as recorded:
                dataset.io.recreate_overviews()
            assert Path(recorded[0].filename).name == Path(__file__).name, (
                f"warning blamed {recorded[0].filename}, expected this test file"
            )
        finally:
            dataset.close()

    def test_existing_overviews_regenerate_without_warning(
        self, era5_raster_path, tmp_path
    ):
        """The happy path stays silent — no warning when every band has overviews.

        Test scenario:
            Build overviews on a writable copy, then regenerate them — expected: no
            `UserWarning`, so a future refactor cannot start warning on every call.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "happy.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            dataset.create_overviews(overview_levels=[2])
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                dataset.recreate_overviews()
            user_warnings = [w for w in recorded if issubclass(w.category, UserWarning)]
            assert not user_warnings, (
                f"happy path should not warn, got {[str(w.message) for w in user_warnings]}"
            )
        finally:
            dataset.close()


class TestGetOverviewDataset:
    """`get_overview_dataset` returns an overview level as a first-class Dataset (#784).

    `get_overview` yields a raw `gdal.Band` with no geotransform, CRS or no-data, so an
    overview could not be plotted, written or reprojected without dropping to GDAL.
    """

    def test_file_backed_level_scales_the_grid(self, tmp_path):
        """A file-backed level keeps the CRS and no-data and halves the resolution.

        Test scenario:
            Level 0 of a 64x64 raster at cell 0.5 — expected: 32x32 at cell 1.0, the
            same origin, EPSG and no-data, and all three bands carried over.
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path))
        overview = None
        try:
            overview = dataset.get_overview_dataset(overview_index=0)
            assert (overview.rows, overview.columns) == (32, 32), (
                f"expected 32x32, got {(overview.rows, overview.columns)}"
            )
            assert overview.cell_size == 1.0, f"cell size {overview.cell_size} != 1.0"
            assert overview.epsg == dataset.epsg, "EPSG should carry over"
            assert overview.band_count == 3, (
                f"expected 3 bands, got {overview.band_count}"
            )
            assert overview.no_data_value[0] == -9999.0, "no-data should carry over"
            assert overview.top_left_corner == dataset.top_left_corner, (
                "the origin must not move when decimating"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_higher_level_scales_further(self, tmp_path):
        """Level 1 decimates twice as far as level 0.

        Test scenario:
            ``overview_index=1`` on the same raster — expected: 16x16 at cell 2.0, so
            the scaling follows the requested level rather than always level 0.
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path))
        overview = None
        try:
            overview = dataset.get_overview_dataset(overview_index=1)
            assert (overview.rows, overview.columns) == (16, 16), (
                f"expected 16x16, got {(overview.rows, overview.columns)}"
            )
            assert overview.cell_size == 2.0, f"cell size {overview.cell_size} != 2.0"
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_band_selection_returns_that_band(self, tmp_path):
        """Passing `band` returns a single-band Dataset holding that band's pixels.

        Test scenario:
            Each band is filled with a distinct constant, so ``band=1`` must come back
            as one band whose values are 2.0 — proving the subset picks the right band
            rather than defaulting to band 0.
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path))
        overview = None
        try:
            overview = dataset.get_overview_dataset(band=1, overview_index=0)
            assert overview.band_count == 1, (
                f"expected 1 band, got {overview.band_count}"
            )
            values = np.asarray(overview.read_array(band=0))
            assert np.allclose(values, 2.0), (
                f"expected band 1's constant 2.0, got {values[0, 0]}"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_file_backed_level_is_not_materialised(self, tmp_path):
        """A file-backed level is described by a VRT rather than copied into memory.

        Test scenario:
            Inspect the returned dataset's driver and handle — expected: the ``vrt``
            driver (not merely "not memory", which an unrecognised driver would also
            satisfy) and a handle distinct from the parent's, i.e. it came through
            ``OVERVIEW_LEVEL`` without reading the level into RAM.
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path))
        overview = None
        try:
            overview = dataset.get_overview_dataset(overview_index=0)
            assert overview.driver_type == "vrt", (
                f"file-backed level should be a lazy VRT, got {overview.driver_type}"
            )
            assert overview.raster is not dataset.raster, "must be its own handle"
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_lazy_level_refuses_the_reopen_by_path_reads(self, tmp_path):
        """The lazy level fails loudly on reads that would reopen it by path.

        Test scenario:
            `OpenEx(OVERVIEW_LEVEL=...)` alone yields a handle described by the
            *parent's* path, so `threadsafe=`/`chunks=`/pickle would silently read the
            full-resolution raster while the object reported the level's shape. The VRT
            wrapper leaves it pathless — expected: a plain read is correct, and each
            reopen-by-path route raises instead of returning parent pixels.
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path, "gradient.tif"))
        overview = None
        try:
            overview = dataset.get_overview_dataset(band=0, overview_index=1)
            expected = dataset.read_overview_array(band=0, overview_index=1)
            np.testing.assert_array_equal(
                np.asarray(overview.read_array(band=0)),
                expected,
                err_msg="the plain read must return the level's own pixels",
            )
            assert overview.file_name == "", (
                f"the lazy level must carry no path, got {overview.file_name!r}"
            )
            with pytest.raises(ValueError, match="reopenable path"):
                overview.read_array(band=0, threadsafe=True)
            with pytest.raises(TypeError, match="no on-disk path"):
                pickle.dumps(overview)
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_lazy_level_refuses_a_chunked_read(self, tmp_path):
        """A chunked read of the lazy level never hands back full-resolution pixels.

        Test scenario:
            `chunks=` reopens the dataset by path inside each dask task and the VRT
            carries no path, making it the third reopen-by-path route beside
            `threadsafe=` and pickle — expected: the graph is shaped from the level
            (32x32, not the parent's 64x64) and computing it raises, rather than quietly
            filling the blocks from the parent raster. The refusal surfaces on compute
            rather than at the call, unlike its two siblings.
        """
        pytest.importorskip("dask")
        dataset = Dataset.read_file(_overviewed_raster(tmp_path, "chunked.tif"))
        overview = None
        try:
            overview = dataset.get_overview_dataset(band=0, overview_index=0)
            lazy = overview.read_array(band=0, chunks=16)
            assert lazy.shape == (32, 32), (
                f"the graph must be shaped from the level, got {lazy.shape}"
            )
            with pytest.raises(RuntimeError, match="No such file or directory"):
                lazy.compute()
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_network_parent_with_credentials_warns(self, tmp_path, monkeypatch):
        """A network-backed parent carrying credentials warns that the VRT strands them.

        Test scenario:
            The warning must key on *network* backing, not on `is_remote`, which is also
            true for purely local `/vsimem/` and `/vsizip/`. Force the predicate true for
            a local file so the branch is reachable offline — expected: a `UserWarning`
            naming the credentials. The same parent without credentials, and a genuinely
            local one, must stay silent.
        """
        from pyramids.dataset.engines import io as io_module

        dataset = Dataset.read_file(_overviewed_raster(tmp_path, "credentialed.tif"))
        with_env = without_env = local_level = None
        try:
            dataset.attach_gdal_env({"AWS_REQUEST_PAYER": "requester"})
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                local_level = dataset.get_overview_dataset()
            assert not [w for w in recorded if "cloud credentials" in str(w.message)], (
                "a local parent must not warn, even carrying an env"
            )

            monkeypatch.setattr(io_module, "is_network_backed", lambda path: True)
            with pytest.warns(UserWarning, match="cloud credentials"):
                with_env = dataset.get_overview_dataset()

            dataset.attach_gdal_env(None)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                without_env = dataset.get_overview_dataset()
            assert not [w for w in recorded if "cloud credentials" in str(w.message)], (
                "no credentials means nothing can be stranded"
            )
        finally:
            for handle in (with_env, without_env, local_level):
                if handle is not None:
                    handle.close()
            dataset.close()

    def test_from_bytes_label_does_not_reopen_a_same_named_file(
        self, tmp_path, monkeypatch
    ):
        """A cosmetic `from_bytes` name never reopens an unrelated file of that name.

        Test scenario:
            `from_bytes(payload, name="dem.tif")` stamps a label, not an identity. With
            a *different* `dem.tif` in the working directory, keying the lazy path off
            `file_name` returned that decoy's overview — silently, with no error.
            Expected: the level holds the payload's own value.
        """
        monkeypatch.chdir(tmp_path)
        decoy = gdal.GetDriverByName("GTiff").Create(
            "dem.tif", 32, 32, 1, gdal.GDT_Float32
        )
        decoy.SetGeoTransform((0.0, 1.0, 0.0, 32.0, 0.0, -1.0))
        decoy.GetRasterBand(1).WriteArray(np.full((32, 32), 999.0, dtype="float32"))
        decoy = None
        handle = gdal.Open("dem.tif", gdal.GA_Update)
        handle.BuildOverviews("AVERAGE", [2])
        handle = None

        real = str(tmp_path / "real.tif")
        raster = gdal.GetDriverByName("GTiff").Create(real, 32, 32, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((0.0, 1.0, 0.0, 32.0, 0.0, -1.0))
        raster.GetRasterBand(1).WriteArray(np.full((32, 32), 111.0, dtype="float32"))
        raster = None
        handle = gdal.Open(real, gdal.GA_Update)
        handle.BuildOverviews("AVERAGE", [2])
        handle = None

        dataset = Dataset.from_bytes(Path(real).read_bytes(), name="dem.tif")
        overview = None
        try:
            overview = dataset.get_overview_dataset()
            value = float(np.asarray(overview.read_array(band=0))[0, 0])
            assert value == pytest.approx(111.0), (
                f"expected the payload's own value 111.0, got {value} "
                "(999.0 means the decoy dem.tif in the CWD was read)"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_partly_declared_no_data_is_not_completed(self):
        """A band without a no-data value does not gain one from its neighbours.

        Test scenario:
            A parent declaring no-data on band 0 only — expected: band 1 stays `None`.
            Passing the builder a list containing `None` coerced it into a `nan`
            sentinel that masking and statistics would then honour.
        """
        raster = gdal.GetDriverByName("MEM").Create("", 32, 32, 2, gdal.GDT_Float32)
        raster.SetGeoTransform((0.0, 1.0, 0.0, 32.0, 0.0, -1.0))
        raster.GetRasterBand(1).SetNoDataValue(-9999.0)
        for index in range(2):
            raster.GetRasterBand(index + 1).WriteArray(
                np.zeros((32, 32), dtype="float32")
            )
        raster.BuildOverviews("AVERAGE", [2])
        dataset = Dataset(raster)
        overview = None
        try:
            assert dataset.no_data_value[1] is None, "precondition: band 1 has none"
            overview = dataset.get_overview_dataset()
            assert overview.no_data_value[0] == pytest.approx(-9999.0), (
                f"band 0 should keep its no-data, got {overview.no_data_value[0]}"
            )
            assert overview.no_data_value[1] is None, (
                f"band 1 must stay unset, got {overview.no_data_value[1]}"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_vsimem_parent_stays_lazy(self):
        """A /vsimem/ raster reopens under OVERVIEW_LEVEL, so it is not materialised.

        Test scenario:
            `/vsimem/` has a name GDAL can reopen, unlike a bare MEM handle — expected:
            the VRT path, not a second RAM copy of the level.
        """
        path = "/vsimem/overview_lazy.tif"
        raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((10.0, 0.5, 0.0, 80.0, 0.0, -0.5))
        raster.GetRasterBand(1).WriteArray(
            np.arange(4096, dtype="float32").reshape(64, 64)
        )
        raster = None
        dataset = Dataset.read_file(path, read_only=False)
        overview = None
        try:
            dataset.create_overviews(overview_levels=[2])
            overview = dataset.get_overview_dataset(overview_index=0)
            assert overview.driver_type == "vrt", (
                f"/vsimem/ should stay lazy, got driver {overview.driver_type}"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()
            gdal.Unlink(path)

    def test_rotated_grid_scales_the_same_on_both_paths(self, tmp_path):
        """The rotation terms are scaled, so a skewed grid is not sheared.

        Test scenario:
            The same skewed geotransform on disk and in memory — expected: identical
            geotransforms. Scaling only gt[1]/gt[5] left the rotation terms at half
            their correct value, displacing every pixel but the origin.
        """
        geo = (10.0, 0.5, 0.1, 80.0, 0.2, -0.5)
        arr = np.arange(4096, dtype="float32").reshape(64, 64)
        path = str(tmp_path / "skewed.tif")
        raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 1, gdal.GDT_Float32)
        raster.SetGeoTransform(geo)
        raster.GetRasterBand(1).WriteArray(arr)
        raster = None
        handle = gdal.Open(path, gdal.GA_Update)
        handle.BuildOverviews("AVERAGE", [2])
        handle = None

        on_disk = Dataset.read_file(path)
        in_memory = Dataset.create_from_array(arr, geo=geo, epsg=4326)
        in_memory.create_overviews(overview_levels=[2])
        lazy_level = memory_level = None
        try:
            lazy_level = on_disk.get_overview_dataset(overview_index=0)
            memory_level = in_memory.get_overview_dataset(overview_index=0)
            np.testing.assert_allclose(
                np.asarray(memory_level.geotransform),
                np.asarray(lazy_level.geotransform),
                err_msg="the two paths must agree on the scaled geotransform",
            )
            assert lazy_level.geotransform[2] == pytest.approx(0.2), (
                f"gt[2] should scale to 0.2, got {lazy_level.geotransform[2]}"
            )
        finally:
            for handle in (lazy_level, memory_level):
                if handle is not None:
                    handle.close()
            on_disk.close()
            in_memory.close()

    def test_absent_no_data_is_not_fabricated(self):
        """A parent with no no-data value does not gain one.

        Test scenario:
            A MEM parent created with `no_data_value=None` — expected: the level's
            no-data stays `None`, rather than a list of `None`s being coerced into a
            `nan` sentinel that masking and statistics would then honour.
        """
        dataset = Dataset.create_from_array(
            np.stack([np.zeros((32, 32), dtype="float32")] * 2),
            top_left_corner=(0.0, 32.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=None,
        )
        overview = None
        try:
            dataset.create_overviews(overview_levels=[2])
            overview = dataset.get_overview_dataset()
            assert all(value is None for value in overview.no_data_value), (
                f"no-data should stay absent, got {overview.no_data_value}"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_band_metadata_is_carried(self, tmp_path):
        """Band names, units, scale and offset survive, so packed data still decodes.

        Test scenario:
            A raster whose bands carry a scale/offset pair and names — expected: the
            level keeps them, since `to_file` on a level that lost its scale writes
            values that no longer decode to physical units.
        """
        path = _overviewed_raster(tmp_path, "packed.tif")
        writable = gdal.Open(path, gdal.GA_Update)
        for index in range(3):
            band = writable.GetRasterBand(index + 1)
            band.SetDescription(["red", "green", "nir"][index])
            band.SetScale(0.5)
            band.SetOffset(3.0)
        writable = None

        dataset = Dataset.read_file(path)
        overview = single = None
        try:
            overview = dataset.get_overview_dataset()
            assert overview.band_names == ["red", "green", "nir"], (
                f"band names lost: {overview.band_names}"
            )
            assert overview.scale == [0.5, 0.5, 0.5], f"scale lost: {overview.scale}"
            assert overview.offset == [3.0, 3.0, 3.0], f"offset lost: {overview.offset}"
            single = dataset.get_overview_dataset(band=2)
            assert single.band_names == ["nir"], (
                f"a band subset should keep that band's name, got {single.band_names}"
            )
        finally:
            for handle in (overview, single):
                if handle is not None:
                    handle.close()
            dataset.close()

    def test_dataset_metadata_is_carried_but_never_invented(self):
        """The parent's dataset metadata is copied over, and absence stays absence.

        Test scenario:
            The materialised path builds the level with `create_from_array`, which
            starts with an empty metadata dictionary — expected: a parent describing
            itself with `units`/`source` passes both on, while a parent with no metadata
            leaves the level's dictionary empty instead of gaining keys of its own.
        """
        described = Dataset.create_from_array(
            np.zeros((32, 32), dtype="float32"),
            top_left_corner=(0.0, 32.0),
            cell_size=1.0,
            epsg=4326,
        )
        described.meta_data = {"units": "K", "source": "reanalysis"}
        bare = Dataset.create_from_array(
            np.zeros((32, 32), dtype="float32"),
            top_left_corner=(0.0, 32.0),
            cell_size=1.0,
            epsg=4326,
        )
        described_level = bare_level = None
        try:
            described.create_overviews(overview_levels=[2])
            bare.create_overviews(overview_levels=[2])
            described_level = described.get_overview_dataset()
            expected = {"units": "K", "source": "reanalysis"}
            assert described_level.meta_data == expected, (
                f"dataset metadata lost: {described_level.meta_data}"
            )
            bare_level = bare.get_overview_dataset()
            assert bare_level.meta_data == {}, (
                f"a metadata-less parent must not gain any: {bare_level.meta_data}"
            )
        finally:
            for handle in (described_level, bare_level):
                if handle is not None:
                    handle.close()
            described.close()
            bare.close()

    def test_crs_without_an_epsg_code_survives(self):
        """A CRS with no EPSG authority code is not silently cleared.

        Test scenario:
            A MEM parent in a geostationary projection, which has no EPSG code, so
            `epsg` alone is `None` — expected: the level still carries a projection,
            rather than `create_from_array` clearing it.
        """
        geostationary = (
            "+proj=geos +h=35785831 +lon_0=0 +sweep=y +ellps=GRS80 +units=m +no_defs"
        )
        dataset = Dataset.create_from_array(
            np.zeros((32, 32), dtype="float32"),
            top_left_corner=(0.0, 32.0),
            cell_size=1000.0,
            epsg=geostationary,
        )
        overview = None
        try:
            dataset.create_overviews(overview_levels=[2])
            assert dataset.epsg is None, "precondition: this CRS has no EPSG code"
            overview = dataset.get_overview_dataset()
            assert overview.crs, "the level lost its CRS entirely"
            assert "geos" in overview.crs.lower(), (
                f"expected the parent's geostationary projection, got {overview.crs[:60]}"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_invalid_band_and_negative_level_raise(self, tmp_path):
        """Bad `band` and negative `overview_index` raise the documented ValueError.

        Test scenario:
            `band` out of range previously raised `IndexError` from `_iloc`, and a
            negative `overview_index` made GDAL hand back the *full-resolution* parent
            labelled as an overview — expected: `ValueError` for both.
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path))
        try:
            with pytest.raises(ValueError, match="out of range"):
                dataset.get_overview_dataset(band=9)
            with pytest.raises(ValueError, match="out of range"):
                dataset.get_overview_dataset(band=-1)
            with pytest.raises(ValueError, match="must not be negative"):
                dataset.get_overview_dataset(overview_index=-1)
        finally:
            dataset.close()

    def test_all_bands_rejected_when_one_lacks_overviews(self, tmp_path):
        """An all-bands request is rejected if any single band has no overviews.

        Test scenario:
            The mixed ``[1, 0]`` VRT — expected: ``band=None`` raises, since the level
            cannot be assembled from a band that has none, while ``band=0`` still
            succeeds. The guard loop runs per selected band, so only the all-bands path
            reaches the empty one.
        """
        dataset = Dataset.read_file(_mixed_overview_vrt(tmp_path))
        overview = None
        try:
            assert dataset.overview_count == [1, 0], (
                f"fixture must produce mixed counts, got {dataset.overview_count}"
            )
            with pytest.raises(ValueError, match="no overviews"):
                dataset.get_overview_dataset()
            overview = dataset.get_overview_dataset(band=0)
            assert overview.band_count == 1, "the populated band should still work"
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_band_less_dataset_raises(self):
        """A dataset with no bands says so instead of failing inside numpy.

        Test scenario:
            A zero-band MEM raster — expected: the same clear message
            `recreate_overviews` gives, not `need at least one array to stack`.
        """
        dataset = Dataset(gdal.GetDriverByName("MEM").Create("", 4, 4, 0))
        try:
            with pytest.raises(ValueError, match="no bands"):
                dataset.get_overview_dataset()
        finally:
            dataset.close()

    def test_path_less_dataset_falls_back_to_the_array(self):
        """An in-memory raster has no path to reopen, so its level is materialised.

        Test scenario:
            A `create_from_array` raster (MEM driver, empty `file_name`) — expected: the
            level still comes back with the scaled cell size and the right values, via
            the array fallback rather than ``OVERVIEW_LEVEL``.
        """
        arr = np.stack(
            [np.full((64, 64), float(index + 1), dtype="float32") for index in range(3)]
        )
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(10.0, 80.0), cell_size=0.5, epsg=4326
        )
        overview = None
        try:
            dataset.create_overviews(overview_levels=[2, 4])
            overview = dataset.get_overview_dataset(band=1, overview_index=1)
            assert (overview.rows, overview.columns) == (16, 16), (
                f"expected 16x16, got {(overview.rows, overview.columns)}"
            )
            assert overview.cell_size == 2.0, f"cell size {overview.cell_size} != 2.0"
            assert overview.epsg == 4326, f"epsg {overview.epsg} != 4326"
            assert np.allclose(np.asarray(overview.read_array(band=0)), 2.0), (
                "the fallback must read the requested band, not band 0"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()

    def test_written_level_round_trips(self, tmp_path):
        """The returned Dataset is writable like any other, with the scaled grid.

        Test scenario:
            ``to_file`` the level and reopen it — expected: the saved raster keeps the
            decimated shape and scaled cell size, which is the motivating use case
            (`get_overview` could not do this at all).
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path))
        out = str(tmp_path / "level0.tif")
        try:
            level = dataset.get_overview_dataset(overview_index=0)
            level.to_file(out)
            level.close()
        finally:
            dataset.close()
        saved = Dataset.read_file(out)
        try:
            assert (saved.rows, saved.columns) == (32, 32), (
                f"expected 32x32 on disk, got {(saved.rows, saved.columns)}"
            )
            assert saved.cell_size == 1.0, f"cell size {saved.cell_size} != 1.0"
        finally:
            saved.close()

    def test_missing_overviews_raise(self):
        """A raster with no overviews raises, matching `get_overview`.

        Test scenario:
            Call the accessor before `create_overviews` — expected: the same
            `ValueError` its `gdal.Band` sibling raises, not an empty Dataset.
        """
        dataset = Dataset.create_from_array(
            np.zeros((8, 8), dtype="float32"),
            top_left_corner=(0.0, 8.0),
            cell_size=1.0,
            epsg=4326,
        )
        try:
            with pytest.raises(ValueError, match="no overviews"):
                dataset.get_overview_dataset()
        finally:
            dataset.close()

    def test_out_of_range_level_raises(self, tmp_path):
        """An overview_index past the built levels raises, matching `get_overview`.

        Test scenario:
            Two levels exist; ask for index 9 — expected: `ValueError` naming the bound.
        """
        dataset = Dataset.read_file(_overviewed_raster(tmp_path))
        try:
            with pytest.raises(ValueError, match="should be less than"):
                dataset.get_overview_dataset(overview_index=9)
        finally:
            dataset.close()


class TestReadOverviewArray:
    def test_single_band_valid_overview(self, rhine_raster):
        dataset = Dataset(rhine_raster)
        # Test with single-band dataset and valid overview
        arr = dataset.read_overview_array(band=0, overview_index=0)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (63, 47)
        # test if the band is None
        arr = dataset.read_overview_array(band=None, overview_index=0)
        assert isinstance(arr, np.ndarray)

    def test_multi_band_all_valid_overview(
        self, era5_image_internal_overviews_read_only_true
    ):
        dataset = Dataset(era5_image_internal_overviews_read_only_true)
        # Test with all bands in a multi-band dataset
        arr = dataset.read_overview_array(band=None, overview_index=0)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == dataset.band_count
        assert arr.shape[1] == 2
        assert arr.shape[2] == 1

    def test_valid_band_no_overview(
        self, modis_surf_temp: gdal.Dataset, clean_overview_after_test
    ):
        dataset = Dataset(modis_surf_temp)
        # Assuming band 0 has no overviews
        with pytest.raises(ValueError):
            dataset.read_overview_array(band=None, overview_index=0)

        with pytest.raises(ValueError):
            dataset.read_overview_array(band=0, overview_index=0)
