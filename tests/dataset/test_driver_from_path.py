"""Every public write path derives its format from the destination path.

#1075 made "the extension names the format" a stated contract, declared on
`RasterBase.from_array` and taught in `docs/migration.md`. Six write paths
predated it and quietly ignored it, each hardcoding a driver so that a `.nc`
or `.jpg` destination received a GeoTIFF under a foreign name -- a file whose
extension lies about its contents, with no warning of any kind.

These tests pin the contract across all of them at once, so a future writer
that hardcodes `GetDriverByName("GTiff")` fails here rather than being found
by the next reviewer.
"""

from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import (
    DriverNotExistError,
    FileFormatNotSupportedError,
    ReadOnlyError,
)
from pyramids.dataset import Dataset, GeoReference
from pyramids.errors import DtypeNarrowingWarning

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326)


@pytest.fixture
def float_raster() -> Dataset:
    """A small in-memory float32 raster.

    Returns:
        Dataset: 4x4 single-band float32 at EPSG:4326.
    """
    return Dataset.from_array(
        np.arange(16, dtype="float32").reshape(4, 4), geo_ref=GEO_REF
    )


@pytest.fixture
def byte_raster() -> Dataset:
    """A small in-memory uint8 raster, writable to PNG and JPEG.

    Returns:
        Dataset: 4x4 single-band uint8 at EPSG:4326.
    """
    return Dataset.from_array(
        np.arange(16, dtype="uint8").reshape(4, 4), geo_ref=GEO_REF
    )


def driver_of(path) -> str:
    """Return the GDAL driver short name a written file actually has.

    Args:
        path: The file to open.

    Returns:
        str: The driver short name, which is the question every test here asks.
    """
    opened = gdal.Open(str(path))
    assert opened is not None, f"{path} was not written"
    return opened.GetDriver().ShortName


class TestTranslateHonoursTheExtension:
    """`translate` is the method whose whole purpose is format conversion."""

    @pytest.mark.parametrize(
        "extension, driver",
        [("tif", "GTiff"), ("nc", "netCDF"), ("img", "HFA")],
    )
    def test_the_written_format_follows_the_path(
        self, float_raster, tmp_path, extension, driver
    ):
        """Each destination is written in the format its extension names.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.
            extension: The destination extension.
            driver: The GDAL driver it must resolve to.

        Test scenario:
            `translate` passed `format="GTiff"` to `gdal.Translate`, which is
            exactly what disables GDAL's own extension inference -- so the
            "Convert Between Formats" its docstring advertises was unreachable
            and every destination produced a GTiff.
        """
        if gdal.GetDriverByName(driver) is None:
            pytest.skip(f"{driver} is not in this GDAL build")
        out = tmp_path / f"t.{extension}"
        float_raster.translate(path=str(out))
        assert driver_of(out) == driver, (
            f"{out.name} should be {driver}, got {driver_of(out)}"
        )

    def test_a_copy_only_format_is_accepted(self, byte_raster, tmp_path):
        """PNG is reachable because `gdal.Translate` writes by copy.

        Args:
            byte_raster: An 8-bit source, which PNG can carry.
            tmp_path: Temporary directory fixture.

        Test scenario:
            The `Creation` catalog flag records `Create` support. A copy-based
            writer must not inherit that refusal, or it rejects a file it can
            produce -- and PNG is the format the docstring names as an example.
        """
        out = tmp_path / "t.png"
        byte_raster.translate(path=str(out))
        assert driver_of(out) == "PNG", f"expected PNG, got {driver_of(out)}"

    def test_no_path_stays_in_memory(self, float_raster):
        """Omitting `path` keeps the result in the MEM driver."""
        assert float_raster.translate().raster.GetDriver().ShortName == "MEM"


class TestMergeRastersHonoursTheExtension:
    """One `dst` must not mean two formats depending on `method`."""

    @pytest.fixture
    def two_tiles(self, tmp_path):
        """Two adjacent single-band rasters on one grid.

        Args:
            tmp_path: Temporary directory fixture.

        Returns:
            list[str]: Paths to the two sources.
        """
        paths = []
        for i, x in enumerate((0.0, 4.0)):
            path = tmp_path / f"tile{i}.tif"
            Dataset.from_array(
                np.full((4, 4), float(i + 1), dtype="float32"),
                geo_ref=GeoReference(
                    top_left_corner=(x, 4.0), cell_size=1.0, epsg=4326
                ),
                path=path,
            ).close()
            paths.append(str(path))
        return paths

    @pytest.mark.parametrize("method", ["first", "last", "min", "max"])
    @pytest.mark.parametrize("extension, driver", [("tif", "GTiff"), ("nc", "netCDF")])
    def test_the_format_follows_the_path_for_every_method(
        self, two_tiles, tmp_path, method, extension, driver
    ):
        """`method` selects the reduction, never the output format.

        Args:
            two_tiles: The two source rasters.
            tmp_path: Temporary directory fixture.
            method: The merge method under test.
            extension: The destination extension.
            driver: The GDAL driver it must resolve to.

        Test scenario:
            The reduction path (`min` / `max` / `sum`) hardcoded
            `GetDriverByName("GTiff").Create`, while the z-order path
            (`first` / `last`) let `gdal.Translate` infer -- so the same `dst`
            produced a netCDF for one method and a GTiff for the other, and a
            caller could not know which without reopening the file.
        """
        from pyramids.dataset.merge import merge_rasters

        out = tmp_path / f"m_{method}.{extension}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            merge_rasters(two_tiles, str(out), method=method)
        assert driver_of(out) == driver, (
            f"method={method} to {out.name} should be {driver}, got {driver_of(out)}"
        )


class TestCopyPathsHonourTheExtension:
    """The remaining `CreateCopy`-based writers."""

    def test_copy_writes_the_format_the_extension_names(self, float_raster, tmp_path):
        """`Dataset.copy(path=...)` is not always a GeoTIFF.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.

        Test scenario:
            `copy` hardcoded GTiff for any path, and its docstring said only
            "Destination path to save the copied dataset" -- so `.nc` produced
            a GTiff and nothing disclosed it.
        """
        out = tmp_path / "cp.nc"
        float_raster.copy(path=str(out))
        assert driver_of(out) == "netCDF", f"expected netCDF, got {driver_of(out)}"

    def test_change_no_data_value_writes_the_named_format(self, float_raster, tmp_path):
        """`change_no_data_value(path=...)` follows the extension too.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.

        Test scenario:
            It hardcoded `"GTiff" if path is not None else "MEM"`, and its
            docstring asserted the `.tif` restriction in prose without
            enforcing it, so a `.nc` produced a mislabelled GeoTIFF. The
            result is closed first because the netCDF driver only materialises
            the file on close.
        """
        out = tmp_path / "nd.nc"
        result = float_raster.change_no_data_value(
            -1.0, float_raster.no_data_value[0], path=str(out)
        )
        result.close()
        assert driver_of(out) == "netCDF", f"expected netCDF, got {driver_of(out)}"


class TestNetCDFFromArrayRefusesAForeignExtension:
    """A netCDF store must not be written under another format's name."""

    @pytest.mark.parametrize("filename", ["lies.tif", "lies.img"])
    def test_a_non_netcdf_extension_raises(self, tmp_path, filename):
        """`NetCDF.from_array` refuses a path naming another driver.

        Args:
            tmp_path: Temporary directory fixture.
            filename: A destination naming a non-netCDF format.

        Test scenario:
            It builds a multidimensional store, which only the netCDF driver
            carries -- so the driver is fixed rather than resolved. What the
            extension decides is whether the caller asked for something else:
            `lies.tif` used to produce a netCDF under a GeoTIFF name silently.
        """
        from pyramids.netcdf import NetCDF

        with pytest.raises(FileFormatNotSupportedError, match="netCDF"):
            NetCDF.from_array(
                np.ones((1, 4, 4), dtype="float32"),
                geo_ref=GEO_REF,
                path=str(tmp_path / filename),
            )

    @pytest.mark.parametrize("extension", ["nc", "nc4"])
    def test_a_netcdf_extension_is_accepted(self, tmp_path, extension):
        """Both catalogued netCDF spellings work.

        Args:
            tmp_path: Temporary directory fixture.
            extension: The netCDF spelling under test.
        """
        from pyramids.netcdf import NetCDF

        out = tmp_path / f"ok.{extension}"
        NetCDF.from_array(
            np.ones((1, 4, 4), dtype="float32"), geo_ref=GEO_REF, path=str(out)
        )
        assert out.exists(), f"{out} was not written"


class TestToFileMatchesTheConstructors:
    """`to_file` reads the same catalog, so it must agree on case."""

    @pytest.mark.parametrize("filename", ["c.TIF", "d.TIFF", "e.tiff"])
    def test_an_upper_case_extension_is_writable(
        self, float_raster, tmp_path, filename
    ):
        """A raster that can be *built* as `x.TIF` can be *written* to it.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.
            filename: A destination differing in case or spelling.

        Test scenario:
            The branch added a case-folding resolver for the constructors
            beside the existing non-folding one behind `to_file`, so
            `from_array(path="x.TIF")` built while `to_file("x.TIF")` raised
            `DriverNotExistError`. On `main` the two agreed only by accident,
            because the constructors refused anything but `.tif`.
        """
        out = tmp_path / filename
        float_raster.to_file(str(out))
        assert driver_of(out) == "GTiff", f"expected GTiff, got {driver_of(out)}"

    @pytest.mark.parametrize("driver", ["geotiff", "GTiff"])
    def test_an_explicit_driver_takes_a_key_or_a_gdal_name(
        self, float_raster, tmp_path, driver
    ):
        """`to_file(driver=...)` accepts both spellings, like the collection.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.
            driver: The driver spelling under test.

        Test scenario:
            The counterpart of the collection fix: `Dataset.to_file` already
            normalised a GDAL short name to its catalog key, but nothing
            exercised that branch, which is how the collection sibling drifted
            into crashing on the same argument.
        """
        out = tmp_path / f"{driver}.tif"
        float_raster.to_file(str(out), driver=driver)
        assert driver_of(out) == "GTiff", f"expected GTiff, got {driver_of(out)}"

    def test_an_unknown_explicit_driver_raises(self, float_raster, tmp_path):
        """A name that is neither a catalog key nor a GDAL driver is refused.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.
        """
        with pytest.raises(DriverNotExistError, match="not in the driver catalog"):
            float_raster.to_file(str(tmp_path / "x.tif"), driver="NotADriver")

    def test_a_narrowing_dtype_warns(self, float_raster, tmp_path):
        """Writing float32 to PNG warns that the values will not survive.

        Args:
            float_raster: A float32 source.
            tmp_path: Temporary directory fixture.

        Test scenario:
            Correcting the catalog's extension rows widened `to_file`, an
            untouched public method: `.png` went from `DriverNotExistError` to
            writing an 8-bit file. GDAL reports the Float32 -> Byte conversion
            only as a `RuntimeWarning`, which a caller filtering on their own
            categories never sees, so a float DEM was quietly destroyed.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            float_raster.to_file(str(tmp_path / "narrow.png"))
        narrowing = [w for w in caught if issubclass(w.category, DtypeNarrowingWarning)]
        assert narrowing, "a lossy dtype conversion must warn"
        message = str(narrowing[0].message)
        assert "Float32" in message, (
            f"the message must name the source dtype: {message}"
        )
        assert "PNG" in message, f"the message must name the driver: {message}"

    def test_a_carryable_dtype_does_not_warn(self, byte_raster, tmp_path):
        """An 8-bit raster to PNG is lossless, so nothing is reported.

        Args:
            byte_raster: A uint8 source.
            tmp_path: Temporary directory fixture.

        Test scenario:
            Writing an 8-bit image to PNG is a legitimate thing to do; warning
            on it would train callers to filter the category out.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            byte_raster.to_file(str(tmp_path / "fine.png"))
        assert not [
            w for w in caught if issubclass(w.category, DtypeNarrowingWarning)
        ], "a dtype the driver carries must not warn"


class TestCollectionToFileNormalisesTheDriver:
    """The collection sibling accepts what `Dataset.to_file` accepts."""

    @pytest.fixture
    def collection(self, float_raster):
        """A two-timestep in-memory collection.

        Args:
            float_raster: The template raster.

        Returns:
            DatasetCollection: Two timesteps sharing one grid.
        """
        from pyramids.dataset.collection import DatasetCollection

        return DatasetCollection.from_dataset(float_raster, 2)

    @pytest.mark.parametrize("driver", ["geotiff", "GTiff"])
    def test_a_catalog_key_or_a_gdal_name_both_work(self, collection, tmp_path, driver):
        """Both spellings resolve, as they do for `Dataset.to_file`.

        Args:
            collection: The collection under test.
            tmp_path: Temporary directory fixture.
            driver: The driver spelling under test.

        Test scenario:
            `Dataset.to_file` accepts a catalog key or a GDAL short name; the
            collection sibling crashed with an unhandled `AttributeError` on
            the latter, because `get_driver` returned None and was then
            dereferenced.
        """
        out = tmp_path / driver
        collection.to_file(str(out), driver=driver)
        assert sorted(p.name for p in out.iterdir()) == ["0.tif", "1.tif"], (
            f"driver={driver!r} did not write the expected files"
        )

    def test_an_unknown_driver_is_refused(self, collection, tmp_path):
        """A name that is neither a catalog key nor a GDAL driver raises.

        Args:
            collection: The collection under test.
            tmp_path: Temporary directory fixture.

        Test scenario:
            The normalisation must not turn an outright typo into a confusing
            `AttributeError` further down.
        """
        with pytest.raises(DriverNotExistError, match="not in the driver catalog"):
            collection.to_file(str(tmp_path / "bad"), driver="NotADriver")

    def test_an_explicit_path_list_needs_no_driver_extension(
        self, collection, tmp_path
    ):
        """A driver with no catalogued extension is fine when paths are given.

        Args:
            collection: The collection under test.
            tmp_path: Temporary directory fixture.

        Test scenario:
            Only the directory branch derives file names from the driver, so
            only it needs an extension. Checking earlier refused `cog` even
            here -- while advising the caller to "pass an explicit list of
            paths", which is exactly what they had done.
        """
        paths = [str(tmp_path / f"g{i}.tif") for i in range(collection.time_length)]
        collection.to_file(paths, driver="cog")
        for path in paths:
            assert Path(path).exists(), f"{path} was not written"

    def test_a_driver_with_no_extension_is_refused(self, collection, tmp_path):
        """A key with no catalogued extension raises instead of writing "0.None".

        Args:
            collection: The collection under test.
            tmp_path: Temporary directory fixture.

        Test scenario:
            `cog` is a real catalog key with no `extension`, so filename
            construction produced files literally named `0.None`.
        """
        with pytest.raises(DriverNotExistError, match="no file extension"):
            collection.to_file(str(tmp_path / "cogdir"), driver="cog")


class TestTheDtypeCheckAsksTheDriverRatherThanItsMetadata:
    """Capability comes from a probe, because the advertised list lies."""

    def test_an_unresolvable_driver_name_is_ignored(self, float_raster):
        """A driver GDAL does not know is skipped, not crashed on.

        Args:
            float_raster: Any source raster.

        Test scenario:
            The check runs before the write and must never be the thing that
            fails it. A name GDAL cannot resolve is somebody else's error to
            report, with a better message than this helper could give.
        """
        from pyramids.dataset.ops.io import _warn_if_driver_narrows_dtype

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_driver_narrows_dtype(float_raster, "NoSuchDriver", "x.zzz")
        assert not caught, f"an unknown driver must not warn: {caught}"

    def test_a_bandless_dataset_is_ignored(self, float_raster, mocker):
        """A handle with no bands has no dtype to judge.

        Args:
            float_raster: A source raster, whose handle is stubbed.
            mocker: pytest-mock fixture.

        Test scenario:
            The check dereferenced `GetRasterBand(1)` unguarded, which is a
            crash on a container or an emptied handle rather than the silence
            an advisory check owes.
        """
        from pyramids.dataset.ops.io import _warn_if_driver_narrows_dtype

        raster = mocker.Mock()
        raster.RasterCount = 0
        mocker.patch.object(type(float_raster), "raster", raster)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_driver_narrows_dtype(float_raster, "PNG", "x.png")
        assert not caught, f"a bandless dataset must not warn: {caught}"

    @pytest.mark.parametrize(
        "driver, dtype, preserved",
        [
            ("GTiff", gdal.GDT_Int64, True),
            ("GTiff", gdal.GDT_Float64, True),
            ("PNG", gdal.GDT_Float32, False),
            ("PNG", gdal.GDT_Byte, True),
        ],
    )
    def test_the_probe_answers_what_the_driver_actually_does(
        self, driver, dtype, preserved
    ):
        """Capability is measured, not read off `DMD_CREATIONDATATYPES`.

        Args:
            driver: The GDAL driver short name.
            dtype: The `gdal.GDT_*` code under test.
            preserved: Whether the driver stores it unchanged.

        Test scenario:
            The advertised list is not exhaustive -- this build's GTiff omits
            `Int64` and stores it faithfully anyway. Trusting the metadata made
            the check warn about the commonest write in the library, wrongly.
        """
        from pyramids.dataset.ops.io import _driver_preserves_dtype

        assert _driver_preserves_dtype(driver, dtype) is preserved, (
            f"{driver} + {gdal.GetDataTypeName(dtype)}: probe disagrees"
        )

    def test_gtiff_advertises_no_int64_but_stores_it(self):
        """The exact metadata gap that caused the false positive.

        Test scenario:
            Pinned so a future GDAL that *does* advertise Int64 cannot quietly
            turn this into a tautology -- if the advertisement changes, the
            assertion about the gap fails and says so.
        """
        from pyramids.dataset.ops.io import _driver_preserves_dtype

        advertised = gdal.GetDriverByName("GTiff").GetMetadataItem(
            "DMD_CREATIONDATATYPES"
        )
        if "Int64" in (advertised or "").split():
            # A newer GDAL closed the gap. That is good news, not a failure:
            # the probe still answers correctly, there is simply no longer a
            # discrepancy to demonstrate.
            pytest.skip("this GDAL advertises Int64; the metadata gap is closed")
        assert _driver_preserves_dtype("GTiff", gdal.GDT_Int64) is True, (
            "GTiff stores Int64 despite not advertising it"
        )


class TestTheNarrowingWarningDoesNotCryWolf:
    """The two false positives the warning shipped with, and the true one."""

    @pytest.mark.parametrize(
        "array, extension",
        [
            (np.arange(16).reshape(4, 4), "tif"),
            (np.full((4, 4), 1 / 3, dtype="float64"), "asc"),
            (np.full((4, 4), 1 / 3, dtype="float64"), "tif"),
            (np.arange(16, dtype="uint8").reshape(4, 4), "png"),
        ],
        ids=["int64-tif", "float64-asc", "float64-tif", "uint8-png"],
    )
    def test_a_lossless_write_is_silent(self, tmp_path, array, extension):
        """No warning when the target stores the dtype unchanged.

        Args:
            tmp_path: Temporary directory fixture.
            array: The source array.
            extension: The destination extension.

        Test scenario:
            `int64 -> .tif` is the library's commonest write (NumPy's default
            integer dtype) and round-trips exactly; `.asc` never reaches a GDAL
            driver at all, since `_io.to_ascii` writes full precision with
            `str()`. Both warned, and the advice they gave -- convert
            deliberately -- would have corrupted data that was fine.
        """
        ds = Dataset.from_array(array, geo_ref=GEO_REF)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds.to_file(str(tmp_path / f"out.{extension}"))
        narrowing = [w for w in caught if issubclass(w.category, DtypeNarrowingWarning)]
        assert not narrowing, (
            f"{array.dtype} -> .{extension} is lossless but warned: "
            f"{[str(w.message) for w in narrowing]}"
        )

    def test_the_asc_write_keeps_full_float64_precision(self, tmp_path):
        """The claim behind skipping the ascii branch, asserted not assumed.

        Args:
            tmp_path: Temporary directory fixture.

        Test scenario:
            The check is skipped for `.asc` because that branch bypasses GDAL
            entirely. If it ever stopped writing full precision, skipping would
            become the wrong call -- so the precision is pinned here.
        """
        out = tmp_path / "p.asc"
        Dataset.from_array(
            np.full((4, 4), 1 / 3, dtype="float64"), geo_ref=GEO_REF
        ).to_file(str(out))
        assert "0.3333333333333333" in out.read_text(), (
            "the ascii writer no longer keeps float64 precision"
        )

    def test_a_genuinely_lossy_write_still_warns(self, tmp_path):
        """The case the warning exists for is unaffected.

        Args:
            tmp_path: Temporary directory fixture.
        """
        ds = Dataset.from_array(
            np.arange(16, dtype="float32").reshape(4, 4), geo_ref=GEO_REF
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds.to_file(str(tmp_path / "lossy.png"))
        narrowing = [w for w in caught if issubclass(w.category, DtypeNarrowingWarning)]
        assert narrowing, "float32 -> PNG must still warn"
        assert "Float32" in str(narrowing[0].message)

    def test_the_warning_blames_the_caller_not_pyramids(self, tmp_path):
        """`stacklevel` points at the user's `to_file` line.

        Args:
            tmp_path: Temporary directory fixture.

        Test scenario:
            A warning attributed to pyramids' own source tells the reader
            nothing about where their write is. The depth is easy to get wrong:
            the engine facade inserts a `wrapper` frame a hand count misses.
        """
        ds = Dataset.from_array(
            np.arange(16, dtype="float32").reshape(4, 4), geo_ref=GEO_REF
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds.to_file(str(tmp_path / "blame.png"))
        narrowing = [w for w in caught if issubclass(w.category, DtypeNarrowingWarning)]
        assert narrowing, "expected the narrowing warning"
        blamed = Path(narrowing[0].filename).name
        assert blamed == Path(__file__).name, (
            f"the warning should blame this test file, got {blamed}"
        )


class TestForCopyIsNarrowerThanItLooks:
    """`for_copy=True` is only safe on a single-path, write-once method."""

    def test_a_method_that_writes_after_copying_refuses_a_copy_only_format(
        self, byte_raster, tmp_path
    ):
        """`change_no_data_value` refuses `.png` even though it uses CreateCopy.

        Args:
            byte_raster: An 8-bit source PNG could otherwise carry.
            tmp_path: Temporary directory fixture.

        Test scenario:
            It copies and *then* streams the no-data swap into the clone, and a
            copy-only driver hands back a read-only handle -- so `.png` passed
            the relaxed gate and died with `ReadOnlyError: The Dataset is open
            with a read only`, an error naming nothing about the format. The
            strict gate refuses it up front with one that does.
        """
        with pytest.raises(FileFormatNotSupportedError):
            byte_raster.change_no_data_value(2, 1, path=str(tmp_path / "n.png"))

    @pytest.mark.parametrize("method", ["last", "min"])
    def test_merge_refuses_a_copy_only_format_for_every_method(self, tmp_path, method):
        """`merge_rasters` answers alike whichever internal path `method` picks.

        Args:
            tmp_path: Temporary directory fixture.
            method: The merge method under test.

        Test scenario:
            The z-order path writes with `gdal.Translate` and could produce a
            PNG; the reduction path builds with `Create` and could not. Letting
            `method` decide what `dst` may be is the same defect as letting it
            decide the format, which is what H2 was filed for.
        """
        from pyramids.dataset.merge import merge_rasters

        sources = []
        for i, x in enumerate((0.0, 4.0)):
            path = tmp_path / f"src{i}.tif"
            Dataset.from_array(
                np.full((4, 4), i + 1, dtype="uint8"),
                geo_ref=GeoReference(
                    top_left_corner=(x, 4.0), cell_size=1.0, epsg=4326
                ),
                path=path,
            ).close()
            sources.append(str(path))
        with pytest.raises(FileFormatNotSupportedError):
            merge_rasters(sources, str(tmp_path / f"m.{method}.png"), method=method)

    @pytest.mark.parametrize("extension", ["png", "jpg"])
    def test_a_single_path_write_once_method_still_accepts_them(
        self, byte_raster, tmp_path, extension
    ):
        """`copy` and `translate` keep the relaxed gate, and should.

        Args:
            byte_raster: An 8-bit source.
            tmp_path: Temporary directory fixture.
            extension: A copy-only format.

        Test scenario:
            Each has exactly one write path and never writes after copying, so
            the refusal would be a capability they actually have -- and for
            `translate` it is the documented feature.
        """
        copied = tmp_path / f"c.{extension}"
        byte_raster.copy(path=str(copied))
        assert copied.exists(), f"{copied} was not written"
        translated = tmp_path / f"t.{extension}"
        byte_raster.translate(path=str(translated))
        assert translated.exists(), f"{translated} was not written"


class TestEveryWriterAgreesOnAReferenceOnlyFormat:
    """`.vrt` is refused by whichever writer you reach for."""

    @pytest.mark.parametrize(
        "writer",
        ["to_file", "copy", "translate"],
    )
    def test_a_vrt_destination_is_refused(self, float_raster, tmp_path, writer):
        """No public writer produces a `.vrt`.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.
            writer: The method under test.

        Test scenario:
            The refusal was added to `resolve_output_driver`, but `to_file` had
            its own second resolver with no format gate at all -- so the same
            destination was illegal for `copy` and legal for `to_file`, which
            then wrote a VRT with an empty `<SourceFilename>` that GDAL refuses
            to reopen, and reported success. Exercised through the public API
            rather than the resolver, which is how that gap survived.
        """
        out = tmp_path / f"{writer}.vrt"
        method = getattr(float_raster, writer)
        # Bound outside the block so only the call under test can throw inside
        # it -- `to_file` takes the path positionally, the other two by keyword.
        call = (
            partial(method, str(out))
            if writer == "to_file"
            else partial(method, path=str(out))
        )
        with pytest.raises(FileFormatNotSupportedError):
            call()

    def test_nothing_is_left_on_disk_when_it_is_refused(self, float_raster, tmp_path):
        """The refusal happens before anything is written.

        Args:
            float_raster: The source raster.
            tmp_path: Temporary directory fixture.
        """
        out = tmp_path / "nothing.vrt"
        with pytest.raises(FileFormatNotSupportedError):
            float_raster.to_file(str(out))
        assert not out.exists(), f"{out} should not have been created"


class TestToFileLabelsAccessFromTheDriver:
    """A copy that cannot be written back must not claim it can."""

    @pytest.mark.parametrize(
        "extension, expected_access",
        [("tif", "write"), ("png", "read_only")],
    )
    def test_the_access_mode_matches_the_handle(
        self, byte_raster, tmp_path, extension, expected_access
    ):
        """`to_file` derives the access mode instead of asserting `"write"`.

        Args:
            byte_raster: An 8-bit source, writable to both formats.
            tmp_path: Temporary directory fixture.
            extension: The destination extension.
            expected_access: The access mode the result should report.

        Test scenario:
            `copy` and `translate` were fixed to derive this; `to_file` still
            asserted `"write"` unconditionally, so a PNG result claimed to be
            writable and `write_array` leaked a raw GDAL error past the
            package's own guard.
        """
        out = tmp_path / f"a.{extension}"
        byte_raster.to_file(str(out))
        assert byte_raster.access == expected_access, (
            f".{extension} should report access={expected_access}, "
            f"got {byte_raster.access}"
        )

    def test_writing_to_a_read_only_result_raises_the_package_error(
        self, byte_raster, tmp_path
    ):
        """The guard fires instead of GDAL.

        Args:
            byte_raster: An 8-bit source.
            tmp_path: Temporary directory fixture.
        """
        byte_raster.to_file(str(tmp_path / "ro.png"))
        with pytest.raises(ReadOnlyError):
            byte_raster.write_array(np.zeros((4, 4), dtype="uint8"))

    def test_the_dataset_is_repointed_at_the_written_file(self, byte_raster, tmp_path):
        """A copy-only write still repoints the dataset.

        Args:
            byte_raster: An 8-bit source.
            tmp_path: Temporary directory fixture.

        Test scenario:
            The read-only reopen was unreachable, then reachable but
            unguarded -- both times leaving the dataset on its old in-memory
            handle with `file_name == ''` and no indication anything was
            skipped.
        """
        out = tmp_path / "repointed.png"
        byte_raster.to_file(str(out))
        assert byte_raster.raster.GetDriver().ShortName == "PNG"
        assert Path(byte_raster.file_name).name == out.name, (
            f"expected file_name to name {out.name}, got {byte_raster.file_name!r}"
        )
