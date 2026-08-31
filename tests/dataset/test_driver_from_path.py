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

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import DriverNotExistError, FileFormatNotSupportedError
from pyramids.dataset import Dataset, GeoReference
from pyramids.errors import DtypeNarrowingWarning

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326)


@pytest.fixture(scope="function")
def float_raster() -> Dataset:
    """A small in-memory float32 raster.

    Returns:
        Dataset: 4x4 single-band float32 at EPSG:4326.
    """
    return Dataset.from_array(
        np.arange(16, dtype="float32").reshape(4, 4), geo_ref=GEO_REF
    )


@pytest.fixture(scope="function")
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

    @pytest.fixture(scope="function")
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

    @pytest.fixture(scope="function")
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
            collection.to_file(str(tmp_path / "cog"), driver="cog")


class TestTheDtypeCheckIsQuietWhenItCannotAnswer:
    """The narrowing check declines rather than guessing."""

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

    def test_a_driver_advertising_no_type_list_is_ignored(self, float_raster, mocker):
        """A driver with no `DMD_CREATIONDATATYPES` is left alone.

        Args:
            float_raster: A float32 source.
            mocker: pytest-mock fixture.

        Test scenario:
            Every driver in this GDAL build happens to advertise a type list,
            so the empty case is forced. With nothing to compare against,
            warning would be a guess about a conversion the check cannot
            actually predict.
        """
        from pyramids.dataset.ops import io as io_ops

        driver = mocker.Mock()
        driver.GetMetadataItem.return_value = None
        mocker.patch.object(io_ops.gdal, "GetDriverByName", return_value=driver)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            io_ops._warn_if_driver_narrows_dtype(float_raster, "Whatever", "x.tif")
        assert not caught, f"a driver with no advertised types must not warn: {caught}"
