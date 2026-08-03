"""Tests for the Dataset class overview methods."""

import contextlib
import pickle
import shutil
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.dataset.engines.io import _DESCRIPTION_EXCERPT, IO
from pyramids.errors import OverviewTargetError, PyramidsError, ReadOnlyError
from pyramids.netcdf import Container, NetCDF

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


def _plain_raster(tmp_path, name: str) -> str:
    """Write a georeferenced 2-band raster that carries no overviews at all.

    Complements :func:`_overviewed_raster` for the cases that must start from an empty
    overview count — a VRT over an overviewed source inherits the source's levels, which
    hides both the "nothing to regenerate" warning and a sidecar of the VRT's own.
    """
    path = str(tmp_path / name)
    raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 2, gdal.GDT_Float32)
    raster.SetGeoTransform((10.0, 0.5, 0.0, 80.0, 0.0, -0.5))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    raster.SetProjection(srs.ExportToWkt())
    for index in range(2):
        band = raster.GetRasterBand(index + 1)
        band.WriteArray(np.full((64, 64), float(index + 1), dtype="float32"))
    raster = None
    return path


def _raster_with_overviews(
    tmp_path,
    name: str,
    bands: list[np.ndarray],
    levels: list[int],
    resampling: str = "AVERAGE",
    no_data_value: float | None = None,
) -> str:
    """Write a georeferenced raster holding ``bands`` and build ``levels`` overviews on it.

    Complements :func:`_overviewed_raster` for the cases that need the band values, the
    number of levels, the resampling method or a declared no-data value picked per test
    rather than fixed.
    """
    path = str(tmp_path / name)
    rows, columns = bands[0].shape
    raster = gdal.GetDriverByName("GTiff").Create(
        path, columns, rows, len(bands), gdal.GDT_Float32
    )
    raster.SetGeoTransform((10.0, 0.5, 0.0, 80.0, 0.0, -0.5))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    raster.SetProjection(srs.ExportToWkt())
    for index, values in enumerate(bands):
        band = raster.GetRasterBand(index + 1)
        if no_data_value is not None:
            band.SetNoDataValue(no_data_value)
        band.WriteArray(values)
    raster = None

    handle = gdal.Open(path, gdal.GA_Update)
    handle.BuildOverviews(resampling, levels)
    handle = None
    return path


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    """Average ``values`` over non-overlapping ``factor`` by ``factor`` blocks.

    Both dimensions of ``values`` must divide by ``factor``; the reshape below silently
    misaligns the blocks otherwise, so callers pick power-of-two shapes.
    """
    rows, columns = values.shape
    averaged = (
        values.astype("float64")
        .reshape(rows // factor, factor, columns // factor, factor)
        .mean(axis=(1, 3))
    )
    return averaged


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


def _mixed_ownership_vrt(tmp_path) -> str:
    """Build a 2-band VRT whose band 1 *stores* its level and whose band 2 *computes* it.

    Band 1 sources a raster carrying an external `.ovr` and names that sidecar in an
    explicit `<Overview>`, so GDAL serves its level from the sidecar GTiff. Band 2 has no
    `<Overview>` of its own and takes its level from the dataset-level `<OverviewList>`,
    which GDAL serves from the VRT. One handle, one level per band, two different owners
    — the shape a per-band classifier and a dataset-wide one disagree about.
    """
    driver = gdal.GetDriverByName("GTiff")
    sources = []
    for name in ("stored_owner.tif", "computed_owner.tif"):
        path = str(tmp_path / name)
        raster = driver.Create(path, 64, 64, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((10.0, 0.5, 0.0, 80.0, 0.0, -0.5))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        raster.SetProjection(srs.ExportToWkt())
        raster.GetRasterBand(1).WriteArray(np.full((64, 64), 1.0, dtype="float32"))
        raster = None
        sources.append(path)
    stored, computed = sources
    # Read-only sends the levels to an external `.ovr`, which is the GTiff that then owns
    # band 1's level; the update mode used for `computed` keeps them internal.
    handle = gdal.Open(stored, gdal.GA_ReadOnly)
    handle.BuildOverviews("AVERAGE", [2])
    handle = None
    handle = gdal.Open(computed, gdal.GA_Update)
    handle.BuildOverviews("AVERAGE", [2])
    handle = None

    vrt = tmp_path / "mixed_ownership.vrt"
    vrt.write_text(
        f'<VRTDataset rasterXSize="64" rasterYSize="64">'
        f'<VRTRasterBand dataType="Float32" band="1">'
        f'<SimpleSource><SourceFilename relativeToVRT="0">{stored}</SourceFilename>'
        f"<SourceBand>1</SourceBand></SimpleSource>"
        f'<Overview><SourceFilename relativeToVRT="0">{stored}.ovr</SourceFilename>'
        f"<SourceBand>1</SourceBand></Overview></VRTRasterBand>"
        f'<VRTRasterBand dataType="Float32" band="2">'
        f'<SimpleSource><SourceFilename relativeToVRT="0">{computed}</SourceFilename>'
        f"<SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>"
        f"<OverviewList>2</OverviewList></VRTDataset>"
    )
    return str(vrt)


def _level_owner_driver(band: gdal.Band, index: int = 0) -> str | None:
    """Name the driver of the dataset owning `band`'s overview `index`, or None."""
    owner = band.GetOverview(index).GetDataset()
    driver = owner.GetDriver() if owner is not None else None
    return None if driver is None else driver.ShortName


def _refuse_to_resolve(self, *args) -> gdal.Band:
    """Stand in for `gdal.Band.GetOverview` and raise GDAL's write refusal instead.

    Fails *before* a single level is resolved, so the classification that follows has no
    levels of this band to look at — the shape the level list's pre-binding exists for.
    """
    raise RuntimeError("Attempt to write to a read only dataset")


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


# The exact stranding -- levels dropped, a file named `.ovr` left in the working directory
# -- was measured on GDAL 3.13.1. The win_arm64 wheel ships 3.12.4 (the vcpkg port ceiling),
# so the assertions that pin GDAL's own misbehaviour are gated the way #892 gated the
# equivalent NetCDF sidecar assertions. The guard itself is not gated; only this canary.
_GDAL_STRANDS_PATHLESS_VRT_OVERVIEWS = int(gdal.VersionInfo("VERSION_NUM")) >= 3130000

# Same ceiling, different capability: on 3.12.4 `create_overviews()` on a read-only NetCDF
# variable view builds queryable levels but writes no external `<container>.0.ovr` sidecar.
# Only the sidecar *file* assertion is version-gated -- the level count and the "no stray
# `.ovr`" check are platform-independent and are what prove the guard does not over-fire,
# so they stay ungated. See tests/netcdf/samples/test_inherited_ops.py, which #892 had to
# gate for exactly this.
_GDAL_WRITES_EXTERNAL_NETCDF_OVR = int(gdal.VersionInfo("VERSION_NUM")) >= 3130000


@pytest.fixture
def pathless_level(tmp_path, monkeypatch):
    """Yield `(level, parent)` for a path-less VRT level, with the CWD isolated.

    The shape most of the guard tests need: `get_overview_dataset` describes the level
    lazily as a VRT with an empty description, which is exactly what the refusal targets.
    The parent comes with it because several tests need a handle the guard does *not*
    refuse, to contrast against.
    """
    monkeypatch.chdir(tmp_path)
    source = _overviewed_raster(tmp_path, "pathless_level_src.tif")
    dataset = Dataset.read_file(source)
    level = None
    try:
        level = dataset.get_overview_dataset(overview_index=0)
        yield level, dataset
    finally:
        if level is not None:
            level.close()
        dataset.close()


class TestCreateOverviewsPathlessGuard:
    """Overview targets a dataset cannot hold, across both build and regenerate.

    Two related refusals live here, since they share the predicates and the message:

    - **#917** — `create_overviews` on a plain VRT with nowhere to put a sidecar. GDAL
      names an external `.ovr` after the dataset description; a plain VRT owns no pixel
      storage, so with an unusable description GDAL wrote a file called literally `.ovr`
      into the working directory, attached nothing, and dropped the levels the handle
      already exposed. A warped VRT is exempt here: it keeps its levels in RAM.
    - **#922** — `recreate_overviews` on a level a VRT *computes* rather than stores (a
      warped band, or one inherited from the source). GDAL refuses those with the same
      `CPLE_NoWriteAccess` it uses for a read-only dataset, so they were misreported as an
      access-mode problem with advice the caller could not act on.

    The cases that must keep working — `MEM`, path-ful and `/vsimem/` VRTs, regular-grid
    NetCDF views, and stored levels in a genuinely read-only handle — are pinned here too,
    since they are what the discriminators must not over-fire on.
    """

    def test_pathless_vrt_refuses_instead_of_writing_a_stray_sidecar(
        self, tmp_path, monkeypatch
    ):
        """A path-less VRT raises, keeps its levels, and leaves the CWD alone.

        Test scenario:
            `get_overview_dataset` returns a VRT-described level with no path — expected:
            `OverviewTargetError` naming the cause, `overview_count` unchanged (it
            previously went 1 -> 0), and no `.ovr` in the working directory.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "guard_src.tif")
        dataset = Dataset.read_file(source)
        level = None
        try:
            level = dataset.get_overview_dataset(overview_index=0)
            assert level.driver_type == "vrt", "precondition: the level is a VRT"
            before = list(level.overview_count)
            with pytest.raises(OverviewTargetError, match="nowhere to go"):
                level.create_overviews(overview_levels=[2])
            assert level.overview_count == before, (
                f"the refusal must not drop the levels, {before} -> {level.overview_count}"
            )
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            if level is not None:
                level.close()
            dataset.close()

    def test_in_memory_dataset_still_builds_overviews(self, tmp_path, monkeypatch):
        """A MEM raster is unaffected: it stores its overviews internally.

        Test scenario:
            `create_from_array` also has an empty description, so guarding on that alone
            would break it — expected: the build still succeeds.
        """
        monkeypatch.chdir(tmp_path)
        dataset = Dataset.create_from_array(
            np.arange(4096, dtype="float32").reshape(64, 64),
            top_left_corner=(0.0, 64.0),
            cell_size=1.0,
            epsg=4326,
        )
        try:
            dataset.create_overviews(overview_levels=[2])
            assert dataset.overview_count == [1], (
                f"a MEM raster should still build, got {dataset.overview_count}"
            )
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            dataset.close()

    @pytest.mark.parametrize("method", ["to_crs", "warped_view"])
    def test_pathless_warped_vrt_still_builds_overviews(
        self, tmp_path, monkeypatch, method
    ):
        """A warped VRT has no path either, but holds its overviews in RAM.

        Test scenario:
            `to_crs` / `warped_view` return a `subClass="VRTWarpedDataset"` handle with an
            empty description — expected: the build succeeds, the levels are readable, and
            nothing is written to the working directory.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, f"warped_src_{method}.tif")
        dataset = Dataset.read_file(source)
        view = None
        try:
            view = getattr(dataset, method)(3857)
            assert view.driver_type == "vrt", "precondition: the warped view is a VRT"
            assert not view.raster.GetDescription(), "precondition: it has no path"
            view.create_overviews("average", overview_levels=[2])
            assert all(count > 0 for count in view.overview_count), (
                f"a warped VRT should still build, got {view.overview_count}"
            )
            level = view.read_overview_array(band=0, overview_index=0)
            assert level.size > 0 and np.isfinite(level).any(), (
                "the built level must carry readable data"
            )
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            if view is not None:
                view.close()
            dataset.close()

    def test_netcdf_variable_view_still_builds_overviews(self, tmp_path, monkeypatch):
        """A file-backed NetCDF view over a *regular* grid is not VRT-wrapped, so it builds.

        Test scenario:
            A regular grid keeps the classic view out of index space, so no VRT wrapper is
            installed — expected: the build succeeds and the sidecar is named after the
            container, not dropped in the working directory. The index-space counterpart
            is VRT-wrapped and refused; see `test_index_space_netcdf_view_is_refused`.
        """
        monkeypatch.chdir(tmp_path)
        source = tmp_path / "guard_var.nc"
        container = NetCDF.create_from_array(
            arr=np.random.default_rng(11).random((16, 16)).astype(np.float64),
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
            no_data_value=-9999.0,
            variable_name="elevation",
        )
        container.to_file(str(source))
        container.close()
        backed = NetCDF.read_file(str(source))
        view = backed.get_variable("elevation")
        try:
            assert not view.raster.GetDescription(), "precondition: the view has no path"
            view.create_overviews(overview_levels=[2])
            assert all(count > 0 for count in view.overview_count), (
                f"a NetCDF variable view should still build, got {view.overview_count}"
            )
            if _GDAL_WRITES_EXTERNAL_NETCDF_OVR:
                sidecars = sorted(p.name for p in tmp_path.glob("*.ovr"))
                assert sidecars == ["guard_var.nc.0.ovr"], (
                    f"the sidecar must be named after the container, found {sidecars}"
                )
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            view.close()
            backed.close()

    def test_the_refusal_is_distinguishable_from_an_argument_error(self, pathless_level):
        """The refusal is its own type, but is still caught as a `ValueError`.

        Test scenario:
            An argument error is worth retrying with different arguments; this one is a
            property of the dataset — expected: `OverviewTargetError` for the dataset, a
            bare `ValueError` for a bad level, and the former still caught by a handler
            written for the latter.
        """
        level, parent = pathless_level
        with pytest.raises(OverviewTargetError):
            level.create_overviews(overview_levels=[2])
        with pytest.raises(ValueError) as excinfo:
            parent.create_overviews(overview_levels=[3])
        assert not isinstance(excinfo.value, OverviewTargetError), (
            "a bad level is an argument error, not an unusable target"
        )

    def test_vsimem_vrt_still_builds_overviews(self, tmp_path, monkeypatch):
        """A `/vsimem/` VRT has a real path, so it names its sidecar after it.

        Test scenario:
            The guard's docstring and the migration entry both single this out as
            unaffected — expected: the build succeeds and writes the sidecar in the
            virtual filesystem, not into the working directory.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "vsimem_src.tif")
        vsi_path = "/vsimem/guard_vsimem.vrt"
        gdal.Translate(vsi_path, str(source), format="VRT").FlushCache()
        dataset = Dataset.read_file(vsi_path)
        try:
            dataset.create_overviews(overview_levels=[2])
            assert all(count > 0 for count in dataset.overview_count), (
                f"a /vsimem/ VRT should still build, got {dataset.overview_count}"
            )
            assert gdal.VSIStatL(f"{vsi_path}.ovr") is not None, (
                "the sidecar must land in /vsimem/ beside the VRT"
            )
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            dataset.close()
            gdal.Unlink(f"{vsi_path}.ovr")
            gdal.Unlink(vsi_path)

    def test_recreate_overviews_refuses_with_the_same_diagnosis(
        self, pathless_level, tmp_path
    ):
        """The sibling method refuses the same shape instead of blaming the access mode.

        Test scenario:
            A pathless VRT reaching GDAL was refused as a write and diagnosed as
            read-only, advising a reopen that a handle with no path cannot perform —
            expected: the same actionable `OverviewTargetError` `create_overviews` gives.
        """
        level, _ = pathless_level
        with pytest.raises(OverviewTargetError, match="nowhere to go") as excinfo:
            level.recreate_overviews()
        assert not isinstance(excinfo.value, ReadOnlyError), (
            "the access mode is not the blocker; a pathless handle cannot be reopened"
        )
        assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"

    def test_index_space_netcdf_view_is_refused(self, tmp_path, monkeypatch):
        """An index-space NetCDF variable view is VRT-wrapped, so it is refused.

        Test scenario:
            An irregular `lon` defeats the geotransform guess, so the classic view comes
            back in index space and `_georeference_index_subset` wraps it in a plain
            pathless VRT — expected: the same refusal as any other pathless VRT, since
            this shape previously produced the stray-sidecar damage.
        """
        monkeypatch.chdir(tmp_path)
        store = gdal.GetDriverByName("MEM").CreateMultiDimensional("m")
        root = store.GetRootGroup()
        y_dim = root.CreateDimension("lat", None, None, 4)
        x_dim = root.CreateDimension("lon", None, None, 5)
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float32)
        lat = root.CreateMDArray("lat", [y_dim], dtype)
        lat.WriteArray(np.array([1.0, 2.0, 3.0, 4.0], "f4"))
        lat.SetUnit("degrees_north")
        lon = root.CreateMDArray("lon", [x_dim], dtype)
        lon.WriteArray(np.array([1.0, 2.0, 4.0, 8.0, 16.0], "f4"))
        lon.SetUnit("degrees_east")
        y_dim.SetIndexingVariable(lat)
        x_dim.SetIndexingVariable(lon)
        root.CreateMDArray("v", [y_dim, x_dim], dtype).WriteArray(
            np.arange(20, dtype="f4").reshape(4, 5)
        )
        view = Container(store).get_variable("v")
        try:
            assert view.raster.GetDriver().ShortName == "VRT", (
                "precondition: the index-space view is VRT-wrapped"
            )
            with pytest.raises(OverviewTargetError, match="nowhere to go"):
                view.create_overviews(overview_levels=[2])
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            view.close()
            del store

    def test_inline_xml_vrt_is_refused_too(self, tmp_path, monkeypatch):
        """An inline-XML VRT has no path either, despite a non-empty description.

        Test scenario:
            GDAL stores the XML document verbatim as the description — expected: the same
            actionable `OverviewTargetError`, not a raw GDAL `RuntimeError` naming a file
            whose name is the whole document.
        """
        monkeypatch.chdir(tmp_path)
        xml = (
            '<VRTDataset rasterXSize="64" rasterYSize="64">'
            '<VRTRasterBand dataType="Float32" band="1"/>'
            "</VRTDataset>"
        )
        dataset = Dataset.read_file(xml)
        try:
            assert dataset.raster.GetDescription().startswith("<"), (
                "precondition: the description is the XML document"
            )
            with pytest.raises(OverviewTargetError, match="nowhere to go"):
                dataset.create_overviews(overview_levels=[2])
        finally:
            dataset.close()

    def test_vrt_with_a_path_still_builds_overviews(self, tmp_path, monkeypatch):
        """A VRT that has a real path names its sidecar after it and is left alone.

        Test scenario:
            The guard keys on a missing description, so a saved `.vrt` is the over-fire
            canary — expected: the build succeeds and writes `<name>.vrt.ovr` beside it.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "vrt_path_src.tif")
        vrt_path = tmp_path / "with_path.vrt"
        gdal.Translate(str(vrt_path), str(source), format="VRT").FlushCache()
        dataset = Dataset.read_file(str(vrt_path))
        try:
            dataset.create_overviews(overview_levels=[2])
            assert (tmp_path / "with_path.vrt.ovr").exists(), (
                "the sidecar must be named after the VRT, which is the point of the case"
            )
            assert dataset.overview_count == [1, 1, 1], (
                f"a path-ful VRT should still build, got {dataset.overview_count}"
            )
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            dataset.close()

    @pytest.mark.skipif(
        not _GDAL_STRANDS_PATHLESS_VRT_OVERVIEWS,
        reason="the stranding behaviour was measured on GDAL >= 3.13",
    )
    def test_raw_build_overviews_strands_the_levels_without_the_guard(
        self, tmp_path, monkeypatch
    ):
        """Pin the #917 damage itself, so the guard is not silently obsoleted.

        Test scenario:
            Call `BuildOverviews` straight on the GDAL handle, bypassing the guard —
            expected: the level count collapses to 0 and a file named `.ovr` appears in
            the working directory. A GDAL release that fixes this fails this test, which
            is the intended signal that the guard can go.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "raw_src.tif")
        dataset = Dataset.read_file(source)
        level = None
        try:
            level = dataset.get_overview_dataset(overview_index=0)
            before = list(level.overview_count)
            assert before == [1, 1, 1], (
                f"precondition: every band exposes one overview, {before}"
            )
            level.raster.BuildOverviews("nearest", [2])
            assert level.overview_count == [0, 0, 0], (
                "GDAL no longer strands a pathless VRT's levels — the guard may be obsolete"
            )
            assert (tmp_path / ".ovr").exists(), (
                "GDAL no longer writes a stray '.ovr' — the guard may be obsolete"
            )
        finally:
            if level is not None:
                level.close()
            dataset.close()

    def test_plain_vrt_wrapping_a_warped_vrt_is_refused(self, tmp_path, monkeypatch):
        """Only the root element's `subClass` exempts a VRT, never a nested source's.

        Test scenario:
            `BuildVRT` over a warped view serialises the warp inline, so
            `subClass="VRTWarpedDataset"` appears in the document but not on the root —
            expected: still refused, because the wrapper itself is a plain VRT with no
            pixel storage and no path. Matching the whole document instead of the root
            would let this one through.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "nested_warp_src.tif")
        dataset = Dataset.read_file(source)
        warped = dataset.to_crs(3857)
        wrapper = Dataset(gdal.BuildVRT("", [warped.raster]))
        try:
            xml = wrapper.raster.GetMetadata("xml:VRT")[0]
            root = xml[: xml.find(">") + 1]
            assert 'subClass="VRTWarpedDataset"' not in root, (
                f"precondition: the wrapper's own root is a plain VRT, got {root}"
            )
            assert 'subClass="VRTWarpedDataset"' in xml, (
                "precondition: the warped source is serialised inside the document"
            )
            with pytest.raises(OverviewTargetError, match="nowhere to go"):
                wrapper.create_overviews(overview_levels=[2])
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            wrapper.close()
            warped.close()
            dataset.close()

    def test_whitespace_only_description_is_refused(self, tmp_path, monkeypatch):
        """A description made of blanks is not a path either, so it is refused too.

        Test scenario:
            Set the pathless level's description to spaces — expected: the same refusal,
            because the guard strips before deciding; an unstripped check would read the
            blanks as a usable path and let GDAL strand the levels.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "blank_desc_src.tif")
        dataset = Dataset.read_file(source)
        level = None
        try:
            level = dataset.get_overview_dataset(overview_index=0)
            level.raster.SetDescription("   ")
            assert level.raster.GetDescription() == "   ", (
                "precondition: the description holds blanks, not a path"
            )
            with pytest.raises(OverviewTargetError, match="nowhere to go"):
                level.create_overviews(overview_levels=[2])
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            if level is not None:
                level.close()
            dataset.close()

    @pytest.mark.parametrize(
        "metadata, blocked",
        [
            (None, True),
            ([], True),
            (['<VRTDataset rasterXSize="8"'], True),
            (['<VRTDataset subClass="VRTWarpedDataset"'], True),
            (
                ['<?xml version="1.0"?><VRTDataset subClass="VRTWarpedDataset" x="1">'],
                False,
            ),
        ],
        ids=[
            "domain-absent",
            "domain-empty",
            "unterminated-plain",
            "unterminated-warped",
            "declaration-prefixed-warped",
        ],
    )
    def test_an_unparseable_xml_document_falls_back_to_refusing(self, metadata, blocked):
        """A root element the predicate cannot read is refused rather than exempted.

        Test scenario:
            GDAL always serialises a well-formed document, so these shapes are driven
            through a stub handle — expected: refusal whenever the root start tag cannot
            be isolated, because failing to prove a handle is warped must not let it
            strand its levels. The declaration-prefixed case is exempted, since anchoring
            on `<VRTDataset` still finds the root.
        """
        handle = MagicMock(spec=gdal.Dataset)
        handle.GetMetadata.return_value = metadata
        handle.GetDescription.return_value = ""
        stub = MagicMock(spec=Dataset, driver_type="vrt", raster=handle)
        engine = IO(stub)
        assert engine._has_nowhere_for_an_overview_sidecar() is blocked, (
            f"expected blocked={blocked} for xml:VRT metadata {metadata!r}"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"overview_levels": 4},
            {"overview_levels": [3, 5]},
            {"resampling_method": "NOPE"},
        ],
        ids=["defaults", "non-list-levels", "unsupported-levels", "unknown-method"],
    )
    def test_the_refusal_precedes_argument_validation(
        self, pathless_level, tmp_path, kwargs
    ):
        """The dataset is the blocker, so no argument spelling reports something else.

        Test scenario:
            Call the pathless level with the default levels and with arguments that would
            each raise on their own (`TypeError`, an unsupported factor, an unknown
            resampling method) — expected: the guard's `OverviewTargetError` every time,
            so a typo'd argument never masks the real blocker.
        """
        level, _ = pathless_level
        with pytest.raises(OverviewTargetError, match="nowhere to go"):
            level.create_overviews(**kwargs)
        assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"

    @pytest.mark.parametrize(
        "driver_type, description",
        [("vrt", "mosaic.vrt"), ("gtiff", "")],
        ids=["path-ful-vrt", "not-a-vrt"],
    )
    def test_a_decidable_handle_never_serialises_the_document(
        self, driver_type, description
    ):
        """A handle the description alone settles is decided without reading `xml:VRT`.

        Test scenario:
            Drive the predicate through a stub that records its calls — expected: no
            `GetMetadata` at all, because serialising a mosaic with many sources costs
            milliseconds and neither a VRT with a real path nor a non-VRT can be blocked.
        """
        handle = MagicMock(spec=gdal.Dataset)
        handle.GetDescription.return_value = description
        stub = MagicMock(spec=Dataset, driver_type=driver_type, raster=handle)
        assert IO(stub)._has_nowhere_for_an_overview_sidecar() is False, (
            f"a {driver_type} handle described as {description!r} must not be blocked"
        )
        handle.GetMetadata.assert_not_called()

    @pytest.mark.parametrize("method", ["create_overviews", "recreate_overviews"])
    def test_the_refusal_is_catchable_as_a_pyramids_error(
        self, tmp_path, monkeypatch, method
    ):
        """Both methods raise something a pyramids-wide handler catches.

        Test scenario:
            Raise the refusal from each method and catch it with `except PyramidsError`
            — expected: caught, and the same instance is a `ValueError`, so neither a
            pyramids-wide handler nor a stdlib one has to be widened for it.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, f"pyramids_error_{method}.tif")
        dataset = Dataset.read_file(source)
        level = None
        try:
            level = dataset.get_overview_dataset(overview_index=0)
            caught = None
            try:
                getattr(level, method)()
            except PyramidsError as error:
                caught = error
            assert isinstance(caught, OverviewTargetError), (
                f"`except PyramidsError` must catch {method}'s refusal, got {caught!r}"
            )
            assert isinstance(caught, ValueError), (
                "the same instance must stay catchable as a ValueError"
            )
        finally:
            if level is not None:
                level.close()
            dataset.close()

    def test_recreate_refuses_before_the_resampling_method_is_validated(
        self, pathless_level, tmp_path
    ):
        """An unknown resampling method does not mask the unusable target either.

        Test scenario:
            Call the pathless level with a resampling method that would raise on its own
            — expected: the guard's refusal, matching the ordering its sibling
            `create_overviews` already pins.
        """
        level, _ = pathless_level
        with pytest.raises(OverviewTargetError, match="nowhere to go"):
            level.recreate_overviews(resampling_method="NOPE")
        assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"

    def test_recreate_argument_error_on_a_usable_target_stays_an_argument_error(
        self, tmp_path, monkeypatch
    ):
        """A bad method on an ordinary raster is still a plain argument error.

        Test scenario:
            The same unknown resampling method on the on-disk source — expected: a bare
            `ValueError`, so the new refusal cannot start swallowing the argument check
            for datasets that are perfectly able to hold overviews.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "recreate_arg_src.tif")
        dataset = Dataset.read_file(source)
        try:
            with pytest.raises(ValueError, match="resampling_method") as excinfo:
                dataset.recreate_overviews(resampling_method="NOPE")
            assert not isinstance(excinfo.value, OverviewTargetError), (
                "a bad method is worth retrying with a different one, not an "
                "unusable target"
            )
        finally:
            dataset.close()

    def test_recreate_still_regenerates_a_path_ful_vrt(self, tmp_path, monkeypatch):
        """A saved `.vrt` regenerates the sidecar it owns, untouched by the refusal.

        Test scenario:
            Build a `.vrt` over a source with no overviews of its own, give it levels, and
            regenerate them — expected: a silent, successful pass that keeps the counts
            and the `<name>.vrt.ovr` sidecar, which is the over-fire canary for the
            `recreate_overviews` half of the guard.
        """
        monkeypatch.chdir(tmp_path)
        source = _plain_raster(tmp_path, "recreate_vrt_src.tif")
        vrt_path = tmp_path / "recreate_with_path.vrt"
        gdal.Translate(str(vrt_path), source, format="VRT").FlushCache()
        dataset = Dataset.read_file(str(vrt_path))
        try:
            dataset.create_overviews(overview_levels=[2])
            assert dataset.overview_count == [1, 1], (
                f"precondition: the VRT owns one level per band, {dataset.overview_count}"
            )
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                dataset.recreate_overviews(resampling_method="average")
            assert not [w for w in recorded if issubclass(w.category, UserWarning)], (
                f"the path-ful VRT should regenerate silently, got {recorded}"
            )
            assert dataset.overview_count == [1, 1], (
                f"regeneration must keep the levels, got {dataset.overview_count}"
            )
            assert (tmp_path / "recreate_with_path.vrt.ovr").exists(), (
                "the sidecar the VRT regenerated through must survive"
            )
        finally:
            dataset.close()

    def test_to_zarr_with_overview_factors_surfaces_the_refusal(
        self, tmp_path, monkeypatch
    ):
        """`to_zarr(overview_factors=...)` builds levels, so the refusal escapes there.

        Test scenario:
            An inline-XML VRT is the one refused shape whose base array still writes —
            GDAL reopens it from its own description, where a description-less VRT fails
            first in `to_zarr`'s base-array write — so `_write_overview_levels` is reached and
            `create_overviews` refuses. Expected: `OverviewTargetError` out of `to_zarr`.
        """
        pytest.importorskip("zarr")
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "zarr_guard_src.tif")
        xml = gdal.Translate("", str(source), format="VRT").GetMetadata("xml:VRT")[0]
        dataset = Dataset.read_file(xml)
        try:
            with pytest.raises(OverviewTargetError, match="nowhere to go"):
                dataset.to_zarr(str(tmp_path / "out.zarr"), overview_factors=[2])
        finally:
            dataset.close()

    def test_to_zarr_refuses_the_target_before_writing_anything(
        self, tmp_path, monkeypatch
    ):
        """The refusal is taken pre-flight, so no half-written store is left behind.

        Test scenario:
            The same inline-XML VRT, whose base array `to_zarr` used to commit before
            `_write_overview_levels` refused — expected: `OverviewTargetError` carrying
            the *building* diagnosis, since `to_zarr` builds the levels rather than
            rewriting them, and no store on disk at all. A store written without its
            `multiscales` looks complete to the next reader.
        """
        pytest.importorskip("zarr")
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "zarr_preflight_src.tif")
        xml = gdal.Translate("", str(source), format="VRT").GetMetadata("xml:VRT")[0]
        store = tmp_path / "preflight.zarr"
        dataset = Dataset.read_file(xml)
        try:
            with pytest.raises(OverviewTargetError, match="nowhere to go") as excinfo:
                dataset.to_zarr(str(store), overview_factors=[2])
            assert "stores no pixels of its own" in str(excinfo.value), (
                f"to_zarr builds levels, so it must carry the building diagnosis, got: {excinfo.value}"
            )
            assert not store.exists(), (
                "the refusal must leave no store behind, found "
                f"{sorted(p.name for p in store.iterdir())}"
            )
        finally:
            dataset.close()

    def test_to_zarr_reports_the_compute_argument_before_the_unusable_target(
        self, tmp_path, monkeypatch
    ):
        """`compute=False` is rejected first, ahead of the target the caller cannot fix.

        Test scenario:
            A call wrong in both ways — a refused target *and* `overview_factors` with
            `compute=False` — expected: the plain `ValueError` naming `compute`, not the
            `OverviewTargetError`, and still no store, since both guards run before any
            write. This is the opposite order from `recreate_overviews`, which reports
            the unusable target ahead of its own argument check.
        """
        pytest.importorskip("zarr")
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "zarr_ordering_src.tif")
        xml = gdal.Translate("", str(source), format="VRT").GetMetadata("xml:VRT")[0]
        store = tmp_path / "ordering.zarr"
        dataset = Dataset.read_file(xml)
        try:
            with pytest.raises(ValueError, match="compute=True") as excinfo:
                dataset.to_zarr(str(store), overview_factors=[2], compute=False)
            assert not isinstance(excinfo.value, OverviewTargetError), (
                "the argument the caller can change is reported first, got "
                f"{type(excinfo.value).__name__}"
            )
            assert not store.exists(), "neither guard may leave a store behind"
        finally:
            dataset.close()

    def test_to_zarr_still_writes_a_pyramid_for_a_vrt_with_a_path(
        self, tmp_path, monkeypatch
    ):
        """The pre-flight refuses unusable targets, not the VRT driver.

        Test scenario:
            A saved `.vrt` names its sidecar after its own path, so it can hold the
            levels `to_zarr` builds — expected: the store carries `data` and `data_2`,
            the over-fire canary for a pre-flight that refused every VRT outright.
        """
        zarr = pytest.importorskip("zarr")
        monkeypatch.chdir(tmp_path)
        source = _plain_raster(tmp_path, "zarr_vrt_src.tif")
        vrt_path = tmp_path / "zarr_usable.vrt"
        gdal.Translate(str(vrt_path), source, format="VRT").FlushCache()
        store = tmp_path / "usable.zarr"
        dataset = Dataset.read_file(str(vrt_path))
        try:
            dataset.to_zarr(str(store), overview_factors=[2])
            keys = set(zarr.open_group(str(store), mode="r").array_keys())
            assert {"data", "data_2"} <= keys, (
                f"a path-ful VRT can hold its levels, so the pyramid must be written, got {keys}"
            )
        finally:
            dataset.close()

    def test_a_refusal_gdals_wording_does_not_cover_is_still_classified(
        self, tmp_path, monkeypatch
    ):
        """The classification must be taken before any GDAL call resets the error number.

        Test scenario:
            `_is_write_refusal` prefers `CPLE_NoWriteAccess` and falls back to a short
            list of message phrases. Drive a refusal that sets the number but whose
            wording matches none of those phrases — the shape a future driver could
            produce — so only the number can classify it, and any GDAL call made between
            the failure and the check would erase it. Read-only, so the classification
            lands on `ReadOnlyError` rather than the writable-handle branch, keeping this
            about the ordering. Expected: `ReadOnlyError`, not the bare `RuntimeError` a
            late check gives.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "unknown_wording_src.tif")
        dataset = Dataset.read_file(source, read_only=True)

        def refuse(band, overviews, method):
            with gdal.ExceptionMgr(useExceptions=False):
                gdal.Error(gdal.CE_Failure, gdal.CPLE_NoWriteAccess, "driver said no")
            raise RuntimeError("driver said no")

        monkeypatch.setattr(gdal, "RegenerateOverviews", refuse)
        try:
            with pytest.raises(ReadOnlyError, match="read-only"):
                dataset.recreate_overviews("average")
        finally:
            dataset.close()

    def test_a_warped_refusal_is_classified_without_the_error_number(
        self, tmp_path, monkeypatch
    ):
        """The wording alone must classify a warped refusal, because the number is racy.

        Test scenario:
            Raise the warped band's real wording while GDAL's last-error number is `0` —
            the state measured intermittently on the genuine call, which made the verdict
            depend on which run it was. Expected: `OverviewTargetError` from the phrase
            fallback, not the bare `RuntimeError` the number alone would leave.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "racy_errno_src.tif")
        dataset = Dataset.read_file(source)
        view = None
        try:
            view = dataset.to_crs(3857)
            view.create_overviews("average", overview_levels=[2])

            def refuse(band, overviews, method):
                gdal.ErrorReset()
                raise RuntimeError(
                    "GDALRasterBand::RasterIO(): attempt to write to a "
                    "VRTWarpedRasterBand."
                )

            monkeypatch.setattr(gdal, "RegenerateOverviews", refuse)
            with pytest.raises(OverviewTargetError, match="belong to a VRT"):
                view.recreate_overviews("average")
        finally:
            if view is not None:
                view.close()
            dataset.close()

    def test_read_only_vrt_still_reports_the_access_mode(self, tmp_path, monkeypatch):
        """A genuine access-mode refusal on a VRT raises `ReadOnlyError`.

        Test scenario:
            The VRT path had no coverage — the only access-mode guard used a GTiff — and
            a VRT reaches the warped branch's neighbourhood. Expected: `ReadOnlyError`.
            This shape survives a late classification on its own, because GDAL words it
            "read-only mode", which the message fallback knows; the sibling test above
            is the one that pins the ordering.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "ro_vrt_src.tif")
        vrt_path = tmp_path / "ro_guard.vrt"
        gdal.Translate(str(vrt_path), str(source), format="VRT").FlushCache()
        writable = Dataset.read_file(str(vrt_path), read_only=False)
        try:
            writable.create_overviews("average", overview_levels=[2])
        finally:
            writable.close()
        dataset = Dataset.read_file(str(vrt_path), read_only=True)
        try:
            assert dataset.driver_type == "vrt", "precondition: the handle is a VRT"
            assert all(count > 0 for count in dataset.overview_count), (
                f"precondition: the sidecar levels are visible, {dataset.overview_count}"
            )
            with pytest.raises(ReadOnlyError, match="read-only"):
                dataset.recreate_overviews("average")
        finally:
            dataset.close()

    def test_wrap_longitude_on_a_file_backed_source_is_refused(
        self, tmp_path, monkeypatch
    ):
        """The lazy longitude roll is a plain pathless VRT, so it is refused.

        Test scenario:
            `_wrap_longitude_vrt` builds the roll with `VRT.Create("")`, which produced
            the stray-sidecar damage before the guard — expected: the refusal, and
            nothing written to the working directory. The in-memory source returns `MEM`
            and is covered by the sibling MEM case.
        """
        monkeypatch.chdir(tmp_path)
        source = tmp_path / "global.tif"
        raster = gdal.GetDriverByName("GTiff").Create(
            str(source), 64, 32, 1, gdal.GDT_Float32
        )
        raster.SetGeoTransform((0.0, 5.625, 0.0, 90.0, 0.0, -5.625))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        raster.SetProjection(srs.ExportToWkt())
        raster.GetRasterBand(1).WriteArray(
            np.arange(64 * 32, dtype="float32").reshape(32, 64)
        )
        raster.FlushCache()
        del raster
        dataset = Dataset.read_file(str(source))
        rolled = None
        try:
            rolled = dataset.wrap_longitude()
            assert rolled.driver_type == "vrt", "precondition: the roll is a VRT"
            assert not rolled.raster.GetDescription(), "precondition: it has no path"
            with pytest.raises(OverviewTargetError, match="nowhere to go"):
                rolled.create_overviews(overview_levels=[2])
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            if rolled is not None:
                rolled.close()
            dataset.close()

    def test_warped_create_overviews_ignores_the_resampling_method(
        self, tmp_path, monkeypatch
    ):
        """Document that a warped VRT resamples with the warper, not the given method.

        Test scenario:
            Two fresh warped views of one raster, built with `average` and `nearest` —
            expected: identical levels, because GDAL uses the warper's own algorithm.
            The refusal message and `create_overviews`' notes both disclose this; the day
            GDAL honours the argument, this test fails and the disclosures can go.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "warp_resample.tif")
        levels = {}
        for method in ("average", "nearest"):
            parent = Dataset.read_file(source)
            view = parent.to_crs(3857)
            try:
                view.create_overviews(method, overview_levels=[2])
                levels[method] = view.read_overview_array(band=0, overview_index=0).copy()
            finally:
                view.close()
                parent.close()
        assert np.array_equal(levels["average"], levels["nearest"]), (
            "GDAL now honours resampling_method on a warped VRT — drop the caveat from "
            "the refusal message and the create_overviews notes"
        )

    def test_recreate_on_a_vrt_inheriting_source_levels_is_not_an_access_error(
        self, tmp_path, monkeypatch
    ):
        """A plain VRT over an overviewed source computes its levels, so reopening cannot help.

        Test scenario:
            `gdal.Translate(..., format="VRT")` over a raster that already has overviews
            inherits them as virtual levels owned by the VRT. Opened *writable*, GDAL
            still refuses the rewrite — the read-only dataset in its message is the
            source. Expected: `OverviewTargetError`, never advice the caller has already
            followed.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "inherit_src.tif")
        vrt_path = tmp_path / "inherit.vrt"
        gdal.Translate(str(vrt_path), str(source), format="VRT").FlushCache()
        dataset = Dataset.read_file(str(vrt_path), read_only=False)
        try:
            assert dataset._access == "write", "precondition: already opened writable"
            assert all(count > 0 for count in dataset.overview_count), (
                f"precondition: the inherited levels are visible, {dataset.overview_count}"
            )
            assert not (tmp_path / "inherit.vrt.ovr").exists(), (
                "precondition: the levels are inherited, not owned by a sidecar"
            )
            with pytest.raises(OverviewTargetError, match="belong to a VRT") as excinfo:
                dataset.recreate_overviews("average")
            assert not isinstance(excinfo.value, ReadOnlyError), (
                "the access mode is not the blocker; the handle is already writable"
            )
            assert "read_only=False" not in str(excinfo.value), (
                "advising a reopen the caller has already done is the #922 defect"
            )
        finally:
            dataset.close()

    def test_recreate_on_an_internal_overview_still_reports_the_access_mode(
        self, tmp_path, monkeypatch
    ):
        """A stored level in a read-only raster is the genuine access-mode case.

        Test scenario:
            An internal overview belongs to the dataset itself, not to a VRT, so
            reopening writable really is the fix — expected: `ReadOnlyError`, proving the
            widened discriminator did not swallow the case it must keep.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "internal_ro.tif")
        dataset = Dataset.read_file(source, read_only=True)
        try:
            assert all(count > 0 for count in dataset.overview_count), (
                f"precondition: the internal levels are visible, {dataset.overview_count}"
            )
            with pytest.raises(ReadOnlyError, match="read-only") as excinfo:
                dataset.recreate_overviews("average")
            assert not isinstance(excinfo.value, OverviewTargetError), (
                "a stored level in a read-only handle is fixable by reopening"
            )
        finally:
            dataset.close()

    def test_a_stored_band_is_not_condemned_by_a_computed_sibling(
        self, tmp_path, monkeypatch
    ):
        """The verdict follows the failing band's own levels, not the dataset's.

        Test scenario:
            One VRT whose band 0 keeps its level in an external `.ovr` GTiff while band 1
            takes its level from the VRT itself, opened read-only so a reopen is still
            worth suggesting. GDAL refuses band 0 first, and that band is stored —
            expected: `ReadOnlyError` naming band 0. A classifier that scanned the whole
            dataset would find band 1's VRT-owned level and hand band 0 the unactionable
            "rebuild them" advice instead.
        """
        monkeypatch.chdir(tmp_path)
        vrt_path = _mixed_ownership_vrt(tmp_path)
        dataset = Dataset.read_file(vrt_path, read_only=True)
        try:
            assert dataset.overview_count == [1, 1], (
                f"precondition: both bands carry a level, {dataset.overview_count}"
            )
            assert _level_owner_driver(dataset._iloc(0)) == "GTiff", (
                "precondition: band 0's level is stored in the external sidecar"
            )
            assert _level_owner_driver(dataset._iloc(1)) == "VRT", (
                "precondition: band 1's level is computed by the VRT"
            )
            with pytest.raises(ReadOnlyError, match="band 0") as excinfo:
                dataset.recreate_overviews("average")
            assert not isinstance(excinfo.value, OverviewTargetError), (
                "band 0's own level is stored, so its sibling must not decide for it"
            )
        finally:
            dataset.close()

    def test_a_writable_handle_is_never_told_to_reopen(self, tmp_path, monkeypatch):
        """Ownership cannot prove the access mode is the blocker; the handle can disprove it.

        Test scenario:
            A VRT serving an explicit `<Overview>` owns a real, on-disk-writable `.ovr`,
            so it classifies as stored — yet GDAL opens VRT sources read-only and refuses
            however the parent was opened. Opened writable, the caller has already done
            the only thing `ReadOnlyError` would suggest — expected: `OverviewTargetError`
            and no mention of `read_only=False`.
        """
        monkeypatch.chdir(tmp_path)
        vrt_path = _mixed_ownership_vrt(tmp_path)
        dataset = Dataset.read_file(vrt_path, read_only=False)
        try:
            assert dataset._access == "write", "precondition: already open for writing"
            assert _level_owner_driver(dataset._iloc(0)) == "GTiff", (
                "precondition: band 0's level classifies as stored"
            )
            with pytest.raises(OverviewTargetError, match="already open for writing") as excinfo:
                dataset.recreate_overviews("average")
            assert not isinstance(excinfo.value, ReadOnlyError), (
                "a writable handle cannot be fixed by reopening it writable"
            )
            assert "read_only=False" not in str(excinfo.value), (
                "advising the caller's own last move is the #922 defect"
            )
        finally:
            dataset.close()

    @pytest.mark.parametrize(
        "read_only, expected",
        [(True, ReadOnlyError), (False, OverviewTargetError)],
        ids=["read-only-handle", "writable-handle"],
    )
    def test_a_band_whose_levels_will_not_resolve_is_still_classified(
        self, tmp_path, monkeypatch, read_only, expected
    ):
        """Resolving the levels can fail too, and the verdict must rest on *this* band.

        Test scenario:
            `GetOverview` refuses on band 0 before a single level comes back, so the
            classification runs with nothing resolved — expected: the same verdict a
            refused write earns on that handle, naming band 0 and chaining GDAL's error.
            The empty list reads as stored, which is the conservative direction. Bound at
            the assignment instead of before the `try`, the name would be unbound on band
            0 and the handler would die classifying, losing GDAL's error entirely.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, f"unresolved_{read_only}.tif")
        dataset = Dataset.read_file(source, read_only=read_only)
        monkeypatch.setattr(gdal.Band, "GetOverview", _refuse_to_resolve)
        try:
            with pytest.raises(expected) as excinfo:
                dataset.recreate_overviews("average")
            assert type(excinfo.value) is expected, (
                f"expected exactly {expected.__name__}, got {type(excinfo.value).__name__}"
            )
            assert "band 0" in str(excinfo.value), (
                f"the failing band must be named, got: {excinfo.value}"
            )
            assert isinstance(excinfo.value.__cause__, RuntimeError), (
                "the error raised while resolving the levels must stay chained as __cause__"
            )
        finally:
            dataset.close()

    def test_the_write_refusal_is_classified_without_serialising_the_document(
        self, tmp_path, monkeypatch
    ):
        """Deciding a refusal never reads `xml:VRT`, which the ownership check replaced.

        Test scenario:
            A read-only path-ful VRT whose levels live in the sidecar it owns — the
            genuine access-mode case on a VRT handle. Expected: `ReadOnlyError`, reached
            without `_is_warped_vrt` running at all: the ownership check replaced it, and
            serialising the document on a mosaic with many sources costs milliseconds in
            a failure path. Both reset the CPL error number, which is why the refusal is
            classified before either of them runs.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, "no_serialise_src.tif")
        vrt_path = tmp_path / "no_serialise.vrt"
        gdal.Translate(str(vrt_path), str(source), format="VRT").FlushCache()
        writable = Dataset.read_file(str(vrt_path), read_only=False)
        try:
            writable.create_overviews("average", overview_levels=[2])
        finally:
            writable.close()
        serialise = MagicMock(return_value=False)
        monkeypatch.setattr(IO, "_is_warped_vrt", serialise)
        dataset = Dataset.read_file(str(vrt_path), read_only=True)
        try:
            assert _level_owner_driver(dataset._iloc(0)) == "GTiff", (
                "precondition: the levels are stored in the sidecar the VRT owns"
            )
            with pytest.raises(ReadOnlyError, match="read-only"):
                dataset.recreate_overviews("average")
            serialise.assert_not_called()
        finally:
            dataset.close()

    @pytest.mark.parametrize("parent_read_only", [True, False], ids=["ro", "writable"])
    def test_recreate_on_a_warped_view_does_not_blame_the_access_mode(
        self, tmp_path, monkeypatch, parent_read_only
    ):
        """A warped band is never writable, so the access mode is not the blocker.

        Test scenario:
            GDAL reports an unwritable `VRTWarpedRasterBand` with the same
            `CPLE_NoWriteAccess` it uses for a read-only dataset — expected:
            `OverviewTargetError` naming the warp, never a `ReadOnlyError` telling the
            caller to reopen. The writable-parent case proves the point: reopening has
            already been done and it changes nothing.
        """
        monkeypatch.chdir(tmp_path)
        source = _overviewed_raster(tmp_path, f"warp_recreate_{parent_read_only}.tif")
        dataset = Dataset.read_file(source, read_only=parent_read_only)
        view = None
        try:
            view = dataset.to_crs(3857)
            view.create_overviews("average", overview_levels=[2])
            assert all(count > 0 for count in view.overview_count), (
                f"precondition: the warped view holds levels, {view.overview_count}"
            )
            with pytest.raises(OverviewTargetError, match="belong to a VRT") as excinfo:
                view.recreate_overviews("average")
            assert not isinstance(excinfo.value, ReadOnlyError), (
                f"the access mode is not the blocker (parent read_only={parent_read_only})"
            )
            assert "read_only=False" not in str(excinfo.value), (
                "a pathless warped view cannot act on that advice"
            )
            assert isinstance(excinfo.value.__cause__, RuntimeError), (
                "GDAL's own error must stay chained"
            )
        finally:
            if view is not None:
                view.close()
            dataset.close()

    def test_recreate_still_warns_for_a_warped_view_without_overviews(
        self, tmp_path, monkeypatch
    ):
        """The refusal does not pre-empt the no-overviews warning on an exempt shape.

        Test scenario:
            A pathless warped view that was never given levels — expected: the
            "call create_overviews() first" warning it emitted before the guard existed,
            since a warped view can act on that advice and build them in RAM.
        """
        monkeypatch.chdir(tmp_path)
        dataset = Dataset.create_from_array(
            np.arange(4096, dtype="float32").reshape(64, 64),
            top_left_corner=(0.0, 64.0),
            cell_size=1.0,
            epsg=4326,
        )
        view = None
        try:
            view = dataset.warped_view(3857)
            assert view.overview_count == [0], (
                f"precondition: the view has no levels, {view.overview_count}"
            )
            with pytest.warns(UserWarning, match=r"call create_overviews\(\) first"):
                view.recreate_overviews()
        finally:
            if view is not None:
                view.close()
            dataset.close()

    def test_recreate_refuses_a_pathless_vrt_rather_than_advising_create_overviews(
        self, tmp_path, monkeypatch
    ):
        """A plain pathless VRT with no levels is refused instead of warned at.

        Test scenario:
            The empty-count warning tells the caller to run `create_overviews()`, which
            refuses this very shape — expected: the refusal instead, and no `UserWarning`,
            so the caller is not sent round a loop that cannot terminate.
        """
        monkeypatch.chdir(tmp_path)
        source = _plain_raster(tmp_path, "pathless_empty_src.tif")
        wrapper = Dataset(gdal.BuildVRT("", [source]))
        try:
            assert wrapper.overview_count == [0, 0], (
                f"precondition: nothing to regenerate, {wrapper.overview_count}"
            )
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with pytest.raises(OverviewTargetError, match="nowhere to go"):
                    wrapper.recreate_overviews()
            assert not [w for w in recorded if issubclass(w.category, UserWarning)], (
                f"the dead advice must not be emitted, got {[str(w.message) for w in recorded]}"
            )
            assert not (tmp_path / ".ovr").exists(), "a stray '.ovr' was written"
        finally:
            wrapper.close()

    @pytest.mark.parametrize(
        "description, truncated",
        [
            ("", False),
            ("   ", False),
            ('<VRTDataset rasterXSize="8" rasterYSize="8">', False),
            ("<" + "x" * 79, False),
            ("<" + "x" * 80, True),
            ("<VRTDataset " + "x" * 200 + ">", True),
        ],
        ids=[
            "empty",
            "blank",
            "short-document",
            "exactly-80",
            "one-over-80",
            "long-document",
        ],
    )
    def test_the_refusal_quotes_the_description_cut_at_80_characters(
        self, description, truncated
    ):
        """The message quotes the description that caused it, cut at 80 characters.

        Test scenario:
            Drive the shared message builder with each description the guard refuses —
            expected: the description is quoted verbatim up to 80 characters, so a
            caller holding several handles can tell which one failed, and a longer
            one is cut with an ellipsis marking it. The 80/81-character pair pins the
            boundary itself: one that exactly fits is quoted whole and unmarked, so
            the marker cannot start announcing a cut that never happened.
        """
        handle = MagicMock(spec=gdal.Dataset)
        handle.GetDescription.return_value = description
        stub = MagicMock(spec=Dataset, driver_type="vrt", raster=handle)
        message = IO(stub)._no_sidecar_message()
        marker = "..." if truncated else ""
        assert f"Description: {description[:_DESCRIPTION_EXCERPT]!r}{marker}. Save it first" in message, (
            f"the message must quote {description[:_DESCRIPTION_EXCERPT]!r}{marker}, got: {message}"
        )
        if description:
            assert (description in message) is not truncated, (
                f"a {len(description)}-character description must "
                f"{'not ' if truncated else ''}appear in full, got: {message}"
            )

    def test_both_methods_share_one_refusal_message(self, tmp_path, monkeypatch):
        """Both methods refuse in exactly the same words.

        Test scenario:
            An inline-XML VRT is refused by both methods and carries a description long
            enough to truncate — expected: one shared message quoting its first 80
            characters, so the two cannot drift apart the way the copied literals
            could. The recovery clause names `create_overviews` for both, because
            `to_file` drops overviews and regenerating on the saved raster would no-op.
            The diagnoses deliberately differ: only the recovery is shared.
        """
        monkeypatch.chdir(tmp_path)
        xml = (
            '<VRTDataset rasterXSize="64" rasterYSize="64">'
            '<VRTRasterBand dataType="Float32" band="1"/>'
            "</VRTDataset>"
        )
        dataset = Dataset.read_file(xml)
        try:
            description = dataset.raster.GetDescription()
            assert len(description) > _DESCRIPTION_EXCERPT, (
                f"precondition: the description outruns the quote, {len(description)}"
            )
            with pytest.raises(OverviewTargetError) as build:
                dataset.create_overviews(overview_levels=[2])
            with pytest.raises(OverviewTargetError) as regenerate:
                dataset.recreate_overviews()
            built, regenerated = str(build.value), str(regenerate.value)
            assert f"Description: {description[:_DESCRIPTION_EXCERPT]!r}" in built, (
                f"the refusal must quote the description that caused it, got: {built}"
            )
            assert description not in built, (
                f"the whole document must not be reproduced, got: {built}"
            )
            assert "create_overviews()" in built, (
                f"the recovery must name create_overviews, not regenerate, got: {built}"
            )
            assert "regenerate" not in built, (
                f"to_file drops overviews, so regenerating would no-op, got: {built}"
            )
            recovery = (
                "Save it first with to_file(path) and build the overviews on the saved "
                "raster with create_overviews()."
            )
            assert built.endswith(recovery) and regenerated.endswith(recovery), (
                f"both refusals must share the recovery clause: {built} vs {regenerated}"
            )
            assert f"Description: {description[:_DESCRIPTION_EXCERPT]!r}" in regenerated, (
                f"both refusals must quote the description, got: {regenerated}"
            )
            assert built != regenerated, (
                "the diagnoses must differ: building has nowhere to put a sidecar, while "
                "regenerating is blocked by the VRT computing the levels it exposes"
            )
            assert "computes them rather than storing them" in regenerated, (
                f"the regenerate diagnosis must name ownership, got: {regenerated}"
            )
        finally:
            dataset.close()

    def test_a_short_description_is_quoted_whole_in_both_diagnoses(self):
        """The excerpt and its ellipsis are shared, so a short description survives intact.

        Test scenario:
            `_no_sidecar_message` on a description well inside the excerpt, asked both
            ways — the shape the public API cannot produce, since the only refused
            description long enough to reach that test is a whole inline VRT document.
            Expected: both messages quote it in full with no ellipsis, both end with the
            one recovery clause, and only the building one blames the description while
            only the regenerating one blames ownership.
        """
        handle = MagicMock()
        handle.GetDescription.return_value = "short.vrt"
        engine = IO(MagicMock(spec=Dataset, driver_type="vrt", raster=handle))
        built = engine._no_sidecar_message()
        regenerated = engine._no_sidecar_message(regenerating=True)
        for message in (built, regenerated):
            assert "Description: 'short.vrt'. Save it first" in message, (
                f"a description inside the excerpt must be quoted whole, got: {message}"
            )
            assert message.endswith("create_overviews()."), (
                f"the recovery clause is shared by both diagnoses, got: {message}"
            )
        assert "GDAL names the external sidecar after the description" in built, (
            f"building is blocked by having nowhere to put a sidecar, got: {built}"
        )
        assert "computes them rather than storing them" in regenerated, (
            f"regenerating is blocked by the VRT owning the levels, got: {regenerated}"
        )
        assert "stores no pixels of its own" not in regenerated, (
            f"the two diagnoses must not drift back into one, got: {regenerated}"
        )


@pytest.fixture
def level_owners(tmp_path, monkeypatch):
    """Yield real GDAL bands keyed by the kind of dataset that owns their pixels.

    The classifier reads `level.GetDataset().GetDriver()`, so every entry is a genuine
    band taken from a real handle rather than a stub: the `.ovr` GTiff of an external
    sidecar, an internal overview (whose owner reports no driver at all), a `MEM`
    dataset, a plain GTiff and a VRT.
    """
    monkeypatch.chdir(tmp_path)
    external = _plain_raster(tmp_path, "external_owner.tif")
    handle = gdal.Open(external, gdal.GA_ReadOnly)
    handle.BuildOverviews("AVERAGE", [2])
    handle = None
    internal = _overviewed_raster(tmp_path, "internal_owner.tif")
    vrt_path = tmp_path / "virtual_owner.vrt"
    gdal.Translate(str(vrt_path), internal, format="VRT").FlushCache()

    plain = gdal.Open(_plain_raster(tmp_path, "plain_owner.tif"))
    sidecar = gdal.Open(external)
    stacked = gdal.Open(internal)
    virtual = gdal.Open(str(vrt_path))
    memory = gdal.GetDriverByName("MEM").Create("", 8, 8, 1, gdal.GDT_Float32)
    owners = {
        "gtiff": plain.GetRasterBand(1),
        "external-ovr": sidecar.GetRasterBand(1).GetOverview(0),
        "internal-ovr": stacked.GetRasterBand(1).GetOverview(0),
        "mem": memory.GetRasterBand(1),
        "vrt": virtual.GetRasterBand(1).GetOverview(0),
    }
    try:
        yield owners
    finally:
        # Each band keeps its owning dataset alive, so dropping the datasets alone would
        # leave the files open — on Windows that is what turns tmp_path cleanup into an
        # "access is denied".
        owners.clear()
        plain = sidecar = stacked = virtual = memory = None


class TestOverviewTargetIsVirtual:
    """The predicate that splits the one `CPLE_NoWriteAccess` refusal into two answers.

    `TestCreateOverviewsPathlessGuard` drives it through the shapes GDAL actually
    produces; this class drives it directly — one owner kind at a time, plus the inputs
    those shapes cannot reach: a band with no levels, a level with no owning dataset at
    all, and a list whose *later* entry is the VRT-owned one.
    """

    @pytest.mark.parametrize(
        "owner, driver, virtual",
        [
            ("gtiff", "GTiff", False),
            ("external-ovr", "GTiff", False),
            ("internal-ovr", None, False),
            ("mem", "MEM", False),
            ("vrt", "VRT", True),
        ],
    )
    def test_only_a_vrt_owner_makes_a_level_virtual(
        self, level_owners, owner, driver, virtual
    ):
        """A level is computed only when a VRT owns its pixels.

        Args:
            owner: Key into the `level_owners` fixture.
            driver: The short name the owning dataset really reports, or None.
            virtual: The verdict the predicate must return.

        Test scenario:
            Feed the predicate one real band per owner kind — expected: True for the
            VRT-owned level alone. The internal overview is the case worth stating: its
            owner reports no driver at all, and that must read as *stored*, because
            rebuilding is the wrong advice for a level a writable handle could rewrite.
        """
        level = level_owners[owner]
        actual = level.GetDataset().GetDriver()
        assert (None if actual is None else actual.ShortName) == driver, (
            f"precondition: the {owner} level's owner must report driver {driver}"
        )
        assert IO._overview_target_is_virtual([level]) is virtual, (
            f"a level owned by a {driver} dataset must read as "
            f"{'computed' if virtual else 'stored'}"
        )

    def test_no_levels_at_all_read_as_stored(self):
        """An empty level list is not virtual, so it cannot invent a refusal.

        Test scenario:
            The band-level loop skips a band with no levels, so the predicate never sees
            an empty list in production — expected: False anyway, since "no evidence of a
            VRT" must never default to the answer that tells the caller to rebuild.
        """
        assert IO._overview_target_is_virtual([]) is False, (
            "with no levels to inspect there is nothing to prove computed"
        )

    @pytest.mark.parametrize(
        "first, second",
        [("external-ovr", "vrt"), ("vrt", "internal-ovr")],
        ids=["stored-then-computed", "computed-then-stored"],
    )
    def test_a_stored_level_does_not_end_the_search(self, level_owners, first, second):
        """One VRT-owned level anywhere in the chain condemns the whole batch.

        Args:
            first: Key of the level handed to the predicate first.
            second: Key of the level handed to it second.

        Test scenario:
            Both orders of a mixed pair — expected: True either way. A band's levels go
            to GDAL as one batch, so a scan that stopped at the first stored level would
            call the batch writable and raise the access-mode error on a chain that has a
            computed level in it. The `internal-ovr` entry pairs a driver-less owner with
            the VRT one, so the None-driver guard cannot short-circuit the scan either.
        """
        levels = [level_owners[first], level_owners[second]]
        assert IO._overview_target_is_virtual(levels) is True, (
            f"a {first} level followed by a {second} one must read as computed"
        )

    def test_a_level_with_no_owning_dataset_reads_as_stored(self):
        """A level that names no owner is answered, not crashed on.

        Test scenario:
            Every band GDAL hands back today reports an owning dataset, so this input is
            unreachable through the public API and has to be built directly — expected:
            False, and no `AttributeError` from asking a missing owner for its driver,
            since the predicate runs inside an exception handler where a second failure
            would replace GDAL's own diagnosis.
        """
        level = MagicMock(spec=gdal.Band)
        level.GetDataset.return_value = None
        assert IO._overview_target_is_virtual([level]) is False, (
            "an owner-less level cannot be proven computed"
        )


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
                OverviewTargetError,
                "already open for writing",
            ),
            (
                "/mnt/read-only-archive/dem.tif, band 1: No space left on device",
                RuntimeError,
                "No space left on device",
            ),
        ],
        ids=[
            "unrelated-failure-propagates",
            "read-only-wording-on-a-writable-handle-is-not-an-access-error",
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
            `gdal.RegenerateOverviews` is patched to raise a `RuntimeError` while a
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

            monkeypatch.setattr(gdal, "RegenerateOverviews", raise_runtime_error)
            with pytest.raises(expected_error) as excinfo:
                dataset.recreate_overviews()
            assert type(excinfo.value) is expected_error, (
                f"expected exactly {expected_error.__name__}, got {type(excinfo.value).__name__}"
            )
            assert expected_text in str(excinfo.value), (
                f"expected {expected_text!r} in the message, got: {excinfo.value}"
            )
            if expected_error is not RuntimeError:
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
            leaves `RegenerateOverviews` *returning* `CE_Failure` instead of raising.
            Patch it to do exactly that on a writable dataset that does have overviews
            — expected: a `RuntimeError` naming the band, since the
            try/except alone would let this through as the silent no-op the method
            exists to remove.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "status_failure.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            dataset.create_overviews(overview_levels=[2])
            monkeypatch.setattr(
                gdal, "RegenerateOverviews", lambda *args, **kwargs: gdal.CE_Failure
            )
            with pytest.raises(RuntimeError) as excinfo:
                dataset.recreate_overviews()
            assert type(excinfo.value) is RuntimeError, (
                f"a failing status must not be relabelled, got {type(excinfo.value).__name__}"
            )
            assert "overviews of band 0" in str(excinfo.value), (
                f"the error must name the band, got: {excinfo.value}"
            )
        finally:
            dataset.close()

    def test_propagated_failure_is_noted_with_the_band_and_level(
        self, era5_raster_path, tmp_path, monkeypatch
    ):
        """A propagated GDAL failure carries a note saying where regeneration stopped.

        Test scenario:
            An unrelated `RuntimeError` is raised from `gdal.RegenerateOverviews` and
            re-raised unchanged — expected: `__notes__` names the band and warns that
            says earlier bands may already have been rewritten, because the loop
            rewrites in place and leaves the dataset half-regenerated.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "noted_failure.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            dataset.create_overviews(overview_levels=[2])

            def raise_runtime_error(*args, **kwargs):
                raise RuntimeError("Failed to write overview block: disk full")

            monkeypatch.setattr(gdal, "RegenerateOverviews", raise_runtime_error)
            with pytest.raises(RuntimeError) as excinfo:
                dataset.recreate_overviews()
            notes = getattr(excinfo.value, "__notes__", [])
            assert any("overviews of band 0" in note for note in notes), (
                f"the note must name the band, got {notes}"
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
            Record every `gdal.RegenerateOverviews` call on the mixed `[1, 0]` VRT —
            expected: exactly one call, carrying band 0's levels as a batch, proving the
            mixed path warns *and* carries on rather than returning early, and that the
            empty band is skipped instead of being handed an empty list. The recorder
            also keeps the call off the VRT's read-only `.ovr`, which is why the sibling
            warning test has to suppress that failure.
        """
        dataset = Dataset.read_file(_mixed_overview_vrt(tmp_path), read_only=True)
        try:
            calls = []

            def record(band, overviews, method):
                calls.append((len(overviews), method))
                return gdal.CE_None

            monkeypatch.setattr(gdal, "RegenerateOverviews", record)
            with pytest.warns(UserWarning, match=r"call create_overviews\(\) first"):
                dataset.recreate_overviews(resampling_method="average")
            assert calls == [(1, "average")], (
                "band 0's levels should regenerate in one batched call and the empty "
                f"band be skipped, got {calls}"
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


class TestRecreateOverviewsBatching:
    """A band's levels are regenerated in one GDAL pass, not one pass per level (#783).

    `gdal.RegenerateOverviews` reads the full-resolution band once and fills every level
    from that single pass; the singular call it replaced re-read the source per level.
    """

    def test_a_bands_levels_go_out_in_a_single_call(self, tmp_path, monkeypatch):
        """Three bands carrying three levels produce three calls of three overviews each.

        Test scenario:
            Record every `gdal.RegenerateOverviews` call on a 3-band raster with levels
            `[2, 4, 8]` — expected: one call per band, in band order, each handed all
            three of that band's overviews. The per-level loop this replaced would make
            nine one-overview calls, and batching every band together would make one.
            Bands are told apart by their constant fill (band `i` holds `i + 1`), so a
            call landing on the wrong band is caught as well.
        """
        constants = [np.full((64, 64), i + 1, dtype="float32") for i in range(3)]
        path = _raster_with_overviews(tmp_path, "batched.tif", constants, [2, 4, 8])
        dataset = Dataset.read_file(path, read_only=False)
        try:
            calls = []

            def record(band, overviews, method):
                fill = int(band.ReadAsArray(0, 0, 1, 1).item())
                calls.append((fill, len(overviews), method))
                return gdal.CE_None

            monkeypatch.setattr(gdal, "RegenerateOverviews", record)
            dataset.recreate_overviews(resampling_method="average")
            expected = [(1, 3, "average"), (2, 3, "average"), (3, 3, "average")]
            assert calls == expected, (
                f"each band's three levels should go out in one call per band, got {calls}"
            )
        finally:
            dataset.close()

    def test_every_level_is_rewritten_from_its_own_band(self, tmp_path):
        """Batched regeneration fills each level with that band's own averaged values.

        Test scenario:
            Build the levels with NEAREST over three different random grids, then
            regenerate with AVERAGE — expected: levels 0 and 1 of every band become the
            2x2 and 4x4 block means of *that* band's grid, which the nearest-neighbour
            picks they started from are not. A batch that filled only the first level, or
            crossed one band's source into another's overviews, would satisfy the
            call-shape assertions above but not these values.
        """
        rng = np.random.default_rng(1337)
        grids = [rng.random((64, 64)).astype("float32") for _ in range(3)]
        path = _raster_with_overviews(tmp_path, "values.tif", grids, [2, 4], "NEAREST")
        dataset = Dataset.read_file(path, read_only=False)
        try:
            before = dataset.read_overview_array(band=0, overview_index=0)
            assert not np.allclose(before, _block_mean(grids[0], 2)), (
                "the NEAREST-built level must start out different from the average, "
                "otherwise regenerating it would prove nothing"
            )
            dataset.recreate_overviews(resampling_method="average")
            for index, grid in enumerate(grids):
                for level, factor in enumerate((2, 4)):
                    actual = dataset.read_overview_array(
                        band=index, overview_index=level
                    )
                    np.testing.assert_allclose(
                        actual,
                        _block_mean(grid, factor),
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg=(
                            f"level {level} of band {index} should hold the "
                            f"{factor}x{factor} block means of that band"
                        ),
                    )
        finally:
            dataset.close()

    def test_each_band_is_batched_with_its_own_level_count(self, tmp_path, monkeypatch):
        """A band with fewer levels than its neighbour is batched with its own count.

        Test scenario:
            Two bands whose non-zero level counts differ — band 0 carries two levels and
            band 1 one, via a per-band `<Overview>` VRT — expected: calls of 2 and 1
            overviews respectively. Every other fixture gives both bands the same count,
            so indexing the snapshot by the wrong band survives them unnoticed.
        """
        # The <Overview> entries below alias the full-resolution band: this test
        # monkeypatches RegenerateOverviews, so nothing ever reads their pixels -- only
        # the per-band *count* the engine snapshots matters here.
        sources = []
        for name, levels in (("two.tif", [2, 4]), ("one.tif", [2])):
            path = str(tmp_path / name)
            raster = gdal.GetDriverByName("GTiff").Create(
                path, 16, 16, 1, gdal.GDT_Float32
            )
            raster.GetRasterBand(1).WriteArray(
                np.arange(256, dtype="float32").reshape(16, 16)
            )
            raster = None
            handle = gdal.Open(path, gdal.GA_Update)
            handle.BuildOverviews("AVERAGE", levels)
            handle = None
            sources.append(path)

        vrt = tmp_path / "uneven.vrt"
        vrt.write_text(
            f'<VRTDataset rasterXSize="16" rasterYSize="16">'
            + "".join(
                f'<VRTRasterBand dataType="Float32" band="{position + 1}">'
                f'<SimpleSource><SourceFilename relativeToVRT="0">{source}'
                f"</SourceFilename><SourceBand>1</SourceBand></SimpleSource>"
                + "".join(
                    f'<Overview><SourceFilename relativeToVRT="0">{source}'
                    f"</SourceFilename><SourceBand>1</SourceBand></Overview>"
                    for _ in range(count)
                )
                + "</VRTRasterBand>"
                for position, (source, count) in enumerate(zip(sources, (2, 1)))
            )
            + "</VRTDataset>"
        )
        dataset = Dataset.read_file(str(vrt), read_only=True)
        try:
            assert dataset.overview_count == [2, 1], (
                f"fixture must give the bands different counts, got {dataset.overview_count}"
            )
            batches = []
            monkeypatch.setattr(
                gdal,
                "RegenerateOverviews",
                lambda band, overviews, method: (
                    batches.append(len(overviews)) or gdal.CE_None
                ),
            )
            dataset.recreate_overviews(resampling_method="average")
            assert batches == [2, 1], (
                f"each band must be batched with its own level count, got {batches}"
            )
        finally:
            dataset.close()

    def test_read_only_refusal_on_a_multi_level_batch(self, tmp_path):
        """A read-only refusal is still classified when the batch carries many levels.

        Test scenario:
            The unmocked read-only path is otherwise only exercised on a single-level
            fixture, so nothing showed that the wider single call still yields a
            `CPLE_NoWriteAccess` GDAL can be asked about — expected: `ReadOnlyError`
            naming the band, with the original chained.
        """
        values = np.arange(4096, dtype="float32").reshape(64, 64)
        path = _raster_with_overviews(tmp_path, "ro_batch.tif", [values], [2, 4, 8])
        dataset = Dataset.read_file(path, read_only=True)
        try:
            assert dataset.overview_count == [3], (
                f"fixture must carry three levels, got {dataset.overview_count}"
            )
            with pytest.raises(ReadOnlyError, match="overviews of band 0") as excinfo:
                dataset.recreate_overviews(resampling_method="average")
            assert isinstance(excinfo.value.__cause__, RuntimeError), (
                "the original GDAL error must stay chained"
            )
        finally:
            dataset.close()

    def test_create_overviews_cascades_the_same_way(self, tmp_path):
        """`create_overviews` produces the same deep level, which is what makes the
        cascade defensible.

        Test scenario:
            Build the levels from scratch and regenerate them on a copy of the same
            raster — expected: identical level 1. The justification for accepting the
            behaviour change is that `BuildOverviews` already cascades, so
            `recreate_overviews` now agrees with how the levels were built; nothing
            asserted that, leaving the argument resting on prose.
        """
        values = np.arange(64, dtype="float32").reshape(8, 8)
        values[0, 0] = -9999.0
        built = _raster_with_overviews(
            tmp_path, "built.tif", [values], [2, 4], no_data_value=-9999.0
        )
        regenerated = _raster_with_overviews(
            tmp_path, "regenerated.tif", [values], [2, 4], no_data_value=-9999.0
        )

        dataset = Dataset.read_file(regenerated, read_only=False)
        try:
            dataset.recreate_overviews(resampling_method="average")
            after = np.asarray(dataset.get_overview(0, 1).ReadAsArray())
        finally:
            dataset.close()
        reference = Dataset.read_file(built)
        try:
            expected = np.asarray(reference.get_overview(0, 1).ReadAsArray())
        finally:
            reference.close()
        np.testing.assert_allclose(
            after,
            expected,
            err_msg="regenerating must land where create_overviews already builds",
        )

    def test_a_deep_level_holds_the_cascaded_values(self, tmp_path):
        """A deeper level is decimated from the level above, not from the source.

        Test scenario:
            `gdal.RegenerateOverviews` cascades: level 1 is built from level 0, whereas
            the per-level call it replaced built every level from the full-resolution
            band. On a raster carrying a no-data gap the two disagree — expected: level
            1 matches the mean of level 0's cells, not the mean of the source window it
            covers. This is the behaviour change #783 trades for the single read pass;
            it also makes recreate_overviews agree with how create_overviews built the
            levels in the first place.
        """
        values = np.arange(64, dtype="float32").reshape(8, 8)
        values[0, 0] = -9999.0
        path = _raster_with_overviews(
            tmp_path,
            "cascade.tif",
            [values],
            [2, 4],
            no_data_value=-9999.0,
        )

        dataset = Dataset.read_file(path, read_only=False)
        try:
            dataset.recreate_overviews(resampling_method="average")
            level_0 = np.asarray(dataset.get_overview(0, 0).ReadAsArray())
            level_1 = np.asarray(dataset.get_overview(0, 1).ReadAsArray())
            cascaded = float(level_0[:2, :2].mean())
            assert float(level_1.ravel()[0]) == pytest.approx(cascaded, rel=1e-6), (
                f"level 1 should decimate level 0 ({cascaded}), got {level_1.ravel()[0]}"
            )
            from_source = float(values[:4, :4][values[:4, :4] != -9999.0].mean())
            assert float(level_1.ravel()[0]) != pytest.approx(from_source, rel=1e-6), (
                "level 1 must no longer be computed from the source window; the "
                "fixture's no-data gap is what makes the two differ"
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

    def test_a_source_that_now_holds_another_grid_is_not_trusted(self):
        """A description that reopens to a different grid is refused, not described.

        Test scenario:
            The `from_bytes` sibling covers a label naming another file; this covers the
            name itself going stale — the `/vsimem/` raster is unlinked and recreated at
            16x16 under the open handle. Expected: `_reopenable_source` returns `None`
            and the level is materialised from the parent's own 64x64 pixels (7.0),
            rather than described from the 16x16 impostor now sitting at that name.
        """
        path = "/vsimem/overview_swapped.tif"
        raster = gdal.GetDriverByName("GTiff").Create(path, 64, 64, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((10.0, 0.5, 0.0, 80.0, 0.0, -0.5))
        raster.GetRasterBand(1).WriteArray(np.full((64, 64), 7.0, dtype="float32"))
        raster = None
        dataset = Dataset.read_file(path, read_only=False)
        overview = None
        try:
            dataset.create_overviews(overview_levels=[2])
            gdal.Unlink(path)
            swap = gdal.GetDriverByName("GTiff").Create(
                path, 16, 16, 1, gdal.GDT_Float32
            )
            swap.SetGeoTransform((0.0, 2.0, 0.0, 0.0, 0.0, -2.0))
            swap.GetRasterBand(1).WriteArray(np.full((16, 16), 99.0, dtype="float32"))
            swap = None

            assert dataset.io._reopenable_source() is None, (
                "a name that reopens to another grid is not an identity"
            )
            overview = dataset.get_overview_dataset()
            assert overview.driver_type == "memory", (
                f"the level should be materialised, got driver {overview.driver_type}"
            )
            assert (overview.rows, overview.columns) == (32, 32), (
                f"expected the parent's own level, got "
                f"{(overview.rows, overview.columns)}"
            )
            value = float(np.asarray(overview.read_array(band=0))[0, 0])
            assert value == pytest.approx(7.0), (
                f"expected the parent's own value 7.0, got {value} "
                "(99.0 means the impostor at the same /vsimem/ name was described)"
            )
        finally:
            if overview is not None:
                overview.close()
            dataset.close()
            gdal.Unlink(path)

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

    def test_colour_table_and_interpretation_are_carried(self, tmp_path):
        """A palette-indexed band keeps its colour table and its interpretation.

        Test scenario:
            The same paletted band on disk (described by a VRT) and in memory
            (materialised) — expected: both levels come back `palette_index` carrying
            the parent's two entries. `create_from_array` sets neither, so a
            materialised level rendered as a plain grey band and wrote out a raster
            whose class colours were gone.
        """
        table = gdal.ColorTable()
        table.SetColorEntry(0, (10, 20, 30, 255))
        table.SetColorEntry(1, (200, 100, 50, 255))
        path = str(tmp_path / "palette.tif")
        on_disk = gdal.GetDriverByName("GTiff").Create(path, 32, 32, 1, gdal.GDT_Byte)
        in_memory = gdal.GetDriverByName("MEM").Create("", 32, 32, 1, gdal.GDT_Byte)
        for target in (on_disk, in_memory):
            target.SetGeoTransform((0.0, 1.0, 0.0, 32.0, 0.0, -1.0))
            band = target.GetRasterBand(1)
            band.SetRasterColorTable(table)
            band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
            band.WriteArray(np.zeros((32, 32), dtype="uint8"))
        on_disk = None
        handle = gdal.Open(path)
        handle.BuildOverviews("NEAREST", [2])
        handle = None
        in_memory.BuildOverviews("NEAREST", [2])

        lazy_parent = Dataset.read_file(path)
        memory_parent = Dataset(in_memory)
        levels = {}
        try:
            levels["lazy"] = lazy_parent.get_overview_dataset()
            levels["materialised"] = memory_parent.get_overview_dataset()
            for tag, level in levels.items():
                assert level.band_color == {0: "palette_index"}, (
                    f"{tag} level lost its colour interpretation: {level.band_color}"
                )
                # GTiff pads its palette out to 256 entries and MEM does not, so compare
                # the two the parent actually declared.
                entries = level.color_table[["red", "green", "blue", "alpha"]]
                assert entries.values.tolist()[:2] == [
                    [10, 20, 30, 255],
                    [200, 100, 50, 255],
                ], f"{tag} level lost its colour table:\n{level.color_table.head()}"
        finally:
            for level in levels.values():
                level.close()
            lazy_parent.close()
            memory_parent.close()

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
