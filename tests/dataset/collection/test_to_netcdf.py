"""Tests for :meth:`pyramids.dataset.DatasetCollection.to_netcdf` (PY-4).

``to_netcdf`` requires xarray, so the whole module is ``xarray``-marked and runs
only in the xarray CI job. The *missing*-xarray branch (the
``OptionalPackageDoesNotExist`` path) lives in ``test_to_netcdf_missing_xarray.py``
as a ``core`` test so it still runs in the extras-free suite.

Inspection round-trip is done with :func:`osgeo.gdal.OpenEx` in
``OF_MULTIDIM_RASTER`` mode so the assertions don't require an xarray
NetCDF engine (xarray in CI may not pull ``netcdf4``).
"""

from __future__ import annotations

import datetime as dt
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.xarray


def _root_attrs(path: str) -> dict:
    """Read the root-group attribute dict of a NetCDF via the multidim API.

    Args:
        path: Path to the NetCDF file.

    Returns:
        dict: ``{attr_name: value}`` for the root group.
    """
    g = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER).GetRootGroup()
    return {a.GetName(): a.Read() for a in g.GetAttributes()}


def _array_attrs(path: str, name: str) -> dict:
    """Read the attribute dict of a single MD-array variable.

    Args:
        path: Path to the NetCDF file.
        name: Name of the multidim array to inspect.

    Returns:
        dict: ``{attr_name: value}`` for the variable.
    """
    g = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER).GetRootGroup()
    arr = g.OpenMDArray(name)
    return {a.GetName(): a.Read() for a in arr.GetAttributes()}


def _array_values(path: str, name: str) -> np.ndarray:
    """Read the values of an MD-array variable.

    Args:
        path: Path to the NetCDF file.
        name: Name of the multidim array.

    Returns:
        np.ndarray: Variable values.
    """
    g = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER).GetRootGroup()
    return np.asarray(g.OpenMDArray(name).ReadAsArray())


def _make_int16_collection(tmp_path, count: int = 2, no_data_value: int = -9999):
    """Build a small int16 file-backed collection.

    Args:
        tmp_path: pytest temp directory.
        count: Number of timesteps to materialise.
        no_data_value: Value stamped as nodata on each timestep.

    Returns:
        tuple[DatasetCollection, list[str]]: the collection plus its
        backing paths, so tests can introspect ``_files``.
    """
    paths = []
    for i in range(count):
        arr = np.arange(20, dtype="int16").reshape(4, 5) + 100 * i
        p = os.path.join(str(tmp_path), f"t{i}.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=no_data_value,
            path=p,
        ).close()
        paths.append(p)
    return DatasetCollection.from_files(paths), paths


@pytest.mark.xarray
class TestToNetcdfDefaults:
    """Default-path tests (positional time index, var-per-band, CF root attrs)."""

    def test_writes_a_real_file(self, tmp_path):
        """A successful call writes a non-empty ``.nc`` file at the given path.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write the cube with no options — expected: the file exists and is
            non-empty.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "cube.nc"
        col.to_netcdf(str(out))
        assert out.exists(), "to_netcdf did not write the output file"
        assert out.stat().st_size > 0, "output file is empty"

    def test_returns_none(self, tmp_path):
        """The method is side-effectful — it returns ``None``.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Capture the return value — expected: ``None``.
        """
        col, _ = _make_int16_collection(tmp_path)
        result = col.to_netcdf(str(tmp_path / "x.nc"))
        assert result is None, f"to_netcdf should return None, got {result!r}"

    def test_default_var_per_band_writes_one_var_per_band_name(self, tmp_path):
        """A single-band collection writes one ``Band_1`` data variable.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Default args on a single-band collection — expected: only
            ``Band_1`` appears among the MD-arrays (plus the three coords).
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "default.nc"
        col.to_netcdf(str(out))
        g = gdal.OpenEx(str(out), gdal.OF_MULTIDIM_RASTER).GetRootGroup()
        names = sorted(g.GetMDArrayNames())
        assert names == [
            "Band_1",
            "time",
            "x",
            "y",
        ], f"unexpected MD-array names: {names}"

    def test_root_attrs_include_cf_and_geobox(self, tmp_path):
        """Root group carries ``Conventions=CF-1.8`` plus ``crs_wkt`` / ``epsg`` / ``GeoTransform``.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Inspect the root attrs — expected: all four keys present, with
            the right ``epsg`` and ``Conventions`` values.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "attrs.nc"
        col.to_netcdf(str(out))
        attrs = _root_attrs(str(out))
        assert (
            attrs.get("Conventions") == "CF-1.8"
        ), f"missing Conventions=CF-1.8: {attrs!r}"
        assert attrs.get("epsg") == 4326, f"unexpected epsg: {attrs.get('epsg')!r}"
        assert "crs_wkt" in attrs and attrs["crs_wkt"], "crs_wkt root attr missing"
        assert "GeoTransform" in attrs, "GeoTransform root attr missing"

    def test_time_dim_defaults_to_positional_index(self, tmp_path):
        """Without ``time_coords`` the time axis is ``0..T-1`` with a ``note`` attr.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Default call — expected: time values are ``[0, 1]`` and the
            ``time`` variable carries a ``note`` attr flagging it positional.
        """
        col, _ = _make_int16_collection(tmp_path, count=2)
        out = tmp_path / "time.nc"
        col.to_netcdf(str(out))
        assert _array_values(str(out), "time").tolist() == [
            0,
            1,
        ], "time values not 0..T-1"
        time_attrs = _array_attrs(str(out), "time")
        assert "note" in time_attrs, f"missing positional-time note: {time_attrs}"
        assert (
            "positional" in time_attrs["note"].lower()
        ), f"unexpected note: {time_attrs['note']!r}"

    def test_data_round_trips_per_timestep(self, tmp_path):
        """Each timestep's source array survives the write/read cycle.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``Band_1`` MD-array shape is ``(T, Y, X)`` and per-timestep
            values match the source rasters.
        """
        col, paths = _make_int16_collection(tmp_path, count=2)
        out = tmp_path / "rt.nc"
        col.to_netcdf(str(out))
        values = _array_values(str(out), "Band_1")
        assert values.shape == (2, 4, 5), f"expected (2,4,5), got {values.shape}"
        for i, p in enumerate(paths):
            expected = Dataset.read_file(p).read_array()
            assert np.array_equal(
                values[i], expected
            ), f"timestep {i} disk-array mismatch"


@pytest.mark.xarray
class TestToNetcdfTimeCoords:
    """``time_coords`` plumbing: explicit values, datetime coercion, warnings, errors."""

    def test_explicit_integer_time_coords(self, tmp_path):
        """Integer ``time_coords`` are written verbatim with no ``note`` attr.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``time_coords=[10, 20]`` — expected: values pass through; the
            positional-index ``note`` is absent.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "int_t.nc"
        col.to_netcdf(str(out), time_coords=[10, 20])
        assert _array_values(str(out), "time").tolist() == [
            10,
            20,
        ], "int time_coords lost"
        assert "note" not in _array_attrs(
            str(out), "time"
        ), "note attr leaked into explicit time"

    def test_explicit_float_time_coords(self, tmp_path):
        """Float ``time_coords`` pass through as floats.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``time_coords=[0.5, 1.5]`` — expected: float64 round-trip.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "float_t.nc"
        col.to_netcdf(str(out), time_coords=[0.5, 1.5])
        values = _array_values(str(out), "time")
        assert np.allclose(values, [0.5, 1.5]), f"float time_coords mangled: {values!r}"

    def test_pd_daterange_encoded_as_cf_nanoseconds(self, tmp_path):
        """A :class:`pandas.DatetimeIndex` is encoded as int64 nanoseconds + CF units.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``time_coords=pd.date_range("2020-01-01", periods=2, freq="D")`` —
            expected: int64 nanoseconds since the Unix epoch (so the round-trip
            preserves the full ``datetime64[ns]`` resolution); GDAL stores CF
            ``units`` via :meth:`MDArray.SetUnit` (not in the attribute dict),
            so probe it there; ``calendar`` stays in attrs.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "dates.nc"
        col.to_netcdf(
            str(out), time_coords=pd.date_range("2020-01-01", periods=2, freq="D")
        )
        values = _array_values(str(out), "time")
        expected_ns = [1577836800 * 1_000_000_000, 1577923200 * 1_000_000_000]
        assert values.tolist() == expected_ns, f"unexpected time values: {values!r}"
        g = gdal.OpenEx(str(out), gdal.OF_MULTIDIM_RASTER).GetRootGroup()
        time_arr = g.OpenMDArray("time")
        unit = time_arr.GetUnit()
        assert (
            unit == "nanoseconds since 1970-01-01 00:00:00"
        ), f"missing CF unit: {unit!r}"
        attrs = {a.GetName(): a.Read() for a in time_arr.GetAttributes()}
        assert (
            attrs.get("calendar") == "proleptic_gregorian"
        ), f"missing calendar: {attrs!r}"

    def test_list_of_datetime_objects_coerced(self, tmp_path):
        """A list of ``datetime`` objects gets coerced to datetime64 then encoded.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``time_coords=[datetime(2020, 1, 1), datetime(2020, 1, 2)]`` —
            expected: same encoded int64-nanoseconds values as the equivalent
            ``date_range``.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "list_dates.nc"
        col.to_netcdf(
            str(out),
            time_coords=[dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)],
        )
        values = _array_values(str(out), "time")
        expected_ns = [1577836800 * 1_000_000_000, 1577923200 * 1_000_000_000]
        assert values.tolist() == expected_ns, f"unexpected: {values!r}"

    def test_subsecond_datetime_roundtrips(self, tmp_path):
        """Sub-second timestamps survive the nanosecond CF encoding (L1 regression).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Two datetimes 500ms apart — expected: the on-disk int64 values
            preserve the millisecond delta exactly.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "subsec.nc"
        t0 = np.datetime64("2020-01-01T00:00:00.000", "ns")
        t1 = np.datetime64("2020-01-01T00:00:00.500", "ns")
        col.to_netcdf(str(out), time_coords=np.array([t0, t1], dtype="datetime64[ns]"))
        values = _array_values(str(out), "time")
        # 500ms in nanoseconds:
        assert (
            int(values[1] - values[0]) == 500_000_000
        ), f"sub-second delta lost: {values!r}"

    def test_generator_time_coords_materialised(self, tmp_path):
        """Generator ``time_coords`` are materialised via ``list()`` (L3 regression).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Pass an iterator (no ``__len__``) — expected: ``to_netcdf``
            consumes it and writes the right values, instead of raising a
            cryptic IndexError from ``np.asarray(generator)``.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "gen.nc"
        col.to_netcdf(str(out), time_coords=iter([10, 20]))
        assert _array_values(str(out), "time").tolist() == [
            10,
            20,
        ], "generator time_coords were not materialised"

    def test_length_mismatch_raises_value_error(self, tmp_path):
        """``len(time_coords) != self.time_length`` raises ``ValueError``.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            2-step collection with ``time_coords=[1, 2, 3]`` — expected:
            ``ValueError`` mentioning the offending count.
        """
        col, _ = _make_int16_collection(tmp_path, count=2)
        with pytest.raises(ValueError, match=r"has 3 entries but"):
            col.to_netcdf(str(tmp_path / "bad.nc"), time_coords=[1, 2, 3])

    def test_non_monotonic_time_coords_warns(self, tmp_path):
        """A non-monotonic ``time_coords`` triggers a :class:`UserWarning`.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``time_coords=[5, 3]`` — expected: a UserWarning mentioning
            ``monotonically``.
        """
        col, _ = _make_int16_collection(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            col.to_netcdf(str(tmp_path / "unsorted.nc"), time_coords=[5, 3])
        assert any(
            "monotonically" in str(w.message) for w in caught
        ), f"missing monotonic warning, got: {[str(w.message) for w in caught]}"

    def test_duplicate_time_coords_warns(self, tmp_path):
        """Duplicate ``time_coords`` triggers a :class:`UserWarning`.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``time_coords=[1, 1]`` — expected: a UserWarning mentioning
            ``duplicate``.
        """
        col, _ = _make_int16_collection(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            col.to_netcdf(str(tmp_path / "dupes.nc"), time_coords=[1, 1])
        assert any(
            "duplicate" in str(w.message) for w in caught
        ), f"missing duplicate warning, got: {[str(w.message) for w in caught]}"

    def test_custom_time_dim_name(self, tmp_path):
        """``time_dim`` renames the time dimension verbatim.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``time_dim="t"`` — expected: a ``t`` MD-array exists, ``time``
            does not.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "t_dim.nc"
        col.to_netcdf(str(out), time_dim="t")
        names = (
            gdal.OpenEx(str(out), gdal.OF_MULTIDIM_RASTER)
            .GetRootGroup()
            .GetMDArrayNames()
        )
        assert "t" in names, f"custom dim missing: {names}"
        assert "time" not in names, f"default 'time' should not appear: {names}"


@pytest.mark.xarray
class TestToNetcdfVarPerBand:
    """``var_per_band`` branch behaviour."""

    def test_var_per_band_false_writes_one_data_variable_with_band_dim(self, tmp_path):
        """``var_per_band=False`` writes a 4-D ``data`` variable with an integer ``band`` dim.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Default 1-band collection — expected: a ``data`` MD-array of
            shape ``(T, B, Y, X)`` plus a ``band`` MD-array of integer dtype.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "4d.nc"
        col.to_netcdf(str(out), var_per_band=False)
        g = gdal.OpenEx(str(out), gdal.OF_MULTIDIM_RASTER).GetRootGroup()
        names = sorted(g.GetMDArrayNames())
        assert names == ["band", "data", "time", "x", "y"], f"unexpected names: {names}"
        data = g.OpenMDArray("data").ReadAsArray()
        assert data.shape == (2, 1, 4, 5), f"unexpected 4D shape: {data.shape}"

    def test_var_per_band_false_writes_band_names_root_attr(self, tmp_path):
        """``var_per_band=False`` stores the band names as a root ``band_names`` attr.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Inspect the root attrs after a ``var_per_band=False`` write —
            expected: a ``band_names`` attr equal to ``"Band_1"`` (single-band
            collection).
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "bn.nc"
        col.to_netcdf(str(out), var_per_band=False)
        attrs = _root_attrs(str(out))
        assert attrs.get("band_names") == "Band_1", f"unexpected band_names: {attrs!r}"

    def test_var_per_band_true_no_band_dim(self, tmp_path):
        """``var_per_band=True`` (default) does NOT create a ``band`` dimension.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Default call — expected: no ``band`` MD-array.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "vpb.nc"
        col.to_netcdf(str(out))
        names = (
            gdal.OpenEx(str(out), gdal.OF_MULTIDIM_RASTER)
            .GetRootGroup()
            .GetMDArrayNames()
        )
        assert (
            "band" not in names
        ), f"unwanted band dim leaked into var_per_band=True: {names}"


@pytest.mark.xarray
class TestToNetcdfNoData:
    """No-data propagation through the writer (the ``_FillValue`` workaround)."""

    def test_nodata_on_root_group(self, tmp_path):
        """Source no-data is mirrored on the root ``nodata`` attribute.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            int16 collection with ``no_data_value=-9999`` — expected:
            ``root.attrs["nodata"] == -9999``.
        """
        col, _ = _make_int16_collection(tmp_path, no_data_value=-9999)
        out = tmp_path / "nd_root.nc"
        col.to_netcdf(str(out))
        attrs = _root_attrs(str(out))
        assert (
            attrs.get("nodata") == -9999
        ), f"nodata root attr missing/wrong: {attrs!r}"

    def test_nodata_on_each_variable(self, tmp_path):
        """Every data variable carries a per-variable ``nodata`` attribute.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Inspect ``Band_1.attrs`` — expected: ``nodata`` key matching the
            source raster's no-data value.
        """
        col, _ = _make_int16_collection(tmp_path, no_data_value=-9999)
        out = tmp_path / "nd_var.nc"
        col.to_netcdf(str(out))
        attrs = _array_attrs(str(out), "Band_1")
        assert (
            attrs.get("nodata") == -9999
        ), f"per-var nodata attr missing/wrong: {attrs!r}"

    def test_nodata_on_var_per_band_false(self, tmp_path):
        """In the 4-D layout the ``nodata`` attr lives on the single ``data`` variable.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``var_per_band=False`` — expected: ``data.attrs["nodata"]`` is
            populated.
        """
        col, _ = _make_int16_collection(tmp_path, no_data_value=-9999)
        out = tmp_path / "nd_4d.nc"
        col.to_netcdf(str(out), var_per_band=False)
        attrs = _array_attrs(str(out), "data")
        assert attrs.get("nodata") == -9999, f"4D nodata attr missing: {attrs!r}"

    def test_no_nodata_when_source_has_none(self, tmp_path):
        """A source raster with ``no_data_value=None`` writes no ``nodata`` attr.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Build a raster with ``no_data_value=None`` — expected: neither
            root nor variable carries a ``nodata`` attribute after the write.
        """
        p = os.path.join(str(tmp_path), "no_nd.tif")
        Dataset.create_from_array(
            np.arange(20, dtype="int16").reshape(4, 5),
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=None,
            path=p,
        ).close()
        col = DatasetCollection.from_files([p])
        out = tmp_path / "no_nd.nc"
        col.to_netcdf(str(out))
        assert "nodata" not in _root_attrs(
            str(out)
        ), "nodata leaked when source has no nodata"
        assert "nodata" not in _array_attrs(
            str(out), "Band_1"
        ), "per-var nodata leaked when source has no nodata"


@pytest.mark.xarray
class TestToNetcdfNoFilesPath:
    """Support for collections that have no ``_files`` (e.g. ``create_cube``)."""

    def test_create_cube_collection_writes_successfully(self, tmp_path):
        """A ``create_cube``-backed collection (no file list) can still be written.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Build via :meth:`DatasetCollection.create_cube` (legacy path
            that stamps a single ``Dataset`` repeated T times) — expected:
            the writer materialises from ``self.datasets`` and produces a
            real file.
        """
        src_path = os.path.join(str(tmp_path), "src.tif")
        Dataset.create_from_array(
            np.arange(20, dtype="int16").reshape(4, 5),
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
            path=src_path,
        ).close()
        src = Dataset.read_file(src_path)
        col = DatasetCollection.create_cube(src, 3)
        out = tmp_path / "nf.nc"
        col.to_netcdf(str(out))
        assert out.exists(), "no-files write did not produce a file"
        values = _array_values(str(out), "Band_1")
        assert values.shape == (3, 4, 5), f"unexpected shape: {values.shape}"


@pytest.mark.xarray
class TestToNetcdfRoundTrip:
    """End-to-end: re-open the written file via :class:`NetCDF` and compare."""

    def test_round_trip_via_netcdf_read_file(self, tmp_path):
        """:meth:`NetCDF.read_file` reopens the file with the expected vars + epsg.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write → ``NetCDF.read_file`` — expected: the variable list
            contains ``Band_1`` and ``epsg == 4326``.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "rt.nc"
        col.to_netcdf(str(out))
        nc = NetCDF.read_file(str(out))
        assert (
            "Band_1" in nc.variables
        ), f"Band_1 missing from variables: {list(nc.variables)}"
        assert nc.epsg == 4326, f"epsg lost: {nc.epsg}"

    def test_single_timestep_writes_length_one_time_dim(self, tmp_path):
        """``time_length == 1`` is supported and yields a length-1 time axis.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Single-file collection — expected: ``time`` MD-array has length 1
            and the data array has shape ``(1, Y, X)``.
        """
        col, _ = _make_int16_collection(tmp_path, count=1)
        out = tmp_path / "one.nc"
        col.to_netcdf(str(out))
        time_vals = _array_values(str(out), "time")
        assert time_vals.shape == (1,), f"expected length-1 time, got {time_vals.shape}"
        data = _array_values(str(out), "Band_1")
        assert data.shape == (1, 4, 5), f"expected (1,4,5), got {data.shape}"
