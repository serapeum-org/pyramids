"""Tests for :meth:`pyramids.dataset.DatasetCollection.to_netcdf` (PY-4).

``to_netcdf`` writes through pyramids' own GDAL multidimensional NetCDF writer
and needs no third-party NetCDF engine, so the whole
module is ``core`` and runs in the extras-free suite. The companion
``test_to_netcdf_no_engine.py`` pins that contract by masking the engine
and asserting the write still succeeds.

Inspection round-trip is done with :func:`osgeo.gdal.OpenEx` in
``OF_MULTIDIM_RASTER`` mode so the assertions never require a NetCDF engine.
"""

from __future__ import annotations

import datetime as dt
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.netcdf import NetCDF
from tests.dataset.collection._helpers import (
    make_int16_collection as _make_int16_collection,
)

pytestmark = pytest.mark.core


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
        assert attrs.get("Conventions") == "CF-1.8", (
            f"missing Conventions=CF-1.8: {attrs!r}"
        )
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
        assert "positional" in time_attrs["note"].lower(), (
            f"unexpected note: {time_attrs['note']!r}"
        )

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
            assert np.array_equal(values[i], expected), (
                f"timestep {i} disk-array mismatch"
            )


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
        assert "note" not in _array_attrs(str(out), "time"), (
            "note attr leaked into explicit time"
        )

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
        assert unit == "nanoseconds since 1970-01-01 00:00:00", (
            f"missing CF unit: {unit!r}"
        )
        attrs = {a.GetName(): a.Read() for a in time_arr.GetAttributes()}
        assert attrs.get("calendar") == "proleptic_gregorian", (
            f"missing calendar: {attrs!r}"
        )

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

    def test_time_axis_used_as_default_time_coords(self, tmp_path):
        """A dated collection exports its own ``time`` axis when ``time_coords`` is omitted.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``col.time`` set to two dates, ``to_netcdf`` with no ``time_coords`` —
            expected: the same int64-nanosecond encoding as passing those dates
            explicitly, and no positional ``note`` attr.
        """
        col, _ = _make_int16_collection(tmp_path, count=2)
        col.time = [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)]
        out = tmp_path / "default_time.nc"
        col.to_netcdf(str(out))  # no time_coords -> falls back to self.time
        values = _array_values(str(out), "time")
        expected_ns = [1577836800 * 1_000_000_000, 1577923200 * 1_000_000_000]
        assert values.tolist() == expected_ns, f"self.time not used: {values!r}"
        assert "note" not in _array_attrs(str(out), "time"), (
            "positional note leaked into a dated export"
        )

    def test_explicit_time_coords_override_time_axis(self, tmp_path):
        """An explicit ``time_coords`` beats the collection's own ``time`` axis.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``col.time`` set to dates, but ``to_netcdf(time_coords=[10, 20])`` —
            expected: the explicit integers win over the dated default.
        """
        col, _ = _make_int16_collection(tmp_path, count=2)
        col.time = [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)]
        out = tmp_path / "override.nc"
        col.to_netcdf(str(out), time_coords=[10, 20])
        assert _array_values(str(out), "time").tolist() == [
            10,
            20,
        ], "explicit time_coords did not override the time axis"

    def test_integer_time_axis_exported_as_is(self, tmp_path):
        """Integer ``time`` order keys export as an integer axis, not ``0..T-1``.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A ``date=False`` read leaves ``self.time`` holding integer order keys;
            ``to_netcdf`` forwards them verbatim (they are real parsed values, not
            a made-up positional index).
        """
        col, _ = _make_int16_collection(tmp_path, count=2)
        col.time = [5, 9]
        out = tmp_path / "int_axis.nc"
        col.to_netcdf(str(out))
        assert _array_values(str(out), "time").tolist() == [
            5,
            9,
        ], "integer time axis not forwarded verbatim"

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
        assert int(values[1] - values[0]) == 500_000_000, (
            f"sub-second delta lost: {values!r}"
        )

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
        path = str(tmp_path / "bad.nc")
        with pytest.raises(ValueError, match=r"has 3 entries but"):
            col.to_netcdf(path, time_coords=[1, 2, 3])

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
        assert any("monotonically" in str(w.message) for w in caught), (
            f"missing monotonic warning, got: {[str(w.message) for w in caught]}"
        )

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
        assert any("duplicate" in str(w.message) for w in caught), (
            f"missing duplicate warning, got: {[str(w.message) for w in caught]}"
        )

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
        assert "band" not in names, (
            f"unwanted band dim leaked into var_per_band=True: {names}"
        )


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
        assert attrs.get("nodata") == -9999, (
            f"nodata root attr missing/wrong: {attrs!r}"
        )

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
        assert attrs.get("nodata") == -9999, (
            f"per-var nodata attr missing/wrong: {attrs!r}"
        )

    def test_data_variable_declares_cf_fill_value(self, tmp_path):
        """The data variable declares a CF ``_FillValue`` so CF readers mask the fill (#1061).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            int16 collection with ``no_data_value=-9999`` — expected: the classic (CF) view a
            reader like Panoply sees carries ``Band_1#_FillValue == -9999``. A bare ``nodata``
            attribute is not honored by CF readers, so without ``_FillValue`` the fill folds into
            the color scale and hides the data.
        """
        col, _ = _make_int16_collection(tmp_path, no_data_value=-9999)
        out = tmp_path / "nd_fill.nc"
        col.to_netcdf(str(out))
        md = gdal.Open(str(out)).GetMetadata()
        fill = md.get("Band_1#_FillValue")
        assert fill is not None, f"data variable must declare a CF _FillValue: {md}"
        assert float(fill) == -9999.0, (
            f"_FillValue should equal the nodata, got {fill!r}"
        )

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

    def test_var_per_band_false_declares_cf_fill_value(self, tmp_path):
        """The single 4-D ``data`` variable also declares a CF ``_FillValue`` (#1061).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``var_per_band=False`` int16 collection with ``no_data_value=-9999`` —
            expected: the classic (CF) view carries ``data#_FillValue == -9999`` so a CF
            reader masks the fill in the 4-D ``data`` variable too, not only in the
            per-band ``Band_1`` layout. Without the ``_FillValue`` the bare ``nodata``
            attribute is ignored and the fill folds into the color scale.
        """
        col, _ = _make_int16_collection(tmp_path, no_data_value=-9999)
        out = tmp_path / "nd_fill_4d.nc"
        col.to_netcdf(str(out), var_per_band=False)
        md = gdal.Open(str(out)).GetMetadata()
        fill = md.get("data#_FillValue")
        assert fill is not None, (
            f"the 4-D data variable must declare a CF _FillValue: {md}"
        )
        assert float(fill) == -9999.0, (
            f"_FillValue should equal the nodata, got {fill!r}"
        )

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
        assert "nodata" not in _root_attrs(str(out)), (
            "nodata leaked when source has no nodata"
        )
        assert "nodata" not in _array_attrs(str(out), "Band_1"), (
            "per-var nodata leaked when source has no nodata"
        )

    def test_nodata_round_trips_via_read_file(self, tmp_path):
        """The written ``nodata`` attribute is restored as ``no_data_value`` on read.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write an int16 collection with ``no_data_value=-9999`` and reopen it with
            :meth:`NetCDF.read_file` — expected: every band of ``Band_1`` reports ``-9999``
            (GDAL's classic view carries no band no-data, so the read falls back to the
            ``nodata`` attribute). Regression test for #935.
        """
        col, _ = _make_int16_collection(tmp_path, no_data_value=-9999)
        out = tmp_path / "nd_rt.nc"
        col.to_netcdf(str(out))
        var = NetCDF.read_file(str(out)).get_variable("Band_1")
        assert all(v == -9999 for v in var.no_data_value), (
            f"nodata did not round-trip: {var.no_data_value!r}"
        )

    def test_no_nodata_round_trips_as_none(self, tmp_path):
        """A source without no-data reopens with ``no_data_value`` still ``None`` (no false positive).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Source raster with ``no_data_value=None`` — expected: the reopened variable
            reports ``None`` for every band, i.e. the attribute fallback invents nothing.
        """
        p = os.path.join(str(tmp_path), "no_nd_rt.tif")
        Dataset.create_from_array(
            np.arange(20, dtype="int16").reshape(4, 5),
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=None,
            path=p,
        ).close()
        out = tmp_path / "no_nd_rt.nc"
        DatasetCollection.from_files([p]).to_netcdf(str(out))
        var = NetCDF.read_file(str(out)).get_variable("Band_1")
        assert all(v is None for v in var.no_data_value), (
            f"expected all-None no_data_value, got {var.no_data_value!r}"
        )

    def test_nodata_round_trips_var_per_band_false(self, tmp_path):
        """The single 4-D ``data`` variable also restores its no-data on read.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``var_per_band=False`` write with ``no_data_value=-9999`` — expected: the reopened
            ``data`` variable reports ``-9999`` on every band, i.e. the attribute fallback covers
            the 4-D layout too. Regression test for #935.
        """
        col, _ = _make_int16_collection(tmp_path, no_data_value=-9999)
        out = tmp_path / "nd_4d_rt.nc"
        col.to_netcdf(str(out), var_per_band=False)
        var = NetCDF.read_file(str(out)).get_variable("data")
        assert all(v == -9999 for v in var.no_data_value), (
            f"4-D nodata did not round-trip: {var.no_data_value!r}"
        )

    @staticmethod
    def _float_collection(tmp_path, fill: float) -> DatasetCollection:
        """Build a float32 collection whose nodata is ``fill`` and one cell stamped to it."""
        paths = []
        for i in range(2):
            arr = (np.arange(20, dtype="float32").reshape(4, 5) + i).astype("float32")
            arr[0, 0] = fill
            p = os.path.join(str(tmp_path), f"f{i}.tif")
            Dataset.create_from_array(
                arr,
                top_left_corner=(0, 0),
                cell_size=0.05,
                epsg=4326,
                no_data_value=fill,
                path=p,
            ).close()
            paths.append(p)
        return DatasetCollection.from_files(paths)

    def test_float_flt_max_fill_value_matches_in_band(self, tmp_path):
        """A float32 ``-FLT_MAX`` nodata becomes a CF ``_FillValue`` equal to the in-band fill (#1061).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The reported dtype — a float32 collection whose nodata is ``-FLT_MAX`` (the fill that hid
            the Coello evap data), with one cell stamped to it. Expected: the classic (CF) view carries
            ``Band_1#_FillValue == -FLT_MAX`` *and* the read-back band still holds ``-FLT_MAX`` there,
            so a CF reader masks that cell instead of scaling to it. Guards the ``double``->``float32``
            cast the int-only tests cannot.
        """
        fill = float(np.finfo("float32").min)
        out = tmp_path / "float_fill.nc"
        self._float_collection(tmp_path, fill).to_netcdf(str(out))
        declared = gdal.Open(str(out)).GetMetadata().get("Band_1#_FillValue")
        assert declared is not None, "float data variable must declare a CF _FillValue"
        assert np.float32(declared) == np.float32(fill), (
            f"_FillValue should equal -FLT_MAX, got {declared!r}"
        )
        in_band = _array_values(str(out), "Band_1")[0, 0, 0]
        assert np.float32(in_band) == np.float32(fill), (
            f"the stamped nodata cell should read back as the fill, got {in_band!r}"
        )

    def test_nan_nodata_declares_nan_fill_value(self, tmp_path):
        """A NaN nodata becomes a NaN CF ``_FillValue`` (#1061).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A float32 collection with ``no_data_value=nan`` — expected: ``Band_1#_FillValue`` parses to
            NaN, confirming ``SetNoDataValueDouble(nan)`` is accepted (not a silent no-op) so a CF
            reader can mask NaN fills.
        """
        out = tmp_path / "nan_fill.nc"
        self._float_collection(tmp_path, float("nan")).to_netcdf(str(out))
        declared = gdal.Open(str(out)).GetMetadata().get("Band_1#_FillValue")
        assert declared is not None, "NaN nodata must still declare a CF _FillValue"
        # Check the classic-view string directly so the assertion cannot pass for a missing value
        # (`np.float32(None)` is itself NaN); GDAL writes a NaN _FillValue as the literal "nan".
        assert str(declared).strip().lower() == "nan", (
            f"_FillValue should be NaN, got {declared!r}"
        )


class TestToNetcdfNoFilesPath:
    """Support for collections that have no ``_files`` (e.g. ``create``)."""

    def test_create_collection_writes_successfully(self, tmp_path):
        """A ``create``-backed collection (no file list) can still be written.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Build via :meth:`DatasetCollection.create` (legacy path
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
        col = DatasetCollection.from_dataset(src, 3)
        out = tmp_path / "nf.nc"
        col.to_netcdf(str(out))
        assert out.exists(), "no-files write did not produce a file"
        values = _array_values(str(out), "Band_1")
        assert values.shape == (3, 4, 5), f"unexpected shape: {values.shape}"


class TestToNetcdfStreaming:
    """ARC-46: ``to_netcdf`` streams timestep-by-timestep, never the full cube."""

    def test_does_not_stack_the_full_cube(self, tmp_path, monkeypatch):
        """``to_netcdf`` must not materialise the whole cube via ``np.stack``.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Patch ``np.stack`` in the collection module to raise, then write.
            The eager implementation stacked every timestep and would trip it;
            the streaming implementation writes one slab per timestep, so the
            file is produced correctly with ``np.stack`` disabled.
        """
        col, _ = _make_int16_collection(tmp_path, count=3)
        import pyramids.dataset.collection as _collection

        def _no_stack(*_args, **_kwargs):
            raise AssertionError("to_netcdf must not materialise the full cube")

        monkeypatch.setattr(_collection.np, "stack", _no_stack)
        out = tmp_path / "streamed.nc"
        col.to_netcdf(str(out))
        assert out.exists(), "streaming write produced no file"
        values = _array_values(str(out), "Band_1")
        assert values.shape == (3, 4, 5), f"unexpected shape: {values.shape}"

    def test_multi_band_multi_timestep_streams_var_per_band_false(self, tmp_path):
        """A multi-band cube streams into the 4-D ``data`` variable correctly.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Three 2-band timesteps written with ``var_per_band=False`` round-trip
            to a ``(T, band, y, x)`` array equal to the source, exercising the
            per-timestep ``(B, Y, X)`` slab path.
        """
        arrays = []
        paths = []
        for t in range(3):
            arr = np.stack(
                [
                    np.full((4, 5), t + 1, dtype="int16"),
                    np.full((4, 5), 100 + t, dtype="int16"),
                ],
                axis=0,
            )
            p = str(tmp_path / f"mb_{t}.tif")
            Dataset.create_from_array(
                arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326, path=p
            ).close()
            arrays.append(arr)
            paths.append(p)
        col = DatasetCollection.from_files(paths)
        out = tmp_path / "mb.nc"
        col.to_netcdf(str(out), var_per_band=False)
        values = _array_values(str(out), "data")
        assert values.shape == (3, 2, 4, 5), f"unexpected shape: {values.shape}"
        for t in range(3):
            np.testing.assert_array_equal(
                values[t], arrays[t], err_msg=f"timestep {t} mismatch"
            )

    def test_mismatched_timestep_shape_raises_alignment_error(self, tmp_path):
        """A timestep whose grid differs from the template raises AlignmentError.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A 4x5 template followed by a 3x3 raster (``from_files`` does not
            validate headers by default) — expected: a clear AlignmentError
            naming the offending file, and no partial file left on disk.
        """
        paths = []
        for t, shape in enumerate([(4, 5), (3, 3)]):
            p = str(tmp_path / f"h_{t}.tif")
            Dataset.create_from_array(
                np.zeros(shape, dtype="int16"),
                top_left_corner=(0, 0),
                cell_size=0.05,
                epsg=4326,
                path=p,
            ).close()
            paths.append(p)
        col = DatasetCollection.from_files(paths)
        out = tmp_path / "bad.nc"
        with pytest.raises(AlignmentError, match="must share the base grid"):
            col.to_netcdf(str(out))
        assert not out.exists(), "a mismatched timestep must leave no partial file"

    def test_failed_overwrite_preserves_existing_file(self, tmp_path):
        """A failed overwrite leaves the pre-existing file untouched (atomic).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write a valid cube, then overwrite the same path from a collection
            whose 2nd timestep mismatches — expected: AlignmentError, and the
            original file's bytes are unchanged (temp write + os.replace).
        """
        out = tmp_path / "cube.nc"
        good, _ = _make_int16_collection(tmp_path, count=2)
        good.to_netcdf(str(out))
        original = out.read_bytes()

        paths = []
        for i, shape in enumerate([(4, 5), (3, 3)]):
            p = str(tmp_path / f"ov_{i}.tif")
            Dataset.create_from_array(
                np.zeros(shape, dtype="int16"),
                top_left_corner=(0, 0),
                cell_size=0.05,
                epsg=4326,
                path=p,
            ).close()
            paths.append(p)
        bad = DatasetCollection.from_files(paths)
        with pytest.raises(AlignmentError):
            bad.to_netcdf(str(out))
        assert out.exists(), "existing file must survive a failed overwrite"
        assert out.read_bytes() == original, "existing file was modified"

    def test_no_temp_file_left_after_success(self, tmp_path):
        """A successful write leaves no temporary sibling file behind.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            After a normal ``to_netcdf`` the atomic temp has been ``os.replace``-d
            onto the destination — expected: only the ``.nc`` output remains, no
            leftover ``.tmp`` file.
        """
        col, _ = _make_int16_collection(tmp_path, count=2)
        out = tmp_path / "clean.nc"
        col.to_netcdf(str(out))
        assert out.exists(), "output not written"
        strays = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert not strays, f"stray temp file(s) left: {strays}"

    def test_empty_collection_raises_value_error(self, tmp_path):
        """An empty collection (``time_length == 0``) raises a clear ValueError.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``from_dataset(base, 0)`` builds a zero-timestep collection — expected:
            ``to_netcdf`` raises a ValueError naming the empty collection and
            writes no file.
        """
        base = Dataset.create_from_array(
            np.zeros((4, 5), dtype="int16"),
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
        )
        col = DatasetCollection.from_dataset(base, 0)
        out = tmp_path / "empty.nc"
        with pytest.raises(ValueError, match="empty collection"):
            col.to_netcdf(str(out))
        assert not out.exists(), "no file should be written for an empty collection"

    def test_var_per_band_true_multi_timestep_streams_each_band(self, tmp_path):
        """``var_per_band=True`` streams each band into its own variable per step.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Three 2-band timesteps written with ``var_per_band=True`` — expected:
            each band becomes a ``(T, y, x)`` variable equal to that band across
            timesteps, exercising the per-band-per-timestep slab loop.
        """
        band0, band1, paths = [], [], []
        for t in range(3):
            b0 = np.full((4, 5), t + 1, dtype="int16")
            b1 = np.full((4, 5), 100 + t, dtype="int16")
            p = str(tmp_path / f"vb_{t}.tif")
            Dataset.create_from_array(
                np.stack([b0, b1], axis=0),
                top_left_corner=(0, 0),
                cell_size=0.05,
                epsg=4326,
                path=p,
            ).close()
            band0.append(b0)
            band1.append(b1)
            paths.append(p)
        col = DatasetCollection.from_files(paths)
        names = (
            list(col.meta.band_names) if col.meta.band_names else ["band_1", "band_2"]
        )
        out = tmp_path / "vb.nc"
        col.to_netcdf(str(out), var_per_band=True)
        v0 = _array_values(str(out), names[0])
        v1 = _array_values(str(out), names[1])
        assert v0.shape == (3, 4, 5), f"{names[0]} shape: {v0.shape}"
        for t in range(3):
            np.testing.assert_array_equal(v0[t], band0[t], err_msg=f"{names[0]} t={t}")
            np.testing.assert_array_equal(v1[t], band1[t], err_msg=f"{names[1]} t={t}")


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
        assert "Band_1" in nc.variables, (
            f"Band_1 missing from variables: {list(nc.variables)}"
        )
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

    def test_calendar_time_axis_decodes_via_reader(self, tmp_path):
        """The nanosecond-encoded time axis decodes back to calendar dates via the reader.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write with a :class:`pandas.DatetimeIndex` (encoded as CF ``nanoseconds since
            1970-01-01``) and reopen — expected: :meth:`NetCDF.get_time_variable` and the
            ``time_stamp`` property return the original dates rather than raising on the
            ``nanoseconds`` unit. Regression test for #936.
        """
        col, _ = _make_int16_collection(tmp_path, count=2)
        out = tmp_path / "cal.nc"
        col.to_netcdf(
            str(out), time_coords=pd.date_range("1979-01-01", periods=2, freq="D")
        )
        nc = NetCDF.read_file(str(out))
        assert nc.get_time_variable() == ["1979-01-01", "1979-01-02"], (
            f"time axis did not decode: {nc.get_time_variable()!r}"
        )
        assert nc.time_stamp == ["1979-01-01", "1979-01-02"], (
            f"time_stamp did not decode: {nc.time_stamp!r}"
        )

    def test_geotransform_round_trips_nonunit_cell_size(self, tmp_path):
        """``NetCDF.geotransform`` round-trips a non-unit projected cell size on both axes.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write a collection on a projected 5000 m grid (EPSG:4647) and reopen —
            expected: ``nc.geotransform`` equals the source transform on *both*
            axes, not an index-space ``pixel width 1.0`` with a half-pixel-shifted
            x origin. Regression test for #1014, where the reader took the x pixel
            size from the index-space ``cell_size`` while y used the real coord
            spacing.
        """
        nd, cell, ox, oy = -9999.0, 5000.0, 32239263.70388, 5756081.42235
        paths = []
        for i in range(3):
            arr = np.full((5, 4), nd, dtype="float32")
            arr[1:4, 1:3] = 10.0 + i
            path = os.path.join(str(tmp_path), f"g{i}.tif")
            Dataset.create_from_array(
                arr,
                geo=(ox, cell, 0.0, oy, 0.0, -cell),
                epsg=4647,
                no_data_value=nd,
            ).to_file(path)
            paths.append(path)
        out = tmp_path / "gt.nc"
        DatasetCollection.from_files(paths).to_netcdf(str(out))
        nc = NetCDF.read_file(str(out))
        expected = (ox, cell, 0.0, oy, 0.0, -cell)
        got = tuple(float(v) for v in nc.geotransform)
        assert got == pytest.approx(expected), (
            f"geotransform did not round-trip: {got} != {expected}"
        )
        assert got[1] == pytest.approx(cell), (
            f"x pixel size not the real spacing: {got[1]}"
        )


class TestToNetcdfCfCoordinates:
    """CF-complete coordinates + grid_mapping so third-party CF readers (Panoply) work."""

    @staticmethod
    def _classic_md(path) -> dict:
        """Return the classic-mode metadata dict (the CF view a tool like Panoply reads)."""
        return gdal.Open(str(path)).GetMetadata()

    def test_geographic_coords_carry_cf_axis_attributes(self, tmp_path):
        """A geographic grid writes CF ``axis``/``standard_name``/degrees units on x/y.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            EPSG:4326 collection — expected: ``x`` carries ``degrees_east``/``longitude``/
            ``X`` and ``y`` carries ``degrees_north``/``latitude``/``Y``, so a CF reader can
            identify the axes (the #1017 Panoply "X-dimension index is not set" failure).
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "geo.nc"
        col.to_netcdf(str(out))
        md = self._classic_md(out)
        assert md.get("x#units") == "degrees_east", f"x#units={md.get('x#units')!r}"
        assert md.get("x#standard_name") == "longitude"
        assert md.get("x#axis") == "X"
        assert md.get("y#units") == "degrees_north"
        assert md.get("y#axis") == "Y"

    def test_grid_mapping_variable_written_and_linked(self, tmp_path):
        """A CF grid_mapping variable is written and the data variable references it.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Expected: ``Band_1#grid_mapping`` names a variable that carries
            ``grid_mapping_name`` and ``crs_wkt``, so the CRS is CF-discoverable.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "gm.nc"
        col.to_netcdf(str(out))
        md = self._classic_md(out)
        gm = md.get("Band_1#grid_mapping")
        assert gm, f"data variable should reference a grid_mapping var: {md}"
        assert md.get(f"{gm}#grid_mapping_name"), (
            "grid_mapping var lacks grid_mapping_name"
        )
        assert md.get(f"{gm}#crs_wkt"), "grid_mapping var lacks crs_wkt"

    def test_grid_mapping_var_not_exposed_and_crs_round_trips(self, tmp_path):
        """The auto grid_mapping variable stays out of variable_names and the CRS round-trips.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Expected: ``variable_names == ["Band_1"]`` (no ``crs``/``spatial_ref`` leak) and
            ``epsg == 4326`` on reopen — the fix does not regress the reader.
        """
        col, _ = _make_int16_collection(tmp_path)
        out = tmp_path / "rt.nc"
        col.to_netcdf(str(out))
        nc = NetCDF.read_file(str(out))
        assert nc.variable_names == ["Band_1"], (
            f"grid_mapping var leaked into variable_names: {nc.variable_names}"
        )
        assert nc.epsg == 4326, f"CRS did not round-trip: {nc.epsg}"

    def test_projected_coords_use_metre_units(self, tmp_path):
        """A projected grid writes metre units and projection_x/y_coordinate standard names.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            EPSG:32632 (UTM 32N) collection — expected: ``x``/``y`` carry ``m`` and
            ``projection_x_coordinate`` / ``projection_y_coordinate``.
        """
        paths = []
        for i in range(2):
            arr = np.arange(20, dtype="int16").reshape(4, 5) + 100 * i
            p = os.path.join(str(tmp_path), f"p{i}.tif")
            Dataset.create_from_array(
                arr,
                top_left_corner=(400000.0, 5800000.0),
                cell_size=5000.0,
                epsg=32632,
                no_data_value=-9999,
                path=p,
            ).close()
            paths.append(p)
        out = tmp_path / "proj.nc"
        DatasetCollection.from_files(paths).to_netcdf(str(out))
        md = self._classic_md(out)
        assert md.get("x#units") == "m", f"x#units={md.get('x#units')!r}"
        assert md.get("x#standard_name") == "projection_x_coordinate"
        assert md.get("y#standard_name") == "projection_y_coordinate"

    def test_streaming_op_writes_cf_coords(self, tmp_path):
        """A streaming raster op (``resample`` with ``path=``) also writes CF-complete coords.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``to_netcdf`` a collection, reopen, ``resample(..., path=)`` on the container —
            expected: the streamed output carries the CF x/y attrs and a ``grid_mapping``,
            since the fix covers the streaming ops, not only ``to_netcdf``.
        """
        col, _ = _make_int16_collection(tmp_path)
        src = tmp_path / "src.nc"
        col.to_netcdf(str(src))
        out = tmp_path / "resampled.nc"
        NetCDF.read_file(str(src)).resample(0.1, path=str(out))
        md = self._classic_md(out)
        assert md.get("x#units") == "degrees_east", f"x#units={md.get('x#units')!r}"
        assert md.get("Band_1#grid_mapping"), "streamed op should write a grid_mapping"
