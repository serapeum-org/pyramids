"""Unit tests for the container / variable summary helpers (#1090).

`__str__` used to be defined once on `NetCDF` and inherited by both subclasses, so a container
fell through to `rows` / `columns` / `cell_size` — which for an MDIM store have no raster behind
them and return GDAL's in-memory placeholder. `NetCDF.__str__` now picks the summary by **type**,
through `_summary_text`, which `Container` and `Variable` override — a band-count test would
mislabel a classic-mode container, which carries the store's bands directly.

These cover the helpers directly; `tests/netcdf/test_netcdf_core.py::TestStr` covers the dispatch
end to end.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF
from pyramids.netcdf.models import MAX_DISPLAY_VARIABLES
from pyramids.netcdf.netcdf import (
    _both_nan,
    _capped_join,
    _collapse_uniform,
    _container_summary,
    _has_georeference,
    _store_label,
    _variable_summary,
)

pytestmark = pytest.mark.core


def _write_store(path: str, variable_count: int = 1, unit: str | None = "K") -> str:
    """Write a small NetCDF-4 store with `variable_count` data variables.

    Args:
        path: Output ``.nc`` path.
        variable_count: How many 2-D data variables to create.
        unit: CF ``units`` to stamp on each variable, or ``None`` for none.

    Returns:
        str: ``path``, for chaining.
    """
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path, [], ["FORMAT=NC4"])
    rg = ds.GetRootGroup()
    d_y = rg.CreateDimension("y", "", "", 2)
    d_x = rg.CreateDimension("x", "", "", 3)
    lat = rg.CreateMDArray("y", [d_y], gdal.ExtendedDataType.Create(gdal.GDT_Float64))
    lat.Write(np.array([51.0, 50.75]))
    lon = rg.CreateMDArray("x", [d_x], gdal.ExtendedDataType.Create(gdal.GDT_Float64))
    lon.Write(np.array([3.0, 3.25, 3.5]))
    for arr, std, axis, units in (
        (lat, "latitude", "Y", "degrees_north"),
        (lon, "longitude", "X", "degrees_east"),
    ):
        for key, value in (("standard_name", std), ("axis", axis), ("units", units)):
            attr = arr.CreateAttribute(key, [], gdal.ExtendedDataType.CreateString())
            attr.Write(value)
    made = []
    for index in range(variable_count):
        var = rg.CreateMDArray(
            f"v{index}", [d_y, d_x], gdal.ExtendedDataType.Create(gdal.GDT_Float32)
        )
        var.Write(np.zeros((2, 3), "float32"))
        if unit is not None:
            attr = var.CreateAttribute(
                "units", [], gdal.ExtendedDataType.CreateString()
            )
            attr.Write(unit)
        made.append(var)
    made = lat = lon = d_y = d_x = rg = None
    ds.Close()
    del ds
    gc.collect()
    return path


class TestCollapseUniform:
    """`_collapse_uniform` — one value when every band agrees, the sequence when they differ."""

    def test_uniform_sequence_collapses(self):
        """A sequence whose entries all match reports the single value.

        Test scenario:
            The reason the helper exists: a 12-band variable reports `['float32'] * 12`, which is
            true and unreadable in a summary.
        """
        assert _collapse_uniform(["float32"] * 12) == "float32"

    def test_mixed_sequence_is_left_alone(self):
        """A genuinely mixed sequence keeps its per-band detail.

        Test scenario:
            Collapsing here would hide the information the caller needs.
        """
        assert _collapse_uniform(["float32", "int16"]) == ["float32", "int16"]

    def test_all_nan_sequence_collapses(self):
        """An all-`nan` no-data tuple collapses despite `nan != nan`.

        Test scenario:
            `no_data_value` comes back as a tuple of `nan`; plain equality would call it mixed and
            print the whole tuple.
        """
        # Distinct nan objects on purpose: `(float("nan"),) * 12` repeats one object, so the
        # identity check would collapse it even with the nan handling removed.
        result = _collapse_uniform(tuple(float("nan") for _ in range(12)))
        assert isinstance(result, float), f"expected a single float, got {result!r}"
        assert np.isnan(result), (
            f"expected the collapsed value to be nan, got {result!r}"
        )

    def test_empty_sequence_is_none(self):
        """An empty sequence yields `None`, so the caller omits the line."""
        assert _collapse_uniform([]) is None

    @pytest.mark.parametrize("value", ["float32", 3], ids=["str", "int"])
    def test_scalar_passes_through(self, value):
        """A non-sequence is returned unchanged.

        Args:
            value: A scalar the property might already have collapsed.

        Test scenario:
            Not every per-band property is a list; the helper must be safe to apply blindly.
        """
        assert _collapse_uniform(value) == value


class TestBothNan:
    """`_both_nan` — the one comparison `==` gets wrong."""

    def test_two_nans_match(self):
        """Two `nan`s are treated as equal."""
        assert _both_nan(float("nan"), float("nan")) is True

    @pytest.mark.parametrize(
        "left, right", [(float("nan"), 1.0), (1.0, 1.0)], ids=["one-nan", "no-nan"]
    )
    def test_non_nan_pairs_do_not_match(self, left, right):
        """Anything that is not a pair of `nan`s is False.

        Args:
            left: First value.
            right: Second value.
        """
        assert _both_nan(left, right) is False

    def test_non_numeric_is_survivable(self):
        """A value `np.isnan` cannot handle returns False rather than raising.

        Test scenario:
            `band_units` holds strings, and the helper is applied to every per-band sequence, so
            it must not raise on one.
        """
        assert _both_nan("K", "K") is False


class TestCrsLabel:
    """`NetCDF._crs_label` — a short label, never the raw WKT."""

    def test_epsg_is_preferred(self, tmp_path):
        """A store with an EPSG reports `EPSG:<code>`."""
        nc = NetCDF.read_file(_write_store(str(tmp_path / "epsg.nc")))
        try:
            assert nc._crs_label() == f"EPSG:{nc.epsg}"
        finally:
            nc.close()

    def test_falls_back_to_the_crs_name(self, tmp_path, monkeypatch):
        """With no EPSG, the CRS *name* is used — still never the WKT.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: Used to blank the EPSG so the fallback runs.

        Test scenario:
            A CRS with no authority code (a spherical-earth GRIB GEOGCS) has no `epsg`; dumping
            its WKT instead would drown every other line of the summary.
        """
        nc = NetCDF.read_file(_write_store(str(tmp_path / "name.nc")))
        try:
            monkeypatch.setattr(
                type(nc), "epsg", property(lambda self: None), raising=True
            )
            label = nc._crs_label()
            assert label not in ("", "unknown"), f"expected a CRS name, got {label!r}"
            assert "GEOGCS" not in label, f"the WKT leaked into the label: {label}"
        finally:
            nc.close()

    def test_unknown_when_nothing_resolves(self, tmp_path, monkeypatch):
        """No EPSG and no CRS yields `unknown` rather than an empty label."""
        nc = NetCDF.read_file(_write_store(str(tmp_path / "none.nc")))
        try:
            monkeypatch.setattr(
                type(nc), "epsg", property(lambda self: None), raising=True
            )
            monkeypatch.setattr(type(nc), "crs", property(lambda self: ""))
            assert nc._crs_label() == "unknown"
        finally:
            nc.close()

    def test_a_raising_crs_is_survivable(self, tmp_path, monkeypatch):
        """A property that raises degrades to `unknown` instead of breaking `str()`.

        Test scenario:
            `str()` runs in debuggers and pytest introspection, so it must stay total.
        """
        nc = NetCDF.read_file(_write_store(str(tmp_path / "raise.nc")))

        def _boom(self):
            raise RuntimeError("no CRS")

        try:
            monkeypatch.setattr(type(nc), "epsg", property(_boom))
            assert nc._crs_label() == "unknown"
        finally:
            nc.close()


class TestContainerSummary:
    """`_container_summary` — the branches a one-variable store does not reach."""

    def test_units_are_shown_when_declared(self, tmp_path):
        """A variable's CF units appear on its row."""
        nc = NetCDF.read_file(_write_store(str(tmp_path / "units.nc"), unit="K"))
        try:
            assert " K" in _container_summary(nc)
        finally:
            nc.close()

    def test_variable_list_is_truncated(self, tmp_path):
        """More variables than the display cap are summarised, not listed.

        Test scenario:
            A store with dozens of variables would otherwise turn `print(nc)` into a wall of text.
        """
        count = MAX_DISPLAY_VARIABLES + 3
        nc = NetCDF.read_file(
            _write_store(str(tmp_path / "many.nc"), variable_count=count)
        )
        try:
            summary = _container_summary(nc)
            assert "... " in summary, f"expected a truncation marker, got:\n{summary}"
            assert " more" in summary, (
                f"expected a hidden-count suffix, got:\n{summary}"
            )
            listed = sum(1 for line in summary.split("\n") if line.startswith("    v"))
            assert listed <= MAX_DISPLAY_VARIABLES, (
                f"listed {listed} variables, cap is {MAX_DISPLAY_VARIABLES}:\n{summary}"
            )
        finally:
            nc.close()

    def test_falls_back_to_variable_names(self, tmp_path, monkeypatch):
        """With no multidim metadata, the variable list comes from `variable_names`.

        Test scenario:
            Classic (non-MDIM) mode publishes no `meta_data.variables` at all, so a summary that
            only consulted it printed `variables : none` about a store that has variables — a
            positive false claim. `variable_names` is the list classic mode does answer.
        """
        nc = NetCDF.read_file(_write_store(str(tmp_path / "fallback.nc")))
        try:
            monkeypatch.setattr(nc.meta_data, "variables", {})
            assert "variables  : v0" in _container_summary(nc), _container_summary(nc)
        finally:
            nc.close()

    def test_reports_none_only_when_the_store_published_nothing(
        self, tmp_path, monkeypatch
    ):
        """`none` is printed when it is a fact, i.e. the store published an empty list.

        Test scenario:
            An MDIM store that genuinely holds no variables should say so; the word is only
            wrong when the mode cannot answer the question at all.
        """
        nc = NetCDF.read_file(_write_store(str(tmp_path / "empty.nc")))
        try:
            monkeypatch.setattr(nc.meta_data, "variables", {})
            monkeypatch.setattr(type(nc), "variable_names", property(lambda self: []))
            summary = _container_summary(nc)
            assert "variables  : none" in summary, summary
        finally:
            nc.close()

    def test_classic_mode_omits_what_it_cannot_know(self):
        """A classic-mode container reports its grid, and claims nothing it cannot see.

        Test scenario:
            The classic driver exposes the store's bands directly, so `band_count >= 1` on a
            `Container` — which is why the summary is chosen by type, not by band count. It
            publishes no multidim metadata, so `dimensions` / `groups` are omitted rather than
            reported as `none`.
        """
        path = "tests/data/netcdf/cf__5v__1d4-3d1__geog__y-desc.nc"
        nc = NetCDF.read_file(path, open_as_multi_dimensional=False)
        try:
            assert nc.band_count > 0, (
                "classic container must carry bands for this to bite"
            )
            summary = str(nc)
            assert summary.startswith("<Container "), f"wrong header: {summary}"
            assert "?" not in summary.split("\n")[0], f"placeholder name: {summary}"
            assert "grid       :" in summary, f"real grid not reported: {summary}"
            for claim in ("dimensions : none", "groups     : none"):
                assert claim not in summary, f"asserted an unknown as fact: {summary}"
        finally:
            nc.close()

    def test_grouped_store_qualifies_every_variable(self):
        """A grouped store labels each row by its full path, so no two rows look alike.

        Test scenario:
            `meta_data.variables` spans sub-groups and is keyed by `group/name`, but every group
            in this fixture carries the same leaf names, so rendering `info.name` printed `CO`,
            `O3`, `UTC_time` and `air_press` two and three times with nothing to tell them apart.
            The labels must carry the group path, the way `variable_names` spells its own names.
        """
        path = "tests/data/netcdf/none__35v__1d35__groups-nc4.nc"
        nc = NetCDF.read_file(path)
        try:
            assert len(nc.group_names) > 1, "fixture must be grouped for this to bite"
            summary = _container_summary(nc)
            rows = [
                line.strip()
                for line in summary.split("\n")
                if line.startswith("    ") and not line.strip().startswith("...")
            ]
            labels = [row.split("  ")[0] for row in rows]
            assert labels, f"no variable rows rendered:\n{summary}"
            assert len(labels) == len(set(labels)), (
                f"duplicate variable labels {labels} in:\n{summary}"
            )
            qualified = [label for label in labels if "/" in label]
            assert qualified, f"no group-qualified label in:\n{summary}"
            groups = set(nc.group_names)
            for label in qualified:
                assert label.rsplit("/", 1)[0] in groups, (
                    f"{label!r} is not qualified by one of the store's groups"
                )
        finally:
            nc.close()

    def test_a_real_file_without_a_suffix_is_not_called_in_memory(self, tmp_path):
        """An extensionless path is still a real file, and the header must name it.

        Test scenario:
            The label used to be chosen by `Path(source).suffix`, so `mkstemp` output, an
            OPeNDAP endpoint or a content-typed download -- all extensionless -- were reported
            as `in-memory`, a false claim about provenance. The driver is the reliable signal.
        """
        source = Path("tests/data/netcdf/cf__5v__1d4-3d1__geog__y-desc.nc")
        target = tmp_path / "cube"
        target.write_bytes(source.read_bytes())
        nc = NetCDF.read_file(str(target))
        try:
            header = str(nc).split("\n")[0]
            assert "in-memory" not in header, f"real file called in-memory: {header}"
            assert "cube" in header, f"header does not name the file: {header}"
        finally:
            nc.close()

    def test_an_unreferenced_store_reports_no_cell_size(self):
        """A store with no affine mapping prints its shape without inventing a cell size.

        Test scenario:
            A curvilinear or unstructured store has no geotransform, so `cell_size` reads 1.0
            by construction. Printing `@ 1` there is the same class of placeholder-as-fact as
            the `512 x 512 @ 1.0` this summary exists to stop printing.
        """
        path = "tests/data/netcdf/ugrid__1v__3d1.nc"
        nc = NetCDF.read_file(path, open_as_multi_dimensional=False)
        try:
            assert nc.band_count > 0, "fixture must carry bands to reach the grid line"
            assert not _has_georeference(nc), "fixture must be unreferenced to bite"
            grid = [
                ln
                for ln in _container_summary(nc).split("\n")
                if ln.strip().startswith("grid")
            ]
            assert grid, "no grid line rendered"
            assert "@" not in grid[0], f"invented a cell size: {grid[0]}"
            assert "8 x 4" in grid[0], f"lost the real shape: {grid[0]}"
        finally:
            nc.close()

    def test_a_georeferenced_store_keeps_a_unit_cell_size(self):
        """A genuine 1-unit grid still prints `@ 1`; only the null transform is suppressed."""
        path = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
        nc = NetCDF.read_file(path, open_as_multi_dimensional=False)
        try:
            assert _has_georeference(nc), (
                "fixture must be georeferenced for this to mean it"
            )
            grid = [
                ln
                for ln in _container_summary(nc).split("\n")
                if ln.strip().startswith("grid")
            ]
            assert "@ 1," in grid[0], f"dropped a real unit cell size: {grid[0]}"
        finally:
            nc.close()

    def test_long_lists_are_capped_by_width(self):
        """`dimensions` and `groups` stay readable on a store with many long names.

        Test scenario:
            The variable rows were capped, but `dimensions` and `groups` were joined whole:
            22 dimensions rendered a 372-character line, and seven flight-path group names a
            289-character one. A summary meant for a debugger or a log line cannot be that.
        """
        path = "tests/data/netcdf/none__111v__1d96-2d13-3d2__str.nc"
        nc = NetCDF.read_file(path)
        try:
            summary = _container_summary(nc)
            dims = [
                ln for ln in summary.split("\n") if ln.strip().startswith("dimensions")
            ]
            assert dims, f"no dimensions line:\n{summary}"
            assert len(dims[0]) < 120, f"line is {len(dims[0])} chars: {dims[0]}"
            assert "more" in dims[0], f"long list was not truncated: {dims[0]}"
        finally:
            nc.close()


class TestCappedJoin:
    """`_capped_join` truncates a one-line list by count and by width."""

    def test_a_short_list_is_joined_whole(self):
        """Nothing is cut when the list fits."""
        assert _capped_join(["time=12", "y=5", "x=5"]) == "time=12, y=5, x=5"

    def test_an_empty_list_joins_to_nothing(self):
        """An empty list yields the empty string, so the caller can decide to omit the line."""
        assert _capped_join([]) == ""

    def test_a_single_overlong_entry_is_still_shown(self):
        """The first entry is never dropped, so the line is never a bare count."""
        result = _capped_join(["x" * 200, "y"])
        assert result.startswith("x" * 200), result
        assert result.endswith("... 1 more"), result

    def test_the_count_cap_applies_below_the_width_cap(self):
        """Many short names are cut at `MAX_DISPLAY_VARIABLES`, not only by width."""
        result = _capped_join([f"d{n}" for n in range(MAX_DISPLAY_VARIABLES + 4)])
        assert result.count(",") == MAX_DISPLAY_VARIABLES, result
        assert result.endswith("... 4 more"), result


class TestStoreLabel:
    """`_store_label` reports the label and the in-memory verdict separately."""

    def test_the_memory_driver_decides_not_the_suffix(self):
        """A `memory`-driver store is in-memory whatever its `file_name` says."""

        class _InMemory:
            file_name = "netcdf"
            driver_type = "memory"

        assert _store_label(_InMemory()) == ("in-memory", True)

    def test_a_signed_remote_path_is_redacted(self):
        """A signature in the store path must not reach the header.

        Test scenario:
            `file_name` is the GDAL description, so a remote open keeps its query string and
            the base name still carries the credential. `str()` reaches every log handler and
            pytest's assertion output.
        """

        class _Signed:
            file_name = "/vsis3/bucket/cube.nc?X-Amz-Signature=deadbeef"
            driver_type = "netcdf"

        label, in_memory = _store_label(_Signed())
        assert "deadbeef" not in label, f"signature leaked into the header: {label}"
        assert label == "cube.nc?X-Amz-Signature=<redacted>", label
        assert in_memory is False, "a signed remote path is not in-memory"

    def test_a_group_scoped_in_memory_store_still_reports_in_memory(self):
        """The group suffix must not hide the in-memory verdict from the caller.

        Test scenario:
            The variable header used to recover the verdict with `label == "in-memory"`. A
            group-scoped store labels as `in-memory:/grp`, so that comparison failed and the
            parent-store fallback was skipped. The verdict is returned, not re-derived.
        """

        class _InMemoryGroup:
            file_name = ""
            driver_type = "memory"
            _group_path = "grp"

        assert _store_label(_InMemoryGroup()) == ("in-memory:/grp", True)


class _FakeVariable:
    """The attributes `_variable_summary` reads, and nothing else.

    A formatting function is worth testing against a stand-in: building a real NetCDF for every
    combination of present/absent optional field would cost far more than it proves, and the
    combinations are exactly what the branches are.
    """

    def __init__(self, **overrides):
        """Start from a fully-populated variable and override the fields under test.

        Args:
            **overrides: Any attribute to replace on this stand-in.
        """
        self.file_name = "cube.nc"
        self._parent_nc = None
        self._source_var_name = "t2m"
        self.cell_size = 0.25
        self.rows = 5
        self.columns = 5
        self.band_count = 12
        self._band_dim_name = "time"
        self.band_units = ["K"] * 12
        self.dtype = ["float32"] * 12
        # Distinct objects: a repeated one would pass the no-data assertion by identity
        # rather than by the nan handling under test.
        self.no_data_value = tuple(float("nan") for _ in range(12))
        self.__dict__.update(overrides)

    def _crs_label(self) -> str:
        """The CRS label the real method would produce."""
        return "EPSG:4326"


class TestVariableSummary:
    """`_variable_summary` — the optional lines appear only when there is something to say."""

    def test_every_field_present(self):
        """A fully-populated variable reports grid, bands, units, dtype and no-data."""
        summary = _variable_summary(_FakeVariable())
        assert summary.startswith("<Variable t2m - cube.nc>"), summary
        for fragment in (
            "grid    : 5 x 5 @ 0.25, EPSG:4326",
            "bands   : 12 along time",
            "units   : K",
            "dtype   : float32",
            "no-data : nan",
        ):
            assert fragment in summary, f"missing {fragment!r} in:\n{summary}"

    def test_optional_fields_are_omitted_when_absent(self):
        """Absent units / band axis / file name drop their line rather than print a blank.

        Test scenario:
            A variable built in memory has no path, a 2-D one has no band dimension, and an
            unlabelled band has no units. None of those should leave a dangling label.
        """
        summary = _variable_summary(
            _FakeVariable(
                file_name="",
                _band_dim_name=None,
                band_units=[""] * 12,
                no_data_value=(None,) * 12,
            )
        )
        assert summary.startswith("<Variable t2m>"), summary
        assert "units" not in summary, f"empty units should be omitted:\n{summary}"
        assert "along" not in summary, f"absent band axis should be omitted:\n{summary}"
        assert "no-data" not in summary, f"absent no-data should be omitted:\n{summary}"
        assert "bands   : 12" in summary, f"band count is not optional:\n{summary}"

    def test_file_name_falls_back_to_the_parent(self):
        """A variable subset has no path of its own; the container it came from does.

        Test scenario:
            `get_variable` returns a view over an in-memory MDArray, so `file_name` is empty and
            the header would otherwise lose the store it belongs to.
        """
        parent = _FakeVariable(file_name="/tmp/store/cube.nc")
        summary = _variable_summary(_FakeVariable(file_name="", _parent_nc=parent))
        assert summary.startswith("<Variable t2m - cube.nc>"), summary
