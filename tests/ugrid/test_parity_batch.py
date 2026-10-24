"""to_image / convert_units / to_dask_dataframe / open_mfdataset on a mesh (issue #1378).

The lower-priority parity batch: four members the raster and cube sides had and the mesh did
not. Each is built on something that already exists rather than reimplemented -- the
rasterisation for ``to_image``, the shared affine unit table for ``convert_units``, the tidy
frame for ``to_dask_dataframe``, and ``concat`` for ``open_mfdataset`` -- so these tests
concentrate on the mesh-specific wiring and the edges.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(values, units=None, nodata=None, location="face"):
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"t": np.asarray(values, dtype="float64")},
        data_locations={"t": location},
        epsg=32636,
    )
    if units is not None:
        ds["t"].units = units
    if nodata is not None:
        ds["t"].nodata = nodata
    return ds


class TestConvertUnits:
    """Values converted through the table the raster side already uses."""

    pytestmark = pytest.mark.core

    def test_kelvin_to_celsius(self):
        """273.15 K and 283.15 K are 0 and 10 degrees Celsius."""
        result = _mesh([273.15, 283.15], units="K").convert_units("celsius")
        assert result["t"].data.tolist() == pytest.approx([0.0, 10.0])

    def test_the_new_unit_is_recorded(self):
        result = _mesh([273.15, 283.15], units="K").convert_units("celsius")
        assert result["t"].units == "celsius"
        assert result["t"].attributes["units"] == "celsius"

    def test_the_source_is_left_untouched(self):
        source = _mesh([273.15, 283.15], units="K")
        source.convert_units("celsius")
        assert source["t"].data.tolist() == pytest.approx([273.15, 283.15])
        assert source["t"].units == "K"

    def test_nodata_keeps_its_sentinel(self):
        """Converting the sentinel would turn -9999 into a plausible-looking temperature."""
        ds = _mesh([273.15, -9999.0], units="K", nodata=-9999.0)
        result = ds.convert_units("celsius")
        assert result["t"].data.tolist() == pytest.approx([0.0, -9999.0])

    def test_a_variable_already_in_the_target_unit_passes_through(self):
        ds = _mesh([1.0, 2.0], units="celsius")
        assert ds.convert_units("celsius")["t"].data.tolist() == pytest.approx(
            [1.0, 2.0]
        )

    def test_naming_a_variable_with_no_unit_is_refused(self):
        """An explicit request must not pass silently."""
        with pytest.raises(ValueError, match="declares no units"):
            _mesh([1.0, 2.0]).convert_units("celsius", variable_name="t")

    def test_a_sweep_skips_variables_with_no_unit(self):
        """Converting everything should not fail on a variable that has nothing to convert."""
        result = _mesh([1.0, 2.0]).convert_units("celsius")
        assert result["t"].data.tolist() == pytest.approx([1.0, 2.0])
        assert not result["t"].units

    def test_an_unsupported_pair_is_refused(self):
        with pytest.raises(ValueError):
            _mesh([1.0, 2.0], units="K").convert_units("metres")


class TestToImage:
    """Rasterise, then colour-map through the raster's own exporter."""

    pytestmark = pytest.mark.plot

    def test_it_returns_an_image_of_the_mesh(self):
        image = _mesh([1.0, 5.0]).to_image()
        assert image.size[0] > 1 and image.size[1] > 1

    def test_the_default_resolution_is_about_512_px_on_the_long_side(self):
        """So a caller need not reason about a cell size in unknown coordinate units."""
        image = _mesh([1.0, 5.0]).to_image()
        assert max(image.size) == pytest.approx(512, abs=2)

    def test_an_explicit_cell_size_sets_the_size(self):
        image = _mesh([1.0, 5.0]).to_image(cell_size=0.1)
        assert max(image.size) == pytest.approx(10, abs=1)

    def test_a_temporal_variable_is_refused_with_a_hint(self):
        """An image is one picture, so it needs one value per element."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"t": np.zeros((3, 2))},
            data_locations={"t": "face"},
            epsg=32636,
        )
        with pytest.raises(ValueError, match="isel"):
            ds.to_image()


class TestToDaskDataframe:
    """The lazy sibling of the tidy frame."""

    pytestmark = pytest.mark.core

    def test_it_returns_a_dask_dataframe(self):
        dd = pytest.importorskip("dask.dataframe")
        frame = _mesh([1.0, 2.0]).to_dask_dataframe()
        assert isinstance(frame, dd.DataFrame)

    def test_it_computes_to_the_same_rows_as_to_dataframe(self):
        """The lazy and eager paths must agree, or one of them is wrong."""
        pytest.importorskip("dask.dataframe")
        ds = _mesh([1.0, 2.0])
        eager = ds.to_dataframe()
        lazy = ds.to_dask_dataframe().compute()
        assert list(lazy.columns) == list(eager.columns)
        assert len(lazy) == len(eager)

    def test_partitions_are_honoured(self):
        pytest.importorskip("dask.dataframe")
        assert _mesh([1.0, 2.0]).to_dask_dataframe(npartitions=2).npartitions == 2

    def test_zero_partitions_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            _mesh([1.0, 2.0]).to_dask_dataframe(npartitions=0)


class TestOpenMfdataset:
    """Many files, one topology, stacked along time."""

    pytestmark = pytest.mark.core

    @staticmethod
    def _write_run(directory, dated):
        """Three single-step files whose names sort WRONG lexically: t10 before t2."""
        paths = []
        for label, day in (("t9", 9), ("t10", 10), ("t2", 2)):
            ds = UgridDataset.from_arrays(
                node_x=_NODE_X,
                node_y=_NODE_Y,
                face_node_connectivity=_TRIANGLES,
                data={"d": np.array([[float(day), float(day) * 2]])},
                data_locations={"d": "face"},
            )
            ds["d"].dimensions = ("time", "mesh2d_nFaces")
            if dated:
                ds["d"].attributes = {
                    **ds["d"].attributes,
                    "time_values": [np.datetime64(f"2001-01-{day:02d}")],
                    TIME_UNITS_ATTR: "days since 2001-01-01",
                    TIME_CALENDAR_ATTR: "standard",
                }
            path = directory / f"run_{label}.nc"
            ds.to_file(str(path))
            paths.append(path)
        return paths

    def test_a_glob_stacks_every_file(self, tmp_path):
        self._write_run(tmp_path, dated=True)
        stacked = UgridDataset.open_mfdataset(str(tmp_path / "run_*.nc"))
        assert stacked["d"].n_time_steps == 3

    def test_a_dated_run_comes_back_in_time_order_not_lexical(self, tmp_path):
        """The reason the sort exists: `run_t10` sorts before `run_t2` as text."""
        self._write_run(tmp_path, dated=True)
        stacked = UgridDataset.open_mfdataset(str(tmp_path / "run_*.nc"))
        assert stacked["d"].data[:, 0].tolist() == [2.0, 9.0, 10.0]

    def test_sort_false_keeps_the_given_order(self, tmp_path):
        self._write_run(tmp_path, dated=True)
        stacked = UgridDataset.open_mfdataset(str(tmp_path / "run_*.nc"), sort=False)
        assert stacked["d"].data[:, 0].tolist() == [10.0, 2.0, 9.0]

    def test_undated_files_keep_their_given_position(self, tmp_path):
        """Positional indices say nothing about order, so the sort must not pretend they do -
        a caller with undated files has to pass an ordered sequence."""
        self._write_run(tmp_path, dated=False)
        stacked = UgridDataset.open_mfdataset(str(tmp_path / "run_*.nc"))
        assert stacked["d"].data[:, 0].tolist() == [10.0, 2.0, 9.0]

    def test_an_explicit_sequence_is_stacked_in_that_order(self, tmp_path):
        paths = self._write_run(tmp_path, dated=False)
        ordered = [paths[2], paths[0], paths[1]]  # t2, t9, t10
        stacked = UgridDataset.open_mfdataset([str(p) for p in ordered], sort=False)
        assert stacked["d"].data[:, 0].tolist() == [2.0, 9.0, 10.0]

    def test_a_single_path_is_accepted(self, tmp_path):
        paths = self._write_run(tmp_path, dated=True)
        assert UgridDataset.open_mfdataset(str(paths[0]))["d"].n_time_steps == 1

    def test_a_pattern_matching_nothing_is_refused(self, tmp_path):
        """Rather than answering an empty dataset that looks like a successful read."""
        with pytest.raises(ValueError, match="matches no file"):
            UgridDataset.open_mfdataset(str(tmp_path / "absent_*.nc"))

    def test_an_empty_sequence_is_refused(self):
        with pytest.raises(ValueError, match="nothing to open"):
            UgridDataset.open_mfdataset([])

    def test_a_file_on_a_different_topology_is_refused(self, tmp_path):
        self._write_run(tmp_path, dated=True)
        other = UgridDataset.from_arrays(
            node_x=np.array([0.0, 9.0, 9.0]),
            node_y=np.array([0.0, 0.0, 9.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"d": np.array([[1.0]])},
            data_locations={"d": "face"},
        )
        other["d"].dimensions = ("time", "mesh2d_nFaces")
        other.to_file(str(tmp_path / "other.nc"))
        with pytest.raises(Exception, match="(?i)mesh|topolog|align"):
            UgridDataset.open_mfdataset(str(tmp_path / "*.nc"))

    def test_a_mixed_set_keeps_undated_files_in_place(self, tmp_path):
        """Review M3: the docstring promises an undated file "keeps its given position even
        with sort=True", but every dated file sorted before every undated one.

        Passed as [undated, dated], the undated file must stay first.
        """
        undated = tmp_path / "zz_undated.nc"
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.array([[1.0, 2.0]])},
            data_locations={"d": "face"},
        )
        ds["d"].dimensions = ("time", "mesh2d_nFaces")
        ds.to_file(str(undated))

        dated = tmp_path / "aa_dated.nc"
        stamped = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.array([[9.0, 8.0]])},
            data_locations={"d": "face"},
        )
        stamped["d"].dimensions = ("time", "mesh2d_nFaces")
        stamped["d"].attributes = {
            **stamped["d"].attributes,
            "time_values": [np.datetime64("2001-01-01")],
            TIME_UNITS_ATTR: "days since 2001-01-01",
            TIME_CALENDAR_ATTR: "standard",
        }
        stamped.to_file(str(dated))

        stacked = UgridDataset.open_mfdataset([str(undated), str(dated)])
        assert stacked["d"].data[:, 0].tolist() == [1.0, 9.0], (
            "the undated file keeps its given position, as the docstring promises"
        )

    def test_a_mixed_set_still_orders_the_dated_files_among_themselves(self, tmp_path):
        """Sorting the dated subset is the point of sort=True; only the undated ones hold."""
        paths = []
        for label, day in (("t9", 9), ("t2", 2)):
            ds = UgridDataset.from_arrays(
                node_x=_NODE_X,
                node_y=_NODE_Y,
                face_node_connectivity=_TRIANGLES,
                data={"d": np.array([[float(day), 0.0]])},
                data_locations={"d": "face"},
            )
            ds["d"].dimensions = ("time", "mesh2d_nFaces")
            ds["d"].attributes = {
                **ds["d"].attributes,
                "time_values": [np.datetime64(f"2001-01-{day:02d}")],
                TIME_UNITS_ATTR: "days since 2001-01-01",
                TIME_CALENDAR_ATTR: "standard",
            }
            path = tmp_path / f"dated_{label}.nc"
            ds.to_file(str(path))
            paths.append(str(path))
        stacked = UgridDataset.open_mfdataset(paths)
        assert stacked["d"].data[:, 0].tolist() == [2.0, 9.0], (
            "dated files sort by stamp"
        )
