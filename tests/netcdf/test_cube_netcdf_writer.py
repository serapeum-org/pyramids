"""Unit tests for :class:`pyramids.netcdf._cube_netcdf_writer.CubeNetCDFWriter`.

The writer is exercised in isolation with a duck-typed mock collection and a
patched streaming writer, so its three phases (guard + derived state, schema
assembly, streaming write) are pinned independently of real NetCDF I/O. The
end-to-end round-trip is covered by ``tests/dataset/collection/test_to_netcdf.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import AlignmentError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset._cube_time import TimeAxis
from pyramids.netcdf import NetCDF
from pyramids.netcdf._cube_netcdf_writer import CubeNetCDFWriter

_WRITER_MODULE = "pyramids.netcdf._cube_netcdf_writer.open_streaming_multidim_netcdf"


def _stub_base(rows, cols, *, bands=1, scale=None, offset=None):
    """A stand-in for the collection's template `Dataset`.

    A real `Dataset`, not a namespace: the writer asks the template for its `y` / `x`
    axes *and* for each band's CF packing, so a stub that models only the axes drifts
    out of date the moment the writer needs anything else — which is exactly how eight
    of these tests came to raise `AttributeError` instead of exercising the schema.

    Args:
        rows: Length of the `y` axis.
        cols: Length of the `x` axis.
        bands: How many bands the template carries.
        scale: `scale_factor` to declare on every band, or `None` for an unpacked
            template. A per-band list declares a different factor on each.
        offset: `add_offset` to declare, same shape rules as `scale`.

    Returns:
        Dataset: The template.
    """
    array = np.zeros((bands, rows, cols), dtype="int16")
    dataset = Dataset.from_array(
        array if bands > 1 else array[0],
        geo_ref=GeoReference(top_left_corner=(0, rows), cell_size=1.0, epsg=4326),
    )
    if scale is not None:
        dataset.scale = list(scale) if isinstance(scale, list) else [scale] * bands
    if offset is not None:
        dataset.offset = list(offset) if isinstance(offset, list) else [offset] * bands
    return dataset


def _schema_writer(
    *,
    nodata=(-9999,),
    crs=None,
    epsg=4326,
    geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    band_count=2,
    names=("b1", "b2"),
    var_dtype="int16",
    rows=4,
    cols=5,
    scale=None,
    offset=None,
):
    """Build a writer with derived state set, for `_build_schema` unit tests.

    Returns:
        CubeNetCDFWriter: a writer whose ``_meta`` / ``band_count`` / ``names`` /
        ``var_dtype`` and base ``y`` / ``x`` / template band are populated directly.
    """
    base = _stub_base(rows, cols, bands=band_count, scale=scale, offset=offset)
    writer = CubeNetCDFWriter(SimpleNamespace(_base=base))
    writer._meta = SimpleNamespace(
        nodata=nodata, crs=crs, epsg=epsg, geotransform=geotransform
    )
    writer.band_count = band_count
    writer.names = list(names)
    writer.var_dtype = np.dtype(var_dtype)
    return writer


def _stream_writer(
    *, datasets, files, band_count=1, names=("b1",), var_dtype="float32"
):
    """Build a writer over mock datasets/files, for `_stream` unit tests.

    Returns:
        CubeNetCDFWriter: a writer whose collection exposes ``datasets`` / ``files``.
    """
    writer = CubeNetCDFWriter(SimpleNamespace(datasets=datasets, files=files))
    writer.band_count = band_count
    writer.names = list(names)
    writer.var_dtype = np.dtype(var_dtype)
    return writer


def _decision_writer(recipes, *, var_dtype="int16"):
    """Build a writer over timesteps that answer only their CF packing.

    Args:
        recipes: One list of `(scale, offset)` pairs per timestep, one pair per band.
        var_dtype: The stored dtype the writer starts from.

    Returns:
        CubeNetCDFWriter: a writer ready for `_decide_storage`.
    """
    datasets = [
        SimpleNamespace(_effective_packing=pairs.__getitem__) for pairs in recipes
    ]
    writer = CubeNetCDFWriter(SimpleNamespace(datasets=datasets))
    writer.band_count = len(recipes[0])
    writer.var_dtype = np.dtype(var_dtype)
    return writer


def _packed_step(counts, *, no_data, scale, offset):
    """A packed in-memory timestep, for `_physical_block` unit tests.

    Args:
        counts: `(rows, cols)` or `(bands, rows, cols)` stored counts.
        no_data: One sentinel per band.
        scale: One `scale_factor` per band.
        offset: One `add_offset` per band.

    Returns:
        Dataset: The timestep.
    """
    array = np.asarray(counts, dtype="int16")
    step = Dataset.from_array(
        array,
        geo_ref=GeoReference(
            top_left_corner=(0, array.shape[-2]), cell_size=1.0, epsg=4326
        ),
        no_data_value=list(no_data),
    )
    step.scale = list(scale)
    step.offset = list(offset)
    return step


class TestCubeNetCDFWriterWrite:
    """Tests for ``__init__`` and ``write`` (guard + wiring)."""

    def test_init_stores_collection(self):
        """__init__ captures the collection without touching its metadata.

        Test scenario:
            Construction is side-effect-free; no meta is read (an empty collection
            can still be constructed).
        """
        collection = SimpleNamespace(time_length=0)
        writer = CubeNetCDFWriter(collection)
        assert writer._collection is collection, (
            "collection must be stored on the writer"
        )

    def test_write_empty_collection_raises_before_touching_meta(self):
        """write() raises ValueError for an empty collection, before reading meta.

        Test scenario:
            ``time_length == 0`` raises with the ``to_netcdf`` message; ``_meta`` is
            never accessed (the namespace has none, so accessing it would AttributeError).
        """
        writer = CubeNetCDFWriter(SimpleNamespace(time_length=0))
        with pytest.raises(ValueError, match="cannot write an empty collection"):
            writer.write("out.nc")

    def test_write_derives_state_and_streams(self):
        """write() derives band state, opens the writer, and streams each timestep.

        Test scenario:
            A 2-timestep single-band collection: ``band_count``/``names``/``var_dtype``
            are derived from meta, ``open_streaming_multidim_netcdf`` is opened with
            the output path, and one slab per band per timestep is written.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((4, 5))
        # An unpacked timestep: `write` asks each one for its recipe to decide whether
        # one recipe describes the whole cube.
        ds._effective_packing.return_value = (None, None)
        meta = SimpleNamespace(
            shape=(1, 4, 5),
            band_names=["only"],
            dtype="float32",
            nodata=(None,),
            crs=None,
            epsg=None,
            geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        )
        collection = SimpleNamespace(
            time_length=2,
            time=None,
            datasets=[ds, ds],
            files=["f0.tif", "f1.tif"],
            _meta=meta,
            _base=_stub_base(4, 5, bands=1),
        )
        writer = CubeNetCDFWriter(collection)
        sink = Mock()
        with patch(_WRITER_MODULE) as mock_open:
            mock_open.return_value.__enter__.return_value = sink
            writer.write("out.nc", var_per_band=True)

        assert writer.band_count == 1, (
            f"band_count must derive to 1, got {writer.band_count}"
        )
        assert writer.names == ["only"], (
            f"names must come from band_names, got {writer.names}"
        )
        assert mock_open.call_args.args[0] == "out.nc", (
            "writer must open at the given path"
        )
        assert sink.write_slab.call_count == 2, (
            f"1 band x 2 timesteps => 2 slab writes, got {sink.write_slab.call_count}"
        )


class TestCubeNetCDFWriterBuildSchema:
    """Tests for ``_build_schema``."""

    def test_var_per_band_true_shapes_and_specs(self):
        """_build_schema(var_per_band=True) makes one 3-D variable per band.

        Test scenario:
            dims are ``time``/``y``/``x`` (no ``band``); each band name maps to a
            ``(time, y, x)`` spec; time coord carries the axis values + attrs.
        """
        writer = _schema_writer(nodata=None, band_count=2, names=("b1", "b2"))
        axis = TimeAxis(np.array([0, 1, 2]), {"note": "n"})
        dims, coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert dims == {"time": 3, "y": 4, "x": 5}, f"unexpected dims: {dims}"
        assert "band" not in dims, "var_per_band=True must not add a band dim"
        assert set(var_specs) == {"b1", "b2"}, (
            f"expected one var per band, got {set(var_specs)}"
        )
        assert var_specs["b1"][0] == ("time", "y", "x"), (
            "band var dims must be (time, y, x)"
        )
        assert coords["time"][0] is axis.values, "time coord must be the axis values"
        assert coords["time"][1] == {"note": "n"}, (
            "time coord must carry the axis attrs"
        )

    def test_var_per_band_false_single_data_var_with_band_dim(self):
        """_build_schema(var_per_band=False) makes one 4-D `data` variable.

        Test scenario:
            A ``band`` dim/coord is added and a single ``data`` variable spans
            ``(time, band, y, x)``; the human names ride along as a root attr.
        """
        writer = _schema_writer(nodata=None, band_count=2, names=("b1", "b2"))
        axis = TimeAxis(np.array([0, 1]), {})
        dims, coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=False
        )
        assert dims["band"] == 2, "var_per_band=False must add a band dim of size 2"
        assert set(var_specs) == {"data"}, (
            f"expected a single data var, got {set(var_specs)}"
        )
        assert var_specs["data"][0] == ("time", "band", "y", "x"), "data var dims wrong"
        assert root_attrs["band_names"] == "b1,b2", (
            f"band names must ride on the root attr, got {root_attrs.get('band_names')}"
        )

    def test_nodata_present_sets_var_and_root_attr(self):
        """_build_schema surfaces a typed nodata on each var and the root group.

        Test scenario:
            A ``nodata`` of -9999 is cast to the var dtype and attached to both the
            variable spec's attrs and the root attrs.
        """
        writer = _schema_writer(
            nodata=(-9999,), var_dtype="int16", band_count=1, names=("b1",)
        )
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert var_specs["b1"][2]["nodata"] == -9999, (
            "nodata must be on the variable attrs"
        )
        assert root_attrs["nodata"] == -9999, "nodata must be on the root attrs"

    def test_no_nodata_omits_the_attr(self):
        """_build_schema omits nodata attrs entirely when the source has none.

        Test scenario:
            ``nodata`` resolving to None leaves the variable attrs empty and adds no
            root ``nodata`` key.
        """
        writer = _schema_writer(nodata=(None,), band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert var_specs["b1"][2] == {}, (
            f"no nodata => empty var attrs, got {var_specs['b1'][2]}"
        )
        assert "nodata" not in root_attrs, "no nodata => no root nodata attr"

    def test_a_packed_template_puts_its_recipe_on_every_variable(self):
        """A packed collection's `scale_factor` / `add_offset` reach the cube.

        Test scenario:
            The cube is created at the collection's *stored* dtype and the slabs are
            streamed with `unpack=False`, so the recipe has to travel with them as CF
            attributes -- GDAL lifts them back into the MDArray's own slots on the next
            read. Without them the file holds counts that nothing identifies as counts,
            and every value reads back a hundredfold off.
        """
        writer = _schema_writer(
            nodata=(None,), band_count=2, names=("b1", "b2"), scale=0.01, offset=1.5
        )
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, _root = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        for name in ("b1", "b2"):
            attrs = var_specs[name][2]
            assert attrs["scale_factor"] == pytest.approx(0.01), (
                f"{name}: scale_factor not carried, got {attrs.get('scale_factor')}"
            )
            assert attrs["add_offset"] == pytest.approx(1.5), (
                f"{name}: add_offset not carried, got {attrs.get('add_offset')}"
            )

    def test_an_unpacked_template_declares_no_packing(self):
        """The identity must not be written out as a declaration.

        Test scenario:
            A band can declare the identity outright -- and pyramids' own `scale` /
            `offset` report an unset pair as `1.0` / `0.0` -- so carrying the pair
            unconditionally would stamp a meaningless recipe onto every such cube, and
            a reader would dutifully apply it.
        """
        writer = _schema_writer(
            nodata=(None,), band_count=1, names=("b1",), scale=1.0, offset=0.0
        )
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, _root = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert var_specs["b1"][2] == {}, (
            f"an unpacked template declared packing: {var_specs['b1'][2]}"
        )

    def test_the_single_data_var_takes_the_packing_when_every_band_agrees(self):
        """One 4-D variable can carry one recipe, so it takes the shared one.

        Test scenario:
            `var_per_band=False` folds every band into a single `data` variable, which
            has exactly one `scale_factor` slot. When the bands agree that slot is the
            truth for all of them, and dropping it would leave the whole cube as counts.
        """
        writer = _schema_writer(
            nodata=(None,), band_count=2, names=("b1", "b2"), scale=0.01, offset=1.5
        )
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, _root = writer._build_schema(
            axis, time_dim="time", var_per_band=False
        )
        attrs = var_specs["data"][2]
        assert attrs["scale_factor"] == pytest.approx(0.01), (
            f"the shared scale_factor was dropped, got {attrs.get('scale_factor')}"
        )
        assert attrs["add_offset"] == pytest.approx(1.5), (
            f"the shared add_offset was dropped, got {attrs.get('add_offset')}"
        )

    @pytest.mark.parametrize(
        "scale, offset, expected",
        [(None, 1.5, {"add_offset": 1.5}), (0.01, None, {"scale_factor": 0.01})],
        ids=["offset-only", "scale-only"],
    )
    def test_a_half_declared_recipe_writes_only_the_half_that_exists(
        self, scale, offset, expected
    ):
        """CF lets a variable declare one of the pair, so only that one is written.

        Test scenario:
            A missing `scale_factor` means "no factor", not zero, and a missing
            `add_offset` means "no shift", not unpacked. Filling the absent half in
            with an invented `1.0` / `0.0` would be harmless arithmetic but a false
            claim about the file, and a reader comparing attributes would see one.
        """
        writer = _schema_writer(
            nodata=(None,), band_count=1, names=("b1",), scale=scale, offset=offset
        )
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, _root = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert var_specs["b1"][2] == pytest.approx(expected), var_specs["b1"][2]

    def test_disagreeing_bands_each_keep_their_own_recipe_per_variable(self):
        """`var_per_band=True` is the arm that can carry a factor per band."""
        writer = _schema_writer(
            nodata=(None,),
            band_count=2,
            names=("b1", "b2"),
            scale=[0.01, 2.0],
            offset=[1.5, -3.0],
        )
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, var_specs, _root = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert var_specs["b1"][2]["scale_factor"] == pytest.approx(0.01)
        assert var_specs["b2"][2]["scale_factor"] == pytest.approx(2.0)
        assert var_specs["b2"][2]["add_offset"] == pytest.approx(-3.0)

    def test_geobox_root_attrs(self):
        """_build_schema always writes CF-1.8 + GeoTransform, and crs/epsg when present.

        Test scenario:
            A crs with ``to_wkt`` and an epsg populate ``crs_wkt`` / ``epsg``;
            ``Conventions`` and ``GeoTransform`` are always present.
        """
        crs = SimpleNamespace(to_wkt=lambda: "WKT-HERE")
        writer = _schema_writer(crs=crs, epsg=4326, band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, _var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert root_attrs["Conventions"] == "CF-1.8", "must declare CF-1.8"
        assert root_attrs["crs_wkt"] == "WKT-HERE", (
            "crs_wkt must come from crs.to_wkt()"
        )
        assert root_attrs["epsg"] == 4326, "epsg must be written"
        assert root_attrs["GeoTransform"] == "0.0 1.0 0.0 0.0 0.0 -1.0", (
            "GeoTransform wrong"
        )

    def test_crs_none_omits_crs_wkt(self):
        """_build_schema omits crs_wkt when the meta has no CRS.

        Test scenario:
            ``crs=None`` leaves no ``crs_wkt`` root attr (GeoTransform still present).
        """
        writer = _schema_writer(crs=None, band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, _var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert "crs_wkt" not in root_attrs, "no crs => no crs_wkt attr"

    def test_crs_without_to_wkt_is_swallowed(self):
        """_build_schema swallows a non-None CRS that lacks a to_wkt() (AttributeError guard).

        Test scenario:
            A crs object without a ``to_wkt`` attribute triggers the
            ``except AttributeError`` guard, leaving no ``crs_wkt`` root attr.
        """
        writer = _schema_writer(crs=object(), band_count=1, names=("b1",))
        axis = TimeAxis(np.array([0]), {})
        _dims, _coords, _var_specs, root_attrs = writer._build_schema(
            axis, time_dim="time", var_per_band=True
        )
        assert "crs_wkt" not in root_attrs, (
            "a crs without to_wkt() must be swallowed, not written"
        )


class TestCubeNetCDFWriterStream:
    """Tests for ``_stream``."""

    def test_stream_var_per_band_writes_one_slab_per_band(self):
        """_stream(var_per_band=True) writes each band of each timestep as a slab.

        Test scenario:
            Two single-band timesteps => two ``write_slab`` calls addressed to the
            band's variable name with the (rows, cols) plane.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((4, 5))
        writer = _stream_writer(datasets=[ds, ds], files=["f0", "f1"])
        sink = Mock()
        writer._stream(sink, dims={"y": 4, "x": 5}, var_per_band=True)
        assert sink.write_slab.call_count == 2, (
            f"1 band x 2 timesteps => 2 writes, got {sink.write_slab.call_count}"
        )
        assert sink.write_slab.call_args_list[0].args[0] == "b1", (
            "slab must address band var"
        )

    def test_stream_var_per_band_false_writes_the_data_var(self):
        """_stream(var_per_band=False) writes the whole block to the `data` var.

        Test scenario:
            The (band, rows, cols) block is written under the ``data`` variable name.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((4, 5))
        writer = _stream_writer(datasets=[ds], files=["f0"])
        sink = Mock()
        writer._stream(sink, dims={"y": 4, "x": 5}, var_per_band=False)
        assert sink.write_slab.call_args.args[0] == "data", (
            "single-var write must target 'data'"
        )

    def test_stream_mismatched_shape_raises_alignment_error_naming_file(self):
        """_stream raises AlignmentError naming the offending file on a shape mismatch.

        Test scenario:
            A timestep whose band count differs from the template raises, and the
            message names the source file.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((2, 4, 5))
        writer = _stream_writer(datasets=[ds], files=["bad.tif"], band_count=1)
        sink = Mock()
        with pytest.raises(AlignmentError, match="bad.tif"):
            writer._stream(sink, dims={"y": 4, "x": 5}, var_per_band=True)

    def test_stream_mismatch_without_files_names_timestep(self):
        """_stream falls back to 'timestep N' in the error when files is None.

        Test scenario:
            With no ``files`` list, a shape mismatch names the positional timestep.
        """
        ds = Mock()
        ds.read_array.return_value = np.zeros((3, 4, 5))
        writer = _stream_writer(datasets=[ds], files=None, band_count=1)
        sink = Mock()
        with pytest.raises(AlignmentError, match="timestep 0"):
            writer._stream(sink, dims={"y": 4, "x": 5}, var_per_band=True)

    def test_a_materialised_cube_streams_physical_slabs(self):
        """With `_materialise` set, each slab is the timestep's physical block.

        Test scenario:
            The materialised form writes `float64` physical values with the gaps as
            NaN, so the counts `read_array(unpack=False)` would give never reach the
            file: the slab holds `2.5` where the store holds `100`.
        """
        step = _packed_step(
            [[-9999, 100], [200, 300]], no_data=[-9999], scale=[0.01], offset=[1.5]
        )
        writer = _stream_writer(datasets=[step], files=["f0"], var_dtype="float64")
        writer._materialise = True
        sink = Mock()

        writer._stream(sink, dims={"y": 2, "x": 2}, var_per_band=True)

        name, index, slab = sink.write_slab.call_args.args
        assert (name, index) == ("b1", 0), (name, index)
        np.testing.assert_allclose(slab, [[np.nan, 2.5], [3.5, 4.5]])


class TestCubeNetCDFWriterDecideStorage:
    """Tests for ``_decide_storage``: the compact counts + recipe, or a materialised cube."""

    @pytest.mark.parametrize("var_per_band", [True, False], ids=["per-band", "data"])
    def test_every_timestep_unpacked_keeps_the_compact_form(self, var_per_band):
        """No timestep declares a recipe, so the counts are written as they are.

        Test scenario:
            The identity reaches this in two spellings -- GDAL's unset `None` and an
            explicit `1.0` / `0.0` -- and both mean "unpacked". Neither may switch on
            the materialised `float64` form, which would double the cube for nothing.
        """
        writer = _decision_writer(
            [[(None, None), (1.0, 0.0)], [(1.0, 0.0), (None, None)]]
        )

        writer._decide_storage(var_per_band)

        assert writer._materialise is False, "an unpacked cube was materialised"
        assert writer._recipe == [(None, None), (None, None)], writer._recipe
        assert writer.var_dtype == np.dtype("int16"), writer.var_dtype

    def test_timesteps_packed_alike_keep_their_shared_recipe(self):
        """One recipe describes every slab, so the counts are stored with it."""
        writer = _decision_writer([[(0.01, 1.5)], [(0.01, 1.5)]])

        writer._decide_storage(True)

        assert writer._materialise is False, "a consistent cube was materialised"
        assert writer._recipe == [(0.01, 1.5)], writer._recipe
        assert writer.var_dtype == np.dtype("int16"), writer.var_dtype

    def test_timesteps_packed_differently_are_materialised(self):
        """A second timestep at `0.1` cannot be labelled with the first's `0.01`."""
        writer = _decision_writer([[(0.01, 0.0)], [(0.1, 0.0)]])

        writer._decide_storage(True)

        assert writer._materialise is True, "disagreeing timesteps kept one recipe"
        assert writer._recipe is None, writer._recipe
        assert writer.var_dtype == np.dtype("float64"), writer.var_dtype

    @pytest.mark.parametrize(
        ("var_per_band", "materialise"),
        [(True, False), (False, True)],
        ids=["per-band-carries-each", "data-var-cannot"],
    )
    def test_bands_packed_differently_fit_only_one_variable_per_band(
        self, var_per_band, materialise
    ):
        """A variable per band can carry a recipe per band; one 4-D variable cannot."""
        writer = _decision_writer([[(0.01, 0.0), (0.1, 0.0)]])

        writer._decide_storage(var_per_band)

        assert writer._materialise is materialise, writer._materialise
        expected_dtype = np.dtype("float64" if materialise else "int16")
        assert writer.var_dtype == expected_dtype, writer.var_dtype

    def test_the_recipe_comes_from_the_timesteps_not_the_template(self):
        """A template packed unlike its timesteps cannot mislabel them.

        Test scenario:
            The template declares `0.5`; every timestep declares `0.01`. The schema
            takes the recipe `_decide_storage` resolved from the timesteps, since those
            are what the slabs are read from.
        """
        writer = _schema_writer(
            nodata=(None,), band_count=1, names=("b1",), scale=0.5, offset=0.0
        )
        writer._collection.datasets = [
            SimpleNamespace(_effective_packing=[(0.01, 1.5)].__getitem__)
        ]

        writer._decide_storage(True)
        _dims, _coords, var_specs, _root = writer._build_schema(
            TimeAxis(np.array([0]), {}), time_dim="time", var_per_band=True
        )

        attrs = var_specs["b1"][2]
        assert attrs["scale_factor"] == pytest.approx(0.01), attrs
        assert attrs["add_offset"] == pytest.approx(1.5), attrs


class TestCubeNetCDFWriterPhysicalBlock:
    """Tests for ``_physical_block``: one timestep in physical units, gaps as NaN."""

    def test_a_single_band_timestep_is_lifted_to_three_dimensions(self):
        """A 2-D read comes back as `(1, rows, cols)` `float64`, unpacked, gap as NaN."""
        step = _packed_step(
            [[-9999, 100], [200, 300]], no_data=[-9999], scale=[0.01], offset=[1.5]
        )

        block = CubeNetCDFWriter._physical_block(step)

        assert block.shape == (1, 2, 2), block.shape
        assert block.dtype == np.float64, block.dtype
        np.testing.assert_allclose(block, [[[np.nan, 2.5], [3.5, 4.5]]])

    def test_each_band_takes_its_own_recipe_and_sentinel(self):
        """Band by band: each is masked with its own sentinel and unpacked with its recipe.

        Test scenario:
            Band 1 declares `-32768` and holds a real `-9999`, and it is unpacked at the
            identity. Masking it with band 0's sentinel would blank a measurement, and
            unpacking it with band 0's recipe would shift every value.
        """
        step = _packed_step(
            [[[-9999, 100], [200, 300]], [[-32768, 100], [-9999, 300]]],
            no_data=[-9999, -32768],
            scale=[0.01, 1.0],
            offset=[1.5, 0.0],
        )

        block = CubeNetCDFWriter._physical_block(step)

        np.testing.assert_allclose(block[0], [[np.nan, 2.5], [3.5, 4.5]])
        np.testing.assert_allclose(block[1], [[np.nan, 100.0], [-9999.0, 300.0]])


def _packed_timesteps(tmp_path, scales, *, bands=1):
    """Write one packed GeoTIFF per timestep and open them as a collection.

    Args:
        tmp_path: Where the files go.
        scales: One list of per-band factors per timestep.
        bands: Band count of every timestep.

    Returns:
        DatasetCollection: The stack.
    """
    counts = np.array([[100, -9999], [200, 300]], dtype="int16")
    for index, per_band in enumerate(scales):
        stack = np.stack([counts] * bands) if bands > 1 else counts
        step = Dataset.from_array(
            stack,
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
            no_data_value=[-9999] * bands,
        )
        step.scale = list(per_band)
        step.offset = [0.0] * bands
        step.to_file(str(tmp_path / f"t{index}.tif"))
    return DatasetCollection.from_files(str(tmp_path), glob="*.tif")


class TestTheCubeReadsBackItsValues:
    """Round trips through a real file: what goes in is what comes out."""

    def test_timesteps_packed_differently_keep_their_own_values(self, tmp_path):
        """A cube no single recipe describes is written in physical units.

        Test scenario:
            The recipe came from the template and the slabs from each timestep, so a
            second timestep packed at 0.1 was labelled with the first's 0.01 and read
            back ten times too small. The gap comes back as NaN, since one declared
            sentinel cannot name gaps whose physical value differs per timestep.
        """
        collection = _packed_timesteps(tmp_path, [[0.01], [0.1]])
        collection.to_netcdf(str(tmp_path / "out.nc"), var_per_band=True)

        variable = NetCDF.read_file(str(tmp_path / "out.nc")).get_variable("Band_1")
        values = np.asarray(variable.read_array(), dtype="float64").reshape(2, -1)

        np.testing.assert_allclose(
            values, [[1.0, np.nan, 2.0, 3.0], [10.0, np.nan, 20.0, 30.0]]
        )

    def test_bands_packed_differently_survive_the_single_data_variable(self, tmp_path):
        """One 4-D variable cannot hold two recipes, so it holds physical values.

        Test scenario:
            With `var_per_band=False` bands packed at 0.01 and 0.1 shared one variable
            holding their raw counts under no recipe, and both read back as 100. The
            old schema test pinned that as intended.
        """
        collection = _packed_timesteps(tmp_path, [[0.01, 0.1]], bands=2)
        collection.to_netcdf(str(tmp_path / "out.nc"), var_per_band=False)

        data = NetCDF.read_file(str(tmp_path / "out.nc")).get_variable("data")
        values = np.asarray(data.read_array(), dtype="float64").ravel()

        assert np.nanmax(values) == pytest.approx(30.0), values
        assert np.nanmin(values) == pytest.approx(1.0), values

    def test_a_consistent_recipe_keeps_the_compact_form(self, tmp_path):
        """When one recipe fits every slab, the counts are stored with it."""
        collection = _packed_timesteps(tmp_path, [[0.01], [0.01]])
        collection.to_netcdf(str(tmp_path / "out.nc"), var_per_band=True)

        variable = NetCDF.read_file(str(tmp_path / "out.nc")).get_variable("Band_1")
        assert variable._scale == pytest.approx(0.01), variable._scale
        np.testing.assert_allclose(
            np.asarray(variable.read_array(), dtype="float64").reshape(2, -1)[
                :, [0, 2, 3]
            ],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        )
