"""Tests for :mod:`pyramids.dataset.engines`.

These tests pin down the Stage 1 contract documented in the module's
docstring and in the L-2 plan:

* The seven collaborator classes are accessible as attributes on a
  ``Dataset`` (``ds.io``, ``ds.spatial``, ``ds.bands``, ``ds.analysis``,
  ``ds.cell``, ``ds.vectorize``, ``ds.cog``).
* Every public method on every collaborator is a forwarder — calling
  ``ds.<collab>.<method>(...)`` invokes ``ds.<method>(...)`` with the same
  positional and keyword arguments and returns the same value.
* Read-only and read/write properties on the collaborators forward in
  both directions.
* The back-reference is a :class:`weakref.proxy` so the parent ``Dataset``
  can be garbage-collected while a collaborator instance is still
  referenced (otherwise the GDAL handle leaks and Windows file-unlink
  fails in tests).
* ``Dataset`` survives a pickle round-trip (covered more broadly in
  :mod:`tests.dataset.test_pickle`); the round-tripped instance carries
  fresh collaborator instances of the right type.
* Pickling a collaborator *directly* yields a ``_Placeholder`` on
  unpickle rather than crashing or producing a circular pickle through
  the parent ``Dataset``.
* :meth:`Analysis.normalize` (the only collaborator method with a real
  body in Stage 1) min-max scales arrays into the [0, 1] range.
"""

from __future__ import annotations

import gc
import pickle
import weakref

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.dataset.engines import (
    COG,
    IO,
    Analysis,
    Bands,
    Cell,
    Spatial,
    Vectorize,
)
from pyramids.dataset.engines._base import (
    _Engine,
    _Placeholder,
    _recreate_placeholder,
)


@pytest.fixture
def in_memory_dataset() -> Dataset:
    """Build a small in-memory ``Dataset`` for collaborator method tests.

    Returns:
        Dataset: A 4x4 float32 dataset in EPSG:4326 backed by GDAL's MEM
        driver. Suitable for any test that does not need to round-trip
        through pickle (in-memory datasets cannot pickle by design).
    """
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return Dataset.create_from_array(
        arr=arr,
        top_left_corner=(0.0, 0.0),
        cell_size=1.0,
        epsg=4326,
    )


@pytest.fixture
def file_backed_dataset(tmp_path) -> Dataset:
    """Write a tiny GeoTIFF to ``tmp_path`` and return it as a ``Dataset``.

    Args:
        tmp_path: pytest-provided per-test temporary directory.

    Returns:
        Dataset: A file-backed dataset suitable for pickle round-trips
        (``RasterBase.__reduce__`` re-opens via ``cls.read_file(path)``,
        which only works for paths that exist on disk).
    """
    path = str(tmp_path / "tiny.tif")
    drv = gdal.GetDriverByName("GTiff")
    raster = drv.Create(path, 3, 4, 1, gdal.GDT_Float32)
    raster.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    raster.SetProjection(srs.ExportToWkt())
    raster.GetRasterBand(1).WriteArray(np.arange(12, dtype=np.float32).reshape(4, 3))
    raster.FlushCache()
    raster = None
    return Dataset.read_file(path)


class TestPlaceholder:
    """Tests for :class:`_Placeholder` and :func:`_recreate_placeholder`."""

    def test_recreate_placeholder_returns_placeholder(self):
        """The factory must return a fresh ``_Placeholder`` instance.

        Test scenario:
            ``_recreate_placeholder()`` is the unpickle target referenced
            from ``_Engine.__reduce__``; calling it directly should
            yield a usable placeholder.
        """
        result = _recreate_placeholder()
        assert isinstance(
            result, _Placeholder
        ), f"Expected _Placeholder instance, got {type(result).__name__}"

    def test_each_call_returns_distinct_instance(self):
        """Successive calls must produce distinct objects, not a singleton.

        Test scenario:
            Two separate calls to the factory should yield two distinct
            placeholders so a buggy callsite cannot accidentally share
            state across collaborator instances.
        """
        first = _recreate_placeholder()
        second = _recreate_placeholder()
        assert (
            first is not second
        ), "Successive calls to _recreate_placeholder returned the same instance"


class TestCollaboratorBase:
    """Tests for :class:`_Engine` (base class shared by all collaborators)."""

    def test_init_stores_weak_proxy_to_dataset(self, in_memory_dataset):
        """Constructor must wrap the Dataset in a ``weakref.proxy``.

        Test scenario:
            After construction, attribute access through ``self._ds`` should
            transparently resolve to the wrapped Dataset's attributes.
        """
        collab = _Engine(in_memory_dataset)
        assert (
            collab._ds.epsg == in_memory_dataset.epsg
        ), f"Proxy did not resolve epsg: {collab._ds.epsg} != {in_memory_dataset.epsg}"
        assert (
            collab._ds.rows == in_memory_dataset.rows
        ), "Proxy did not resolve rows attribute"

    def test_proxy_does_not_keep_dataset_alive(self):
        """Strong-cycle safety: deleting the only Dataset ref must release it.

        Test scenario:
            A ``weakref.ref`` set up on the Dataset should expire after the
            local Dataset variable is deleted, even while a collaborator
            still holds the back-reference. If this fails, GDAL handles
            leak and Windows file-unlink in tests intermittently fails.
        """
        arr = np.zeros((4, 4), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr=arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
        )
        ref = weakref.ref(ds)
        collab = _Engine(ds)
        del ds
        gc.collect()
        assert ref() is None, (
            "Dataset was kept alive by collaborator's _ds back-reference; "
            "the back-reference must use weakref.proxy"
        )
        # Referencing the proxy after the parent is gone must raise.
        with pytest.raises(ReferenceError):
            _ = collab._ds.epsg

    def test_reduce_returns_placeholder_recipe(self, in_memory_dataset):
        """``__reduce__`` must return ``(_recreate_placeholder, ())``.

        Test scenario:
            Direct collaborator pickle should not attempt to serialize the
            parent Dataset — it should reduce to the placeholder factory
            with no arguments so unpickle yields a ``_Placeholder``.
        """
        collab = _Engine(in_memory_dataset)
        recipe = collab.__reduce__()
        assert recipe == (
            _recreate_placeholder,
            (),
        ), f"Unexpected reduce recipe: {recipe}"

    def test_slots_only_contains_ds(self):
        """``__slots__`` must be exactly ``('_ds',)``.

        Test scenario:
            The slots declaration prevents accidental attribute attachment
            (e.g., a future bug that sets ``self.foo = bar`` on a
            collaborator) and keeps the back-reference the only state.
        """
        assert _Engine.__slots__ == (
            "_ds",
        ), f"Expected __slots__ == ('_ds',), got {_Engine.__slots__!r}"


class TestCollaboratorAttachment:
    """Tests that ``Dataset.__init__`` attaches every collaborator."""

    @pytest.mark.parametrize(
        "attr_name, expected_type",
        [
            ("io", IO),
            ("spatial", Spatial),
            ("bands", Bands),
            ("analysis", Analysis),
            ("cell", Cell),
            ("vectorize", Vectorize),
            ("cog", COG),
        ],
    )
    def test_dataset_exposes_collaborator(
        self, in_memory_dataset, attr_name, expected_type
    ):
        """Each collaborator must be accessible as a Dataset attribute.

        Args:
            attr_name: The attribute name on the Dataset (e.g. ``"io"``).
            expected_type: The collaborator class that attribute should hold.

        Test scenario:
            After ``Dataset.__init__`` runs, the seven collaborators are
            wired in and reachable by attribute access.
        """
        collab = getattr(in_memory_dataset, attr_name)
        assert isinstance(
            collab, expected_type
        ), f"ds.{attr_name} should be {expected_type.__name__}, got {type(collab).__name__}"


# Stage 1 forwarders: collaborator method delegates to the same-named
# Dataset method (the mixin is still in Dataset's MRO). PR 2.1 (cell)
# migrated method bodies into the collaborator and inverted the
# direction — those methods are tested in TestFacadeDelegation below.
FORWARDING_METHODS: list[tuple[str, str]] = []

# Stage 2 facades: Dataset method delegates to the collaborator method
# (the mixin has been removed from Dataset's MRO). PR 2.1 — cell, PR 2.2 —
# cog, PR 2.3 — vectorize, PR 2.4 — analysis, PR 2.5 — spatial.
FACADE_METHODS = [
    ("bands", "get_attribute_table"),
    ("bands", "set_attribute_table"),
    ("bands", "add_band"),
    ("bands", "get_band_by_color"),
    ("bands", "change_no_data_value"),
    ("io", "read_array"),
    ("io", "write_array"),
    ("io", "to_file"),
    ("io", "to_raster"),
    ("io", "get_block_arrangement"),
    ("io", "get_tile"),
    ("io", "map_blocks"),
    ("io", "to_xyz"),
    ("io", "create_overviews"),
    ("io", "recreate_overviews"),
    ("io", "get_overview"),
    ("io", "read_overview_array"),
    ("spatial", "crop"),
    ("spatial", "to_crs"),
    ("spatial", "set_crs"),
    ("spatial", "convert_longitude"),
    ("spatial", "resample"),
    ("spatial", "align"),
    ("spatial", "fill_gaps"),
    ("cell", "get_cell_coords"),
    ("cell", "get_cell_polygons"),
    ("cell", "get_cell_points"),
    ("cell", "map_to_array_coordinates"),
    ("cell", "array_to_map_coordinates"),
    ("cog", "to_cog"),
    ("cog", "validate_cog"),
    ("vectorize", "to_feature_collection"),
    ("vectorize", "translate"),
    ("vectorize", "cluster"),
    ("vectorize", "cluster2"),
    ("analysis", "stats"),
    ("analysis", "count_domain_cells"),
    ("analysis", "apply"),
    ("analysis", "fill"),
    ("analysis", "extract"),
    ("analysis", "overlay"),
    ("analysis", "get_mask"),
    ("analysis", "footprint"),
    ("analysis", "get_histogram"),
    ("analysis", "plot"),
]


class TestForwardingParity:
    """Each public collaborator forwarder method delegates to the same-named Dataset method."""

    @pytest.mark.parametrize("collab_attr, method_name", FORWARDING_METHODS)
    def test_method_forwards_args_and_return(
        self,
        in_memory_dataset,
        mocker,
        collab_attr,
        method_name,
    ):
        """Forwarder calls the underlying Dataset method with identical args.

        Args:
            collab_attr: Collaborator attribute name on the Dataset
                (one of ``io``, ``spatial``, ``bands``, ``analysis``,
                ``vectorize``, ``cog``).
            method_name: Public method to test on that collaborator.

        Test scenario:
            Patch ``Dataset.<method_name>`` to return a sentinel. Call
            ``ds.<collab_attr>.<method_name>(1, 2, foo="bar")``. The
            forwarder must (a) pass the positional and keyword arguments
            through unchanged and (b) return the same sentinel.
        """
        sentinel = object()
        mock = mocker.patch.object(
            in_memory_dataset, method_name, return_value=sentinel
        )
        collab = getattr(in_memory_dataset, collab_attr)
        bound = getattr(collab, method_name)

        result = bound(1, 2, foo="bar")

        assert result is sentinel, (
            f"{collab_attr}.{method_name} did not return the underlying "
            f"call's value (got {result!r})"
        )
        mock.assert_called_once_with(1, 2, foo="bar")


class TestFacadeDelegation:
    """Each migrated Dataset facade method delegates to the collaborator method."""

    @pytest.mark.parametrize("collab_attr, method_name", FACADE_METHODS)
    def test_facade_calls_collaborator(
        self,
        in_memory_dataset,
        mocker,
        collab_attr,
        method_name,
    ):
        """``ds.<method>(...)`` should invoke ``ds.<collab>.<method>(...)``.

        Args:
            collab_attr: Collaborator attribute name on the Dataset.
            method_name: Public method that has been migrated onto that
                collaborator.

        Test scenario:
            For methods migrated to a collaborator (Stage 2), the
            Dataset method is now a thin facade. Patch the collaborator
            method to return a sentinel; calling ``ds.<method>(...)``
            should invoke the patched collaborator method with the same
            args and return the sentinel.
        """
        sentinel = object()
        collab = getattr(in_memory_dataset, collab_attr)
        mock = mocker.patch.object(collab, method_name, return_value=sentinel)

        facade = getattr(in_memory_dataset, method_name)
        result = facade(1, 2, foo="bar")

        assert (
            result is sentinel
        ), f"Dataset.{method_name} facade did not return the collaborator's value"
        mock.assert_called_once_with(1, 2, foo="bar")


READONLY_PROPERTIES = []

READWRITE_PROPERTIES: list[tuple[str, str]] = []

# Stage 2 facade properties: Dataset property delegates to a same-named
# property on the collaborator. PR 2.2 — cog.is_cog. PR 2.6 — io.overview_count.
FACADE_PROPERTIES = [
    ("cog", "is_cog"),
    ("io", "overview_count"),
    ("bands", "band_color"),
    ("bands", "color_table"),
]


class TestPropertyForwarding:
    """Properties on collaborators forward to the same-named Dataset property."""

    @pytest.mark.parametrize("collab_attr, prop_name", READONLY_PROPERTIES)
    def test_readonly_property_reads_from_dataset(
        self,
        in_memory_dataset,
        mocker,
        collab_attr,
        prop_name,
    ):
        """Read-only collaborator property delegates to ``Dataset.<prop>``.

        Args:
            collab_attr: Collaborator attribute name (e.g. ``"io"``).
            prop_name: Property on that collaborator (e.g. ``"overview_count"``).

        Test scenario:
            Patch the property on the Dataset class with a ``PropertyMock``
            returning a sentinel. Reading the same-named property through
            the collaborator should produce that sentinel.
        """
        sentinel = object()
        mocker.patch.object(
            type(in_memory_dataset),
            prop_name,
            new_callable=mocker.PropertyMock,
            return_value=sentinel,
        )
        collab = getattr(in_memory_dataset, collab_attr)
        result = getattr(collab, prop_name)
        assert (
            result is sentinel
        ), f"{collab_attr}.{prop_name} getter did not forward to Dataset.{prop_name}"

    @pytest.mark.parametrize("collab_attr, prop_name", FACADE_PROPERTIES)
    def test_facade_property_reads_from_collaborator(
        self,
        in_memory_dataset,
        mocker,
        collab_attr,
        prop_name,
    ):
        """``ds.<prop>`` should read from ``ds.<collab>.<prop>``.

        Args:
            collab_attr: Collaborator attribute name (e.g. ``"cog"``).
            prop_name: Property migrated onto that collaborator.

        Test scenario:
            For Stage 2 properties, the Dataset property is now a thin
            facade reading from the collaborator. Patch the collaborator
            class property with a sentinel value; reading ``ds.<prop>``
            should produce that sentinel.
        """
        sentinel = object()
        collab = getattr(in_memory_dataset, collab_attr)
        mocker.patch.object(
            type(collab),
            prop_name,
            new_callable=mocker.PropertyMock,
            return_value=sentinel,
        )
        result = getattr(in_memory_dataset, prop_name)
        assert result is sentinel, (
            f"Dataset.{prop_name} facade did not read from {collab_attr}.{prop_name} "
            f"(got {result!r})"
        )

    @pytest.mark.parametrize("collab_attr, prop_name", READWRITE_PROPERTIES)
    def test_readwrite_property_get_and_set(
        self,
        in_memory_dataset,
        mocker,
        collab_attr,
        prop_name,
    ):
        """Read/write collaborator property forwards both getter and setter.

        Args:
            collab_attr: Collaborator attribute name.
            prop_name: Read/write property name.

        Test scenario:
            Replace the underlying Dataset property with a ``PropertyMock``.
            Reading via the collaborator must return the mock's value;
            setting via the collaborator must call the mock with the new
            value (PropertyMock records sets as calls with one positional
            arg).
        """
        getter_value = object()
        new_value = {"red": 1, "green": 2}
        prop_mock = mocker.patch.object(
            type(in_memory_dataset),
            prop_name,
            new_callable=mocker.PropertyMock,
            return_value=getter_value,
        )
        collab = getattr(in_memory_dataset, collab_attr)

        observed = getattr(collab, prop_name)
        assert (
            observed is getter_value
        ), f"{collab_attr}.{prop_name} getter did not forward (got {observed!r})"

        setattr(collab, prop_name, new_value)
        prop_mock.assert_any_call(new_value)


class TestAnalysisNormalize:
    """Direct tests for ``Analysis.normalize`` — the only Stage 1 method with a real body."""

    def test_normalize_1d_simple(self):
        """1D array maps min->0 and max->1 with linear scaling between.

        Test scenario:
            Input ``[0, 5, 10]`` with min=0 and max=10 should produce
            ``[0.0, 0.5, 1.0]``.
        """
        out = Analysis.normalize(np.array([0.0, 5.0, 10.0]))
        np.testing.assert_array_equal(
            out,
            np.array([0.0, 0.5, 1.0]),
            err_msg=f"Unexpected normalize output: {out.tolist()}",
        )

    def test_normalize_2d_extrema_and_shape(self):
        """2D array preserves shape and hits 0 and 1 at the extrema.

        Test scenario:
            A 2x2 array ``[[2, 4], [6, 8]]`` should normalize to a 2x2 array
            with min=0.0 and max=1.0 (linear scaling preserves rank).
        """
        out = Analysis.normalize(np.array([[2.0, 4.0], [6.0, 8.0]]))
        assert float(out.min()) == 0.0, f"Expected min 0.0, got {out.min()}"
        assert float(out.max()) == 1.0, f"Expected max 1.0, got {out.max()}"
        assert out.shape == (2, 2), f"Shape mismatch: {out.shape}"

    def test_normalize_signed_values(self):
        """Negative values are normalized into [0, 1] alongside positives.

        Test scenario:
            Input ``[-10, 0, 10]`` (range 20, min -10) should map to
            ``[0.0, 0.5, 1.0]``.
        """
        out = Analysis.normalize(np.array([-10.0, 0.0, 10.0]))
        np.testing.assert_array_equal(
            out,
            np.array([0.0, 0.5, 1.0]),
            err_msg=f"Unexpected normalize output for signed input: {out.tolist()}",
        )

    def test_normalize_returns_ndarray(self):
        """Return value is a NumPy ndarray regardless of input dtype.

        Test scenario:
            Integer input should not be returned as a Python list or
            Python scalar; the staticmethod always returns ``np.ndarray``.
        """
        out = Analysis.normalize(np.array([1, 2, 3, 4]))
        assert isinstance(
            out, np.ndarray
        ), f"Expected numpy ndarray, got {type(out).__name__}"


class TestPickleRoundTrip:
    """Pickle behaviour of collaborators and of the parent Dataset."""

    def test_dataset_round_trip_yields_fresh_collaborators(self, file_backed_dataset):
        """A round-tripped Dataset has fresh collaborators of the right types.

        Test scenario:
            ``RasterBase.__reduce__`` reduces a Dataset to a recipe
            ``(reconstruct_fn, (cls, path, access))`` and re-opens it via
            ``cls.read_file(...)``, which calls ``Dataset.__init__``, which
            instantiates fresh collaborators. The unpickled instance must
            therefore expose all seven collaborators with the correct
            types — never ``_Placeholder`` instances.
        """
        roundtripped = pickle.loads(pickle.dumps(file_backed_dataset))
        assert isinstance(
            roundtripped.io, IO
        ), f"Roundtripped ds.io is wrong type: {type(roundtripped.io).__name__}"
        assert isinstance(
            roundtripped.spatial, Spatial
        ), f"Roundtripped ds.spatial is wrong type: {type(roundtripped.spatial).__name__}"
        assert isinstance(
            roundtripped.bands, Bands
        ), f"Roundtripped ds.bands is wrong type: {type(roundtripped.bands).__name__}"
        assert isinstance(
            roundtripped.analysis, Analysis
        ), f"Roundtripped ds.analysis is wrong type: {type(roundtripped.analysis).__name__}"
        assert isinstance(
            roundtripped.cell, Cell
        ), f"Roundtripped ds.cell is wrong type: {type(roundtripped.cell).__name__}"
        assert isinstance(
            roundtripped.vectorize, Vectorize
        ), f"Roundtripped ds.vectorize is wrong type: {type(roundtripped.vectorize).__name__}"
        assert isinstance(
            roundtripped.cog, COG
        ), f"Roundtripped ds.cog is wrong type: {type(roundtripped.cog).__name__}"

    def test_round_tripped_collaborators_are_functional(self, file_backed_dataset):
        """After round-trip, calling a forwarder still works end-to-end.

        Test scenario:
            ``ds.io.read_array()`` on the unpickled Dataset should return
            the same array data as the original. This confirms that the
            collaborators on the unpickled Dataset are wired to a working
            GDAL handle (i.e., the weakref proxy points at the
            reconstructed Dataset, not at a stale reference).
        """
        original_array = file_backed_dataset.io.read_array()
        roundtripped = pickle.loads(pickle.dumps(file_backed_dataset))
        roundtripped_array = roundtripped.io.read_array()
        np.testing.assert_array_equal(
            roundtripped_array,
            original_array,
            err_msg="Roundtripped collaborator did not read identical array data",
        )

    @pytest.mark.parametrize(
        "collab_attr",
        ["io", "spatial", "bands", "analysis", "cell", "vectorize", "cog"],
    )
    def test_directly_pickled_collaborator_yields_placeholder(
        self,
        in_memory_dataset,
        collab_attr,
    ):
        """Pickling a collaborator directly produces a ``_Placeholder``.

        Args:
            collab_attr: Collaborator attribute name on the Dataset.

        Test scenario:
            ``pickle.dumps(ds.io)`` must not attempt to pickle the parent
            Dataset (that would either explode the payload size or fail
            on the GDAL handle). Instead, ``_Engine.__reduce__``
            short-circuits to ``_recreate_placeholder``, so the unpickled
            object is a ``_Placeholder``.
        """
        collab = getattr(in_memory_dataset, collab_attr)
        unpickled = pickle.loads(pickle.dumps(collab))
        assert isinstance(unpickled, _Placeholder), (
            f"Direct pickle of {collab_attr} collaborator yielded "
            f"{type(unpickled).__name__}, expected _Placeholder"
        )
