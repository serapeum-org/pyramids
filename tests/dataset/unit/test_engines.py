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

from pyramids.base.crs import sr_from_epsg
from pyramids.dataset import Dataset
from pyramids.dataset.engines.bands import _is_read_only_error
from pyramids.dataset.engines.cog import _cached_transformer
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
        assert isinstance(result, _Placeholder), (
            f"Expected _Placeholder instance, got {type(result).__name__}"
        )

    def test_each_call_returns_distinct_instance(self):
        """Successive calls must produce distinct objects, not a singleton.

        Test scenario:
            Two separate calls to the factory should yield two distinct
            placeholders so a buggy callsite cannot accidentally share
            state across collaborator instances.
        """
        first = _recreate_placeholder()
        second = _recreate_placeholder()
        assert first is not second, (
            "Successive calls to _recreate_placeholder returned the same instance"
        )


class TestCollaboratorBase:
    """Tests for :class:`_Engine` (base class shared by all collaborators)."""

    def test_init_stores_weak_proxy_to_dataset(self, in_memory_dataset):
        """Constructor must wrap the Dataset in a ``weakref.proxy``.

        Test scenario:
            After construction, attribute access through ``self._ds`` should
            transparently resolve to the wrapped Dataset's attributes.
        """
        collab = _Engine(in_memory_dataset)
        assert collab._ds.epsg == in_memory_dataset.epsg, (
            f"Proxy did not resolve epsg: {collab._ds.epsg} != {in_memory_dataset.epsg}"
        )
        assert collab._ds.rows == in_memory_dataset.rows, (
            "Proxy did not resolve rows attribute"
        )

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
        assert _Engine.__slots__ == ("_ds",), (
            f"Expected __slots__ == ('_ds',), got {_Engine.__slots__!r}"
        )


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
        assert isinstance(collab, expected_type), (
            f"ds.{attr_name} should be {expected_type.__name__}, got {type(collab).__name__}"
        )


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
    ("spatial", "wrap_longitude"),
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
    ("vectorize", "to_polygons"),
    ("analysis", "stats"),
    ("analysis", "count_domain_cells"),
    ("analysis", "apply"),
    ("analysis", "fill"),
    ("analysis", "extract"),
    ("analysis", "overlay"),
    ("analysis", "get_mask"),
    ("analysis", "footprint"),
    ("analysis", "get_histogram"),
    # NB: ``("analysis", "plot")`` is intentionally excluded — ``Dataset.plot`` is
    # no longer a pure pass-through facade since PR-1 (D-0). It now applies the
    # GeoTIFF/Sentinel band-resolution policy (``_resolve_plot_band``) before
    # delegating, so its signature and call shape differ from the engine's. The
    # band-resolution behaviour is covered by ``TestResolvePlotBand`` in
    # ``tests/dataset/test_plot.py``.
]


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

        assert result is sentinel, (
            f"Dataset.{method_name} facade did not return the collaborator's value"
        )
        mock.assert_called_once_with(1, 2, foo="bar")


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
        assert float(out.min()) == pytest.approx(0.0), (
            f"Expected min 0.0, got {out.min()}"
        )
        assert float(out.max()) == pytest.approx(1.0), (
            f"Expected max 1.0, got {out.max()}"
        )
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
        assert isinstance(out, np.ndarray), (
            f"Expected numpy ndarray, got {type(out).__name__}"
        )


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
        assert isinstance(roundtripped.io, IO), (
            f"Roundtripped ds.io is wrong type: {type(roundtripped.io).__name__}"
        )
        assert isinstance(roundtripped.spatial, Spatial), (
            f"Roundtripped ds.spatial is wrong type: {type(roundtripped.spatial).__name__}"
        )
        assert isinstance(roundtripped.bands, Bands), (
            f"Roundtripped ds.bands is wrong type: {type(roundtripped.bands).__name__}"
        )
        assert isinstance(roundtripped.analysis, Analysis), (
            f"Roundtripped ds.analysis is wrong type: {type(roundtripped.analysis).__name__}"
        )
        assert isinstance(roundtripped.cell, Cell), (
            f"Roundtripped ds.cell is wrong type: {type(roundtripped.cell).__name__}"
        )
        assert isinstance(roundtripped.vectorize, Vectorize), (
            f"Roundtripped ds.vectorize is wrong type: {type(roundtripped.vectorize).__name__}"
        )
        assert isinstance(roundtripped.cog, COG), (
            f"Roundtripped ds.cog is wrong type: {type(roundtripped.cog).__name__}"
        )

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


class TestIsReadOnlyError:
    """ARC-68: the read-only classification lives in one place now."""

    @pytest.mark.parametrize(
        "refusing_call",
        ["GDALRasterBand::Fill()", "GDALRasterBand::RasterIO()"],
    )
    def test_a_real_gdal_refusal_is_recognised(self, refusing_call, tmp_path):
        """The message GDAL actually emits classifies as read-only.

        Args:
            refusing_call: The GDAL entry point named in the message.
            tmp_path: pytest's temporary directory fixture.

        Test scenario:
            Reproduces the exact string GDAL 3.13 produces -- file, band and
            refusing call, then the reason -- for both the Fill and RasterIO
            paths. The check was repeated at three no-data setters, each testing
            a different subset of the wordings, so two of them never recognised
            a refusal and surfaced a raw RuntimeError instead.
        """
        message = (
            f"ro.tif, band 1: {refusing_call}: attempt to write to dataset "
            "opened in read-only mode."
        )
        assert _is_read_only_error(RuntimeError(message)), (
            f"{message!r} is a read-only refusal and must be classified as one"
        )

    def test_the_classification_matches_a_live_gdal_error(self, tmp_path):
        """The pattern is checked against GDAL itself, not a remembered string.

        Test scenario:
            The wording is GDAL's, so a hard-coded copy drifts silently on a
            version bump. Opens a real raster read-only, provokes the refusal,
            and asserts the classifier recognises whatever GDAL raised.
        """
        path = tmp_path / "ro.tif"
        Dataset.create_from_array(
            np.zeros((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        ).to_file(str(path))
        read_only = gdal.Open(str(path), gdal.GA_ReadOnly)
        with pytest.raises(RuntimeError) as excinfo:
            read_only.GetRasterBand(1).Fill(1.0)
        assert _is_read_only_error(excinfo.value), (
            f"GDAL's own refusal must classify as read-only: {excinfo.value}"
        )

    def test_an_unrelated_error_is_not_classified(self):
        """A different GDAL failure must not be swallowed as read-only.

        Test scenario:
            Misclassifying would turn a genuine I/O failure into a ReadOnlyError
            and send the caller looking at the access mode instead of the disk.
        """
        assert not _is_read_only_error(RuntimeError("Cannot allocate memory")), (
            "an unrelated RuntimeError must not classify as a read-only refusal"
        )


class TestCachedTransformer:
    """ARC-60: the pyproj transformer is built once per CRS pair."""

    def test_the_same_crs_pair_returns_the_same_object(self):
        """A repeat call is a cache hit, not a rebuild.

        Test scenario:
            Building a transformer parses both CRS definitions and resolves a
            pipeline. The COG read paths rebuilt one per tile and per point, so
            a read_tile loop paid that cost on every call.
        """
        _cached_transformer.cache_clear()
        first = _cached_transformer(4326, 3857)
        second = _cached_transformer(4326, 3857)
        assert first is second, "a repeated CRS pair must return the cached object"
        assert _cached_transformer.cache_info().hits == 1, (
            f"expected one cache hit, got {_cached_transformer.cache_info()}"
        )

    def test_a_different_pair_builds_a_new_transformer(self):
        """The cache is keyed on both CRSes, not just the source."""
        _cached_transformer.cache_clear()
        to_mercator = _cached_transformer(4326, 3857)
        to_utm = _cached_transformer(4326, 32636)
        assert to_mercator is not to_utm, "distinct destinations must not share a key"

    def test_a_wkt_string_key_works_and_transforms_correctly(self):
        """A WKT string is a valid, hashable key and yields a working transformer.

        Test scenario:
            Datasets without an EPSG code pass their WKT through instead, so the
            cache has to accept it. Round-trips a point through the transformer
            and back to confirm the cached object is functional, not just
            identical.
        """
        _cached_transformer.cache_clear()
        wkt = sr_from_epsg(4326).ExportToWkt()
        forward = _cached_transformer(wkt, 3857)
        back = _cached_transformer(3857, wkt)
        x, y = forward.transform(12.0, 55.0)
        lon, lat = back.transform(x, y)
        assert round(lon, 6) == 12.0 and round(lat, 6) == 55.0, (
            f"round trip through the cached transformers drifted: {lon}, {lat}"
        )
