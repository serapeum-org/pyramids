"""The shared constructor contract introduced by #1075.

`Dataset` and `NetCDF` declare `from_array` with the same core signature, and
`Dataset.create` / `Dataset.create_empty` take the same `geo_ref`. These pin the
contract itself — that the hierarchy agrees, that a polymorphic caller works,
and that the parameters removed in #1075 are genuinely gone — rather than any
one constructor's behaviour, which its own module already covers.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
CONSTRUCTORS = (Dataset, NetCDF)


@pytest.fixture(scope="function")
def array_2d() -> np.ndarray:
    """A small 2-D float array.

    Returns:
        numpy.ndarray: 3x4 of consecutive values.
    """
    return np.arange(12, dtype="float32").reshape(3, 4)


@pytest.fixture(scope="function")
def array_3d() -> np.ndarray:
    """A single-band 3-D array, the shape NetCDF expects.

    Returns:
        numpy.ndarray: 1x3x4 of consecutive values.
    """
    return np.arange(12, dtype="float32").reshape(1, 3, 4)


class TestTheHierarchyAgrees:
    """The Liskov violation #1075 was filed for."""

    def test_a_polymorphic_caller_works_for_both_classes(self, array_2d, array_3d):
        """The issue's own reproduction, which raised `TypeError` before.

        Test scenario:
            A caller written against the declared base type calls `from_array`
            with the same keywords for either class. On `main` the `NetCDF`
            branch rejected `geo=`, because the two concrete classes had
            diverged onto different keyword sets.
        """

        def build(cls, arr):
            return cls.from_array(arr, geo_ref=GeoReference(geo=GEO, epsg=4326))

        assert build(Dataset, array_2d) is not None, "Dataset branch must build"
        assert build(NetCDF, array_3d) is not None, "NetCDF branch must build"

    @pytest.mark.parametrize("cls", CONSTRUCTORS, ids=lambda c: c.__name__)
    def test_the_shared_core_parameters_are_present_on_both(self, cls):
        """Both concrete classes accept the core the abstract base declares.

        Args:
            cls: The concrete class under test.

        Test scenario:
            `arr`, `geo_ref`, `no_data_value` and `path` are the shared core.
            A subclass may add keyword-only parameters, but removing one of
            these would break substitutability again.
        """
        params = inspect.signature(cls.from_array).parameters
        for name in ("arr", "geo_ref", "no_data_value", "path"):
            assert name in params, f"{cls.__name__}.from_array is missing {name}"

    @pytest.mark.parametrize("cls", CONSTRUCTORS, ids=lambda c: c.__name__)
    def test_geo_ref_is_required_and_keyword_only(self, cls):
        """`geo_ref` has no default, so omitting it fails at the call site.

        Args:
            cls: The concrete class under test.

        Test scenario:
            The engine used to declare `geo_ref: GeoReference | None = None`
            and then turn `None` into an empty `GeoReference` whose
            `resolve_geotransform()` raised — a runtime `ValueError` where a
            call-site `TypeError` belongs.
        """
        param = inspect.signature(cls.from_array).parameters["geo_ref"]
        assert param.default is inspect.Parameter.empty, (
            f"{cls.__name__}.from_array should require geo_ref"
        )
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{cls.__name__}.from_array should take geo_ref keyword-only"
        )

    @pytest.mark.parametrize("cls", CONSTRUCTORS, ids=lambda c: c.__name__)
    def test_the_override_is_not_a_bare_args_kwargs_facade(self, cls):
        """The signature is declared, so a static checker can see it.

        Args:
            cls: The concrete class under test.

        Test scenario:
            The `NetCDF` override was `(*args, **kwargs)`, which made an
            incompatible signature look compatible to mypy — the mechanism that
            let the divergence ship. A regression here would restore that
            blindness, so it is worth asserting directly.
        """
        params = inspect.signature(cls.from_array).parameters
        kinds = {p.kind for p in params.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds, (
            f"{cls.__name__}.from_array must not be a **kwargs facade"
        )
        assert inspect.Parameter.VAR_POSITIONAL not in kinds, (
            f"{cls.__name__}.from_array must not be an *args facade"
        )


class TestRemovedParametersAreGone:
    """The flat keywords and `driver_type` were removed outright."""

    @pytest.mark.parametrize("cls", CONSTRUCTORS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(
        "removed", ["geo", "epsg", "top_left_corner", "cell_size", "driver_type"]
    )
    def test_a_removed_keyword_is_rejected(self, cls, removed, array_2d, array_3d):
        """Passing an old keyword raises rather than being silently ignored.

        Args:
            cls: The concrete class under test.
            removed: The keyword removed in #1075.
            array_2d: 2-D fixture, for `Dataset`.
            array_3d: 3-D fixture, for `NetCDF`.

        Test scenario:
            There is no deprecation alias, so the break must be loud. A
            silently-ignored keyword would be the worst outcome: the caller's
            georeferencing would be dropped without warning.
        """
        arr = array_2d if cls is Dataset else array_3d
        with pytest.raises(TypeError, match=removed):
            cls.from_array(arr, geo_ref=GeoReference(geo=GEO), **{removed: 4326})

    def test_the_old_method_name_is_gone(self):
        """`create_from_array` no longer exists on either class.

        Test scenario:
            A lingering alias would defeat the point of renaming without a
            deprecation window, and would leave two spellings in the wild.
        """
        for cls in CONSTRUCTORS:
            assert not hasattr(cls, "create_from_array"), (
                f"{cls.__name__}.create_from_array should be gone"
            )


class TestDriverComesFromPath:
    """`path` alone decides memory-vs-disk and the format."""

    def test_no_path_builds_in_memory(self, array_2d):
        """Omitting `path` yields a MEM raster."""
        ds = Dataset.from_array(array_2d, geo_ref=GeoReference(geo=GEO))
        assert ds.raster.GetDriver().ShortName == "MEM", (
            f"expected MEM, got {ds.raster.GetDriver().ShortName}"
        )

    def test_a_tif_path_writes_a_geotiff(self, array_2d, tmp_path):
        """A `.tif` destination resolves to GTiff and lands on disk."""
        out = tmp_path / "out.tif"
        ds = Dataset.from_array(array_2d, geo_ref=GeoReference(geo=GEO), path=out)
        assert ds.raster.GetDriver().ShortName == "GTiff", (
            f"expected GTiff, got {ds.raster.GetDriver().ShortName}"
        )
        assert out.exists() and out.stat().st_size > 0, "the file must be written"

    def test_the_array_round_trips_through_a_written_file(self, array_2d, tmp_path):
        """Values survive the write, so the driver change did not corrupt output.

        Test scenario:
            Resolving the driver differently could quietly pick a format that
            writes but does not preserve the data; reading back proves it does.
        """
        out = tmp_path / "roundtrip.tif"
        Dataset.from_array(array_2d, geo_ref=GeoReference(geo=GEO), path=out).close()
        reopened = Dataset.read_file(str(out))
        np.testing.assert_array_equal(np.asarray(reopened.read_array()), array_2d)


class TestGeoRefIsHonoured:
    """The resolved transform and CRS actually reach the raster."""

    def test_an_explicit_transform_reaches_the_raster(self, array_2d):
        """The `geo` form lands on the dataset verbatim."""
        ds = Dataset.from_array(array_2d, geo_ref=GeoReference(geo=GEO, epsg=4326))
        assert tuple(ds.geotransform) == GEO, f"got {ds.geotransform}"
        assert ds.epsg == 4326, f"got epsg {ds.epsg}"

    def test_the_corner_and_cell_size_form_reaches_the_raster(self, array_2d):
        """The composed north-up transform lands on the dataset."""
        ds = Dataset.from_array(
            array_2d,
            geo_ref=GeoReference(
                top_left_corner=(10.0, 50.0), cell_size=0.5, epsg=4326
            ),
        )
        assert tuple(ds.geotransform) == (10.0, 0.5, 0, 50.0, 0, -0.5), (
            f"got {ds.geotransform}"
        )

    def test_an_empty_geo_ref_raises_from_resolution(self, array_2d):
        """Supplying an empty reference is a runtime `ValueError`.

        Test scenario:
            The signature can require the argument but cannot check its
            contents, so this failure necessarily stays at resolution time.
        """
        with pytest.raises(ValueError, match="top_left_corner"):
            Dataset.from_array(array_2d, geo_ref=GeoReference())

    def test_a_none_epsg_leaves_the_raster_without_a_crs(self, array_2d):
        """`epsg=None` means deliberately ungeoreferenced, not "default to WGS 84".

        Test scenario:
            A source can genuinely have no CRS; stamping 4326 on it would be a
            silent, wrong claim about the data.
        """
        ds = Dataset.from_array(array_2d, geo_ref=GeoReference(geo=GEO, epsg=None))
        assert not ds.crs, f"expected no CRS, got {ds.crs!r}"


def _build(constructor, geo_ref, array_2d):
    """Call one of the three raster constructors with a georeference.

    Args:
        constructor: The name of the constructor under test.
        geo_ref: The reference to pass.
        array_2d: The 2-D fixture, used only by `from_array`.

    Returns:
        Dataset: The constructed raster.
    """
    if constructor == "from_array":
        built = Dataset.from_array(array_2d, geo_ref=geo_ref)
    elif constructor == "create":
        built = Dataset.create(3, 4, "float32", 1, geo_ref=geo_ref)
    else:
        built = Dataset.create_empty(3, 4, geo_ref=geo_ref)
    return built


CONSTRUCTOR_NAMES = ("from_array", "create", "create_empty")


class TestTheThreeConstructorsAgreeOnGeoRef:
    """One `GeoReference` must mean the same thing to all three constructors.

    Converging them on a single georeferencing input is the point of #1075,
    so a value object accepted by one and reinterpreted by another would
    defeat it. `create_empty` is the one that can differ, because it alone
    treats `geo_ref` as optional.
    """

    @pytest.mark.parametrize("constructor", CONSTRUCTOR_NAMES)
    def test_an_explicit_transform_reaches_every_constructor(
        self, constructor, array_2d
    ):
        """A complete `geo` lands verbatim whichever constructor is used.

        Args:
            constructor: The constructor under test.
            array_2d: The 2-D fixture.
        """
        ds = _build(constructor, GeoReference(geo=GEO, epsg=4326), array_2d)
        assert tuple(ds.geotransform) == GEO, f"{constructor}: got {ds.geotransform}"
        assert ds.epsg == 4326, f"{constructor}: got epsg {ds.epsg}"

    @pytest.mark.parametrize("constructor", CONSTRUCTOR_NAMES)
    @pytest.mark.parametrize(
        "partial",
        [
            GeoReference(top_left_corner=(10.0, 50.0), epsg=4326),
            GeoReference(cell_size=0.5, epsg=4326),
        ],
        ids=["corner-without-size", "size-without-corner"],
    )
    def test_a_partial_reference_raises_in_every_constructor(
        self, constructor, partial, array_2d
    ):
        """Half a corner/cell-size pair is an error, not a request for the origin.

        Args:
            constructor: The constructor under test.
            partial: A reference with one half of the pair missing.
            array_2d: The 2-D fixture.

        Test scenario:
            `create_empty` used to substitute the identity transform whenever
            it could not resolve one, which silently discarded the half the
            caller *did* supply and placed the raster at (0, 0) with 1-unit
            pixels -- while `from_array` and `create` raised for the very same
            value. A silent wrong georeference propagates through every
            downstream crop / align / to_crs / to_file, so it has to be loud.
        """
        with pytest.raises(ValueError, match="top_left_corner"):
            _build(constructor, partial, array_2d)

    def test_only_create_empty_accepts_a_reference_with_no_transform(self, array_2d):
        """`create_empty` alone defaults an absent transform to the identity.

        Test scenario:
            This is the deliberate asymmetry, and it is narrow: a header-only
            allocation often does not care where it sits. It applies only when
            *no* transform is given at all -- which is what keeps it from
            swallowing the partial references above.
        """
        ds = Dataset.create_empty(3, 4, geo_ref=GeoReference(epsg=3857))
        assert tuple(ds.geotransform) == (0.0, 1.0, 0.0, 0.0, 0.0, -1.0), (
            f"expected the identity transform, got {ds.geotransform}"
        )
        assert ds.epsg == 3857, f"the epsg must survive, got {ds.epsg}"
        for constructor in ("from_array", "create"):
            with pytest.raises(ValueError, match="top_left_corner"):
                _build(constructor, GeoReference(epsg=3857), array_2d)

    @pytest.mark.parametrize("constructor", ["create", "create_empty"])
    @pytest.mark.parametrize(
        "removed", ["geo", "epsg", "top_left_corner", "cell_size", "driver_type"]
    )
    def test_a_removed_keyword_is_rejected_by_create_and_create_empty(
        self, constructor, removed, array_2d
    ):
        """The five dropped keywords are gone from these two as well.

        Args:
            constructor: The constructor under test.
            removed: The keyword removed in #1075.
            array_2d: The 2-D fixture, unused here but required by `_build`.

        Test scenario:
            `from_array`'s rejection is pinned above; these two dropped the
            same five keywords and were not covered.
        """
        call = Dataset.create if constructor == "create" else Dataset.create_empty
        args = (3, 4, "float32", 1) if constructor == "create" else (3, 4)
        with pytest.raises(TypeError, match=removed):
            call(*args, geo_ref=GeoReference(geo=GEO), **{removed: 4326})

    def test_create_requires_geo_ref_keyword_only(self):
        """`create` takes `geo_ref` required and keyword-only, like `from_array`.

        Test scenario:
            `create_empty` is deliberately the exception; `create` must not
            drift into being one too.
        """
        param = inspect.signature(Dataset.create).parameters["geo_ref"]
        assert param.default is inspect.Parameter.empty, "create should require geo_ref"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            "create should take geo_ref keyword-only"
        )

    def test_a_none_epsg_leaves_create_without_a_crs(self):
        """`create(geo_ref=GeoReference(epsg=None))` yields a CRS-less raster.

        Test scenario:
            `create` used to take a required `epsg: int` straight through
            `sr_from_epsg`; routing it through the shared helper made
            `epsg=None` newly expressible, and nothing pinned the result.
        """
        ds = Dataset.create(
            3, 4, "float32", 1, geo_ref=GeoReference(geo=GEO, epsg=None)
        )
        assert not ds.crs, f"expected no CRS, got {ds.crs!r}"
