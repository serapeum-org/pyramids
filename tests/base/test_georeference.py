"""Tests for :mod:`pyramids.base.georeference`.

`GeoReference` became the single georeferencing input for every raster
constructor in #1075, so its two accepted forms — an affine `geo` transform, or
a `top_left_corner` plus `cell_size` — and the precedence between them are now
load-bearing for `Dataset.from_array`, `Dataset.create`, `Dataset.create_empty`
and `NetCDF.from_array` alike.
"""

from __future__ import annotations

import dataclasses

import pytest

from pyramids.base.georeference import GeoReference, GeoTransformTuple

pytestmark = pytest.mark.core

GEO: GeoTransformTuple = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)


class TestGeoReferenceConstruction:
    """Field defaults and the value-object contract."""

    def test_every_field_defaults_so_an_empty_instance_is_constructible(self):
        """`GeoReference()` builds, deferring the error to resolution time.

        Test scenario:
            The dataclass cannot know which of its two mutually exclusive forms
            a caller intends, so it accepts neither and lets
            `resolve_geotransform` report the problem with both spellings named.
        """
        ref = GeoReference()
        assert ref.geo is None, f"geo should default to None, got {ref.geo}"
        assert ref.top_left_corner is None, "top_left_corner should default to None"
        assert ref.cell_size is None, "cell_size should default to None"

    def test_epsg_defaults_to_wgs84(self):
        """`epsg` defaults to 4326 so the common case needs no argument.

        Test scenario:
            Most callers supply only a transform; defaulting the CRS keeps
            `GeoReference(geo=...)` as short as the flat `geo=` keyword it
            replaced.
        """
        assert GeoReference().epsg == 4326, f"expected 4326, got {GeoReference().epsg}"

    def test_it_is_frozen(self):
        """The value object is immutable.

        Test scenario:
            A constructor argument that a callee could mutate would let one
            raster's georeferencing leak into another's.
        """
        ref = GeoReference(geo=GEO)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.geo = None  # type: ignore[misc]

    def test_equality_is_by_value_not_identity(self):
        """Two distinct references with the same fields compare equal.

        Test scenario:
            Built from differently-spelled but equivalent inputs, so the
            assertion cannot pass on identity alone -- comparing two
            syntactically identical expressions would still succeed if
            `__eq__` fell back to `is`, and would be testing `dataclasses`
            rather than this type.
        """
        from_literal = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
        from_tuple_call = GeoReference(geo=tuple(GEO), epsg=4326)
        assert from_literal is not from_tuple_call, "the fixtures must be distinct"
        assert from_literal == from_tuple_call, (
            f"equal fields must compare equal: {from_literal} != {from_tuple_call}"
        )

    def test_a_differing_field_compares_unequal(self):
        """A reference differing in one field is not equal.

        Test scenario:
            The negative half of the contract. Without it, an `__eq__` that
            returned `True` unconditionally would pass the positive case.
        """
        assert GeoReference(geo=GEO, epsg=4326) != GeoReference(geo=GEO, epsg=3857), (
            "references differing only in epsg must not compare equal"
        )
        assert GeoReference(geo=GEO) != GeoReference(
            top_left_corner=(0.0, 3.0), cell_size=1.0
        ), "references that resolve alike but hold different fields are not equal"

    def test_replace_produces_an_updated_copy(self):
        """`dataclasses.replace` works, which `create_empty` relies on.

        Test scenario:
            `create_empty` substitutes an identity transform into a caller's
            reference that carries only an `epsg`, without mutating the original.
        """
        original = GeoReference(epsg=3857)
        updated = dataclasses.replace(original, geo=GEO)
        assert original.geo is None, "the original must not be mutated"
        assert updated.geo == GEO and updated.epsg == 3857, (
            f"replace must keep the other fields, got {updated}"
        )


class TestResolveGeotransform:
    """The two accepted forms, their precedence, and the failure."""

    def test_an_explicit_geo_is_returned_verbatim(self):
        """A supplied affine transform passes through untouched."""
        assert GeoReference(geo=GEO).resolve_geotransform() == GEO, (
            "an explicit geo must be returned unchanged"
        )

    def test_corner_and_cell_size_build_a_north_up_transform(self):
        """`top_left_corner` + `cell_size` compose the usual north-up affine.

        Test scenario:
            The y pixel size is negated, which is what makes the raster
            north-up; getting that sign wrong flips every raster vertically.
        """
        built = GeoReference(
            top_left_corner=(10.0, 50.0), cell_size=0.25
        ).resolve_geotransform()
        assert built == (10.0, 0.25, 0, 50.0, 0, -0.25), f"got {built}"

    def test_geo_takes_precedence_over_corner_and_cell_size(self):
        """When both forms are given, `geo` wins.

        Test scenario:
            The docstring promises this precedence, and callers that pass a
            template's full transform alongside inherited corner/size values
            depend on it.
        """
        ref = GeoReference(geo=GEO, top_left_corner=(99.0, 99.0), cell_size=9.0)
        assert ref.resolve_geotransform() == GEO, (
            "an explicit geo must win over corner + cell size"
        )

    @pytest.mark.parametrize(
        "ref",
        [
            GeoReference(),
            GeoReference(epsg=3857),
            GeoReference(top_left_corner=(0.0, 1.0)),
            GeoReference(cell_size=1.0),
        ],
        ids=["empty", "epsg-only", "corner-without-size", "size-without-corner"],
    )
    def test_an_incomplete_reference_raises_naming_both_forms(self, ref):
        """Half of the corner/size pair is not enough, and the error says so.

        Args:
            ref: An incompletely specified reference.

        Test scenario:
            A message naming only one accepted spelling would send a caller who
            has the other one down the wrong path.
        """
        with pytest.raises(ValueError, match="top_left_corner") as excinfo:
            ref.resolve_geotransform()
        assert "geo" in str(excinfo.value), (
            f"the error must name both accepted forms, got: {excinfo.value}"
        )

    def test_a_zero_cell_size_is_accepted_rather_than_guessed_at(self):
        """A degenerate cell size resolves; validation belongs to GDAL.

        Test scenario:
            This type's job is composing the affine, not judging it. A zero
            pixel size is nonsense, but rejecting it here would duplicate — and
            eventually contradict — GDAL's own validation.
        """
        built = GeoReference(
            top_left_corner=(0.0, 0.0), cell_size=0
        ).resolve_geotransform()
        assert built == (0.0, 0, 0, 0.0, 0, 0), f"got {built}"

    def test_a_negative_cell_size_is_composed_literally(self):
        """A negative cell size flips the sign, and is not silently corrected."""
        built = GeoReference(
            top_left_corner=(0.0, 0.0), cell_size=-2.0
        ).resolve_geotransform()
        assert built[1] == -2.0 and built[5] == 2.0, f"got {built}"


class TestEpsgHandling:
    """`epsg` is carried, not interpreted, by this type."""

    @pytest.mark.parametrize("epsg", [4326, "4326", 3857, None, 0])
    def test_any_epsg_form_is_carried_through(self, epsg):
        """The field accepts every form the constructors accept.

        Args:
            epsg: The CRS spelling under test.

        Test scenario:
            An int, a numeric string, `None` (deliberately no CRS) and `0` all
            reach the constructors, which decide what each means. Interpreting
            them here would put the rule in two places.
        """
        assert GeoReference(geo=GEO, epsg=epsg).epsg == epsg, (
            f"epsg must be carried verbatim, got {GeoReference(geo=GEO, epsg=epsg).epsg}"
        )

    def test_epsg_does_not_affect_the_resolved_transform(self):
        """The CRS and the affine are independent.

        Test scenario:
            Changing the CRS must not perturb the geotransform; they are
            separate facts about the raster that happen to travel together.
        """
        assert GeoReference(geo=GEO, epsg=4326).resolve_geotransform() == (
            GeoReference(geo=GEO, epsg=3857).resolve_geotransform()
        ), "epsg must not change the resolved transform"
