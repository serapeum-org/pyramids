"""Tests for :meth:`pyramids.feature.FeatureCollection.from_bbox`."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


class TestFeatureCollectionFromBbox:
    """Tests for the bbox primitive."""

    def test_returns_feature_collection(self):
        """The result is a :class:`FeatureCollection`, not a bare ``GeoDataFrame``.

        Test scenario:
            ``FeatureCollection.from_bbox((0, 0, 1, 1), epsg=4326)`` — expected:
            a ``FeatureCollection`` of length 1 (the rectangle).
        """
        fc = FeatureCollection.from_bbox((0.0, 0.0, 1.0, 1.0), epsg=4326)
        assert isinstance(fc, FeatureCollection), (
            f"expected FeatureCollection, got {type(fc)}"
        )
        assert len(fc) == 1, f"expected one row, got {len(fc)}"

    def test_geometry_matches_shapely_box(self):
        """The single row's geometry equals :func:`shapely.geometry.box`'s output.

        Test scenario:
            Build with ``(w, s, e, n) = (10, 20, 30, 40)`` — expected: the row's
            geometry equals ``shapely.geometry.box(10, 20, 30, 40)``.
        """
        fc = FeatureCollection.from_bbox((10, 20, 30, 40), epsg=4326)
        expected = box(10, 20, 30, 40)
        assert next(iter(fc.geometry)).equals(expected), (
            f"unexpected geometry: {next(iter(fc.geometry)).wkt}"
        )

    def test_total_bounds_round_trip(self):
        """``total_bounds`` on the result matches the input ``(w, s, e, n)``.

        Test scenario:
            ``FeatureCollection.from_bbox((W, S, E, N), epsg=4326).total_bounds``
            — expected: ``[W, S, E, N]`` byte-for-byte.
        """
        bbox = (31.0, 30.0, 31.1, 30.1)
        fc = FeatureCollection.from_bbox(bbox, epsg=4326)
        bounds = tuple(float(v) for v in fc.total_bounds)
        assert bounds == bbox, f"total_bounds {bounds} != input {bbox}"

    def test_epsg_int_applies(self):
        """An ``int`` ``epsg`` becomes the CRS on the result.

        Test scenario:
            ``epsg=3857`` — expected: ``fc.crs`` reports EPSG 3857.
        """
        fc = FeatureCollection.from_bbox((0, 0, 100, 100), epsg=3857)
        assert fc.crs is not None, "CRS was not set"
        assert fc.crs.to_epsg() == 3857, f"expected 3857, got {fc.crs.to_epsg()}"

    @pytest.mark.parametrize(
        "epsg_form",
        ["EPSG:4326", 4326, "+proj=longlat +datum=WGS84 +no_defs"],
        ids=["string", "int", "proj4"],
    )
    def test_epsg_accepts_geopandas_crs_forms(self, epsg_form):
        """Any ``geopandas``-accepted ``crs=`` form works — string, int, proj4.

        Args:
            epsg_form: CRS spec in one of the supported forms.

        Test scenario:
            ``from_bbox(..., epsg=epsg_form)`` — expected: the resulting CRS
            resolves to EPSG 4326.
        """
        fc = FeatureCollection.from_bbox((0, 0, 1, 1), epsg=epsg_form)
        assert fc.crs is not None, f"CRS not set for {epsg_form!r}"
        assert fc.crs.to_epsg() == 4326, (
            f"expected EPSG 4326 from {epsg_form!r}, got {fc.crs.to_epsg()}"
        )

    def test_accepts_list_in_addition_to_tuple(self):
        """A 4-element ``list`` works as well as a tuple.

        Test scenario:
            ``from_bbox([0, 0, 1, 1], epsg=4326)`` — expected: equivalent to the
            tuple form.
        """
        a = FeatureCollection.from_bbox([0, 0, 1, 1], epsg=4326)
        b = FeatureCollection.from_bbox((0, 0, 1, 1), epsg=4326)
        assert next(iter(a.geometry)).equals(next(iter(b.geometry))), (
            "list and tuple inputs produced different geometries"
        )

    def test_accepts_integer_coordinates(self):
        """Integer coordinates are accepted and cast to float internally.

        Test scenario:
            ``from_bbox((0, 0, 1, 1), epsg=4326)`` (ints) — expected: a valid
            polygon with float ``total_bounds``.
        """
        fc = FeatureCollection.from_bbox((0, 0, 1, 1), epsg=4326)
        assert tuple(float(v) for v in fc.total_bounds) == (
            0.0,
            0.0,
            1.0,
            1.0,
        ), f"unexpected bounds: {fc.total_bounds}"

    def test_none_epsg_raises(self):
        """``epsg=None`` is rejected — a bbox without a CRS is ambiguous.

        Test scenario:
            ``from_bbox((0, 0, 1, 1), epsg=None)`` — expected: ``ValueError``
            mentioning ``epsg``.
        """
        with pytest.raises(ValueError, match="epsg") as exc:
            FeatureCollection.from_bbox((0, 0, 1, 1), epsg=None)
        assert "epsg" in str(exc.value), f"unexpected message: {exc.value}"

    @pytest.mark.parametrize(
        "bad, n",
        [
            ((0, 0, 1), 3),
            ((0, 0, 1, 1, 2), 5),
            ((), 0),
        ],
        ids=["3-tuple", "5-tuple", "empty"],
    )
    def test_wrong_length_raises(self, bad, n):
        """A bbox of length != 4 raises ``ValueError`` reporting the actual count.

        Args:
            bad: A sequence of the wrong length.
            n: Its length.

        Test scenario:
            ``from_bbox(bad, epsg=4326)`` — expected: ``ValueError`` mentioning
            ``4 elements`` and the wrong count.
        """
        with pytest.raises(ValueError, match=r"4 elements") as exc:
            FeatureCollection.from_bbox(bad, epsg=4326)
        assert f"got {n}" in str(exc.value), (
            f"missing actual length in message: {exc.value}"
        )

    def test_non_iterable_raises(self):
        """A non-sequence bbox raises ``ValueError`` from the ``list()`` coercion.

        Test scenario:
            ``from_bbox(42, epsg=4326)`` — expected: ``ValueError`` mentioning
            ``4-element``.
        """
        with pytest.raises(ValueError, match="4-element"):
            FeatureCollection.from_bbox(42, epsg=4326)

    def test_nan_bbox_raises(self):
        """A NaN coordinate (e.g. an empty frame's all-NaN bounds) is rejected."""
        nan = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            FeatureCollection.from_bbox((nan, nan, nan, nan), epsg=4326)
        with pytest.raises(ValueError, match="NaN"):
            FeatureCollection.from_bbox((0, 0, nan, 1), epsg=4326)

    @pytest.mark.parametrize(
        "bad",
        [
            (0, 0, 1, "x"),
            ("a", "b", "c", "d"),
            (0, 0, 1, None),
            (0, 0, 1, object()),
        ],
        ids=["one-string", "all-strings", "none", "object"],
    )
    def test_non_numeric_elements_raise_type_error(self, bad):
        """Any non-numeric element raises ``TypeError`` mentioning ``numbers``.

        Args:
            bad: A 4-tuple containing at least one non-numeric element.

        Test scenario:
            ``from_bbox(bad, epsg=4326)`` — expected: ``TypeError``.
        """
        with pytest.raises(TypeError, match="numbers"):
            FeatureCollection.from_bbox(bad, epsg=4326)

    @pytest.mark.parametrize(
        "bbox",
        [(1, 0, 0, 1), (0, 0, 0, 1)],
        ids=["w-greater", "w-equal"],
    )
    def test_west_not_less_than_east_raises(self, bbox):
        """``west < east`` is required; ``>=`` raises ``ValueError``.

        Args:
            bbox: A bbox with ``west >= east``.

        Test scenario:
            ``from_bbox(bbox, epsg=4326)`` — expected: ``ValueError`` mentioning
            ``west < east``.
        """
        with pytest.raises(ValueError, match=r"west < east"):
            FeatureCollection.from_bbox(bbox, epsg=4326)

    @pytest.mark.parametrize(
        "bbox",
        [(0, 1, 1, 0), (0, 0, 1, 0)],
        ids=["s-greater", "s-equal"],
    )
    def test_south_not_less_than_north_raises(self, bbox):
        """``south < north`` is required; ``>=`` raises ``ValueError``.

        Args:
            bbox: A bbox with ``south >= north``.

        Test scenario:
            ``from_bbox(bbox, epsg=4326)`` — expected: ``ValueError`` mentioning
            ``south < north``.
        """
        with pytest.raises(ValueError, match=r"south < north"):
            FeatureCollection.from_bbox(bbox, epsg=4326)

    def test_independent_calls(self):
        """Two calls produce independent objects (no shared state).

        Test scenario:
            Build two FCs and mutate one — expected: the other is unchanged.
        """
        a = FeatureCollection.from_bbox((0, 0, 1, 1), epsg=4326)
        b = FeatureCollection.from_bbox((10, 10, 11, 11), epsg=4326)
        assert tuple(a.total_bounds) != tuple(b.total_bounds), (
            "from_bbox shared state across calls"
        )
