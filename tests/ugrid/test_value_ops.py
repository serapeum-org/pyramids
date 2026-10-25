"""Generic value operations on a mesh: ``where`` / ``apply`` / ``get_histogram`` (issue #1375).

Three operations the raster side had and the mesh did not. The pointed one is
``get_histogram``: the mesh could already *plot* a histogram but had no way to obtain the
numbers behind the bars.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.analysis import valid_samples

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])


def _mesh(values, nodata=None):
    """A three-face mesh carrying ``values`` as a face variable."""
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.asarray(values)},
        data_locations={"v": "face"},
    )
    if nodata is not None:
        ds["v"].nodata = nodata
    return ds


class TestWhere:
    """``where`` keeps what the condition selects and masks the rest."""

    def test_keeps_selected_masks_rest(self):
        ds = _mesh([1.0, 5.0, 9.0])
        out = ds.where(np.array([False, True, True]))["v"].data
        assert np.isnan(out[0])
        assert out[1] == 5.0
        assert out[2] == 9.0

    def test_condition_from_the_values_themselves(self):
        ds = _mesh([1.0, 5.0, 9.0])
        values = np.asarray(ds["v"].data)
        out = ds.where(values > 4)["v"].data
        assert np.isnan(out[0])
        assert out[1:].tolist() == [5.0, 9.0]

    def test_a_nodata_element_is_never_selected(self):
        """A missing value cannot be "kept" — it masks whatever the condition says."""
        ds = _mesh([-999.0, 5.0, 9.0], nodata=-999.0)
        out = ds.where(np.array([True, True, True]))["v"].data
        assert np.isnan(out[0]), "a nodata element must not survive selection"
        assert out[1:].tolist() == [5.0, 9.0]

    def test_custom_other(self):
        ds = _mesh([1.0, 5.0, 9.0])
        out = ds.where(np.array([False, True, True]), other=0.0)["v"].data
        assert out.tolist() == [0.0, 5.0, 9.0]

    def test_composes_with_the_active_mask(self):
        """The active mask is not applied implicitly; a caller composes it into the condition.

        Test scenario:
            #1331 keeps "dry" distinct from "missing", so `where` must not silently consult the
            mask. This pins the documented composition instead.
        """
        ds = _mesh([1.0, 5.0, 9.0])
        ds["v"].active = np.array([True, False, True])
        values = np.asarray(ds["v"].data)
        out = ds.where(ds.active_mask("v") & (values > 0))["v"].data
        assert out[0] == 1.0
        assert np.isnan(out[1]), (
            "the dry element was excluded by the explicit condition"
        )
        assert out[2] == 9.0
        # and without composing it, the mask is ignored
        plain = ds.where(values > 0)["v"].data
        assert plain.tolist() == [1.0, 5.0, 9.0]


class TestApply:
    """``apply`` maps a function over the values, keeping the shape."""

    def test_vectorised_ufunc(self):
        out = _mesh([1.0, 4.0, 9.0]).apply(np.sqrt)["v"].data
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_lambda(self):
        out = _mesh([1.0, 2.0, 3.0]).apply(lambda a: a * 10)["v"].data
        assert out.tolist() == [10.0, 20.0, 30.0]

    def test_preserves_location_and_dimensions(self):
        ds = _mesh([1.0, 2.0, 3.0])
        out = ds.apply(np.sqrt)["v"]
        assert out.location == "face"
        assert out.dimensions == ds["v"].dimensions

    def test_refuses_a_shape_changing_function(self):
        """A function that drops elements would mis-align the result against the mesh."""
        subject = _mesh([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="changed the array shape"):
            subject.apply(lambda a: a[:2])


class TestGetHistogram:
    """``get_histogram`` returns the distribution as data, not as a picture."""

    def test_counts_and_edges(self):
        counts, edges = _mesh([1.0, 2.0, 9.0]).get_histogram(bins=2)
        assert counts.tolist() == [2, 1]
        assert edges.tolist() == [1.0, 5.0, 9.0]

    def test_edges_count_is_bins_plus_one(self):
        counts, edges = _mesh([1.0, 5.0, 9.0]).get_histogram(bins=4)
        assert len(counts) == 4
        assert len(edges) == 5

    def test_value_range_restricts_the_bins(self):
        counts, edges = _mesh([1.0, 5.0, 9.0]).get_histogram(
            bins=2, value_range=(0.0, 10.0)
        )
        assert edges.tolist() == [0.0, 5.0, 10.0]
        assert counts.sum() == 3

    def test_nodata_is_excluded(self):
        counts, _ = _mesh([-999.0, 5.0, 9.0], nodata=-999.0).get_histogram(bins=2)
        assert counts.sum() == 2, "the nodata element must not be counted"

    def test_counts_describe_the_same_samples_plot_histogram_shows(self):
        """Regression (#1375 DoD): both paths mask through `valid_samples`, so the numbers and
        the bars cannot disagree."""
        ds = _mesh([-999.0, 1.0, 9.0], nodata=-999.0)
        counts, _ = ds.get_histogram(bins=3)
        shared = valid_samples(np.asarray(ds["v"].data), ds["v"].nodata)
        assert counts.sum() == shared.size

    def test_all_nodata_is_refused(self):
        subject = _mesh([-999.0, -999.0, -999.0], nodata=-999.0)
        with pytest.raises(ValueError, match="No valid samples"):
            subject.get_histogram()


class TestWhereDtypePromotion:
    """What happens to the dtype when the fill value does not fit it."""

    def test_a_fractional_fill_promotes_an_integer_array(self):
        """Truncating 0.5 to 0 in an int array would be a silent wrong answer, so the array
        is promoted to float instead."""
        from pyramids.base._array_ops import where_values

        ints = np.array([1, 2, 3, 4], dtype="int32")
        result = where_values(ints, np.array([True, False, True, False]), other=0.5)
        assert result.dtype == np.dtype("float64"), (
            "the array promoted rather than truncating"
        )
        assert result.tolist() == pytest.approx([1.0, 0.5, 3.0, 0.5])

    def test_a_fitting_fill_keeps_the_values_exact(self):
        from pyramids.base._array_ops import where_values

        values = np.array([1.0, 2.0, 3.0])
        result = where_values(values, np.array([True, False, True]), other=-1.0)
        assert result.tolist() == pytest.approx([1.0, -1.0, 3.0])

    @pytest.mark.parametrize(
        "value",
        [[[1, 2], [3]], "text", {"a": 1}, None],
        ids=["ragged", "str", "dict", "none"],
    )
    def test_a_value_numpy_cannot_read_answers_false_rather_than_raising(self, value):
        """`_fits` promises a bool; a ragged list used to propagate numpy's ValueError out of
        it, which is not an answer to "does this fit".

        Args:
            value: Something numpy cannot read as a dtype-compatible scalar.
        """
        from pyramids.base._array_ops import _fits

        assert _fits(value, np.dtype("float64")) is False


class TestWhereKeepsTheDtypeWhenItCan:
    """Promotion should happen only when the fill value needs it (review L4)."""

    def test_an_integer_fill_keeps_an_integer_array(self):
        """`np.asarray(0).dtype` is int64, which never casts safely into int32, so every
        Python int used to force a float64 promotion for a value that fits perfectly."""
        from pyramids.base._array_ops import where_values

        ints = np.array([1, 2, 3, 4], dtype="int32")
        result = where_values(ints, np.array([True, False, True, False]), other=0)
        assert result.dtype == np.dtype("int32"), f"promoted to {result.dtype}"
        assert result.tolist() == [1, 0, 3, 0]

    def test_a_fractional_fill_still_promotes(self):
        from pyramids.base._array_ops import where_values

        ints = np.array([1, 2], dtype="int32")
        result = where_values(ints, np.array([True, False]), other=0.5)
        assert result.dtype == np.dtype("float64")
        assert result.tolist() == pytest.approx([1.0, 0.5])

    def test_an_out_of_range_integer_promotes_rather_than_overflowing(self):
        """`result_type` reports the kind, not the range - it calls 2**40 an int32, and the
        assignment would then raise OverflowError on a value float64 holds exactly."""
        from pyramids.base._array_ops import where_values

        ints = np.array([1, 2], dtype="int32")
        result = where_values(ints, np.array([True, False]), other=2**40)
        assert result.dtype == np.dtype("float64")
        assert result[1] == pytest.approx(float(2**40))


class TestWhereShapeMessage:
    """A mismatched condition must name the problem (review L10)."""

    def test_the_message_names_both_shapes(self):
        """Was numpy's bare "operands could not be broadcast together"."""
        from pyramids.base._array_ops import where_values

        with pytest.raises(ValueError, match=r"cannot be broadcast against the values"):
            where_values(np.array([1.0, 2.0]), np.array([True, True, True]))

    def test_a_condition_on_one_axis_broadcasts(self):
        """A `(n_face,)` condition against a `(time, face)` variable means "these faces, every
        step" - the natural thing to write, and what the docstring promised all along."""
        from pyramids.base._array_ops import where_values

        values = np.ones((2, 2))
        result = where_values(values, np.array([True, False]), other=0.0)
        assert result.tolist() == [[1.0, 0.0], [1.0, 0.0]]

    def test_the_message_suggests_how_to_build_the_condition(self):
        from pyramids.base._array_ops import where_values

        with pytest.raises(ValueError, match=r"ds\[name\]"):
            where_values(np.array([1.0, 2.0]), np.array([True, True, True]))


class TestUnknownVariableMessageWording:
    """The shared message, including the empty-dataset wording (round 2 /test gap)."""

    def test_an_empty_dataset_says_so_rather_than_printing_an_empty_list(self):
        from pyramids.netcdf.ugrid.messages import unknown_variable_message

        message = unknown_variable_message("slope", "z", [])
        assert "holds no data variables" in message, message
        assert "[]" not in message, message

    def test_a_populated_dataset_lists_its_names_sorted(self):
        from pyramids.netcdf.ugrid.messages import unknown_variable_message

        message = unknown_variable_message("polyfit", "dpeth", ["velocity", "depth"])
        assert "['depth', 'velocity']" in message, message
