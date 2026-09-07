"""``Analysis.combine`` on a NetCDF variable view, and where the result can go.

`combine` builds its result with the operand's own class — the same rule `apply`, `crop`
and `resample` follow — so combining two variable views hands back a `Variable`. These
tests pin what that object can and cannot do, so the limitation is a recorded contract
rather than something a user discovers when a write fails.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)


def _variable(tmp_path, name: str, value: float):
    """Write a one-variable NetCDF and return its variable view.

    Args:
        tmp_path: pytest temp directory.
        name: File name to write under `tmp_path`.
        value: Constant filling the variable's single band.

    Returns:
        The variable view, ready to combine.
    """
    path = str(tmp_path / name)
    NetCDF.from_array(
        np.full((4, 4), value, "float32"), geo_ref=GEO_REF, variable_name="t"
    ).to_file(path)
    return NetCDF.read_file(path).get_variable("t")


class TestCombineNetCDFVariable:
    """Two NetCDF variable views combine, and the result behaves like `apply`'s."""

    def test_two_variable_views_subtract(self, tmp_path):
        """The arithmetic itself works on a variable view.

        Test scenario:
            10 minus 4 across a one-variable NetCDF gives 6, on the same grid and CRS.
        """
        left = _variable(tmp_path, "a.nc", 10.0)
        right = _variable(tmp_path, "b.nc", 4.0)

        result = left - right

        assert np.allclose(np.asarray(result.read_array()), 6.0)
        assert result.epsg == left.epsg
        assert result.geotransform == left.geotransform

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            (lambda a, b: a - b, 6.0),
            (lambda a, b: a + b, 14.0),
            (lambda a, b: sum([a, b]), 14.0),
            (lambda a, b: math.prod([a, b]), 40.0),
            (lambda a, b: sum([a]), 10.0),
        ],
        ids=["sub", "add", "sum", "prod", "sum-one"],
    )
    def test_every_operator_works_on_a_variable_view(
        self, tmp_path, expression, expected
    ):
        """`NetCDF` inherits the operators, so they must be exercised on it too.

        Args:
            tmp_path: pytest temp directory.
            expression: The fold or operator under test.
            expected: The mean every result cell must produce.

        Test scenario:
            10 and 4 through each operator. `sum`/`math.prod` matter most: they route
            through `__radd__`/`__rmul__`, which copy the operand, and a plain `Dataset`
            copy behaves differently from a variable view's.
        """
        left = _variable(tmp_path, "a.nc", 10.0)
        right = _variable(tmp_path, "b.nc", 4.0)

        result = expression(left, right)

        assert np.asarray(result.read_array()).mean() == pytest.approx(expected)

    def test_a_comparison_on_variables_gives_a_byte_mask(self, tmp_path):
        """The comparison operators reach `NetCDF` through inheritance as well.

        Test scenario:
            `>=` on two variable views gives the same uint8/255 mask a plain raster does.
        """
        left = _variable(tmp_path, "a.nc", 10.0)
        right = _variable(tmp_path, "b.nc", 4.0)

        mask = left >= right

        assert np.asarray(mask.read_array()).dtype == np.uint8
        assert mask.no_data_value[0] == 255
        assert (np.asarray(mask.read_array()) == 1).all()

    def test_no_netcdf_object_has_a_truth_value(self, tmp_path):
        """The class-wide refusal reaches every `Dataset` subclass.

        Test scenario:
            Both a variable view and the root container raise, so `if nc:` cannot be
            silently true anywhere in the NetCDF hierarchy either. `is not None` is the
            presence check.
        """
        path = str(tmp_path / "root.nc")
        NetCDF.from_array(
            np.full((4, 4), 1.0, "float32"), geo_ref=GEO_REF, variable_name="t"
        ).to_file(path)
        container = NetCDF.read_file(path)

        for obj in (container, container.get_variable("t")):
            with pytest.raises(
                ValueError, match="truth value of a Dataset is ambiguous"
            ):
                bool(obj)
        assert container is not None, "presence is asked with `is not None`"

    def test_metadata_carries_without_coercing_the_netcdf_type(self, tmp_path):
        """`NetCDF.meta_data` is a `NetCDFMetadata`, not a dict.

        Test scenario:
            Carrying dataset metadata through `combine` must assign it through rather
            than rebuild it — coercing with `dict(...)` raised `TypeError:
            'NetCDFMetadata' object is not iterable` and broke every operator on a
            variable view.
        """
        left = _variable(tmp_path, "a.nc", 10.0)
        right = _variable(tmp_path, "b.nc", 4.0)

        result = left - right

        assert result.meta_data is not None

    def test_the_result_matches_what_apply_returns(self, tmp_path):
        """`combine` inherits `apply`'s class rule rather than inventing its own.

        Test scenario:
            Both hand back the same type wrapping a plain in-memory raster with no
            variables of its own, so a `.tif` destination is refused by the NetCDF
            writer. Pinned because the limitation is shared, not introduced here.
        """
        left = _variable(tmp_path, "a.nc", 10.0)
        right = _variable(tmp_path, "b.nc", 4.0)

        combined = left - right
        applied = left.apply(np.abs)

        assert type(combined) is type(applied), "combine must not diverge from apply"
        assert combined.variable_names == applied.variable_names
        with pytest.raises(ValueError, match="Cannot save a multidimensional NetCDF"):
            combined.to_file(str(tmp_path / "out.tif"))

    def test_the_values_round_trip_through_a_plain_dataset(self, tmp_path):
        """The documented way out: read the array and wrap it.

        Test scenario:
            The combined values written as a GeoTIFF through `Dataset.from_array`,
            keeping the operand's grid.
        """
        left = _variable(tmp_path, "a.nc", 10.0)
        right = _variable(tmp_path, "b.nc", 4.0)
        combined = left - right
        destination = str(tmp_path / "diff.tif")

        Dataset.from_array(
            np.asarray(combined.read_array()),
            geo_ref=GeoReference(geo=combined.geotransform, epsg=combined.epsg),
        ).to_file(destination)

        written = Dataset.read_file(destination)
        assert np.allclose(np.asarray(written.read_array()), 6.0)
        assert written.geotransform == left.geotransform

    def test_a_root_container_is_refused_by_the_read(self, tmp_path):
        """A multidimensional container has no single grid to combine.

        Test scenario:
            `nc.combine(nc, ...)` on the root store raises the container guard, the same
            one every spatial operation on a container hits.
        """
        path = str(tmp_path / "root.nc")
        NetCDF.from_array(
            np.full((4, 4), 1.0, "float32"), geo_ref=GEO_REF, variable_name="t"
        ).to_file(path)
        container = NetCDF.read_file(path)

        with pytest.raises(ValueError, match="not supported on the NetCDF container"):
            container.combine(container, np.subtract)
