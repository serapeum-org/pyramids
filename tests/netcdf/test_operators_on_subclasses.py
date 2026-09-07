"""Every operator `Dataset` defines, exercised on the classes that inherit it.

`NetCDF`, `Variable` and `Container` are `Dataset` subclasses, so `__add__`, `__sub__`,
`__mul__`, `__truediv__`, `__radd__`, `__rmul__`, the four comparisons and `__bool__` all
reach them for free — and "for free" is exactly the kind of inheritance that goes untested
until it breaks. It did: carrying `meta_data` through `combine` with `dict(...)` worked for
a plain `Dataset`, whose `meta_data` is a dict, and turned every operator on a NetCDF
variable into `TypeError: 'NetCDFMetadata' object is not iterable`.

`UgridDataset` is not a `Dataset` subclass, so it inherits none of this. That is pinned
here too, so the absence is a recorded fact rather than an assumption.
"""

from __future__ import annotations

import math
import operator
from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf.netcdf import Container, NetCDF, Variable
from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)
UGRID_SAMPLE = (
    Path(__file__).resolve().parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"
)


def _variable(tmp_path, name: str, value: float) -> Variable:
    """Write a one-variable NetCDF and return its variable view.

    Args:
        tmp_path: pytest temp directory.
        name: File name to write under `tmp_path`.
        value: Constant filling the variable's single band.

    Returns:
        Variable: The variable view, ready to combine.
    """
    path = str(tmp_path / name)
    NetCDF.from_array(
        np.full((4, 4), value, "float32"), geo_ref=GEO_REF, variable_name="t"
    ).to_file(path)
    return NetCDF.read_file(path).get_variable("t")


class TestOperatorsOnNetCDFVariables:
    """The inherited operators produce the same answers on a variable view."""

    @pytest.mark.parametrize(
        ("apply_operator", "expected"),
        [
            (operator.sub, 6.0),
            (operator.add, 14.0),
            (operator.mul, 40.0),
            (operator.truediv, 2.5),
        ],
        ids=["sub", "add", "mul", "truediv"],
    )
    def test_arithmetic_matches_the_plain_dataset_answer(
        self, tmp_path, apply_operator, expected
    ):
        """A variable view and a plain raster of the same numbers agree.

        Args:
            tmp_path: pytest temp directory.
            apply_operator: The operator under test, applied through `operator` so the
                binary-op protocol runs rather than the dunder being called directly.
            expected: The value every result cell must hold.

        Test scenario:
            10 and 4 through each operator, computed once on `Variable` operands and once
            on `Dataset` operands, must give the same numbers.
        """
        variable = apply_operator(
            _variable(tmp_path, "a.nc", 10.0), _variable(tmp_path, "b.nc", 4.0)
        )
        plain = apply_operator(
            Dataset.from_array(np.full((4, 4), 10.0, "float32"), geo_ref=GEO_REF),
            Dataset.from_array(np.full((4, 4), 4.0, "float32"), geo_ref=GEO_REF),
        )

        assert np.asarray(variable.read_array()).mean() == pytest.approx(expected)
        np.testing.assert_allclose(
            np.asarray(variable.read_array()), np.asarray(plain.read_array())
        )

    @pytest.mark.parametrize(
        ("fold", "expected"), [(sum, 14.0), (math.prod, 40.0)], ids=["sum", "prod"]
    )
    def test_the_reflected_identities_fold_variable_views(
        self, tmp_path, fold, expected
    ):
        """`sum`/`math.prod` seed with a scalar, so they exercise `__radd__`/`__rmul__`.

        Args:
            tmp_path: pytest temp directory.
            fold: The builtin fold under test.
            expected: The value every result cell must hold.

        Test scenario:
            Those reflected methods answer with `self.copy()`, and a variable view's copy
            is not a plain raster's — this is the path that must work for a subclass.
        """
        result = fold(
            [_variable(tmp_path, "a.nc", 10.0), _variable(tmp_path, "b.nc", 4.0)]
        )

        assert np.asarray(result.read_array()).mean() == pytest.approx(expected)

    def test_folding_one_variable_returns_a_usable_copy(self, tmp_path):
        """The identity step must not alias, and must survive on a subclass.

        Test scenario:
            `sum([v])` copies through `__radd__`; the copy reads back the same values and
            is a distinct object.
        """
        variable = _variable(tmp_path, "a.nc", 10.0)

        total = sum([variable])

        assert total is not variable, "the identity step must copy"
        assert np.allclose(np.asarray(total.read_array()), 10.0)

    @pytest.mark.parametrize(
        ("compare", "expected"),
        [(operator.ge, 1), (operator.gt, 1), (operator.le, 0), (operator.lt, 0)],
        ids=["ge", "gt", "le", "lt"],
    )
    def test_comparisons_give_a_byte_mask(self, tmp_path, compare, expected):
        """A comparison on variable views yields the same Byte mask a raster does.

        Args:
            tmp_path: pytest temp directory.
            compare: The comparison under test.
            expected: The value every mask cell must hold.

        Test scenario:
            10 against 4 under each operator gives uint8 1s or 0s declaring 255.
        """
        mask = compare(
            _variable(tmp_path, "a.nc", 10.0), _variable(tmp_path, "b.nc", 4.0)
        )

        array = np.asarray(mask.read_array())
        assert array.dtype == np.uint8
        assert mask.no_data_value[0] == 255
        assert (array == expected).all()

    def test_numpy_defers_on_a_variable_too(self, tmp_path):
        """`__array_ufunc__ = None` is inherited, so numpy refuses here as well.

        Test scenario:
            An ndarray on the left raises rather than broadcasting into an object array
            of variable copies.
        """
        with pytest.raises(TypeError):
            np.array([0.0]) + _variable(tmp_path, "a.nc", 10.0)


class TestTruthinessAcrossTheHierarchy:
    """`__bool__` refuses for every class that inherits it, and only those."""

    def test_a_variable_and_a_container_both_refuse(self, tmp_path):
        """The breaking change reaches the whole NetCDF hierarchy, by inheritance.

        Test scenario:
            `bool()` raises on a `Variable` and on the root store alike, so `if nc:`
            cannot be silently true anywhere; `is not None` is the presence check.
        """
        path = str(tmp_path / "root.nc")
        NetCDF.from_array(
            np.full((4, 4), 1.0, "float32"), geo_ref=GEO_REF, variable_name="t"
        ).to_file(path)
        container = NetCDF.read_file(path)
        variable = container.get_variable("t")

        assert isinstance(container, Dataset), "the container is a Dataset"
        assert isinstance(variable, Dataset), "and so is a variable view"
        for obj in (container, variable):
            with pytest.raises(
                ValueError, match="truth value of a Dataset is ambiguous"
            ):
                bool(obj)
        assert container is not None

    def test_the_netcdf_classes_really_do_inherit_from_dataset(self):
        """The premise of every test above, asserted rather than assumed.

        Test scenario:
            `NetCDF`, `Variable` and `Container` are `Dataset` subclasses, which is why
            they get the operators for free — and why they have to be tested for free
            too.
        """
        for cls in (NetCDF, Variable, Container):
            assert issubclass(cls, Dataset), f"{cls.__name__} must be a Dataset"


class TestUgridDatasetIsNotARaster:
    """`UgridDataset` inherits none of this, and that is the recorded contract."""

    def test_it_is_not_a_dataset_subclass(self):
        """An unstructured mesh is not a grid, so it gets no raster operators.

        Test scenario:
            `UgridDataset` derives straight from `object`, so nothing in this PR reaches
            it. Pinned so that a future `UgridDataset(Dataset)` cannot silently inherit
            cell-by-cell operators that have no meaning on a mesh.
        """
        assert not issubclass(UgridDataset, Dataset)
        assert UgridDataset.__mro__[1:] == (object,)

    @pytest.mark.skipif(not UGRID_SAMPLE.exists(), reason="ugrid sample not available")
    @pytest.mark.parametrize(
        "apply_operator",
        [operator.sub, operator.add, operator.mul, operator.ge, operator.lt],
        ids=["sub", "add", "mul", "ge", "lt"],
    )
    def test_the_operators_are_unavailable_on_a_real_mesh(self, apply_operator):
        """A real mesh read from disk declines every operator, loudly.

        Args:
            apply_operator: The operator that must not be supported.

        Test scenario:
            Two meshes give Python's own TypeError rather than a half-working answer, so
            a caller reaching for raster arithmetic on a mesh finds out immediately.
        """
        mesh = UgridDataset.read_file(UGRID_SAMPLE)

        with pytest.raises(TypeError):
            apply_operator(mesh, mesh)

    @pytest.mark.skipif(not UGRID_SAMPLE.exists(), reason="ugrid sample not available")
    def test_a_mesh_keeps_its_truth_value(self):
        """`bool()` on a mesh is unaffected by the `Dataset` change.

        Test scenario:
            A mesh is an ordinary object, so `if mesh:` still works — the breaking change
            is scoped to rasters.
        """
        mesh = UgridDataset.read_file(UGRID_SAMPLE)

        assert bool(mesh) is True
        assert mesh is not None
