"""Tests for the NetCDF engine collaborators (issue #615, STR-1).

The engine bodies (``to_xarray`` / ``from_xarray`` → ``Interop``;
``set_variable`` / ``add_variable`` / ``remove_variable`` /
``rename_variable`` / ``create_from_array`` → ``Variables``;
``crop`` / ``sel`` / ``subset`` / ``reduce`` → ``Selection``) are exercised
end-to-end through the public ``NetCDF`` façades by the topic-specific
suites (``test_xarray_interop.py``, ``test_set_variable.py``, ``test_crop*.py``,
``test_sel*.py``, ``test_subset.py``, ``test_reduce.py``).

This module adds what those suites don't: the **wiring invariant** introduced
by the extraction (each façade delegates to the right engine, and the engines'
weakref back-reference survives in-place updates), the **curvilinear crop
path** (previously uncovered), and the cheap **error / edge branches** of the
engine methods.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF
from pyramids.netcdf.engines.interop import Interop
from pyramids.netcdf.engines.selection import Selection
from pyramids.netcdf.engines.variables import Variables

pytestmark = pytest.mark.core

CURVILINEAR_PATH = "tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc"
THREE_D_PATH = "tests/data/netcdf/pyramids-netcdf-3d.nc"
NOAH_PATH = "tests/data/netcdf/noah-precipitation-1979.nc"


@pytest.fixture
def mdim_container():
    """An MDIM root container opened from the 3-D pyramids fixture."""
    return NetCDF.read_file(THREE_D_PATH, open_as_multi_dimensional=True)


@pytest.fixture
def classic_container():
    """A classic-mode (non-MDIM) container — has no GDAL root group."""
    return NetCDF.read_file(NOAH_PATH, open_as_multi_dimensional=False)


@pytest.fixture
def curvilinear_container():
    """A container with 2-D ``xc`` / ``yc`` coordinates (genuine curvilinear grid)."""
    return NetCDF.read_file(CURVILINEAR_PATH, open_as_multi_dimensional=True)


class TestEngineWiring:
    """The façade ↔ engine wiring introduced by the STR-1 extraction."""

    def test_engines_are_wired_with_distinct_attrs(self, mdim_container):
        """Each NetCDF carries the three netcdf engines under their own attrs.

        Test scenario:
            A freshly opened container exposes ``interop`` / ``varops`` /
            ``selection`` bound to the matching engine classes, and ``varops``
            does not shadow the read-side ``variables`` property.
        """
        assert isinstance(mdim_container.interop, Interop), "interop engine missing"
        assert isinstance(mdim_container.varops, Variables), "varops engine missing"
        assert isinstance(mdim_container.selection, Selection), "selection engine missing"
        assert not isinstance(
            mdim_container.variables, Variables
        ), "`variables` must stay the read-side property, not the engine"

    def test_engine_back_reference_points_at_owner(self, mdim_container):
        """Each engine's weakref back-reference resolves to its owning container.

        Test scenario:
            ``engine._ds`` is a transparent ``weakref.proxy`` of the container,
            so identity-comparable attributes (the file name) match.
        """
        for engine in (
            mdim_container.interop,
            mdim_container.varops,
            mdim_container.selection,
        ):
            assert engine._ds.file_name == mdim_container.file_name, (
                "engine back-reference does not resolve to its owner"
            )

    def test_engines_rebind_after_inplace_update(self):
        """The netcdf engines are re-bound after an in-place state swap.

        Test scenario:
            Setting the ``epsg`` on an in-memory container runs
            ``_update_inplace`` (rebuilds internal state via a temporary
            instance); afterwards every engine's back-reference must resolve to
            the surviving container — and stay the right engine type — so a
            subsequent façade call still reaches the live instance. Identity is
            checked through the proxy by comparing engine types and a delegated
            attribute rather than ``is`` (``_ds`` is a ``weakref.proxy``).
        """
        nc = NetCDF.create_from_array(
            np.arange(24.0).reshape(2, 3, 4),
            geo=(0.0, 1.0, 0, 2.0, 0, -1.0),
            epsg=4326,
            variable_name="data",
        )
        nc.epsg = 3857
        assert isinstance(nc.selection, Selection), "selection engine lost on update"
        assert isinstance(nc.varops, Variables), "varops engine lost on update"
        assert isinstance(nc.interop, Interop), "interop engine lost on update"
        # A façade call must still reach the live instance after the swap.
        assert nc.selection._ds.epsg == nc.epsg, "engine back-ref points at stale state"

    def test_facade_delegates_to_engine(self, mdim_container, monkeypatch):
        """``NetCDF.to_xarray`` forwards to ``self.interop.to_xarray``.

        Test scenario:
            Patching the engine method makes the façade return the sentinel,
            proving the façade is a thin delegator rather than a reimplementation.
        """
        monkeypatch.setattr(
            mdim_container.interop, "to_xarray", lambda *a, **k: "SENTINEL"
        )
        assert mdim_container.to_xarray() == "SENTINEL", "façade did not delegate"


class TestInteropEngine:
    """Edge / error branches of :class:`Interop` and the module functions."""

    def test_to_xarray_requires_multidimensional(self, classic_container):
        """``to_xarray`` on a classic container raises ValueError.

        Test scenario:
            A non-MDIM container has no GDAL root group, so the conversion
            cannot proceed and a guiding ValueError is raised.
        """
        with pytest.raises(ValueError, match="multidimensional container"):
            classic_container.to_xarray()

    def test_to_xarray_missing_xarray_raises(self, mdim_container, monkeypatch):
        """``to_xarray`` raises OptionalPackageDoesNotExist when xarray is absent.

        Test scenario:
            Masking ``xarray`` in ``sys.modules`` makes ``import xarray`` fail;
            the engine surfaces the optional-dependency error.
        """
        from pyramids.base._errors import OptionalPackageDoesNotExist

        monkeypatch.setitem(sys.modules, "xarray", None)
        with pytest.raises(OptionalPackageDoesNotExist, match="xarray is required"):
            mdim_container.to_xarray()

    def test_from_xarray_rejects_non_dataset(self):
        """``from_xarray`` raises TypeError when handed a non-Dataset.

        Test scenario:
            Passing a plain object that is not an ``xarray.Dataset`` is a
            programming error and raises TypeError naming the bad type.
        """
        pytest.importorskip("xarray")
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray(object())

    def test_from_xarray_missing_xarray_raises(self, monkeypatch):
        """``from_xarray`` raises OptionalPackageDoesNotExist when xarray is absent.

        Test scenario:
            With ``xarray`` masked, the classmethod façade's engine call fails on
            the optional-dependency import before any conversion happens.
        """
        from pyramids.base._errors import OptionalPackageDoesNotExist

        monkeypatch.setitem(sys.modules, "xarray", None)
        with pytest.raises(OptionalPackageDoesNotExist, match="xarray is required"):
            NetCDF.from_xarray(object())

    def test_from_xarray_skips_non_dimension_coord(self):
        """A coordinate that is not also a dimension is skipped on write.

        Test scenario:
            ``_build_multidim_from_xarray`` only writes coords whose name matches
            a dimension; a scalar/auxiliary coord is silently skipped, so the
            round-trip drops it rather than crashing.
        """
        xr = pytest.importorskip("xarray")
        ds = xr.Dataset(
            data_vars={"t": (("y", "x"), np.arange(6.0).reshape(2, 3))},
            coords={
                "y": ("y", [0.0, 1.0]),
                "x": ("x", [0.0, 1.0, 2.0]),
                "scalar_meta": 42.0,  # not a dimension -> skipped
            },
        )
        nc = NetCDF.from_xarray(ds)
        assert "t" in nc.variable_names, "data variable lost on round-trip"
        assert "scalar_meta" not in nc.variable_names, "non-dim coord should be skipped"

    def test_to_xarray_roundtrip_through_engine(self, mdim_container):
        """``nc.interop.to_xarray()`` and ``nc.to_xarray()`` agree.

        Test scenario:
            Calling the engine directly and through the façade produce datasets
            with the same data variables — the façade adds no behaviour.
        """
        pytest.importorskip("xarray")
        via_engine = mdim_container.interop.to_xarray()
        via_facade = mdim_container.to_xarray()
        assert set(via_engine.data_vars) == set(via_facade.data_vars), (
            "engine and façade disagree on variables"
        )


class TestVariablesEngine:
    """Edge / error branches of :class:`Variables`."""

    def test_set_variable_requires_multidimensional(self, classic_container):
        """``set_variable`` on a classic container raises ValueError.

        Test scenario:
            Writing a variable back needs a GDAL root group; a non-MDIM
            container has none, so a guiding ValueError is raised before any
            mutation is attempted.
        """
        from pyramids.dataset import Dataset

        donor = Dataset.create_from_array(
            np.ones((4, 5), dtype=np.float32), geo=(0, 1, 0, 4, 0, -1), epsg=4326
        )
        with pytest.raises(ValueError, match="multidimensional container"):
            classic_container.set_variable("new", donor)

    def test_rename_variable_unknown_old_name_raises(self, mdim_container):
        """``rename_variable`` raises ValueError for an unknown source name.

        Test scenario:
            Renaming a variable that does not exist names the missing variable
            and lists the available ones.
        """
        with pytest.raises(ValueError, match="not found"):
            mdim_container.rename_variable("does_not_exist", "whatever")

    def test_rename_variable_existing_target_raises(self, mdim_container):
        """``rename_variable`` raises ValueError when the target name is taken.

        Test scenario:
            Renaming onto an existing variable name would clobber it, so the
            engine refuses with a clear error.
        """
        names = mdim_container.variable_names
        if len(names) < 2:
            pytest.skip("fixture needs at least two variables")
        with pytest.raises(ValueError, match="already exists"):
            mdim_container.rename_variable(names[0], names[1])

    def test_create_from_array_requires_geo(self):
        """``create_from_array`` raises ValueError without ``geo`` or corner+size.

        Test scenario:
            Neither a geotransform nor a ``(top_left_corner, cell_size)`` pair is
            supplied, so the geobox is undefined and a ValueError is raised.
        """
        with pytest.raises(ValueError, match="geo.*top_left_corner|top_left_corner"):
            NetCDF.create_from_array(np.zeros((2, 3)))

    def test_create_from_array_corner_and_cell_size(self):
        """``create_from_array`` builds ``geo`` from ``top_left_corner`` + ``cell_size``.

        Test scenario:
            With no explicit ``geo`` but both corner and cell size given, the
            geotransform is synthesised and a Container is returned.
        """
        nc = NetCDF.create_from_array(
            np.arange(12.0).reshape(3, 4),
            top_left_corner=(0.0, 10.0),
            cell_size=1.0,
        )
        assert "data" in nc.variable_names, "default variable not created"


class TestSelectionEngine:
    """Crop (incl. the curvilinear path), and selection error branches."""

    def _middle_bbox(self, nc):
        """Return a ``(x0, y0, x1, y1)`` covering the middle of the curv grid."""
        xc = nc.get_variable("xc").read_array()
        yc = nc.get_variable("yc").read_array()
        x0, x1 = float(np.min(xc)), float(np.max(xc))
        y0, y1 = float(np.min(yc)), float(np.max(yc))
        return (
            x0 + 0.25 * (x1 - x0),
            y0 + 0.25 * (y1 - y0),
            x0 + 0.75 * (x1 - x0),
            y0 + 0.75 * (y1 - y0),
        )

    def test_curvilinear_crop_returns_windowed_subset(self, curvilinear_container):
        """Cropping a curvilinear variable masks + windows on its 2-D coords.

        Test scenario:
            A bbox over the middle of the grid yields a smaller variable that
            carries its windowed 2-D ``_curvilinear_coords`` (stays curvilinear),
            with the band axis preserved.
        """
        var = curvilinear_container.get_variable("Tair")
        full_shape = var.read_array().shape
        bbox = self._middle_bbox(curvilinear_container)
        fc = FeatureCollection.from_bbox(bbox, epsg=var.epsg)
        cropped = var.crop(mask=fc)
        assert cropped._curvilinear_coords is not None, "lost curvilinear coords"
        out = cropped.read_array()
        assert out.shape[0] == full_shape[0], "band axis must be preserved"
        assert out.shape[-1] < full_shape[-1], "crop did not shrink the column window"

    def test_curvilinear_crop_no_overlap_raises(self, curvilinear_container):
        """A bbox disjoint from the grid raises ValueError.

        Test scenario:
            A polygon placed far outside the coordinate extent contains no cell
            centre, so the curvilinear path raises a guiding ValueError.
        """
        var = curvilinear_container.get_variable("Tair")
        xc = curvilinear_container.get_variable("xc").read_array()
        yc = curvilinear_container.get_variable("yc").read_array()
        far_x, far_y = float(np.max(xc)) + 100.0, float(np.max(yc)) + 100.0
        fc = FeatureCollection.from_bbox(
            (far_x, far_y, far_x + 10.0, far_y + 10.0), epsg=var.epsg
        )
        with pytest.raises(ValueError, match="does not overlap the curvilinear grid"):
            var.crop(mask=fc)

    def test_curvilinear_crop_chunks_matches_eager(self, curvilinear_container):
        """The lazy/chunked curvilinear crop equals the eager one.

        Test scenario:
            ``crop(..., chunks="auto")`` reads the bounding window through the
            dask-backed lazy path; the materialised result is value-equal to the
            default eager read.
        """
        pytest.importorskip("dask")
        var = curvilinear_container.get_variable("Tair")
        bbox = self._middle_bbox(curvilinear_container)
        fc = FeatureCollection.from_bbox(bbox, epsg=var.epsg)
        eager = curvilinear_container.get_variable("Tair").crop(mask=fc).read_array()
        lazy = var.crop(mask=fc, chunks="auto").read_array()
        np.testing.assert_array_equal(
            lazy, eager, err_msg="lazy curvilinear crop differs from eager"
        )

    def test_crop_chunks_on_root_container_raises(self, curvilinear_container):
        """``crop(chunks=…)`` on a root container is rejected.

        Test scenario:
            ``chunks`` is a per-variable, curvilinear-only knob; a root container
            fans out over its variables and cannot honour it, so it raises.
        """
        bbox = self._middle_bbox(curvilinear_container)
        with pytest.raises(ValueError, match="not supported on a root container"):
            curvilinear_container.crop(bbox=bbox, chunks="auto")

    def test_crop_chunks_on_rectilinear_raises(self, mdim_container):
        """``crop(chunks=…)`` on a rectilinear variable is rejected.

        Test scenario:
            The affine (rectilinear) crop path is eager; passing ``chunks`` there
            is a usage error and raises a guiding ValueError.
        """
        var = mdim_container.get_variable(mdim_container.variable_names[0])
        bbox = (0.5, 0.5, 2.0, 2.0)
        fc = FeatureCollection.from_bbox(bbox, epsg=var.epsg or 4326)
        with pytest.raises(ValueError, match="only supported for curvilinear"):
            var.crop(mask=fc, chunks="auto")

    def test_crop_requires_mask_or_bbox(self, mdim_container):
        """``crop`` with neither ``mask`` nor ``bbox`` raises TypeError.

        Test scenario:
            Exactly one selector is required; supplying none is a usage error.
        """
        with pytest.raises(TypeError, match="requires a `mask`"):
            mdim_container.crop()

    def test_crop_mask_and_bbox_mutually_exclusive(self, mdim_container):
        """``crop`` rejects both ``mask`` and ``bbox`` together.

        Test scenario:
            Supplying both selectors is ambiguous and raises ValueError.
        """
        bbox = (0.5, 0.5, 2.0, 2.0)
        fc = FeatureCollection.from_bbox(bbox, epsg=mdim_container.epsg or 4326)
        with pytest.raises(ValueError, match="either .* or .* not both"):
            mdim_container.crop(mask=fc, bbox=bbox)

    def test_subset_unknown_variable_raises(self, mdim_container):
        """``subset`` of an absent variable raises ValueError listing the names.

        Test scenario:
            The requested variable is not in the store, so subset names the
            available variables instead of failing opaquely.
        """
        with pytest.raises(ValueError, match="not a variable in this store"):
            mdim_container.subset("no_such_variable")

    def test_subset_rejects_variable_below_two_dims(self, mdim_container):
        """``subset`` of a 1-D variable raises ValueError naming the dim count.

        Test scenario:
            The ``x`` coordinate array has a single dimension; subset needs at
            least ``(y, x)``, so it rejects it with a message stating how many
            dimensions the variable actually has. (Coordinate arrays are openable
            via the multidimensional API even though they are not data variables.)
        """
        with pytest.raises(ValueError, match=r"dimension\(s\); subset\(\) needs"):
            mdim_container.subset("x")

    def test_subset_requires_multidimensional(self, classic_container):
        """``subset`` on a classic container raises ValueError.

        Test scenario:
            Windowed reads need a multidimensional store; a classic container
            has no root group and is rejected.
        """
        with pytest.raises(ValueError, match="requires a multidimensional store"):
            classic_container.subset("anything")

    def test_reduce_unknown_how_raises(self, mdim_container):
        """``reduce`` with an unknown reducer name raises ValueError.

        Test scenario:
            ``how`` must be one of the registered reducers; an unknown value is
            rejected with the list of valid options.
        """
        with pytest.raises(ValueError, match="how must be one of"):
            mdim_container.reduce("time", how="median")
