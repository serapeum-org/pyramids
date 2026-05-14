"""Tests for :meth:`pyramids.netcdf.NetCDF.crop` bbox kwargs (PY-8).

The ``bbox=`` / ``epsg=`` keyword-only arguments mirror PY-5's surface
on :meth:`pyramids.dataset.Dataset.crop`. They route through the shared
:meth:`pyramids.feature.FeatureCollection.from_bbox` primitive and fall
through to the existing polygon / variable-subset paths.
"""

from __future__ import annotations

import pytest

from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

NC_FIXTURE = "tests/data/netcdf/noah-precipitation-1979.nc"
INSIDE_BBOX = (10.0, -50.0, 50.0, -20.0)


@pytest.fixture(scope="module")
def root_nc() -> NetCDF:
    """Open the test NetCDF as a root MDIM container.

    Returns:
        NetCDF: Container with four data variables (``Band1`` … ``Band4``).
    """
    return NetCDF.read_file(NC_FIXTURE)


class TestNetCDFCropBbox:
    """Tests for the ``bbox=`` / ``epsg=`` kwargs on a root container."""

    def test_bbox_in_native_crs_returns_container(self, root_nc: NetCDF):
        """Test bbox crop preserves the four-variable container shape.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            A bbox inside the raster's own CRS must yield a container
            with the same variable list as the source.
        """
        cropped = root_nc.crop(bbox=INSIDE_BBOX)
        assert sorted(cropped.variables) == sorted(root_nc.variables), (
            f"Variables changed: {sorted(cropped.variables)!r}"
        )

    def test_bbox_default_epsg_matches_dataset(self, root_nc: NetCDF):
        """Test explicit ``epsg=`` of the dataset's CRS matches the default.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Omitting ``epsg=`` and passing ``epsg=nc.epsg`` must produce
            byte-identical crop results (FC is built with the same CRS).
        """
        without_epsg = root_nc.crop(bbox=INSIDE_BBOX)
        with_epsg = root_nc.crop(bbox=INSIDE_BBOX, epsg=root_nc.epsg)
        a = without_epsg.get_variable("Band1").read_array()
        b = with_epsg.get_variable("Band1").read_array()
        assert a.shape == b.shape, (
            f"Shape mismatch: {a.shape} vs {b.shape}"
        )

    def test_bbox_equivalent_to_explicit_fc(self, root_nc: NetCDF):
        """Test ``crop(bbox=…)`` matches ``crop(mask=FC.from_bbox(…))``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The ``bbox=`` sugar must be exactly equivalent to building
            the one-row FC by hand — same array values, same shape.
        """
        via_bbox = root_nc.crop(bbox=INSIDE_BBOX).get_variable("Band1").read_array()
        fc = FeatureCollection.from_bbox(INSIDE_BBOX, epsg=root_nc.epsg)
        via_fc = root_nc.crop(mask=fc).get_variable("Band1").read_array()
        assert via_bbox.shape == via_fc.shape, (
            f"Shape mismatch: {via_bbox.shape} vs {via_fc.shape}"
        )

    def test_mask_path_still_works(self, root_nc: NetCDF):
        """Test pre-PY-8 ``mask=`` callers see no regression.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Passing a FC via the positional / ``mask=`` slot must keep
            its original behaviour.
        """
        fc = FeatureCollection.from_bbox(INSIDE_BBOX, epsg=root_nc.epsg)
        cropped = root_nc.crop(mask=fc)
        assert sorted(cropped.variables) == sorted(root_nc.variables)


class TestNetCDFCropVariableSubset:
    """Bbox crop on a single variable (delegates to ``super().crop``)."""

    def test_variable_subset_accepts_bbox(self, root_nc: NetCDF):
        """Test ``nc.get_variable(...).crop(bbox=...)`` works.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The variable-subset branch must accept the same ``bbox=`` /
            ``epsg=`` kwargs and route through ``super().crop``.
        """
        var = root_nc.get_variable("Band1")
        cropped = var.crop(bbox=INSIDE_BBOX)
        arr = cropped.read_array()
        assert arr.ndim in (2, 3), f"Unexpected ndim: {arr.ndim}"


class TestNetCDFCropMutex:
    """Mutual-exclusion + missing-argument errors."""

    def test_mask_and_bbox_together_raises(self, root_nc: NetCDF):
        """Test passing both ``mask=`` and ``bbox=`` raises ``ValueError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Mutex guard must fire before the FC is built.
        """
        fc = FeatureCollection.from_bbox(INSIDE_BBOX, epsg=root_nc.epsg)
        with pytest.raises(ValueError, match="not both"):
            root_nc.crop(mask=fc, bbox=INSIDE_BBOX)

    def test_neither_raises_type_error(self, root_nc: NetCDF):
        """Test calling ``crop()`` with no mask / bbox raises ``TypeError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Missing-argument guard must give a clear, actionable error.
        """
        with pytest.raises(TypeError, match=r"mask.*bbox|bbox.*mask"):
            root_nc.crop()

    def test_invalid_bbox_raises_value_error(self, root_nc: NetCDF):
        """Test ``W>=E`` / ``S>=N`` bbox raises ``ValueError`` via FC.from_bbox.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Validation lives in ``FeatureCollection.from_bbox``; the
            NetCDF override must surface that error unchanged.
        """
        with pytest.raises(ValueError):
            root_nc.crop(bbox=(50.0, -50.0, 10.0, -20.0))  # west >= east


class TestNetCDFReadArrayBbox:
    """Tests for the ``bbox=`` / ``epsg=`` kwargs on ``read_array``."""

    def test_root_container_bbox_routes_to_variable(self, root_nc: NetCDF):
        """Test root-container call with ``variable=`` + ``bbox=`` reads a window.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The container call dispatches to ``get_variable(...)``, and
            both ``bbox=`` and ``variable=`` must flow through.
        """
        full = root_nc.read_array(variable="Band1")
        windowed = root_nc.read_array(
            variable="Band1", bbox=(10.0, -50.0, 50.0, -20.0),
        )
        assert windowed.shape != full.shape, (
            f"Windowed read should differ from full: full={full.shape} "
            f"windowed={windowed.shape}"
        )
        assert windowed.size < full.size, (
            f"Windowed read should be smaller: full={full.size} "
            f"windowed={windowed.size}"
        )

    def test_variable_subset_bbox(self, root_nc: NetCDF):
        """Test ``read_array(bbox=…)`` on a pinned variable subset.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            On a variable subset the call forwards bbox/epsg to
            ``super().read_array`` (Dataset eager path).
        """
        var = root_nc.get_variable("Band1")
        arr = var.read_array(bbox=(10.0, -50.0, 50.0, -20.0))
        assert arr.ndim in (2, 3), f"Unexpected ndim: {arr.ndim}"

    def test_window_and_bbox_together_raises(self, root_nc: NetCDF):
        """Test ``window=`` + ``bbox=`` together raises ``ValueError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Mutex must fire before any reading happens.
        """
        with pytest.raises(ValueError, match="not both"):
            root_nc.read_array(
                variable="Band1",
                window=[0, 0, 10, 10],
                bbox=(10.0, -50.0, 50.0, -20.0),
            )

    def test_chunks_and_bbox_together_raises_type_error(self, root_nc: NetCDF):
        """Test ``chunks=`` + ``bbox=`` together raises ``TypeError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The lazy path doesn't yet honour bbox windowing; the error
            must be clear and actionable.
        """
        with pytest.raises(TypeError, match=r"eager path|drop `chunks`"):
            root_nc.read_array(
                variable="Band1",
                bbox=(10.0, -50.0, 50.0, -20.0),
                chunks="auto",
            )
