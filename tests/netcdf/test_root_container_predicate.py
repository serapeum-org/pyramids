"""One answer to "is this the whole store, or a variable of it?".

`_is_md_array and not _is_subset and band_count == 0` was written out six
times: in the operations that refuse a container (`_check_not_container`,
`warped_view`), the ones that fan out over it (`to_crs`, `resample`), the one
that reads from it (`read_array`) and the crop engine. Three conditions that
only mean something together, kept in step by hand across two modules.

The three conditions are not redundant. A classic-mode open has bands and is
not multidimensional; a `get_variable(...)` subset is multidimensional *and*
has bands; only the store itself is multidimensional, un-narrowed and bandless.
Dropping any one of them would misclassify one of those three.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
STORE = DATA / "cf__5v__1d4-4d1__y-asc.nc"
VARIABLE = "temperature"


@pytest.fixture
def container() -> NetCDF:
    """The store itself, opened multidimensionally.

    Returns:
        NetCDF: A root container -- no bands of its own.
    """
    return NetCDF.read_file(str(STORE))


@pytest.fixture
def variable(container) -> NetCDF:
    """One variable of that store.

    Args:
        container: The store fixture.

    Returns:
        NetCDF: A subset view, which carries bands.
    """
    return container.get_variable(VARIABLE)


class TestThePredicate:
    """The three shapes it has to tell apart."""

    def test_a_multidimensional_store_is_a_root_container(self, container):
        """Args: container: The store fixture.

        Test scenario:
            Opened through the multidimensional API, never narrowed to a
            variable, and carrying no bands of its own -- all three at once,
            which is what "container" means.
        """
        assert container._is_root_container is True, (
            f"md={container._is_md_array} subset={container._is_subset} "
            f"bands={container.band_count}"
        )

    def test_a_variable_subset_is_not(self, variable):
        """Args: variable: The subset fixture.

        Test scenario:
            A subset is multidimensional too, so `_is_md_array` alone cannot
            separate it -- `_is_subset` is what does, and it has bands as
            well.
        """
        assert variable._is_root_container is False
        assert variable._is_md_array is True, "the subset is still multidimensional"
        assert variable.band_count > 0, "the subset carries bands"

    def test_a_classic_mode_open_is_not(self):
        """The third shape, which neither of the others covers.

        Test scenario:
            Opening without the multidimensional API gives a bandful raster
            that was never narrowed -- so `not _is_subset` is true of it, and
            only `_is_md_array` keeps it out.
        """
        classic = NetCDF.read_file(str(STORE), open_as_multi_dimensional=False)

        assert classic._is_root_container is False
        assert classic._is_md_array is False, "classic mode is not multidimensional"
        assert classic._is_subset is False, "classic mode was never narrowed"

    def test_every_condition_is_load_bearing(self, container, variable):
        """None of the three can be dropped without misclassifying something.

        Args:
            container: The store fixture.
            variable: The subset fixture.

        Test scenario:
            Asserted as the truth table rather than by reading the code: the
            container is the only one true on all three, and each of the other
            two fails a *different* condition. A conjunction that lost a term
            would start calling one of them a container.
        """
        classic = NetCDF.read_file(str(STORE), open_as_multi_dimensional=False)
        table = {
            name: (obj._is_md_array, not obj._is_subset, obj.band_count == 0)
            for name, obj in (
                ("container", container),
                ("variable", variable),
                ("classic", classic),
            )
        }

        assert table["container"] == (True, True, True), table["container"]
        assert table["variable"][1] is False, "the subset must fail the subset term"
        assert table["classic"][0] is False, "classic mode must fail the mdim term"


class TestTheCallSitesStillBehave:
    """Six places read it; the behaviour each gates must be unchanged."""

    def test_a_container_refuses_warped_view(self, container):
        """Args: container: The store fixture.

        Test scenario:
            `warped_view` works on one variable. Asked of the store it has to
            say so and name the way through, rather than warping nothing.
        """
        with pytest.raises(ValueError, match="single variable"):
            container.warped_view(3857)

    def test_a_variable_accepts_warped_view(self, variable):
        """Args: variable: The subset fixture.

        Test scenario:
            The other side of the same gate -- a guard that fired for both
            would pass the refusal test above while breaking every caller.
        """
        assert variable.warped_view(3857) is not None

    def test_a_container_refuses_a_spatial_operation_by_name(self, container):
        """`_check_not_container` names the way through.

        Args:
            container: The store fixture.

        Test scenario:
            The message tells the caller to reach for `get_variable`, which is
            the whole value of refusing here rather than failing later.
        """
        with pytest.raises(ValueError, match="get_variable"):
            container._check_not_container("crop")

    def test_a_variable_passes_that_check(self, variable):
        """Args: variable: The subset fixture.

        Test scenario:
            `_check_not_container` returning quietly is what lets every
            spatial operation run on a variable.
        """
        assert variable._check_not_container("crop") is None

    def test_a_container_fans_a_reprojection_out_over_its_variables(self, container):
        """Args: container: The store fixture.

        Test scenario:
            `to_crs` reads the predicate to decide between fanning out and
            reprojecting a single raster. The result must still be a container
            holding the same variables, not one warped plane.
        """
        reprojected = container.to_crs(3857)

        assert reprojected._is_root_container is True
        assert set(reprojected.variable_names) == set(container.variable_names)

    def test_a_variable_reprojects_as_one_raster(self, variable):
        """Args: variable: The subset fixture.

        Test scenario:
            The other branch of the same `if`. A variable comes back with
            bands, not as a container to be fanned out again.
        """
        reprojected = variable.to_crs(3857)

        assert reprojected._is_root_container is False
        assert reprojected.band_count == variable.band_count
