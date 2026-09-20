"""`equals` and `identical` — value equality, which `same_grid` does not provide.

Each verdict is the one xarray gives for the same difference, listed in
`TestTheVerdictsMatchXarray`: a copy is both, one changed cell is neither, and an
attribute-only difference is equal but not identical — the case that separates the two
methods and the plan's "done when".
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, NetCDF
from pyramids.netcdf import GeoReference as NCGeoReference

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
NDV = -9999.0
VALUES = np.array([[1.0, NDV], [3.0, 4.0]])


def _raster(
    array: np.ndarray | None = None, geo_ref: GeoReference = GEO_REF
) -> Dataset:
    """A 2x2 raster with one gap.

    Args:
        array: The values; `VALUES` when omitted.
        geo_ref: The georeference.

    Returns:
        Dataset: The raster.
    """
    return Dataset.from_array(
        VALUES if array is None else array, geo_ref=geo_ref, no_data_value=NDV
    )


class TestTheVerdictsMatchXarray:
    """The five differences xarray was measured on, and the verdict it gives for each."""

    def test_a_copy_is_equal_and_identical(self):
        """Nothing differs, so both hold."""
        raster = _raster()
        assert raster.equals(raster.copy())
        assert raster.identical(raster.copy())

    def test_one_changed_cell_is_neither(self):
        """A single different value is enough to separate two rasters."""
        changed = VALUES.copy()
        changed[1, 1] = 99.0
        assert not _raster().equals(_raster(changed))
        assert not _raster().identical(_raster(changed))

    def test_an_attribute_only_difference_separates_the_two_methods(self):
        """The values agree and the descriptions do not — equal, not identical.

        Test scenario:
            This is the plan's "done when": the pair is worth having only if one of them
            reads the attributes and the other does not.
        """
        raster = _raster()
        relabelled = _raster()
        relabelled.band_names = ["reflectance"]
        assert raster.equals(relabelled)
        assert not raster.identical(relabelled)

    def test_a_metadata_only_difference_also_separates_the_two_methods(self):
        """The dataset tags are attributes too, not only the band names.

        Test scenario:
            The band-name case above would pass an `identical` that read the names and
            ignored the tags. The same rasters under different dataset-level metadata are
            equal and not identical, which is xarray's verdict for differing `attrs`.
        """
        raster = _raster()
        tagged = _raster()
        tagged.meta_data = {"units": "mm"}
        assert raster.meta_data == {}, "precondition: the tags start out empty"
        assert raster.equals(tagged)
        assert not raster.identical(tagged)

    def test_the_same_tags_on_both_sides_are_identical(self):
        """Metadata only separates the two verdicts when it actually differs."""
        raster = _raster()
        raster.meta_data = {"units": "mm"}
        same = _raster()
        same.meta_data = {"units": "mm"}
        assert raster.identical(same)

    def test_a_different_grid_is_neither(self):
        """Same numbers somewhere else on the globe is a different raster."""
        elsewhere = GeoReference(top_left_corner=(50.0, 2.0), cell_size=1.0, epsg=4326)
        assert not _raster().equals(_raster(geo_ref=elsewhere))
        assert not _raster().identical(_raster(geo_ref=elsewhere))

    def test_a_different_shape_is_neither(self):
        """Shapes that do not match cannot hold the same cells."""
        taller = Dataset.from_array(np.ones((3, 2)), geo_ref=GEO_REF, no_data_value=NDV)
        assert not _raster().equals(taller)
        assert not _raster().identical(taller)


class TestGapsCompareEqual:
    """A gap equals a gap, whatever value each raster marks it with."""

    def test_a_gap_equals_a_gap(self):
        """The same cell missing in both is agreement, not disagreement."""
        assert _raster().equals(_raster())

    def test_two_sentinels_for_the_same_missing_cell(self):
        """Marking the same gap with a different number is still the same raster.

        Test scenario:
            Comparing the stored numbers would call these two different, because one holds
            `-9999.0` where the other holds `-32768.0` — but both mean "missing here".
        """
        other = np.array([[1.0, -32768.0], [3.0, 4.0]])
        theirs = Dataset.from_array(other, geo_ref=GEO_REF, no_data_value=-32768.0)
        assert _raster().equals(theirs)

    def test_a_gap_in_one_and_a_value_in_the_other(self):
        """One raster knows a cell the other does not, so they differ."""
        complete = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not _raster().equals(_raster(complete))

    def test_a_nan_gap_equals_itself(self):
        """A NaN is not equal to itself as a number, and must not decide the verdict."""
        values = np.array([[1.0, np.nan], [3.0, 4.0]])
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=np.nan)
        assert raster.equals(raster.copy())


class TestCheapInvariantsFirst:
    """Two rasters that cannot agree are refused without reading their cells."""

    def test_a_band_count_difference_is_refused(self):
        """A one-band raster is not a two-band one."""
        stack = Dataset.from_array(
            np.stack([VALUES, VALUES]), geo_ref=GEO_REF, no_data_value=NDV
        )
        assert not _raster().equals(stack)

    def test_something_that_is_not_a_raster(self):
        """Comparing with anything else answers `False` rather than raising."""
        assert not _raster().equals(42)
        assert not _raster().equals(None)
        assert not _raster().identical("a raster")


class TestOnANetCDFVariable:
    """A variable carries band dimensions, and two cubes stamped differently differ."""

    @staticmethod
    def _variable(stamps: list[float]) -> NetCDF:
        """A two-step variable over `time`.

        Args:
            stamps: The `time` coordinate values.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=NCGeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=stamps),
        ).get_variable("t")

    def test_a_variable_equals_its_own_copy(self):
        """The same cube twice agrees on values and on stamps."""
        variable = self._variable([0.0, 6.0])
        assert variable.equals(variable.copy())

    def test_different_stamps_are_a_different_cube(self):
        """The same numbers at different times are not the same cube.

        Test scenario:
            Only the band dimensions' coordinates differ, so a comparison that read the
            values alone would call these equal.
        """
        assert not self._variable([0.0, 6.0]).equals(self._variable([0.0, 12.0]))

    def test_a_variable_and_a_plain_raster_of_the_same_values(self):
        """A cube with a `time` dimension is not a plain raster of the same cells."""
        variable = self._variable([0.0, 6.0])
        plain = Dataset.from_array(
            np.arange(8.0).reshape(2, 2, 2),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            no_data_value=NDV,
        )
        assert not variable.equals(plain)


class TestAStrayNanUnderANumericSentinel:
    """A NaN that is not the declared sentinel must still compare as itself."""

    @staticmethod
    def _stray() -> Dataset:
        """A raster declaring -9999.0 but holding a NaN in one cell.

        Returns:
            Dataset: The raster.
        """
        values = np.array([[1.0, np.nan], [3.0, 4.0]])
        return Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=NDV)

    def test_it_equals_its_own_copy(self):
        """Reflexivity is the one property a comparison cannot be allowed to fail.

        Test scenario:
            The in-domain values were compared with `np.array_equal`, where a NaN is not
            equal to itself, so a raster holding a NaN that is *not* its declared sentinel
            did not equal its own copy — the exact raster `where(cond, np.nan)` used to
            produce before M1 was fixed.
        """
        raster = self._stray()
        assert raster.equals(raster.copy())

    def test_it_is_identical_to_its_own_copy(self):
        """`identical` reads the same comparison, so it inherits the same property."""
        raster = self._stray()
        assert raster.identical(raster.copy())

    def test_a_different_stray_nan_still_differs(self):
        """Reflexivity must not be bought by calling every NaN-bearing raster equal."""
        other = np.array([[1.0, 2.0], [3.0, np.nan]])
        assert not self._stray().equals(
            Dataset.from_array(other, geo_ref=GEO_REF, no_data_value=NDV)
        )
