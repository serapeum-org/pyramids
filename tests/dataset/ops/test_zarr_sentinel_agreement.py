"""One zarr array holds every band, so `fill_value` may only say what they share.

Two ways of deciding that were both wrong, and both shipped:

- Filtering `None` out before comparing made `(5.0, None)` look unanimous, so
  the store declared `5.0` and GDAL reported it as band 2's no-data. Masking a
  read of that store blanks every genuine `5.0` in a band with no sentinel at
  all -- the exact failure declaring a shared sentinel exists to avoid.
- Comparing through a `set` made NaN's answer depend on object identity. A
  `set` short-circuits on identity, so one band's sentinel deduped to a single
  element while two separately-built NaNs did not: a NaN sentinel survived on a
  one-band raster and vanished on a two-band one.

Separately, the attribute is carried as a `float` because it is written to
JSON, and `float(2**63 - 1)` rounds *up* past what `int64` can hold -- so
promoting it into zarr's `fill_value` raised `OverflowError` and a store that
used to be writable could not be written at all.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.ops._zarr import (
    _agreed_sentinel,
    _is_nan,
    _representable,
    _same_sentinel,
)

try:
    import zarr
except ImportError:  # pragma: no cover - exercised only without the extra
    zarr = None

# Guarded rather than a module-level `importorskip`, so the file still collects
# without the `[lazy]` extra; the tests that need a real store are marked.
needs_zarr = pytest.mark.skipif(zarr is None, reason="requires the [lazy] extra")

# The helper tests are pure functions; only the end-to-end class needs a store,
# and those carry `lazy` as well.
pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)


def _on_disk(array: np.ndarray, no_data, tmp_path: Path) -> Dataset:
    """Write an array to a GeoTIFF and reopen it.

    `to_zarr` reads through the file manager, so a purely in-memory source
    cannot be written.

    Args:
        array: The raster values.
        no_data: The sentinel, per band or shared.
        tmp_path: Where to put the intermediate GeoTIFF.

    Returns:
        Dataset: The reopened, file-backed raster.
    """
    geotiff = tmp_path / "source.tif"
    Dataset.from_array(array, geo_ref=GEO, no_data_value=no_data).to_file(str(geotiff))
    return Dataset.read_file(str(geotiff))


class TestTheAgreementTest:
    """`_agreed_sentinel` on its own, over every shape it decides."""

    def test_bands_that_agree_yield_the_value(self):
        """Test scenario: the ordinary case, and the only one that may declare."""
        assert _agreed_sentinel([-9999.0, -9999.0, -9999.0]) == -9999.0

    def test_a_band_with_no_sentinel_blocks_agreement(self):
        """The regression: `(5.0, None)` used to look unanimous.

        Test scenario:
            Band 2 has no sentinel. Declaring band 1's would tell every reader
            that 5.0 means missing in a band where it means 5.0.
        """
        assert _agreed_sentinel([5.0, None]) is None

    def test_none_in_any_position_blocks_agreement(self):
        """Filtering `None` first is what made position matter.

        Test scenario:
            Leading, trailing or interior -- a band without a sentinel is a
            band without a sentinel wherever it sits.
        """
        assert _agreed_sentinel([None, 5.0]) is None
        assert _agreed_sentinel([5.0, 5.0, None]) is None

    def test_two_separately_built_nans_agree(self):
        """The `set` answered this by object identity.

        Test scenario:
            `float("nan") != float("nan")`, and a `set` deduped them only when
            they happened to be the same object -- so the answer depended on
            how the tuple was built rather than on what it said.
        """
        result = _agreed_sentinel([float("nan"), float("nan")])

        assert result is not None, "two NaN sentinels should agree"
        assert math.isnan(result)

    def test_one_nan_band_agrees_with_itself(self):
        """The single-band case, which used to be the only one that worked.

        Test scenario:
            A one-band raster deduped to a single element whatever the value,
            so NaN worked here and nowhere else. It must still work.
        """
        result = _agreed_sentinel([float("nan")])

        assert result is not None
        assert math.isnan(result)

    def test_a_nan_does_not_agree_with_a_number(self):
        """Consistency must not become permissiveness.

        Test scenario:
            NaN and 5.0 are different sentinels, and a helper that made all
            NaNs agree could easily make NaN agree with anything.
        """
        assert _agreed_sentinel([float("nan"), 5.0]) is None

    @pytest.mark.parametrize(
        ("label", "values"),
        [("empty", []), ("all none", [None, None]), ("differing", [1.0, 2.0])],
    )
    def test_no_agreement_cases(self, label: str, values: list):
        """Args: label: The shape. values: The per-band sentinels.

        Test scenario:
            None of these has a value every band shares, so none may be
            declared.
        """
        assert _agreed_sentinel(values) is None, label


class TestTheNanHelpers:
    """The two predicates the agreement test is built from."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(float("nan"), True), (0.0, False), (-9999.0, False), (None, False)],
    )
    def test_is_nan_never_raises(self, value, expected: bool):
        """Args: value: Any sentinel. expected: Whether it is a float NaN.

        Test scenario:
            It is asked of `None` as well as of numbers, so `math.isnan`
            alone -- which raises on `None` -- cannot be the whole answer.
        """
        assert _is_nan(value) is expected

    def test_same_sentinel_agrees_on_equal_numbers(self):
        """Test scenario: the ordinary comparison must still work."""
        assert _same_sentinel(-9999.0, -9999.0) is True

    def test_same_sentinel_rejects_different_numbers(self):
        """Test scenario: two real sentinels that differ are a disagreement."""
        assert _same_sentinel(-9999.0, 0.0) is False


class TestRepresentableInTheBandDtype:
    """A sentinel `float` cannot hold exactly must not reach `fill_value`."""

    def test_the_int64_maximum_is_not_representable(self):
        """The regression: `to_zarr` raised and the store could not be written.

        Test scenario:
            `float(2**63 - 1)` rounds up to `2**63`, which `int64` cannot
            hold, so zarr's cast raised `OverflowError` on write.
        """
        assert _representable(float(2**63 - 1), "int64") is False

    @pytest.mark.parametrize(
        ("sentinel", "dtype"),
        [(-9999.0, "float32"), (-9999.0, "int32"), (255.0, "uint8"), (0.0, "int16")],
    )
    def test_ordinary_sentinels_are_representable(self, sentinel: float, dtype: str):
        """Args: sentinel: The value. dtype: The band's type.

        Test scenario:
            The guard must not refuse the sentinels every real raster uses;
            refusing them would drop `fill_value` for the whole corpus.
        """
        assert _representable(sentinel, dtype) is True

    def test_a_fractional_sentinel_is_refused_on_an_integer_band(self):
        """Test scenario: 0.5 cannot be stored in an integer band unchanged."""
        assert _representable(0.5, "int32") is False

    def test_an_out_of_range_sentinel_is_refused(self):
        """Test scenario: 300 does not fit `uint8`, so it cannot be declared."""
        assert _representable(300.0, "uint8") is False


class TestEndToEndThroughAWrittenStore:
    """The property that matters: what a reader sees on disk."""

    @pytest.mark.lazy
    @needs_zarr
    def test_an_int64_limit_sentinel_still_writes(self, tmp_path):
        """Args: tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            This raised `OverflowError` -- not a metadata nicety but a store
            that could not be written at all.
        """
        source = _on_disk(
            np.ones((4, 5), dtype="int64"), int(np.iinfo(np.int64).max), tmp_path
        )

        source.to_zarr(str(tmp_path / "limit.zarr"))

        assert (tmp_path / "limit.zarr").exists(), "the store was not written"

    @pytest.mark.lazy
    @needs_zarr
    def test_an_ordinary_sentinel_still_reaches_fill_value(self, tmp_path):
        """Guarding the limits must not drop the common case.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `fill_value` is what GDAL reads, and it is the whole reason the
            sentinel is written there rather than only in pyramids' own
            attribute.
        """
        source = _on_disk(np.ones((4, 5), dtype="float32"), -9999.0, tmp_path)
        store = tmp_path / "ordinary.zarr"

        source.to_zarr(str(store))

        metadata = json.loads((store / "data" / "zarr.json").read_text())
        assert metadata["fill_value"] == -9999.0
        assert tuple(Dataset.read_file(str(store)).no_data_value) == (-9999.0,)
