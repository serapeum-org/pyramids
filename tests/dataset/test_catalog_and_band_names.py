"""Two normalisers that each existed twice.

`Catalog.resolve_key` accepts a catalog key or a GDAL short name. Two writers
carried their own two-branch version of that guard, each with its own copy of
the error message.

`Bands.apply_names` restores band names read back from a store, skipping a
length mismatch with a warning. The Zarr reader had it; the cube reader had the
length check without the warning, so a mismatched list there was dropped
silently.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pyramids.base._errors import DriverNotExistError
from pyramids.base._utils import get_catalog
from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core


class TestCatalogResolveKey:
    """One normaliser for the two spellings of a driver."""

    @pytest.mark.parametrize("spelling", ["geotiff", "GTiff"])
    def test_both_spellings_resolve_to_the_key(self, spelling: str):
        """A catalog key and its GDAL short name give the same answer."""
        assert get_catalog().resolve_key(spelling) == "geotiff"

    def test_an_unknown_driver_raises_and_lists_the_known_ones(self):
        """The error names what was passed and what is available."""
        with pytest.raises(DriverNotExistError, match="not in the driver catalog"):
            get_catalog().resolve_key("NotADriver")

    def test_every_catalog_key_resolves_to_itself(self):
        """Swept over the whole catalog, not just the GeoTiff example.

        Test scenario:
            The first branch is "the key is already a key". Any entry for which
            that fails would fall through to the short-name lookup and either
            resolve to a different driver or raise -- a per-driver bug a single
            example cannot find.
        """
        catalog = get_catalog()

        wrong = {
            key: catalog.resolve_key(key)
            for key in catalog.drivers
            if catalog.resolve_key(key) != key
        }

        assert wrong == {}, f"these keys did not resolve to themselves: {wrong}"

    def test_every_gdal_short_name_resolves_to_its_own_key(self):
        """The second branch, likewise swept rather than sampled.

        Test scenario:
            Each catalog entry's GDAL short name must resolve back to the key
            that declares it. A short name mapping to a *different* key would
            silently write through the wrong driver.
        """
        catalog = get_catalog()

        wrong = {}
        for key in catalog.drivers:
            short_name = catalog.get_gdal_name(key)
            if short_name and catalog.resolve_key(short_name) != key:
                wrong[short_name] = (catalog.resolve_key(short_name), key)

        assert wrong == {}, f"short name -> (resolved, expected): {wrong}"

    def test_resolving_twice_changes_nothing(self):
        """Callers normalise defensively; doing so twice must be safe.

        Test scenario:
            `resolve_key` is applied at several layers, so an already-resolved
            key routinely arrives at another one. The operation has to be
            idempotent for that to be harmless.
        """
        catalog = get_catalog()

        once = catalog.resolve_key("GTiff")

        assert catalog.resolve_key(once) == once

    def test_no_catalog_key_collides_with_a_gdal_short_name(self):
        """The invariant that makes key-first resolution correct, not lucky.

        `resolve_key` tries the key before the short name. If any key were also
        a short name for a *different* driver, that order would silently pick
        the wrong one.
        """
        for raster in (True, False):
            catalog = get_catalog(raster)
            keys = set(catalog.drivers)
            gdal_names = {
                catalog.get_gdal_name(key) for key in keys if catalog.get_gdal_name(key)
            }
            collisions = {
                name
                for name in gdal_names & keys
                if catalog.get_gdal_name(name) != name
            }
            assert not collisions, f"key/short-name collision: {sorted(collisions)}"


class TestBandsApplyNames:
    """Restoring band names from a store, in one place."""

    @pytest.fixture
    def dataset(self) -> Dataset:
        """A three-band raster."""
        return Dataset.from_array(
            np.ones((3, 4, 4), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )

    def test_matching_names_are_applied(self, dataset: Dataset):
        """A list of the right length is applied."""
        applied = dataset.bands.apply_names(["a", "b", "c"], source="test store")

        assert applied is True
        assert list(dataset.band_names) == ["a", "b", "c"]

    def test_a_mismatched_list_is_skipped_with_a_warning(
        self, dataset: Dataset, caplog
    ):
        """The wrong number of names would rename the wrong bands."""
        before = list(dataset.band_names)

        with caplog.at_level(logging.WARNING):
            applied = dataset.bands.apply_names(["a", "b"], source="test store")

        assert applied is False
        assert list(dataset.band_names) == before
        assert "do not match band count" in caplog.text
        assert "test store" in caplog.text

    @pytest.mark.parametrize("names", [None, []])
    def test_no_names_is_a_silent_no_op(self, dataset: Dataset, names, caplog):
        """A store that recorded no names is not a mismatch."""
        before = list(dataset.band_names)

        with caplog.at_level(logging.WARNING):
            applied = dataset.bands.apply_names(names, source="test store")

        assert applied is False
        assert list(dataset.band_names) == before
        assert "do not match" not in caplog.text
