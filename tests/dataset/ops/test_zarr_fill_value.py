"""A pyramids Zarr store declares its no-data where every reader looks.

`to_zarr` recorded the sentinel in a `no_data_value` attribute, which is
pyramids' own spelling and which only `from_zarr` reads. Zarr's array metadata
carries its own `fill_value` field, and it defaults to **0** -- an ordinary
value of every numeric type. GDAL's Zarr driver reads that field, so
`Dataset.read_file(store.zarr)` reported a no-data of `0.0` for a store written
with `-9999.0`, and masking such a read blanked every genuinely-zero cell.

The sentinel now goes in `fill_value` as well, so the two ways of opening the
same store agree, and so does any other GeoZarr reader.

One array holds every band, so a single `fill_value` can only describe a
sentinel the bands agree on. Where they differ it is left off rather than
written wrong, and `from_zarr` recovers the full per-band list from
`no_data_value` either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

try:
    import zarr
except ImportError:  # pragma: no cover - exercised only without the extra
    zarr = None

# Guarded rather than a module-level `importorskip`, so the file still collects
# without the `[lazy]` extra; the tests that need a real store are marked.
needs_zarr = pytest.mark.skipif(zarr is None, reason="requires the [lazy] extra")

SENTINEL = -9999.0


def _written_store(tmp_path: Path, no_data, bands: int = 3) -> str:
    """Write a raster to Zarr and return the store path.

    Round-tripped through a GeoTIFF first: `to_zarr` reads through the file
    manager, so the source has to be on disk rather than purely in memory.

    Args:
        tmp_path: The pytest temporary directory to write under.
        no_data: The no-data value to stamp, or `None` for none.
        bands: How many bands the raster carries.

    Returns:
        str: Path of the written `.zarr` store.
    """
    array = np.arange(bands * 4 * 5, dtype="float32").reshape(bands, 4, 5)
    in_memory = Dataset.from_array(
        array,
        geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
        no_data_value=no_data,
    )
    geotiff = tmp_path / "source.tif"
    in_memory.to_file(str(geotiff))
    store = tmp_path / "store.zarr"
    Dataset.read_file(str(geotiff)).to_zarr(str(store))
    return str(store)


class TestTheSentinelReachesTheArrayMetadata:
    """`fill_value`, not only the pyramids attribute."""

    @pytest.mark.lazy
    @needs_zarr
    def test_the_zarr_metadata_carries_it(self, tmp_path):
        """The field GDAL reads, asserted on disk.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            Asserted against the JSON rather than through a reader, because
            the whole defect was that one reader saw the value and another
            did not. The file is the thing both of them read.
        """
        store = _written_store(tmp_path, SENTINEL)

        metadata = json.loads((Path(store) / "data" / "zarr.json").read_text())

        assert metadata["fill_value"] == SENTINEL, (
            f"zarr fill_value is {metadata['fill_value']}, not the sentinel"
        )

    @pytest.mark.lazy
    @needs_zarr
    def test_both_readers_agree(self, tmp_path):
        """The regression: two ways of opening one store, two answers.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `from_zarr` reads pyramids' own `no_data_value` attribute and
            always got this right; `read_file` goes through GDAL, which reads
            `fill_value` and got `0.0`. Nothing warned.
        """
        store = _written_store(tmp_path, SENTINEL)

        through_pyramids = Dataset.from_zarr(store).no_data_value
        through_gdal = Dataset.read_file(store).no_data_value

        assert tuple(through_pyramids) == tuple(through_gdal), (
            f"from_zarr says {through_pyramids}, read_file says {through_gdal}"
        )
        assert all(value == SENTINEL for value in through_gdal)

    @pytest.mark.lazy
    @needs_zarr
    def test_the_pyramids_attribute_is_still_written(self, tmp_path):
        """Adding one spelling must not drop the other.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `no_data_value` is the only spelling that can carry *differing*
            per-band values, so it stays even where `fill_value` duplicates
            it.
        """
        store = _written_store(tmp_path, SENTINEL)

        attributes = dict(zarr.open_group(store, mode="r")["data"].attrs)

        assert attributes["no_data_value"] == [SENTINEL] * 3
        assert attributes["_FillValue"] == SENTINEL

    @pytest.mark.lazy
    @needs_zarr
    def test_a_raster_without_a_sentinel_declares_none_in_the_attributes(
        self, tmp_path
    ):
        """No sentinel must not be recorded as a sentinel of 0.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            Zarr v3 requires a numeric `fill_value`, so "no no-data" cannot be
            said in that field -- it stays at the spec default of 0, and a
            GDAL read of such a store therefore reports 0.0 whatever pyramids
            does. That is a Zarr limitation, not something the writer chooses,
            and it predates the sentinel work. What pyramids *can* say is said
            in its own attribute, and `from_zarr` reads it: the store records
            no sentinel and reports none.
        """
        store = _written_store(tmp_path, None)

        attributes = dict(zarr.open_group(store, mode="r")["data"].attrs)

        assert "_FillValue" not in attributes, (
            "a sentinel was declared for a raster that has none"
        )
        assert all(v is None for v in attributes["no_data_value"])
        assert all(v is None for v in Dataset.from_zarr(store).no_data_value)

    @pytest.mark.lazy
    @needs_zarr
    def test_differing_per_band_sentinels_are_not_flattened(self, tmp_path):
        """One `fill_value` cannot describe three different sentinels.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            The store is one `(band, y, x)` array, so `fill_value` is a single
            number. Writing one band's sentinel there would tell every reader
            it applied to all three, which is worse than saying nothing --
            so it is omitted, and the per-band list keeps the truth.
        """
        array = np.arange(3 * 4 * 5, dtype="float32").reshape(3, 4, 5)
        in_memory = Dataset.from_array(
            array,
            geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
            no_data_value=[-1.0, -2.0, -3.0],
        )
        geotiff = tmp_path / "mixed.tif"
        in_memory.to_file(str(geotiff))
        store = tmp_path / "mixed.zarr"
        Dataset.read_file(str(geotiff)).to_zarr(str(store))

        attributes = dict(zarr.open_group(str(store), mode="r")["data"].attrs)

        assert "_FillValue" not in attributes, (
            "one band's sentinel was declared for all three"
        )
        assert attributes["no_data_value"] == [-1.0, -2.0, -3.0]
        assert tuple(Dataset.from_zarr(str(store)).no_data_value) == (-1.0, -2.0, -3.0)

    @pytest.mark.lazy
    @needs_zarr
    def test_the_data_survives_the_round_trip(self, tmp_path):
        """Declaring a sentinel must not alter a single cell.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `fill_value` is what zarr returns for an unwritten chunk. Every
            chunk here is written, so it must change nothing that comes back
            -- and the values deliberately include 0, which is what the old
            default claimed was missing.
        """
        store = _written_store(tmp_path, SENTINEL)

        recovered = np.asarray(Dataset.from_zarr(store).read_array())

        expected = np.arange(3 * 4 * 5, dtype="float32").reshape(3, 4, 5)
        assert np.array_equal(recovered, expected)
        assert (recovered == 0).any(), (
            "the fixture must contain a zero to be meaningful"
        )
