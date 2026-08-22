"""Tests for GDAL open options threaded through Dataset.read_file (#1025).

`Dataset.read_file(open_options=...)` forwards driver-specific open options to
GDAL and captures them on the returned dataset, so the read paths that reopen
the file rather than reuse the handle — `threadsafe=True` per-thread handles,
lazy `chunks=` reads inside dask tasks, and unpickling on a worker — reopen with
the same options. `GTiff`'s `GEOREF_SOURCES` gives an observable effect with no
special fixture: `NONE` drops the internal georeference (identity geotransform),
`INTERNAL` keeps it.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from osgeo import gdal

from pyramids._io import normalize_open_options
from pyramids.dataset import Dataset, DatasetCollection
from tests._helpers import write_raster

pytestmark = pytest.mark.core

_REAL_GT = (100.0, 1.0, 0.0, 200.0, 0.0, -1.0)
_IDENTITY_GT = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def georeferenced_tif(tmp_path):
    """A GeoTIFF carrying an internal geotransform.

    Returns:
        str: Path to an 8x8 raster whose internal georef is `_REAL_GT`.
    """
    return write_raster(
        tmp_path / "geo.tif",
        np.ones((8, 8), "float32"),
        (100.0, 200.0),
        cell_size=1.0,
    )


class TestNormalizeOpenOptions:
    """Tests for the `dict` / list / `None` normaliser."""

    def test_dict_becomes_key_value_list(self):
        """A mapping is rendered as GDAL's `KEY=VALUE` list."""
        assert normalize_open_options({"L1B_MODE": "DATASTRIP"}) == ["L1B_MODE=DATASTRIP"]

    def test_list_passes_through(self):
        """A native list is returned as a list, unchanged in content."""
        assert normalize_open_options(["A=1", "B=2"]) == ["A=1", "B=2"]

    def test_none_stays_none(self):
        """`None` (the no-options case) is preserved."""
        assert normalize_open_options(None) is None


class TestReadFileOpenOptions:
    """Tests for `Dataset.read_file(open_options=...)` end to end."""

    def test_default_open_captures_nothing(self, georeferenced_tif):
        """An ordinary open captures no options and keeps the real georef."""
        dataset = Dataset.read_file(georeferenced_tif)
        assert dataset.open_options == [], "an ordinary open must capture nothing"
        assert dataset.geotransform == _REAL_GT

    def test_dict_form_takes_effect(self, georeferenced_tif):
        """A `dict` option reaches GDAL: `GEOREF_SOURCES=NONE` drops the georef.

        Test scenario:
            Without the option the geotransform is `_REAL_GT`; with it, GDAL
            ignores the internal georef and returns the identity transform.
        """
        dataset = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        assert dataset.geotransform == _IDENTITY_GT, "the option did not reach GDAL"
        assert dataset.open_options == ["GEOREF_SOURCES=NONE"]

    def test_list_form_takes_effect(self, georeferenced_tif):
        """GDAL's native `["KEY=VALUE"]` form is accepted too."""
        dataset = Dataset.read_file(
            georeferenced_tif, open_options=["GEOREF_SOURCES=INTERNAL"]
        )
        assert dataset.geotransform == _REAL_GT
        assert dataset.open_options == ["GEOREF_SOURCES=INTERNAL"]

    def test_options_survive_pickle_reopen(self, georeferenced_tif):
        """A worker reopen (unpickle) reapplies the captured options.

        Test scenario:
            Losing the options on the worker would silently change driver
            behaviour there — the geotransform would flip back to `_REAL_GT`.
        """
        dataset = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        reopened = pickle.loads(pickle.dumps(dataset))
        assert reopened.open_options == ["GEOREF_SOURCES=NONE"]
        assert reopened.geotransform == _IDENTITY_GT, "options lost on reopen"

    def test_options_survive_threadsafe_reopen(self, georeferenced_tif):
        """A per-thread reopen carries the options and still reads."""
        dataset = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        result = np.asarray(dataset.read_array(threadsafe=True))
        assert result.shape == (8, 8)

    def test_two_opens_different_options_do_not_share_a_handle(self, georeferenced_tif):
        """The shared cache keys on the options, so the two reads differ.

        Test scenario:
            Read-only opens go through the shared cache; if it ignored the
            options, the second open would inherit the first's georef.
        """
        first = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        second = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "INTERNAL"}
        )
        assert first.geotransform == _IDENTITY_GT
        assert second.geotransform == _REAL_GT


class TestCollectionOpenOptions:
    """Tests for `DatasetCollection.from_files(open_options=...)`."""

    def test_from_files_threads_the_option(self, tmp_path):
        """The option reaches every per-file open in the collection.

        Test scenario:
            `GEOREF_SOURCES=NONE` on a two-file collection must drop the georef
            on the eagerly-opened template.
        """
        paths = [
            write_raster(
                tmp_path / f"g{i}.tif", np.ones((8, 8), "float32"), (100.0, 200.0)
            )
            for i in range(2)
        ]
        collection = DatasetCollection.from_files(
            paths, open_options={"GEOREF_SOURCES": "NONE"}
        )
        assert collection.open_options == ["GEOREF_SOURCES=NONE"]
        assert collection.base.geotransform == _IDENTITY_GT
