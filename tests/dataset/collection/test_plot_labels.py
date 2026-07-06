"""Time axis and animation frame-label routing for ``DatasetCollection`` (#693).

Two connected behaviours are covered:

* ``DatasetCollection`` carries an optional ``time`` axis — parsed from the file
  names by :meth:`read_multiple_files`, or assigned directly.
* :meth:`DatasetCollection.plot` labels animation frames by that time axis when
  present, falls back to an index axis (``range(time_length)``) when absent, and
  lets an explicit ``animation_axis_values`` kwarg override both — without that
  kwarg colliding with the value ``plot`` passes to :func:`render_array`
  internally (before the fix, passing it raised ``TypeError: got multiple values
  for keyword argument``).

``render_array`` is patched in the plot tests so they exercise only the
pyramids-side routing, with no cleopatra dependency and no real rendering.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection

pytestmark = pytest.mark.core


def _collection(count: int = 3, bands: int = 1) -> DatasetCollection:
    """Build a small in-memory collection of ``count`` identical timesteps."""
    arr = np.ones((bands, 4, 5), dtype="float32")
    src = Dataset.create_from_array(
        arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
    )
    return DatasetCollection.create_cube(src, count)


def _dated_files(tmp_path, years=(2000, 2001, 2002)) -> list[str]:
    """Write one single-band raster per year, named ``scene_YYYY0101.tif``."""
    paths = []
    for y in years:
        arr = np.ones((1, 4, 5), dtype="float32")
        p = tmp_path / f"scene_{y}0101.tif"
        Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326, path=str(p)
        ).close()
        paths.append(str(p))
    return paths


class TestPlotAnimationAxisValues:
    def test_defaults_to_index_axis(self):
        """With no time axis and no override, frames are labelled ``0..N-1``."""
        cube = _collection(3)
        assert cube.time is None
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(band=0)
        assert render.call_args.kwargs["animation_axis_values"] == [0, 1, 2]

    def test_defaults_to_time_axis_when_present(self):
        """When the collection carries a time axis, plot labels frames by it."""
        cube = _collection(3)
        cube.time = [2000, 2001, 2002]
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(band=0)
        assert render.call_args.kwargs["animation_axis_values"] == [2000, 2001, 2002]

    def test_explicit_axis_values_respected(self):
        """An explicit ``animation_axis_values`` wins over the default.

        This also guards the kwarg collision: before the fix the same keyword was
        passed both by the caller (via ``**kwargs``) and hardcoded by ``plot``,
        which raised ``TypeError``.
        """
        cube = _collection(3)
        cube.time = [2000, 2001, 2002]  # override must beat the time axis too
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(band=0, animation_axis_values=[7, 8, 9])
        assert render.call_args.kwargs["animation_axis_values"] == [7, 8, 9]

    def test_explicit_axis_values_respected_rgb(self):
        """The override also reaches the RGB (true-colour) animation branch."""
        cube = _collection(2, bands=3)
        labels = ["2000", "2001"]
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(
                rgb_options={"rgb": [0, 1, 2], "surface_reflectance": 255},
                animation_axis_values=labels,
            )
        assert render.call_args.kwargs["animation_axis_values"] == labels

    def test_wrong_length_axis_values_raises(self):
        """An explicit override whose length != frame count raises a clear error."""
        cube = _collection(3)
        with patch("pyramids.dataset.collection.render_array") as render:
            with pytest.raises(ValueError, match="animation_axis_values has 2 labels"):
                cube.plot(band=0, animation_axis_values=[7, 8])
        render.assert_not_called()

    def test_parsed_dates_label_frames_end_to_end(self, tmp_path):
        """A dated collection animates with real dates, no manual relabelling.

        This is the issue #693 scenario: ``read_multiple_files`` parses the dates
        from the file names, and ``plot`` uses them as the frame labels.
        """
        cube = DatasetCollection.read_multiple_files(
            _dated_files(tmp_path),
            date=True,
            regex_string=r"\d{8}",
            file_name_data_fmt="%Y%m%d",
        )
        with patch("pyramids.dataset.collection.render_array") as render:
            cube.plot(band=0)
        assert render.call_args.kwargs["animation_axis_values"] == [
            dt.datetime(2000, 1, 1),
            dt.datetime(2001, 1, 1),
            dt.datetime(2002, 1, 1),
        ]


class TestTimeAxis:
    def test_read_multiple_files_parses_time_without_order(self, tmp_path):
        """``file_name_data_fmt`` alone populates ``time`` (no ``with_order``)."""
        cube = DatasetCollection.read_multiple_files(
            _dated_files(tmp_path),
            date=True,
            regex_string=r"\d{8}",
            file_name_data_fmt="%Y%m%d",
        )
        assert cube.time == [
            dt.datetime(2000, 1, 1),
            dt.datetime(2001, 1, 1),
            dt.datetime(2002, 1, 1),
        ]

    def test_read_multiple_files_with_order_sorts_time(self, tmp_path):
        """``with_order`` sorts files by date and the time axis follows."""
        files = _dated_files(tmp_path, years=(2002, 2000, 2001))
        cube = DatasetCollection.read_multiple_files(
            files,
            with_order=True,
            date=True,
            regex_string=r"\d{8}",
            file_name_data_fmt="%Y%m%d",
        )
        assert cube.time == [
            dt.datetime(2000, 1, 1),
            dt.datetime(2001, 1, 1),
            dt.datetime(2002, 1, 1),
        ]

    def test_date_parsed_from_name_not_path(self, tmp_path):
        """The date regex matches the file name, not stray digits in the directory.

        The result is a plain ``datetime.datetime``, not a pandas ``Timestamp``.
        """
        noisy = tmp_path / "12345678_run"  # an 8-digit run in the PATH, not the name
        noisy.mkdir()
        cube = DatasetCollection.read_multiple_files(
            _dated_files(noisy, years=(2000, 2001)),
            date=True,
            regex_string=r"\d{8}",
            file_name_data_fmt="%Y%m%d",
        )
        assert cube.time == [dt.datetime(2000, 1, 1), dt.datetime(2001, 1, 1)]
        assert all(type(t) is dt.datetime for t in cube.time), "plain datetime"

    def test_read_multiple_files_without_format_has_no_time_axis(self, tmp_path):
        """A default read (no format) leaves ``time`` unset — non-breaking."""
        cube = DatasetCollection.read_multiple_files(
            _dated_files(tmp_path), with_order=False
        )
        assert cube.time is None

    def test_constructor_time_length_mismatch_raises(self):
        """Constructing with a mis-sized ``time`` is rejected."""
        arr = np.ones((1, 4, 5), dtype="float32")
        src = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        with pytest.raises(ValueError, match="time has length"):
            DatasetCollection(src, 3, time=[2000, 2001])

    def test_setter_length_validation_and_clear(self):
        """The ``time`` setter validates length and accepts ``None`` to clear."""
        cube = _collection(3)
        cube.time = [2000, 2001, 2002]
        assert cube.time == [2000, 2001, 2002]
        with pytest.raises(ValueError, match="time has length"):
            cube.time = [2000, 2001]
        cube.time = None
        assert cube.time is None
