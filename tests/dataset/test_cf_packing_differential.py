"""A packed raster must answer every question the way its physical twin does.

The twin holds the same measurements already unpacked, as `float64`, and declares its gap with a
sentinel in those units. Every domain-aware public method should give the two the same answer,
because they hold the same data -- the only difference is how it is stored.

This exists because the unpack-by-default change (#1124) had to be carried into every call site
that compares a read against `no_data_value`, and fixing the sites a review named kept missing
the ones it did not: two review rounds found more than twenty, most in files the change never
touched, and every one of them had answered correctly before the default flipped. A test that
enumerates *methods* rather than *sites* catches the next one without anybody having to find it.

Each case returns something comparable -- a count, an area, the values over the domain -- so a
case only needs its answers to agree, not to know which internal path produced them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

SCALE, OFFSET, SENTINEL = 0.01, 1.5, -9999
COUNTS = np.array(
    [
        [SENTINEL, 100, 200, 300],
        [50, 150, 250, 350],
        [-100, 0, 400, 120],
        [60, 70, 80, 90],
    ],
    dtype="int16",
)
GAP = COUNTS == SENTINEL
GEO = GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.1, epsg=4326)


def _packed() -> Dataset:
    """The raster as a CF producer stores it: `int16` counts, a recipe, a stored sentinel.

    Returns:
        Dataset: The packed raster.
    """
    dataset = Dataset.from_array(COUNTS, geo_ref=GEO, no_data_value=SENTINEL)
    dataset.scale = [SCALE]
    dataset.offset = [OFFSET]
    return dataset


def _twin() -> Dataset:
    """The same measurements already unpacked, declaring its gap in physical units.

    Returns:
        Dataset: The physical twin, holding no packing.
    """
    values = COUNTS.astype("float64") * SCALE + OFFSET
    values[GAP] = float(SENTINEL)
    return Dataset.from_array(values, geo_ref=GEO, no_data_value=float(SENTINEL))


def _domain(array: Any) -> np.ndarray:
    """The cells outside the gap, as a flat `float64` array.

    Args:
        array: A full-grid result.

    Returns:
        np.ndarray: The values at the fifteen data cells.
    """
    grid = np.asarray(array, dtype="float64").reshape(COUNTS.shape)
    return grid[~GAP]


def _quiet(call: Callable[[], Any]) -> Any:
    """Run a call with its deprecation warning silenced.

    Args:
        call: The deprecated call to make.

    Returns:
        Whatever the call returns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return call()


def _to_celsius(dataset: Dataset) -> Dataset:
    """Declare the band as Kelvin and convert it, with the deprecation silenced.

    Args:
        dataset: The raster to convert.

    Returns:
        Dataset: The converted raster.
    """
    dataset.band_units = ["K"]
    return _quiet(lambda: dataset.convert_units("celsius"))


def _zones() -> gpd.GeoDataFrame:
    """One polygon covering the whole raster.

    Returns:
        gpd.GeoDataFrame: A single-zone layer.
    """
    return gpd.GeoDataFrame(
        {"zone": [1]}, geometry=[box(10.0, 49.6, 10.4, 50.0)], crs=4326
    )


def _overlaid(dataset: Dataset) -> np.ndarray:
    """Overlay the raster with a two-class map, flattened class by class.

    The left two columns are class 1 and the right two class 2, so the gap in the
    top-left corner falls inside class 1.

    Args:
        dataset: The raster to overlay.

    Returns:
        np.ndarray: Class 1's values sorted, then class 2's.
    """
    classes = Dataset.from_array(
        np.tile(np.array([1, 1, 2, 2], dtype="int32"), (4, 1)), geo_ref=GEO
    )
    groups = sorted(dataset.overlay(classes).items())
    return np.concatenate(
        [np.sort(np.asarray(values, dtype="float64")) for _, values in groups]
    )


CASES: list[tuple[str, Callable[[Dataset], Any]]] = [
    ("count_domain_cells", lambda ds: ds.count_domain_cells()),
    ("domain_area", lambda ds: ds.domain_area()),
    ("stats", lambda ds: ds.stats(approx_ok=False)[["min", "max", "mean"]].to_numpy()),
    ("extract", lambda ds: np.sort(np.asarray(ds.extract(), dtype="float64").ravel())),
    ("get_cell_coords", lambda ds: len(ds.get_cell_coords(domain_only=True))),
    (
        "to_feature_collection",
        lambda ds: np.sort(
            ds.to_feature_collection(tile=False).iloc[:, 0].to_numpy(dtype="float64")
        ),
    ),
    ("footprint", lambda ds: float(ds.footprint().area.sum())),
    ("focal_mean", lambda ds: _domain(ds.focal_mean(radius=1))),
    ("apply", lambda ds: _domain(ds.apply(lambda a: a * 2).read_array())),
    (
        "apply-tiled",
        lambda ds: _domain(ds.apply(lambda a: a * 2, elementwise=True).read_array()),
    ),
    (
        "map_blocks",
        lambda ds: _domain(ds.map_blocks(lambda t: t * 2, tile_size=2).read_array()),
    ),
    (
        "stream_transform",
        lambda ds: _domain(
            ds.io.stream_transform(lambda t: t * 2, tile_size=2).read_array()
        ),
    ),
    ("get_histogram", lambda ds: ds.get_histogram(band=0, bins=4)[0]),
    (
        "zonal_stats",
        lambda ds: ds.zonal_stats(_zones(), stats=("mean", "count", "min"))[
            ["mean", "count", "min"]
        ].to_numpy(dtype="float64"),
    ),
    ("point", lambda ds: float(ds.point(10.15, 49.95, band=0))),
    ("convert_units", lambda ds: _domain(_to_celsius(ds).read_array())),
    ("get_tile", lambda ds: _domain(next(iter(ds.get_tile(size=4))))),
    (
        "to_feature_collection-tiled",
        lambda ds: np.sort(
            ds.to_feature_collection(tile=True, tile_size=2)
            .iloc[:, 0]
            .to_numpy(dtype="float64")
        ),
    ),
    (
        # A window straddling the raster's north-west corner takes the padded arm, the
        # one every edge `read_tile` goes through. Cell (1, 1) of the raster is inside it.
        "read_part-edge",
        # Edges set mid-pixel (x and y from -1.5 to 2.5 cells) so floor / ceil cannot
        # tip the window a cell either way on any platform: it is always 5x5, and
        # [3, 3] is raster cell (1, 1), a real value rather than the gap.
        lambda ds: float(
            np.asarray(ds.read_part((9.85, 49.75, 10.25, 50.15), band=0))[3, 3]
        ),
    ),
    (
        # An interior window, well clear of every pixel edge. A bbox exactly on the
        # raster's extent puts an edge a hair either side of a pixel boundary (49.6 lands
        # at 3.999999999999986), which `read_part`'s floor / ceil snapping turns into a
        # window one cell wider on some platforms -- macOS read 4x5 where Windows read
        # 4x4. It also excludes the gap cell, so the two rasters compare cell for cell.
        "read_part",
        lambda ds: np.asarray(ds.read_part((10.12, 49.62, 10.38, 49.88), band=0)),
    ),
    ("preview", lambda ds: _domain(ds.preview(band=0))),
    ("overlay", _overlaid),
    (
        "map_blocks-band",
        lambda ds: _domain(
            ds.map_blocks(lambda t: t * 2, tile_size=2, band=0).read_array()
        ),
    ),
    (
        "stream_transform-band",
        lambda ds: _domain(
            ds.io.stream_transform(lambda t: t * 2, band=0, tile_size=2).read_array()
        ),
    ),
]


@pytest.mark.parametrize(("name", "question"), CASES, ids=[name for name, _ in CASES])
def test_the_packed_raster_answers_like_its_twin(
    name: str, question: Callable[[Dataset], Any]
):
    """Asked the same question, the packed raster and its physical twin agree.

    Args:
        name: The method under test.
        question: What to ask each raster.
    """
    packed_answer = question(_packed())
    twin_answer = question(_twin())
    np.testing.assert_allclose(
        np.asarray(packed_answer, dtype="float64"),
        np.asarray(twin_answer, dtype="float64"),
        rtol=1e-9,
        atol=1e-9,
        err_msg=f"{name}: the packed raster disagrees with its physical twin",
    )


@pytest.mark.parametrize(
    ("name", "compute"),
    [
        ("apply", lambda ds: ds.apply(lambda a: a * 2)),
        ("map_blocks", lambda ds: ds.map_blocks(lambda t: t * 2, tile_size=2)),
        (
            "stream_transform",
            lambda ds: ds.io.stream_transform(lambda t: t * 2, tile_size=2),
        ),
        (
            "map_blocks-band",
            lambda ds: ds.map_blocks(lambda t: t * 2, tile_size=2, band=0),
        ),
        (
            "stream_transform-band",
            lambda ds: ds.io.stream_transform(lambda t: t * 2, band=0, tile_size=2),
        ),
    ],
    ids=[
        "apply",
        "map_blocks",
        "stream_transform",
        "map_blocks-band",
        "stream_transform-band",
    ],
)
def test_a_computed_result_still_marks_its_gap(
    name: str, compute: Callable[[Dataset], Dataset]
):
    """The gap cell reads back as the result's declared sentinel, not as data.

    Test scenario:
        A compute path that hands the function physical values and then declares the
        *stored* sentinel on its result has to put that sentinel at the gap, or the gap
        comes back as a measurement the next time anything reads it.

    Args:
        name: The method under test.
        compute: The call producing a new raster.
    """
    result = compute(_packed())
    declared = result.no_data_value[0]
    values = np.asarray(result.read_array(unpack=False), dtype="float64").reshape(
        COUNTS.shape
    )
    assert declared is not None, f"{name}: the result declares no sentinel at all"
    assert values[GAP][0] == pytest.approx(float(declared)), (
        f"{name}: the gap holds {values[GAP][0]} while the result declares {declared}"
    )
