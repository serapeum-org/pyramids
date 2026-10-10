"""Unit tests for the item-level windowed reads (STAC-18).

Every read here is of a local GeoTIFF written into `tmp_path`, so the tests
exercise the real `COG`-engine delegation without touching the network. The
assertions compare each item-level read against the same `Dataset`-level call,
which is the contract: these helpers compose, they do not re-implement.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import OutOfBoundsError, UnsupportedAssetError
from pyramids.dataset import Dataset
from pyramids.stac import _windowed
from pyramids.stac._windowed import (
    STAC_WINDOW_CRS,
    geometry_bounds,
    read_item_feature,
    read_item_part,
    read_item_point,
    read_item_preview,
)

pytestmark = pytest.mark.core

_COG_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


class _Bounded:
    """Shapely-style geometry exposing only `bounds`."""

    def __init__(self, bounds):
        self.bounds = bounds


@pytest.fixture
def raster_path(tmp_path):
    """Write a 10x10 EPSG:4326 raster spanning lon/lat 0..10 and return its path.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written GeoTIFF; pixel (row, col) holds `row * 10 + col`.
    """
    path = str(tmp_path / "window.tif")
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, 10, 10, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(4326)
    dataset.SetProjection(reference.ExportToWkt())
    values = np.arange(100, dtype="float32").reshape(10, 10)
    dataset.GetRasterBand(1).WriteArray(values)
    dataset.FlushCache()
    dataset = None
    return path


@pytest.fixture
def item(raster_path):
    """Return a raw STAC Item whose `data` asset points at the local raster.

    Args:
        raster_path: The GeoTIFF written by the `raster_path` fixture.

    Returns:
        dict: A STAC Item with one `data` asset.
    """
    return {
        "id": "window-1",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "assets": {"data": {"href": raster_path, "type": _COG_TYPE}},
    }


@pytest.fixture
def delegated(monkeypatch):
    """Record every `Dataset` window read the item-level helpers delegate to.

    The engine method still runs, so the arrays stay real; what the recorder
    adds is the call itself — which method, and with which window. Output
    equality alone cannot tell delegation from a reimplementation that happens
    to agree.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        list: One `(method_name, args, kwargs)` tuple per delegated call.
    """
    calls: list[tuple[str, tuple, dict]] = []

    def recorder(name, original):
        def wrapper(self, *args, **kwargs):
            """Record the delegated call, then perform it."""
            calls.append((name, args, kwargs))
            return original(self, *args, **kwargs)

        return wrapper

    for name in ("read_part", "preview", "point"):
        monkeypatch.setattr(Dataset, name, recorder(name, getattr(Dataset, name)))
    return calls


class TestEngineDelegation:
    """Each helper hands its window to the `COG` engine method it documents.

    Test scenario:
        The module's claim is composition: "No windowing arithmetic lives here."
        A hand-rolled reimplementation producing the same numbers would satisfy
        an output comparison, so these pin the call instead — the method name,
        the window, and the options that must survive the hop.
    """

    def test_read_item_part_delegates_the_window(self, item, delegated):
        """`read_item_part` calls `Dataset.read_part` with the bbox it was given."""
        read_item_part(item, "data", (2.0, 6.0, 5.0, 9.0), band=0, dst_width=3)
        assert len(delegated) == 1, f"expected exactly one engine call: {delegated}"
        name, args, kwargs = delegated[0]
        assert name == "read_part", f"delegated to the wrong method: {name}"
        assert args == ((2.0, 6.0, 5.0, 9.0),), f"window not forwarded: {args}"
        assert kwargs == {"bbox_crs": 4326, "band": 0, "dst_width": 3}, (
            f"read options not forwarded verbatim: {kwargs}"
        )

    def test_read_item_preview_delegates_max_size(self, item, delegated):
        """`read_item_preview` calls `Dataset.preview` with `max_size`."""
        read_item_preview(item, "data", max_size=4, band=0)
        assert [name for name, _, _ in delegated] == ["preview"], (
            f"delegated to the wrong method: {delegated}"
        )
        _, args, kwargs = delegated[0]
        assert args == (), f"preview takes keywords only: {args}"
        assert kwargs == {"max_size": 4, "band": 0}, (
            f"preview options not forwarded verbatim: {kwargs}"
        )

    def test_read_item_point_delegates_the_coordinate(self, item, delegated):
        """`read_item_point` calls `Dataset.point` with the split coordinate."""
        read_item_point(item, "data", (2.5, 7.5), band=0)
        assert [name for name, _, _ in delegated] == ["point"], (
            f"delegated to the wrong method: {delegated}"
        )
        _, args, kwargs = delegated[0]
        assert args == (2.5, 7.5), f"coordinate not forwarded: {args}"
        assert kwargs == {"point_crs": 4326, "band": 0}, (
            f"point options not forwarded verbatim: {kwargs}"
        )

    def test_read_item_feature_delegates_the_envelope_to_read_part(
        self, item, delegated
    ):
        """`read_item_feature` reduces the geometry and routes to `read_part`."""
        polygon = {
            "type": "Polygon",
            "coordinates": [[[2.0, 6.0], [5.0, 6.0], [5.0, 9.0], [2.0, 6.0]]],
        }
        read_item_feature(item, "data", polygon, band=0)
        assert [name for name, _, _ in delegated] == ["read_part"], (
            f"a feature read should be one read_part call: {delegated}"
        )
        _, args, kwargs = delegated[0]
        assert args == ((2.0, 6.0, 5.0, 9.0),), f"envelope not forwarded: {args}"
        assert kwargs["bbox_crs"] == 4326, f"geometry_crs not forwarded: {kwargs}"

    def test_a_custom_window_crs_reaches_the_engine(self, item, delegated):
        """An explicit `bbox_crs` is forwarded rather than re-derived here."""
        read_item_part(item, "data", (2.0, 6.0, 5.0, 9.0), bbox_crs=None, band=0)
        _, _, kwargs = delegated[0]
        assert kwargs["bbox_crs"] is None, f"bbox_crs not forwarded: {kwargs}"


class TestGeometryBounds:
    """`geometry_bounds` reduces every accepted geometry shape to an envelope."""

    def test_polygon(self):
        """A GeoJSON polygon reduces to its envelope."""
        polygon = {
            "type": "Polygon",
            "coordinates": [[[1.0, 2.0], [4.0, 2.0], [4.0, 6.0], [1.0, 2.0]]],
        }
        assert geometry_bounds(polygon) == (1.0, 2.0, 4.0, 6.0), "wrong envelope"

    def test_multipolygon(self):
        """Deeper nesting is walked to the positions."""
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                [[[5.0, 5.0], [7.0, 8.0], [5.0, 5.0]]],
            ],
        }
        assert geometry_bounds(geometry) == (0.0, 0.0, 7.0, 8.0), "wrong envelope"

    def test_feature_is_unwrapped(self):
        """A `Feature` wrapper is unwrapped before bounding."""
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
        }
        assert geometry_bounds(feature) == (3.0, 4.0, 3.0, 4.0), "wrong envelope"

    def test_feature_collection_is_unioned(self):
        """A `FeatureCollection` bounds every member."""
        collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 2]},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [5, 9]},
                },
            ],
        }
        assert geometry_bounds(collection) == (1.0, 2.0, 5.0, 9.0), "wrong envelope"

    def test_geometry_collection_is_unioned(self):
        """A `GeometryCollection` bounds every member."""
        collection = {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [2.0, 2.0]},
                {"type": "Point", "coordinates": [8.0, 3.0]},
            ],
        }
        assert geometry_bounds(collection) == (2.0, 2.0, 8.0, 3.0), "wrong envelope"

    def test_elevation_is_dropped(self):
        """A 3-D position contributes only its horizontal ordinates."""
        geometry = {"type": "Point", "coordinates": [1.0, 2.0, 300.0]}
        assert geometry_bounds(geometry) == (1.0, 2.0, 1.0, 2.0), "elevation leaked"

    def test_shapely_style_bounds_are_used(self):
        """An object exposing `bounds` is read through it directly."""
        bounds = geometry_bounds(_Bounded((1.0, 2.0, 3.0, 4.0)))
        assert bounds == (1.0, 2.0, 3.0, 4.0), f"bounds not used: {bounds}"

    def test_bare_coordinate_nesting(self):
        """A bare `coordinates` nesting is bounded without a geometry wrapper.

        Test scenario:
            A ring handed in on its own — no mapping, no `type` — is walked
            straight into the position reader.
        """
        ring = [[1.0, 2.0], [4.0, 2.0], [4.0, 6.0], [1.0, 2.0]]
        assert geometry_bounds(ring) == (1.0, 2.0, 4.0, 6.0), (
            f"bare nesting not bounded: {geometry_bounds(ring)}"
        )

    def test_empty_geometry_raises(self):
        """An empty geometry is an error, not a silent whole-raster read."""
        with pytest.raises(ValueError, match="no coordinates"):
            geometry_bounds({"type": "Polygon", "coordinates": []})


class TestReadItemPart:
    """`read_item_part` delegates to `Dataset.read_part` on the resolved asset."""

    def test_matches_the_dataset_level_read(self, item, raster_path):
        """The item-level window equals the same `Dataset.read_part` call."""
        bbox = (2.0, 6.0, 5.0, 9.0)
        window = read_item_part(item, "data", bbox)
        expected = Dataset.read_file(raster_path).read_part(bbox, bbox_crs=4326)
        assert np.array_equal(window, expected), f"window differs: {window}"

    def test_reads_the_requested_cells(self, item):
        """The window's values are the raster cells it covers."""
        window = read_item_part(item, "data", (2.0, 6.0, 5.0, 9.0), band=0)
        assert window.shape == (3, 3), f"wrong window shape: {window.shape}"
        assert np.array_equal(
            window,
            np.array([[12.0, 13.0, 14.0], [22.0, 23.0, 24.0], [32.0, 33.0, 34.0]]),
        ), f"wrong cells read: {window}"

    def test_default_crs_is_stac_lon_lat(self, item, delegated):
        """The default `bbox_crs` handed to the engine is EPSG:4326.

        Test scenario:
            Comparing two reads of a 4326 raster cannot see this: `bbox_crs=4326`
            and `bbox_crs=None` are identical by construction there, so the
            default could be changed to `None` without the arrays moving. The
            value that reaches the engine is what the contract is about.
        """
        assert STAC_WINDOW_CRS == 4326, "STAC window default is not lon/lat"
        bbox = (2.0, 6.0, 5.0, 9.0)
        read_item_part(item, "data", bbox)
        read_item_part(item, "data", bbox, bbox_crs=None)
        forwarded = [kwargs["bbox_crs"] for _, _, kwargs in delegated]
        assert forwarded == [4326, None], (
            f"the engine should receive the STAC default then the opt-out, got {forwarded}"
        )

    def test_decimation_options_are_forwarded(self, item):
        """`dst_width` / `dst_height` reach the engine and resize the output."""
        window = read_item_part(
            item, "data", (0.0, 0.0, 10.0, 10.0), dst_width=5, dst_height=5, band=0
        )
        assert window.shape == (5, 5), f"decimation not forwarded: {window.shape}"

    def test_return_transform_is_forwarded(self, item):
        """`return_transform=True` comes back as an (array, geotransform) pair."""
        result = read_item_part(
            item, "data", (2.0, 6.0, 5.0, 9.0), band=0, return_transform=True
        )
        array, transform = result
        assert array.shape == (3, 3), f"wrong window shape: {array.shape}"
        assert len(transform) == 6, f"not a geotransform: {transform}"

    def test_accepts_a_bare_asset(self, item, raster_path):
        """A bare asset (no item) is read with `asset_key=None`."""
        asset = {"href": raster_path, "type": _COG_TYPE}
        window = read_item_part(asset, None, (2.0, 6.0, 5.0, 9.0), band=0)
        assert window.shape == (3, 3), f"wrong window shape: {window.shape}"

    def test_prefers_an_alternate_href(self, raster_path):
        """The `alternate-assets` preference reaches `load_asset`."""
        asset = {
            "href": "https://nowhere.invalid/missing.tif",
            "type": _COG_TYPE,
            "alternate": {"local": {"href": raster_path}},
        }
        window = read_item_part(
            asset, None, (2.0, 6.0, 5.0, 9.0), band=0, alternate="local"
        )
        assert window.shape == (3, 3), f"alternate not used: {window.shape}"

    def test_signer_is_forwarded(self, item, raster_path):
        """The signer handed in is the one `load_asset` applies."""

        class _Signer:
            """Signer that records the href and returns it unchanged."""

            def __init__(self):
                self.seen = None

            def sign_href(self, href):
                """Record the href, leaving it readable."""
                self.seen = href
                return href

            def gdal_env(self):
                """Return no extra GDAL config."""
                return {}

        signer = _Signer()
        read_item_part(item, "data", (2.0, 6.0, 5.0, 9.0), band=0, signer=signer)
        assert signer.seen == raster_path, f"signer not applied: {signer.seen}"

    def test_missing_asset_raises(self, item):
        """A missing asset key fails with the usual STAC asset error."""
        with pytest.raises(KeyError, match="B99"):
            read_item_part(item, "B99", (2.0, 6.0, 5.0, 9.0))

    def test_non_dataset_reader_is_rejected(self, item, monkeypatch):
        """An asset that opens as a collection has no windowed read."""
        monkeypatch.setattr(_windowed, "load_asset", lambda *args, **kwargs: object())
        with pytest.raises(UnsupportedAssetError, match="windowed read"):
            read_item_part(item, "data", (2.0, 6.0, 5.0, 9.0))


class TestReadItemPreview:
    """`read_item_preview` delegates to `Dataset.preview`."""

    def test_matches_the_dataset_level_preview(self, item, raster_path):
        """The thumbnail equals the same `Dataset.preview` call."""
        thumb = read_item_preview(item, "data", max_size=5)
        expected = Dataset.read_file(raster_path).preview(max_size=5)
        assert np.array_equal(thumb, expected), f"preview differs: {thumb}"

    def test_max_size_bounds_the_long_edge(self, item):
        """`max_size` reaches the engine."""
        thumb = read_item_preview(item, "data", max_size=4, band=0)
        assert max(thumb.shape) == 4, f"max_size not honoured: {thumb.shape}"

    def test_accepts_a_bare_asset(self, raster_path):
        """A bare asset previews without an item or asset key."""
        asset = {"href": raster_path, "type": _COG_TYPE}
        thumb = read_item_preview(asset, max_size=5, band=0)
        assert thumb.shape == (5, 5), f"wrong thumbnail shape: {thumb.shape}"


class TestReadItemPoint:
    """`read_item_point` delegates to `Dataset.point`."""

    def test_matches_the_dataset_level_sample(self, item, raster_path):
        """The sample equals the same `Dataset.point` call."""
        sampled = read_item_point(item, "data", (2.5, 7.5))
        expected = Dataset.read_file(raster_path).point(2.5, 7.5, point_crs=4326)
        assert np.array_equal(sampled, expected), f"sample differs: {sampled}"

    def test_samples_the_covering_cell(self, item):
        """The value is the cell covering the coordinate."""
        sampled = read_item_point(item, "data", (2.5, 7.5), band=0)
        assert float(sampled) == 22.0, f"wrong cell sampled: {sampled}"

    def test_out_of_extent_raises(self, item):
        """A coordinate outside the asset is an error, not a silent clamp."""
        with pytest.raises(OutOfBoundsError):
            read_item_point(item, "data", (99.0, 99.0), band=0)


class TestReadItemFeature:
    """`read_item_feature` reads a geometry's bounding window."""

    def test_matches_the_envelope_window(self, item):
        """The result equals `read_item_part` over the geometry's envelope."""
        polygon = {
            "type": "Polygon",
            "coordinates": [[[2.0, 6.0], [5.0, 6.0], [5.0, 9.0], [2.0, 6.0]]],
        }
        window = read_item_feature(item, "data", polygon, band=0)
        expected = read_item_part(item, "data", (2.0, 6.0, 5.0, 9.0), band=0)
        assert np.array_equal(window, expected), f"feature window differs: {window}"

    def test_is_not_masked(self, item):
        """Cells inside the envelope but outside the geometry keep their values.

        The documented contract: this is a bounding-window read, not a cut-out.
        """
        triangle = {
            "type": "Polygon",
            "coordinates": [[[2.0, 6.0], [5.0, 6.0], [5.0, 9.0], [2.0, 6.0]]],
        }
        window = read_item_feature(item, "data", triangle, band=0)
        assert np.isfinite(window).all(), f"a cell was masked out: {window}"

    def test_shapely_style_geometry(self, item):
        """A geometry exposing `bounds` is accepted."""
        window = read_item_feature(item, "data", _Bounded((2.0, 6.0, 5.0, 9.0)), band=0)
        assert window.shape == (3, 3), f"wrong window shape: {window.shape}"

    def test_options_are_forwarded(self, item):
        """Engine read options survive the hop through `read_item_part`."""
        window = read_item_feature(
            item,
            "data",
            _Bounded((0.0, 0.0, 10.0, 10.0)),
            band=0,
            dst_width=5,
            dst_height=5,
        )
        assert window.shape == (5, 5), f"options not forwarded: {window.shape}"
