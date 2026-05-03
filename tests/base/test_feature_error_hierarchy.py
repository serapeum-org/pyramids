"""Tests for the vector-side error hierarchy.

`pyramids.base._errors` gained a :class:`FeatureError` base and
three subclasses:

- :class:`InvalidGeometryError` (also a `ValueError`)
- :class:`CRSError` (also a `ValueError`)
- :class:`VectorDriverError` (also a `RuntimeError`)

Every feature-side raise that previously used a bare builtin was
migrated to the matching subclass. The multi-inheritance with the
original builtin means existing `except ValueError:` and
`except RuntimeError:` handlers keep working — we did not break
callers to gain the new type.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, box

from pyramids.base._errors import (
    CRSError,
    FeatureError,
    InvalidGeometryError,
    VectorDriverError,
    _PyramidsError,
)
from pyramids.base.crs import get_epsg_from_prj
from pyramids.feature import FeatureCollection
from pyramids.feature.geometry import get_coords

pytestmark = pytest.mark.core


class TestHierarchy:
    """Class-level inheritance / MRO checks."""

    def test_feature_error_is_pyramids_error(self):
        assert issubclass(FeatureError, _PyramidsError)

    def test_invalid_geometry_is_value_error(self):
        """`except ValueError` must still catch `InvalidGeometryError`."""
        assert issubclass(InvalidGeometryError, ValueError)
        assert issubclass(InvalidGeometryError, FeatureError)

    def test_crs_error_is_value_error(self):
        assert issubclass(CRSError, ValueError)
        assert issubclass(CRSError, FeatureError)

    def test_vector_driver_error_is_runtime_error(self):
        assert issubclass(VectorDriverError, RuntimeError)
        assert issubclass(VectorDriverError, FeatureError)


class TestInvalidGeometryError:
    """MultiPolygon into get_coords raises InvalidGeometryError (/-18)."""

    def test_raised_on_multipolygon(self):
        import pandas as pd

        mp = MultiPolygon([box(0, 0, 1, 1)])
        row = pd.Series({"geometry": mp})
        with pytest.raises(InvalidGeometryError, match="MultiPolygon"):
            get_coords(row, "geometry", "x")

    def test_still_catchable_as_value_error(self):
        """Legacy handlers using `except ValueError` keep working."""
        import pandas as pd

        mp = MultiPolygon([box(0, 0, 1, 1)])
        row = pd.Series({"geometry": mp})
        with pytest.raises(ValueError):
            get_coords(row, "geometry", "x")

    def test_catchable_as_feature_error(self):
        """A broad `except FeatureError:` also catches it."""
        import pandas as pd

        mp = MultiPolygon([box(0, 0, 1, 1)])
        row = pd.Series({"geometry": mp})
        with pytest.raises(FeatureError):
            get_coords(row, "geometry", "x")


class TestCRSError:
    """Empty projection / CRS mismatch raises CRSError."""

    def test_get_epsg_from_prj_empty_raises_crs_error(self):
        with pytest.raises(CRSError, match="empty projection string"):
            get_epsg_from_prj("")

    def test_still_catchable_as_value_error(self):
        """Legacy `except ValueError` still catches it."""
        with pytest.raises(ValueError):
            get_epsg_from_prj("")

    @pytest.mark.plot
    def test_basemap_without_crs_raises_crs_error(self):
        """FC.plot(basemap=True) without CRS raises CRSError."""
        poly = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly])  # no crs=
        fc = FeatureCollection(gdf)
        with pytest.raises(CRSError, match="CRS"):
            fc.plot(basemap=True)

    def test_rasterize_crs_mismatch_raises_crs_error(self):
        """Dataset.from_features with mismatched CRS raises CRSError."""
        from pyramids.dataset import Dataset

        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1]},
                geometry=[Point(0, 0)],
                crs="EPSG:4326",
            )
        )
        template = Dataset.create(
            cell_size=1000.0,
            rows=5,
            columns=5,
            dtype="float32",
            bands=1,
            top_left_corner=(500000.0, 3500000.0),
            epsg=32636,  # different from 4326
            no_data_value=-9999.0,
        )
        with pytest.raises(CRSError, match="not the same EPSG"):
            Dataset.from_features(fc, template=template)


class TestVectorDriverError:
    """OGR translation failures raise VectorDriverError."""

    def test_vectortranslate_none_raises(self, monkeypatch):
        """_ogr.datasource_to_gdf maps a None return to VectorDriverError."""
        from pyramids.feature import _ogr

        monkeypatch.setattr(
            "pyramids.feature._ogr.gdal.VectorTranslate",
            lambda *a, **kw: None,
        )
        # Construct a real (tiny) OGR DS to feed in.
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        with _ogr.as_datasource(gdf) as ds:
            with pytest.raises(VectorDriverError, match="VectorTranslate"):
                _ogr.datasource_to_gdf(ds)

    def test_still_catchable_as_runtime_error(self, monkeypatch):
        """Legacy `except RuntimeError` still catches it."""
        from pyramids.feature import _ogr

        monkeypatch.setattr(
            "pyramids.feature._ogr.gdal.VectorTranslate",
            lambda *a, **kw: None,
        )
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        with _ogr.as_datasource(gdf) as ds:
            with pytest.raises(RuntimeError):
                _ogr.datasource_to_gdf(ds)
