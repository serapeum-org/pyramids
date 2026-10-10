"""Tests for per-band resampling in from_stac / from_band_files (STAC-11)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset._stac import from_stac

pytestmark = pytest.mark.core

# The coarse rasters hold only these four values, so a nearest-neighbour
# resample can produce nothing else and an interpolating one must.
_DISCRETE = {0.0, 10.0, 20.0, 30.0}


@pytest.fixture
def fine_raster(tmp_path):
    """A 4x4 raster at cell size 1.0 — the grid every asset is aligned onto.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "fine.tif")
    Dataset.from_array(
        np.arange(16, dtype="float32").reshape(4, 4),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def coarse_rasters(tmp_path):
    """Two 2x2 rasters at cell size 2.0 over the fine raster's extent.

    Both hold the same four discrete values, so the only thing that can make
    the two bands differ is the resampling method applied to each.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        tuple[str, str]: The `B04` (continuous) and `SCL` (categorical) paths.
    """
    paths = []
    for name in ("B04", "SCL"):
        path = str(tmp_path / f"{name}.tif")
        Dataset.from_array(
            np.array([[0.0, 10.0], [20.0, 30.0]], dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=2.0, epsg=4326),
        ).to_file(path)
        paths.append(path)
    return paths[0], paths[1]


@pytest.fixture
def mixed_item(fine_raster, coarse_rasters):
    """An item whose `grid` asset is fine and whose `B04` / `SCL` are coarse.

    The fine asset comes first in every request below, so it defines the
    output grid and both coarse assets have to be resampled onto it — which is
    what makes a per-band method observable.

    Args:
        fine_raster: The fine 4x4 raster.
        coarse_rasters: The two coarse 2x2 rasters.

    Returns:
        dict: A STAC item with three same-extent assets.
    """
    b04, scl = coarse_rasters
    return {
        "type": "Feature",
        "id": "mixed",
        "bbox": [0.0, 0.0, 4.0, 4.0],
        "properties": {"datetime": "2023-06-01T00:00:00Z"},
        "assets": {
            "grid": {"href": fine_raster},
            "B04": {"href": b04, "eo:bands": [{"name": "red"}]},
            "SCL": {"href": scl},
        },
        "stac_extensions": [],
    }


def _band(collection, index):
    """Return one band of the cube's single timestep as a float array.

    Args:
        collection: The built :class:`DatasetCollection`.
        index: The band index to read.

    Returns:
        np.ndarray: The band's pixels as float64.
    """
    return np.asarray(collection.iloc(0).read_array(band=index), dtype="float64")


class TestFromStacResampling:
    """Tests for `from_stac(resampling=...)` in the multi-asset path."""

    def test_default_is_nearest_for_every_band(self, mixed_item):
        """Without `resampling` both coarse bands stay on nearest neighbour.

        Test scenario:
            Three assets stacked onto the fine grid with today's defaults.
        """
        collection = from_stac([mixed_item], asset=["grid", "B04", "SCL"])
        for index in (1, 2):
            values = _band(collection, index)
            assert set(np.unique(values)) <= _DISCRETE, (
                f"band {index} must keep the discrete source values by default, "
                f"got {np.unique(values)}"
            )

    def test_a_per_band_mapping_is_honoured_band_by_band(self, mixed_item):
        """One band interpolates while the categorical one stays on nearest.

        Test scenario:
            `resampling={"B04": "bilinear"}` — `SCL` is left to the default, so
            the two bands resample the same source values differently.
        """
        collection = from_stac(
            [mixed_item],
            asset=["grid", "B04", "SCL"],
            resampling={"B04": "bilinear", "SCL": "nearest"},
        )
        interpolated = _band(collection, 1)
        categorical = _band(collection, 2)
        assert not set(np.unique(interpolated)) <= _DISCRETE, (
            f"B04 must be interpolated, got {np.unique(interpolated)}"
        )
        assert set(np.unique(categorical)) <= _DISCRETE, (
            f"SCL must stay on nearest, got {np.unique(categorical)}"
        )

    def test_a_scalar_method_applies_to_every_band(self, mixed_item):
        """A bare method name resamples every mismatched band with it.

        Test scenario:
            `resampling="bilinear"` over the same three assets.
        """
        collection = from_stac(
            [mixed_item], asset=["grid", "B04", "SCL"], resampling="bilinear"
        )
        for index in (1, 2):
            values = _band(collection, index)
            assert not set(np.unique(values)) <= _DISCRETE, (
                f"band {index} must be interpolated, got {np.unique(values)}"
            )

    def test_the_output_stays_on_the_first_assets_grid(self, mixed_item):
        """Choosing a method does not move the output off the first asset's grid.

        Test scenario:
            A bilinear build compared against the fine raster's shape.
        """
        collection = from_stac(
            [mixed_item], asset=["grid", "B04", "SCL"], resampling="bilinear"
        )
        timestep = collection.iloc(0)
        assert (timestep.rows, timestep.columns) == (4, 4), (
            f"the cube must stay on the fine 4x4 grid, got "
            f"{timestep.rows}x{timestep.columns}"
        )
        assert timestep.cell_size == pytest.approx(1.0), (
            f"the cell size must stay the first asset's, got {timestep.cell_size}"
        )

    def test_the_mapping_is_keyed_by_asset_key_under_renaming(self, mixed_item):
        """An asset-keyed mapping still applies when `eo:bands` renames the band.

        Test scenario:
            `eo_band_names=True` renames `B04` to `red`, while `resampling`
            is still keyed by `B04`.
        """
        collection = from_stac(
            [mixed_item],
            asset=["grid", "B04", "SCL"],
            resampling={"B04": "bilinear"},
            eo_band_names=True,
        )
        assert collection.iloc(0).band_names == ["grid", "red", "SCL"], (
            f"expected the renamed band, got {collection.iloc(0).band_names}"
        )
        assert not set(np.unique(_band(collection, 1))) <= _DISCRETE, (
            "the asset-keyed method must survive the rename, got "
            f"{np.unique(_band(collection, 1))}"
        )

    def test_an_unknown_method_raises(self, mixed_item):
        """A misspelled method is reported, not passed to GDAL.

        Test scenario:
            `resampling="sinc"`.
        """
        with pytest.raises(ValueError, match="does not exist"):
            from_stac([mixed_item], asset=["grid", "B04"], resampling="sinc")

    def test_an_unknown_band_key_raises(self, mixed_item):
        """A mapping key that names no requested asset is reported.

        Test scenario:
            `resampling={"B08": "bilinear"}` while only `grid` / `B04` are read.
        """
        with pytest.raises(ValueError, match="do not name any band"):
            from_stac(
                [mixed_item], asset=["grid", "B04"], resampling={"B08": "bilinear"}
            )

    def test_rejected_with_align_false(self, fine_raster):
        """`align=False` cannot resample anything, so a method is refused.

        Test scenario:
            Two same-grid assets built with `align=False` and a method.
        """
        item = {
            "type": "Feature",
            "id": "same-grid",
            "bbox": [0.0, 0.0, 4.0, 4.0],
            "properties": {"datetime": "2023-06-01T00:00:00Z"},
            "assets": {"a": {"href": fine_raster}, "b": {"href": fine_raster}},
            "stac_extensions": [],
        }
        with pytest.raises(ValueError, match="resampling only applies"):
            from_stac([item], asset=["a", "b"], align=False, resampling="bilinear")

    def test_rejected_in_single_asset_mode(self, three_local_items):
        """A single-asset cube has no band axis to resample onto.

        Test scenario:
            `asset="data"` with a method.
        """
        with pytest.raises(ValueError, match="resampling only applies"):
            from_stac(three_local_items, asset="data", resampling="bilinear")

    def test_rejected_in_grouped_mode(self, three_local_items):
        """A grouped mosaic refuses the option too.

        Test scenario:
            `groupby="orbit"` with a method.
        """
        with pytest.raises(ValueError, match="resampling only applies"):
            from_stac(
                three_local_items,
                asset="data",
                groupby="orbit",
                resampling="bilinear",
            )


class TestFromBandFilesResampling:
    """Tests for `Dataset.from_band_files(resampling=...)`."""

    def test_none_matches_an_explicit_nearest(self, fine_raster, coarse_rasters):
        """The default stays byte-identical to an explicit nearest build.

        Test scenario:
            The same stack built with `resampling=None` and with `"nearest"`.
        """
        b04, _scl = coarse_rasters
        default = Dataset.from_band_files([fine_raster, b04], align=True)
        explicit = Dataset.from_band_files(
            [fine_raster, b04], align=True, resampling="nearest"
        )
        assert np.array_equal(
            default.read_array(band=1), explicit.read_array(band=1)
        ), "resampling=None must behave exactly like nearest neighbour"

    def test_a_scalar_method_resamples_every_band(self, fine_raster, coarse_rasters):
        """A bare method name is applied to every mismatched input.

        Test scenario:
            A fine band plus a coarse one, stacked with `"bilinear"`.
        """
        b04, _scl = coarse_rasters
        stacked = Dataset.from_band_files(
            [fine_raster, b04], align=True, resampling="bilinear"
        )
        values = np.asarray(stacked.read_array(band=1), dtype="float64")
        assert not set(np.unique(values)) <= _DISCRETE, (
            f"the coarse band must be interpolated, got {np.unique(values)}"
        )

    def test_a_mapping_is_matched_against_the_band_names(
        self, fine_raster, coarse_rasters
    ):
        """Per-band methods are keyed by the output band names.

        Test scenario:
            Two identical coarse bands, one named band interpolated and the
            other left on the default.
        """
        b04, scl = coarse_rasters
        stacked = Dataset.from_band_files(
            [fine_raster, b04, scl],
            band_names=["grid", "continuous", "categorical"],
            align=True,
            resampling={"continuous": "bilinear"},
        )
        continuous = np.asarray(stacked.read_array(band=1), dtype="float64")
        categorical = np.asarray(stacked.read_array(band=2), dtype="float64")
        assert not set(np.unique(continuous)) <= _DISCRETE, (
            f"the mapped band must be interpolated, got {np.unique(continuous)}"
        )
        assert set(np.unique(categorical)) <= _DISCRETE, (
            f"the unmapped band must stay on nearest, got {np.unique(categorical)}"
        )

    def test_derived_band_names_are_valid_mapping_keys(
        self, fine_raster, coarse_rasters
    ):
        """A mapping may use the names derived from the file names.

        Test scenario:
            No `band_names` given, so `B04.tif` becomes the band name `B04`.
        """
        b04, _scl = coarse_rasters
        stacked = Dataset.from_band_files(
            [fine_raster, b04], align=True, resampling={"B04": "bilinear"}
        )
        values = np.asarray(stacked.read_array(band=1), dtype="float64")
        assert stacked.band_names == ["fine", "B04"], (
            f"expected the derived names, got {stacked.band_names}"
        )
        assert not set(np.unique(values)) <= _DISCRETE, (
            f"the mapped band must be interpolated, got {np.unique(values)}"
        )

    def test_align_false_refuses_a_method(self, fine_raster):
        """Without `align=True` nothing is resampled, so a method is refused.

        Test scenario:
            Two same-grid files stacked with `align=False`.
        """
        with pytest.raises(ValueError, match="resampling only applies"):
            Dataset.from_band_files(
                [fine_raster, fine_raster], align=False, resampling="bilinear"
            )

    def test_an_unknown_method_raises(self, fine_raster, coarse_rasters):
        """An unsupported algorithm name is reported up front.

        Test scenario:
            `resampling="sinc"`.
        """
        b04, _scl = coarse_rasters
        with pytest.raises(ValueError, match="does not exist"):
            Dataset.from_band_files([fine_raster, b04], align=True, resampling="sinc")

    def test_an_unknown_method_in_a_mapping_raises(self, fine_raster, coarse_rasters):
        """A mapping's values are validated too.

        Test scenario:
            A valid band key with an invalid method.
        """
        b04, _scl = coarse_rasters
        with pytest.raises(ValueError, match="does not exist"):
            Dataset.from_band_files(
                [fine_raster, b04], align=True, resampling={"B04": "nope"}
            )

    def test_an_unknown_band_key_raises(self, fine_raster, coarse_rasters):
        """A key naming no stacked band is reported with the valid names.

        Test scenario:
            `resampling={"missing": "bilinear"}`.
        """
        b04, _scl = coarse_rasters
        with pytest.raises(ValueError, match="do not name any band"):
            Dataset.from_band_files(
                [fine_raster, b04], align=True, resampling={"missing": "bilinear"}
            )

    def test_an_unusable_spec_type_raises(self, fine_raster, coarse_rasters):
        """A method spec that is neither a string, a mapping nor `None` raises.

        Test scenario:
            `resampling=["bilinear"]`.
        """
        b04, _scl = coarse_rasters
        with pytest.raises(TypeError, match="resampling must be"):
            Dataset.from_band_files(
                [fine_raster, b04], align=True, resampling=["bilinear"]
            )
