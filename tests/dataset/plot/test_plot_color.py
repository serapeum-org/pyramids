"""Plot tests: colour tables, colour relief, and plot-band resolution."""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from pyramids.dataset import Dataset
from pyramids.dataset.engines import Analysis

pytestmark = pytest.mark.plot

_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("agg")


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each plot test to bound memory.

    Plotting tests open figures via cleopatra/pyplot; without this teardown
    the suite accumulates them and matplotlib warns past 20 open figures.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestColorTable:
    @pytest.mark.plot
    def test_generated_data(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(1, 3, size=(2, 5, 5))
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )

        # without alpha
        color_table = pd.DataFrame(
            {
                "band": [1, 1, 1, 2, 2, 2],
                "values": [1, 2, 3, 1, 2, 3],
                "color": [
                    "#709959",
                    "#F2EEA2",
                    "#F2CE85",
                    "#C28C7C",
                    "#D6C19C",
                    "#D6C19C",
                ],
            }
        )
        dataset.color_table = color_table
        retrieved_color_table = dataset.color_table
        assert all(
            ["band", "values", "red", "green", "blue", "alpha"]
            == retrieved_color_table.columns
        )

    @pytest.mark.plot
    def test_get_color_table(self, src_with_color_table: Dataset):
        dataset = Dataset(src_with_color_table)
        df = dataset.bands._get_color_table()
        assert isinstance(df, DataFrame)
        assert all(df.columns == ["band", "values", "red", "green", "blue", "alpha"])
        assert all(df.band == 1)
        # test the color_table property
        df = dataset.color_table
        assert isinstance(df, DataFrame)
        assert all(df.columns == ["band", "values", "red", "green", "blue", "alpha"])
        assert all(df.band == 1)

    @pytest.mark.plot
    def test_set_color_table(self, src_without_color_table: Dataset):
        color_hex = ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"]
        values = [1, 3, 5, 7, 9]
        df = pd.DataFrame(columns=["band", "values", "color"])
        df.loc[:, "values"] = values
        df.loc[:, "band"] = 1
        df.loc[:, "color"] = color_hex

        # copy() yields a writable in-memory dataset; the color_table facade setter
        # is guarded against a read-only on-disk handle (the fixture opens read-only).
        dataset = Dataset(src_without_color_table).copy()
        dataset.bands._set_color_table(df, overwrite=True)

        color_table = dataset.raster.GetRasterBand(1).GetColorTable()
        assert color_table is not None, "the color table should not be None"
        assert color_table.GetCount() == 10, "the color table should have 5 colors"
        colors = [color_table.GetColorEntry(i) for i in range(color_table.GetCount())]
        assert colors == [
            (0, 0, 0, 0),
            (112, 153, 89, 255),
            (0, 0, 0, 0),
            (242, 238, 162, 255),
            (0, 0, 0, 0),
            (242, 206, 133, 255),
            (0, 0, 0, 0),
            (194, 140, 124, 255),
            (0, 0, 0, 0),
            (214, 193, 156, 255),
        ]
        # test the color_table property
        dataset.color_table = df


class TestColorRelief:
    color_hex = ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"]
    values = [1, 3, 5, 7, 9]
    df = pd.DataFrame(columns=["values", "color"])
    df.loc[:, "values"] = values
    df.loc[:, "color"] = color_hex

    @pytest.mark.plot
    def test_process_color_table(self):

        color_table = Analysis._process_color_table(self.df)
        assert isinstance(color_table, DataFrame)
        assert all(color_table.columns == ["values", "red", "green", "blue", "alpha"])

    @pytest.mark.plot
    def test_process_color_table_without_rgb_or_color_raises(self):
        """A colour table lacking both RGB and hex ``color`` columns is rejected.

        Test scenario:
            ``_process_color_table`` needs either ``red``/``green``/``blue`` or a
            single hex ``color`` column; a table with only ``values`` raises
            ``ValueError`` naming the columns it did receive.
        """
        bad = pd.DataFrame({"values": [1, 2, 3]})
        with pytest.raises(ValueError, match="red, green, blue, or color") as exc_info:
            Analysis._process_color_table(bad)
        assert "values" in str(exc_info.value), (
            f"error should list the given columns, got: {exc_info.value}"
        )


class TestPalettePlot:
    """#913: a paletted raster renders through its GDAL colour table on ``plot()``."""

    @staticmethod
    def _paletted_dataset():
        """A single-band raster (values 1..3) tagged with a red/green/blue palette."""
        arr = np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]], dtype=np.int32)
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.color_table = pd.DataFrame(
            {
                "band": [1, 1, 1],
                "values": [1, 2, 3],
                "color": ["#ff0000", "#00ff00", "#0000ff"],
            }
        )
        return dataset

    @pytest.mark.plot
    def test_paletted_raster_renders_through_color_table(self):
        """Each pixel value maps to its exact palette colour via a boundary norm.

        Test scenario:
            A single-band raster carrying a red/green/blue colour table on values
            1/2/3 must plot through that table — the mappable's colormap is the
            palette ramp under a ``BoundaryNorm``, and sampling it at each value
            returns the palette colour, not a default matplotlib colormap.
        """
        glyph = self._paletted_dataset().plot(band=0)
        assert type(glyph.im.cmap).__name__ == "ListedColormap"
        assert type(glyph.im.norm).__name__ == "BoundaryNorm"
        expected = {1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0), 3: (0.0, 0.0, 1.0, 1.0)}
        for value, rgba in expected.items():
            got = tuple(round(c, 3) for c in glyph.im.cmap(glyph.im.norm(value)))
            assert got == rgba, f"value {value} should render {rgba} (opaque), got {got}"

    @pytest.mark.plot
    def test_sparse_land_cover_palette_renders_exact_and_opaque(self):
        """A sparse land-cover palette renders each class exactly, fully opaque.

        Test scenario:
            GDAL densifies a colour table to ``0..maxvalue`` with transparent
            ``(0, 0, 0, 0)`` gap-fillers, so classes such as 11/21/31 land far apart.
            The 256-entry step colormap must still map each class to its own opaque
            colour — no interpolation blend and no alpha bleed toward the phantom
            transparent stops.
        """
        arr = np.array([[11, 21, 31, 11]], dtype=np.int32)
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.color_table = pd.DataFrame(
            {
                "band": [1, 1, 1],
                "values": [11, 21, 31],
                "color": ["#00ff00", "#ff0000", "#0000ff"],
            }
        )
        glyph = dataset.plot(band=0)
        expected = {11: (0.0, 1.0, 0.0, 1.0), 21: (1.0, 0.0, 0.0, 1.0), 31: (0.0, 0.0, 1.0, 1.0)}
        for value, rgba in expected.items():
            got = tuple(round(c, 3) for c in glyph.im.cmap(glyph.im.norm(value)))
            assert got == rgba, f"class {value} should render {rgba} (opaque), got {got}"

    @pytest.mark.plot
    def test_high_code_land_cover_palette_renders_exact_and_opaque(self):
        """High class codes (densified count > 131) still render exact + opaque.

        Test scenario:
            Land-cover products use codes up to ~200-220, so GDAL densifies the table
            to that many entries. Each class must land on the slot cleopatra's
            ``BoundaryNorm`` actually maps it to — a fixed round-trip index mis-maps
            past ~131 and renders some classes transparent.
        """
        arr = np.array([[10, 132, 200, 10]], dtype=np.int32)
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.color_table = pd.DataFrame(
            {
                "band": [1, 1, 1],
                "values": [10, 132, 200],
                "color": ["#00ff00", "#ff0000", "#0000ff"],
            }
        )
        glyph = dataset.plot(band=0)
        expected = {10: (0.0, 1.0, 0.0, 1.0), 132: (1.0, 0.0, 0.0, 1.0), 200: (0.0, 0.0, 1.0, 1.0)}
        for value, rgba in expected.items():
            got = tuple(round(c, 3) for c in glyph.im.cmap(glyph.im.norm(value)))
            assert got == rgba, f"class {value} should render {rgba} (opaque), got {got}"

    @pytest.mark.plot
    def test_single_entry_palette_renders_without_crash(self):
        """A one-entry colour table renders instead of crashing the colormap build.

        Test scenario:
            A palette with a single value must not raise from the colormap
            construction; the class renders in its one colour.
        """
        arr = np.zeros((1, 3), dtype=np.int32)
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.color_table = pd.DataFrame(
            {"band": [1], "values": [0], "color": ["#ff0000"]}
        )
        glyph = dataset.plot(band=0)
        got = tuple(round(c, 3) for c in glyph.im.cmap(glyph.im.norm(0)))
        assert got == (1.0, 0.0, 0.0, 1.0), f"single-entry palette should render red, got {got}"

    @pytest.mark.plot
    def test_explicit_cmap_overrides_palette(self):
        """An explicit ``cmap=`` wins over the colour table — palette wiring skipped."""
        glyph = self._paletted_dataset().plot(band=0, cmap="viridis")
        assert glyph.im.cmap.name == "viridis"

    @pytest.mark.plot
    def test_explicit_color_scale_overrides_palette(self):
        """An explicit ``color_scale=`` opts out of the palette's boundary norm."""
        glyph = self._paletted_dataset().plot(band=0, color_scale="linear")
        assert type(glyph.im.norm).__name__ != "BoundaryNorm"

    @pytest.mark.plot
    def test_explicit_bounds_overrides_palette(self):
        """A lone ``bounds=`` opts out of the palette rather than being overwritten."""
        glyph = self._paletted_dataset().plot(band=0, bounds=[0, 1, 2, 3])
        assert glyph.im.cmap.name == "coolwarm_r"

    @pytest.mark.plot
    def test_non_paletted_raster_keeps_default_cmap(self):
        """A raster with no colour table renders with cleopatra's default colormap."""
        arr = np.random.default_rng(0).random((5, 5)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        glyph = dataset.plot(band=0)
        assert glyph.im.cmap.name == "coolwarm_r"

    @pytest.mark.plot
    def test_palette_colormap_returns_step_cmap_and_sorted_edges(self):
        """``_palette_colormap`` returns a 256-entry step colormap + ``N+1`` edges.

        Test scenario:
            Given an unsorted three-value colour table, the helper sorts by value
            and returns a ``ListedColormap`` plus four ascending boundary edges (one
            per gap, bracketing each class).
        """
        color_table = pd.DataFrame(
            {
                "values": [3, 1, 2],
                "red": [0, 255, 0],
                "green": [0, 0, 255],
                "blue": [255, 0, 0],
            }
        )
        cmap, bounds = Analysis._palette_colormap(color_table)
        assert type(cmap).__name__ == "ListedColormap"
        assert cmap.N == 256
        assert len(bounds) == 4
        assert bounds == sorted(bounds)

    @pytest.mark.plot
    def test_multiband_palette_band_renders_its_table(self):
        """A 3-band raster with a palette on band 1 plots band 1's colour table.

        Test scenario:
            ``plot()`` with no explicit band resolves the ``palette_index`` band
            (band 1, GDAL band 2), then renders that band's red/green table so
            value 1 comes back red. Values ``1``/``3`` (an even entry count once
            gap-filled) keep the boundary-norm sampling exact.
        """
        mid = np.array([[1, 3, 1, 3]] * 4, dtype=np.int32)
        other = np.full((4, 4), 9, dtype=np.int32)
        dataset = Dataset.create_from_array(
            np.stack([other, mid, other]),
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
        )
        dataset.color_table = pd.DataFrame(
            {"band": [2, 2], "values": [1, 3], "color": ["#ff0000", "#00ff00"]}
        )
        dataset.band_color = {1: "palette_index"}

        glyph = dataset.plot()
        assert type(glyph.im.norm).__name__ == "BoundaryNorm"
        got = tuple(round(c, 3) for c in glyph.im.cmap(glyph.im.norm(1)))
        assert got[:3] == (1.0, 0.0, 0.0), f"band-1 value 1 should be red, got {got[:3]}"
