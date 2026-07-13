"""Tests for the expanded resampling-method registry and its resolver.

Covers `pyramids.base._utils.INTERPOLATION_METHODS` (the full GDAL ``GRA_*``
warp set with rasterio-style snake_case names) and
`pyramids.base._utils.resolve_resampling` (case/whitespace-insensitive name ->
constant resolution), plus the wiring through `Dataset.resample` and
`Dataset.to_crs`.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._utils import INTERPOLATION_METHODS, resolve_resampling
from pyramids.dataset import Dataset
from pyramids.dataset.engines.cog import _RESAMPLING_ALG
from pyramids.dataset.merge import merge_rasters

pytestmark = pytest.mark.core

EXPECTED_BASE_METHODS = [
    "nearest neighbor",
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
]


def _checkerboard_dataset(size: int = 8) -> Dataset:
    """Build a small checkerboard raster (values 0/100) for method-sensitivity tests.

    Args:
        size: Number of rows/cols of the square raster.

    Returns:
        Dataset: A single-band float32 EPSG:4326 dataset with a 0/100 checkerboard.
    """
    arr = np.indices((size, size)).sum(axis=0) % 2 * 100.0
    return Dataset.create_from_array(
        arr.astype("float32"), top_left_corner=(0, 0), cell_size=1.0, epsg=4326
    )


class TestInterpolationMethodsRegistry:
    """Tests for the INTERPOLATION_METHODS mapping content."""

    @pytest.mark.parametrize("name", EXPECTED_BASE_METHODS)
    def test_base_methods_present(self, name):
        """Every always-available GDAL warp algorithm is registered.

        Args:
            name: A method name that must exist on any supported GDAL.

        Test scenario:
            The registry contains the full base ``GRA_*`` set with
            rasterio-style snake_case names plus the historical
            ``"nearest neighbor"`` alias.
        """
        assert name in INTERPOLATION_METHODS, f"{name!r} missing from registry"

    def test_legacy_alias_maps_to_nearest(self):
        """``"nearest neighbor"`` and ``"nearest"`` resolve to the same constant.

        Test scenario:
            The historical pyramids name stays as an alias so existing
            callers keep working unchanged.
        """
        assert (
            INTERPOLATION_METHODS["nearest neighbor"]
            == INTERPOLATION_METHODS["nearest"]
            == gdal.GRA_NearestNeighbour
        ), "legacy alias must equal the snake_case name"

    def test_version_guarded_methods_match_gdal(self):
        """``sum``/``rms`` are present exactly when the GDAL build has them.

        Test scenario:
            The registry never references a missing constant (import-safe on
            older GDAL) and never silently drops one that exists.
        """
        assert ("sum" in INTERPOLATION_METHODS) == hasattr(
            gdal, "GRA_Sum"
        ), "'sum' registration must track gdal.GRA_Sum availability"
        assert ("rms" in INTERPOLATION_METHODS) == hasattr(
            gdal, "GRA_RMS"
        ), "'rms' registration must track gdal.GRA_RMS availability"


class TestResolveResampling:
    """Tests for the resolve_resampling name resolver."""

    def test_case_and_whitespace_insensitive(self):
        """Names are normalised before lookup.

        Test scenario:
            ``" Lanczos "`` resolves despite case and padding.
        """
        assert (
            resolve_resampling(" Lanczos ") == gdal.GRA_Lanczos
        ), "resolver must normalise case/whitespace"

    def test_unknown_method_raises_with_valid_names(self):
        """An unsupported name raises ValueError listing valid names.

        Test scenario:
            ``"sinc"`` is not a GDAL warp algorithm; the error message must
            name at least one valid method so it is self-documenting.
        """
        with pytest.raises(ValueError, match="does not exist") as exc:
            resolve_resampling("sinc")
        assert "bilinear" in str(
            exc.value
        ), f"error should list valid methods, got: {exc.value}"

    def test_version_gated_method_names_the_gdal_requirement(self, monkeypatch):
        """A version-gated method missing on the build names the GDAL version (N4).

        Test scenario:
            Simulate an older GDAL without ``GRA_Sum`` by removing ``"sum"`` from
            the registry; resolving it must mention the GDAL version requirement,
            not the generic "does not exist" listing.
        """
        monkeypatch.delitem(INTERPOLATION_METHODS, "sum", raising=False)
        with pytest.raises(ValueError, match="requires GDAL >= 3.1") as exc:
            resolve_resampling("sum")
        assert "GRA_Sum" in str(
            exc.value
        ), f"message should name the constant: {exc.value}"

    @pytest.mark.parametrize("bad_method", [3, None, b"bilinear"])
    def test_non_string_raises_type_error(self, bad_method):
        """A non-string method raises TypeError.

        Args:
            bad_method: Non-string value a caller might pass by mistake.

        Test scenario:
            An int (e.g. a raw GDAL constant), None, and a bytes name are
            all rejected explicitly.
        """
        with pytest.raises(TypeError, match="must be a string"):
            resolve_resampling(bad_method)


class TestResampleAllMethods:
    """End-to-end: every registered method works through Dataset.resample."""

    @pytest.mark.parametrize("method", sorted(INTERPOLATION_METHODS))
    def test_resample_runs_and_shapes(self, method):
        """Each method resamples an 8x8 raster to 2x cell size without error.

        Args:
            method: Registry method name under test.

        Test scenario:
            Doubling the cell size halves rows/cols for every algorithm; no
            method raises and the output grid is the expected 4x4.
        """
        src = _checkerboard_dataset(8)
        dst = src.resample(2.0, method=method)
        assert (dst.rows, dst.columns) == (
            4,
            4,
        ), f"{method}: expected 4x4 output, got {(dst.rows, dst.columns)}"

    def test_average_differs_from_nearest_on_checkerboard(self):
        """``average`` aggregates 0/100 cells; ``nearest`` picks one of them.

        Test scenario:
            Downsampling a checkerboard 2x: averaging mixes the two values
            (every output cell is 50), while nearest keeps 0 or 100 — proving
            the method string actually reaches GDAL.
        """
        src = _checkerboard_dataset(8)
        avg = src.resample(2.0, method="average").read_array()
        near = src.resample(2.0, method="nearest").read_array()
        assert np.allclose(
            avg, 50.0
        ), f"average of 0/100 checkerboard must be 50, got {avg}"
        assert set(np.unique(near)) <= {
            0.0,
            100.0,
        }, f"nearest must keep original values, got {np.unique(near)}"

    def test_to_crs_accepts_new_method(self):
        """``to_crs`` accepts an expanded method name (``average``).

        Test scenario:
            Reprojecting EPSG:4326 -> EPSG:3857 with ``method="average"``
            succeeds, proving the resolver is wired into the warp path.
        """
        src = _checkerboard_dataset(8)
        dst = src.to_crs(3857, method="average")
        assert dst.epsg == 3857, f"expected EPSG:3857 output, got {dst.epsg}"

    @pytest.mark.parametrize(
        ("method", "vrt_alg"),
        [
            ("nearest", "NearestNeighbour"),
            ("average", "Average"),
            ("bilinear", "Bilinear"),
        ],
    )
    def test_to_crs_default_path_applies_method(self, method, vrt_alg):
        """The default (non-aligned) warp branch honours ``method``.

        Args:
            method: Registry method name passed to ``to_crs``.
            vrt_alg: GDAL's serialized algorithm name expected in the VRT.

        Test scenario:
            ``to_crs`` without ``maintain_alignment`` builds a warped VRT; its
            serialized warp options must carry the requested algorithm rather
            than GDAL's nearest default — guarding against the method being
            validated but silently dropped before :func:`gdal.Warp`.
        """
        src = _checkerboard_dataset(8)
        warped = src.to_crs(3857, method=method)
        xml = warped.raster.GetMetadata("xml:VRT")
        xml = xml[0] if isinstance(xml, (list, tuple)) else str(xml)
        assert f"<ResampleAlg>{vrt_alg}</ResampleAlg>" in xml, (
            f"warped VRT must record {vrt_alg!r}; warp options were: "
            f"{[line.strip() for line in xml.splitlines() if 'ResampleAlg' in line]}"
        )


class TestMergeRastersNewMethods:
    """merge_rasters accepts the expanded resampling names on reprojection."""

    def test_reprojecting_merge_with_average(self, tmp_path):
        """A cross-CRS merge with ``resampling="average"`` succeeds.

        Test scenario:
            Two overlapping EPSG:4326 tiles merged onto EPSG:3857 with the
            newly-registered ``average`` algorithm produce a readable output —
            proving the resolver feeds gdal.Warp inside merge_rasters.
        """
        a = Dataset.create_from_array(
            np.full((4, 4), 10.0, dtype="float32"),
            top_left_corner=(0, 4),
            cell_size=1.0,
            epsg=4326,
        )
        b = Dataset.create_from_array(
            np.full((4, 4), 20.0, dtype="float32"),
            top_left_corner=(2, 4),
            cell_size=1.0,
            epsg=4326,
        )
        pa, pb = str(tmp_path / "a.tif"), str(tmp_path / "b.tif")
        a.to_file(pa)
        b.to_file(pb)
        out = tmp_path / "merged.tif"
        merge_rasters([pa, pb], out, dst_crs=3857, resampling="average")
        merged = Dataset.read_file(str(out))
        assert merged.epsg == 3857, f"expected EPSG:3857 output, got {merged.epsg}"
        assert merged.read_array().size > 0, "merged raster should contain data"


class TestCogEngineResamplingNames:
    """COG-engine decimated reads honour the aligned/normalised method names."""

    @pytest.fixture
    def ramp(self) -> Dataset:
        """A 64x64 float32 ramp dataset on EPSG:4326.

        Returns:
            Dataset: In-memory single-band dataset, value == row*64 + col.
        """
        arr = np.arange(64 * 64, dtype="float32").reshape(64, 64)
        return Dataset.create_from_array(
            arr, top_left_corner=(0, 64), cell_size=1.0, epsg=4326
        )

    def test_preview_is_case_insensitive(self, ramp):
        """``preview(resampling=" Average ")`` normalises and succeeds.

        Test scenario:
            The COG engine lowercases/strips the resampling name before the
            registry lookup, mirroring resolve_resampling's behaviour.
        """
        thumb = ramp.preview(max_size=16, resampling=" Average ")
        assert max(thumb.shape) == 16, f"expected 16px long edge, got {thumb.shape}"

    def test_preview_accepts_cubic_spline_alias(self, ramp):
        """``cubic_spline`` (snake_case) works alongside legacy ``cubicspline``.

        Test scenario:
            Both spellings resolve to GRIORA_CubicSpline; the decimated read
            returns identical arrays for the two names.
        """
        legacy = ramp.preview(max_size=16, resampling="cubicspline")
        snake = ramp.preview(max_size=16, resampling="cubic_spline")
        np.testing.assert_array_equal(
            legacy, snake, err_msg="alias must select the same GRIORA algorithm"
        )

    def test_read_part_rejects_unknown_resampling(self, ramp):
        """``read_part`` still rejects unknown names after normalisation.

        Test scenario:
            ``"Sinc"`` is lowercased to ``"sinc"`` which is not registered;
            the error lists the valid choices.
        """
        bbox = tuple(ramp.bbox)
        with pytest.raises(ValueError, match="unknown resampling") as exc:
            ramp.read_part(bbox, bbox_crs=4326, resampling="Sinc")
        assert "bilinear" in str(
            exc.value
        ), f"error should list valid methods, got: {exc.value}"

    def test_read_part_rejects_non_string_resampling(self, ramp):
        """``read_part`` raises TypeError for a non-string resampling.

        Test scenario:
            Passing a raw GDAL constant (int) must produce a clear TypeError,
            not an AttributeError from the normalisation.
        """
        bbox = tuple(ramp.bbox)
        with pytest.raises(TypeError, match="must be a string"):
            ramp.read_part(bbox, bbox_crs=4326, resampling=1)

    def test_preview_rejects_non_string_resampling(self, ramp):
        """``preview`` raises TypeError for a non-string resampling.

        Test scenario:
            Same guard as read_part on the thumbnail path.
        """
        with pytest.raises(TypeError, match="must be a string"):
            ramp.preview(max_size=16, resampling=2)

    def test_griora_guards_match_gdal(self):
        """``gauss``/``rms`` registration tracks the GDAL build's GRIORA set.

        Test scenario:
            The guarded entries are present exactly when the constant exists,
            so imports never break on an older GDAL.
        """
        assert ("gauss" in _RESAMPLING_ALG) == hasattr(
            gdal, "GRIORA_Gauss"
        ), "'gauss' registration must track gdal.GRIORA_Gauss availability"
        assert ("rms" in _RESAMPLING_ALG) == hasattr(
            gdal, "GRIORA_RMS"
        ), "'rms' registration must track gdal.GRIORA_RMS availability"
