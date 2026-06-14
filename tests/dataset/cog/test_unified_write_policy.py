"""Tests for the unified COG write policy (Phase A: ARC-1/2/3/4).

Covers the single-owner write policy now living in
:meth:`pyramids.dataset.engines.cog.COG.to_cog`:

- the shared dtype helpers in :mod:`pyramids.base._utils`
  (:func:`resolve_cog_predictor`, :func:`default_cog_overview_resampling`,
  :func:`is_integer_gdal_dtype`);
- dtype-aware auto-resolution of ``predictor`` and ``overview_resampling``;
- the categorical guardrail firing only on caller-chosen averaging;
- the ``STATISTICS`` retry now owned by the engine
  (:meth:`COG._translate_with_statistics_retry`);
- the ``to_cog`` ≡ ``write_cog`` equivalence regression guard.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import FailedToSaveError
from pyramids.base._utils import (
    INTEGER_GDAL_DTYPES,
    default_cog_overview_resampling,
    is_integer_gdal_dtype,
    resolve_cog_predictor,
)
from pyramids.dataset import Dataset
from pyramids.dataset.cog import write_cog

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def float_dataset() -> Dataset:
    """A 64x64 Float32 Dataset on EPSG:4326 (continuous data).

    Returns:
        Dataset: An in-memory float32 dataset.
    """
    rng = np.random.default_rng(seed=1337)
    arr = (rng.random((64, 64)) * 100.0).astype("float32")
    return Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)


@pytest.fixture
def int_dataset() -> Dataset:
    """A 48x48 Int16 Dataset on EPSG:4326 (categorical-capable data).

    Returns:
        Dataset: An in-memory int16 dataset.
    """
    rng = np.random.default_rng(seed=7)
    arr = rng.integers(0, 50, size=(48, 48)).astype("int16")
    return Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)


def _read_predictor(path: str | Path) -> str:
    """Return the GDAL IMAGE_STRUCTURE PREDICTOR token of a raster.

    Args:
        path: Path to a raster readable by GDAL.

    Returns:
        The predictor token (e.g. ``"2"``/``"3"``), or ``""`` when absent.
    """
    ds = gdal.Open(str(path))
    pred = ds.GetMetadataItem("PREDICTOR", "IMAGE_STRUCTURE") or ""
    ds = None
    return pred


def _first_overview_decimation(path: str | Path) -> int:
    """Return the decimation factor of the first overview, or 0 if none.

    Args:
        path: Path to a raster readable by GDAL.

    Returns:
        ``round(full_width / overview_width)`` for the first overview, else 0.
    """
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    if band.GetOverviewCount() == 0:
        ds = None
        return 0
    ovr = band.GetOverview(0)
    dec = round(ds.RasterXSize / ovr.XSize)
    ds = None
    return dec


class TestResolveCogPredictor:
    """Tests for resolve_cog_predictor."""

    @pytest.mark.parametrize(
        "gdal_dtype",
        sorted(INTEGER_GDAL_DTYPES),
    )
    def test_integer_dtypes_map_to_2(self, gdal_dtype):
        """Every integer GDAL dtype resolves to predictor 2.

        Args:
            gdal_dtype: A GDAL integer data-type code.

        Test scenario:
            Horizontal differencing (predictor 2) is correct for integer data.
        """
        assert (
            resolve_cog_predictor(gdal_dtype) == 2
        ), f"integer dtype {gdal_dtype} should map to predictor 2"

    @pytest.mark.parametrize(
        "gdal_dtype",
        [gdal.GDT_Float32, gdal.GDT_Float64],
    )
    def test_float_dtypes_map_to_3(self, gdal_dtype):
        """Floating-point GDAL dtypes resolve to predictor 3.

        Args:
            gdal_dtype: A GDAL float data-type code.

        Test scenario:
            The floating-point predictor (3) is correct for continuous data.
        """
        assert (
            resolve_cog_predictor(gdal_dtype) == 3
        ), f"float dtype {gdal_dtype} should map to predictor 3"


class TestIsIntegerGdalDtype:
    """Tests for is_integer_gdal_dtype."""

    def test_integer_true(self):
        """An integer dtype returns True.

        Test scenario:
            GDT_Int16 is an integer type.
        """
        assert is_integer_gdal_dtype(gdal.GDT_Int16) is True, "Int16 is integer"

    def test_float_false(self):
        """A float dtype returns False.

        Test scenario:
            GDT_Float32 is not an integer type.
        """
        assert is_integer_gdal_dtype(gdal.GDT_Float32) is False, "Float32 not integer"


class TestDefaultCogOverviewResampling:
    """Tests for default_cog_overview_resampling."""

    def test_integer_is_mode(self):
        """An integer dtype with no colour table defaults to mode.

        Test scenario:
            Integer data is categorical-capable; averaging would corrupt it.
        """
        result = default_cog_overview_resampling(gdal.GDT_Byte, False)
        assert result == "mode", f"integer should default to 'mode', got {result!r}"

    def test_color_table_is_mode(self):
        """A float dtype with a colour table still defaults to mode.

        Test scenario:
            A palette signals categorical data regardless of storage dtype.
        """
        result = default_cog_overview_resampling(gdal.GDT_Float32, True)
        assert result == "mode", f"palette should default to 'mode', got {result!r}"

    def test_continuous_is_average(self):
        """A float dtype with no colour table defaults to average.

        Test scenario:
            Continuous data benefits from averaging overviews.
        """
        result = default_cog_overview_resampling(gdal.GDT_Float32, False)
        assert result == "average", f"float should default to 'average', got {result!r}"


class TestToCogPredictorResolution:
    """Tests for COG.to_cog dtype-aware predictor auto-resolution."""

    def test_float_predictor_auto_3(self, float_dataset, tmp_path):
        """to_cog auto-resolves predictor 3 for a float source.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            With predictor=None (default), a float COG carries PREDICTOR=3.
        """
        out = float_dataset.to_cog(tmp_path / "f.tif")
        assert _read_predictor(out) == "3", "float source should yield PREDICTOR=3"

    def test_int_predictor_auto_2(self, int_dataset, tmp_path):
        """to_cog auto-resolves predictor 2 for an integer source.

        Args:
            int_dataset: Fixture int16 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            With predictor=None (default), an int COG carries PREDICTOR=2.
        """
        out = int_dataset.to_cog(tmp_path / "i.tif")
        assert _read_predictor(out) == "2", "int source should yield PREDICTOR=2"

    def test_explicit_predictor_wins(self, float_dataset, tmp_path):
        """An explicit predictor overrides the dtype-aware default.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            Passing predictor=1 (none) disables the predictor on a float COG.
        """
        out = float_dataset.to_cog(tmp_path / "f1.tif", predictor=1)
        assert _read_predictor(out) in ("", "1"), "explicit predictor=1 must win"


class TestToCogResamplingGuardrail:
    """Tests for the categorical overview-resampling guardrail."""

    def test_default_on_integer_emits_no_warning(self, int_dataset, tmp_path):
        """The auto-resolved default on integer data emits NO warning.

        Args:
            int_dataset: Fixture int16 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            With overview_resampling=None the engine picks 'mode' for integer
            data, so the categorical guardrail must stay silent (this is the
            ARC-3 fix — write_cog previously forced 'average' and always warned).
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            int_dataset.to_cog(tmp_path / "i.tif")
        categorical = [w for w in caught if "categorical" in str(w.message).lower()]
        assert not categorical, f"default int write must not warn, got {categorical}"

    def test_explicit_average_on_integer_warns(self, int_dataset, tmp_path):
        """Explicitly requesting averaging on integer data warns.

        Args:
            int_dataset: Fixture int16 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            When the caller themselves passes overview_resampling='average' on
            integer data, the guardrail must fire.
        """
        with pytest.warns(UserWarning, match="categorical"):
            int_dataset.to_cog(tmp_path / "i_avg.tif", overview_resampling="average")

    def test_default_on_float_emits_no_warning(self, float_dataset, tmp_path):
        """The auto-resolved default on float data emits no categorical warning.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            A float COG written with defaults (which resolve to 'average') must
            produce a valid COG and emit no categorical warning. (Overviews are
            only generated when the raster exceeds the tile size, so a tiny
            fixture legitimately has none — the resampling *choice* is covered
            by TestDefaultCogOverviewResampling.)
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = float_dataset.to_cog(tmp_path / "f.tif")
        categorical = [w for w in caught if "categorical" in str(w.message).lower()]
        assert not categorical, "float default must not warn"
        assert (
            float_dataset.read_file(str(out)).validate_cog().is_valid
        ), "float default write must produce a valid COG"

    def test_large_float_builds_averaged_overviews(self, tmp_path):
        """A float raster larger than the tile size gets averaged overviews.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            With a 600x600 float source (> 512 blocksize) the default write
            builds at least one overview level, confirming overviews are
            generated when the geometry warrants them.
        """
        rng = np.random.default_rng(seed=99)
        arr = (rng.random((600, 600)) * 100.0).astype("float32")
        ds = Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)
        out = ds.to_cog(tmp_path / "big.tif")
        assert _first_overview_decimation(out) >= 2, "large float COG needs overviews"


class TestStatisticsRetry:
    """Tests for COG._translate_with_statistics_retry."""

    def test_retries_without_statistics_on_valid_pixels_error(
        self, float_dataset, tmp_path, monkeypatch
    ):
        """A 'no valid pixels' failure is retried once without STATISTICS.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            The first translate raises the GDAL statistics-sampling error; the
            engine must drop STATISTICS and retry, succeeding on the 2nd call.
            (This logic moved from the facade to the engine in ARC-4.)
        """
        from pyramids.dataset.engines import cog as cog_engine

        calls: list[dict] = []
        real_translate = cog_engine.translate_to_cog

        def fake_translate(src, path, options):
            calls.append(dict(options))
            if "STATISTICS" in options:
                raise RuntimeError("no valid pixels found in sampling")
            return real_translate(src, path, options)

        monkeypatch.setattr(cog_engine, "translate_to_cog", fake_translate)
        out = float_dataset.to_cog(tmp_path / "retry.tif")
        assert len(calls) == 2, f"expected initial call + one retry, got {len(calls)}"
        assert "STATISTICS" in calls[0], "first attempt should carry STATISTICS"
        assert "STATISTICS" not in calls[1], "retry should drop STATISTICS"
        assert Path(out).exists(), "retry should produce the output file"

    def test_retries_on_failed_to_save_error(
        self, float_dataset, tmp_path, monkeypatch
    ):
        """A FailedToSaveError wrapping the statistics error is also retried.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            translate_to_cog wraps CreateCopy RuntimeErrors into
            FailedToSaveError; the retry must catch that wrapped form too.
        """
        from pyramids.dataset.engines import cog as cog_engine

        calls: list[dict] = []
        real_translate = cog_engine.translate_to_cog

        def fake_translate(src, path, options):
            calls.append(dict(options))
            if "STATISTICS" in options:
                raise FailedToSaveError(
                    "GDAL COG CreateCopy failed: no valid pixels found in sampling"
                )
            return real_translate(src, path, options)

        monkeypatch.setattr(cog_engine, "translate_to_cog", fake_translate)
        float_dataset.to_cog(tmp_path / "retry2.tif")
        assert len(calls) == 2, f"expected initial call + one retry, got {len(calls)}"

    def test_unrelated_error_propagates(self, float_dataset, tmp_path, monkeypatch):
        """An unrelated error is not swallowed by the retry guard.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            A failure whose message is not about the statistics pass propagates
            unchanged after a single attempt.
        """
        from pyramids.dataset.engines import cog as cog_engine

        calls: list[dict] = []

        def fake_translate(src, path, options):
            calls.append(dict(options))
            raise RuntimeError("disk full")

        monkeypatch.setattr(cog_engine, "translate_to_cog", fake_translate)
        with pytest.raises(RuntimeError, match="disk full"):
            float_dataset.to_cog(tmp_path / "boom.tif")
        assert len(calls) == 1, "unrelated error must not trigger a retry"


class TestToCogWriteCogEquivalence:
    """ARC-1 regression guard: to_cog and write_cog agree on output."""

    @pytest.mark.parametrize("fixture_name", ["float_dataset", "int_dataset"])
    def test_same_predictor_and_overviews(self, fixture_name, tmp_path, request):
        """to_cog and write_cog produce equivalent COGs for the same source.

        Args:
            fixture_name: Name of the Dataset fixture to exercise.
            tmp_path: pytest temp directory.
            request: pytest request used to resolve the fixture by name.

        Test scenario:
            Writing the same dataset through ds.to_cog(...) and
            write_cog(ds, ...) must yield the same PREDICTOR, the same first
            overview decimation, and both must validate as COGs.
        """
        ds = request.getfixturevalue(fixture_name)
        p1 = tmp_path / "via_to_cog.tif"
        p2 = tmp_path / "via_write_cog.tif"

        ds.to_cog(p1)
        out2, report = write_cog(ds, p2)

        assert _read_predictor(p1) == _read_predictor(
            out2
        ), "predictor must match between to_cog and write_cog"
        assert _first_overview_decimation(p1) == _first_overview_decimation(
            out2
        ), "overview decimation must match between to_cog and write_cog"
        assert report is not None and report.is_valid, "write_cog output must validate"
        assert (
            ds.read_file(str(p1)).validate_cog().is_valid
        ), "to_cog output must validate"
