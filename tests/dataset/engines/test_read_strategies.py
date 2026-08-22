"""Tests for the ``read_array`` read-path strategies.

Covers the :class:`ReadStrategy` set that ``read_array`` dispatches to:
``matches`` selection, each path's window-dependent precondition guards (exact
exception type + message), delegation to the existing ``_*_read`` helpers, the
declared backend per path, the ``READ_STRATEGIES`` selection order that
reproduces the original ``if/elif`` ladder, and ``EagerRead``'s real read
branches (single/multi band, windowed, masked) against a live IO engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, Window
from pyramids.dataset.engines._read_request import ReadRequest
from pyramids.dataset.engines._read_strategies import (
    READ_STRATEGIES,
    BoundlessRead,
    DecimatedRead,
    EagerRead,
    LazyRead,
    ReadStrategy,
    ThreadsafeRead,
)
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


def make_request(**overrides) -> ReadRequest:
    """Build a valid ``ReadRequest`` with overridable option fields.

    Args:
        **overrides: Fields to replace on the valid full-eager-read baseline.

    Returns:
        ReadRequest: A constructed, matrix-valid request.
    """
    base = {
        "band": 0,
        "chunks": None,
        "lock": None,
        "out_shape": None,
        "resampling": "nearest",
        "boundless": False,
        "fill_value": None,
        "masked": False,
        "threadsafe": False,
    }
    base.update(overrides)
    return ReadRequest(**base)


def select(req: ReadRequest) -> ReadStrategy:
    """Return the first strategy in ``READ_STRATEGIES`` that matches ``req``."""
    return next(s for s in READ_STRATEGIES if s.matches(req))


@pytest.fixture(scope="function")
def single_band() -> Dataset:
    """A 6x6 float32 ramp, single band, nodata -9999, one cell set to nodata.

    Returns:
        Dataset: In-memory single-band dataset; value == row*6 + col, except
        cell (0, 0) which is the nodata value.
    """
    arr = np.arange(36, dtype="float32").reshape(6, 6)
    arr[0, 0] = -9999.0
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326, no_data_value=-9999.0
    )


@pytest.fixture(scope="function")
def multi_band() -> Dataset:
    """A 3-band, 4x5 float32 raster; value == band*1000 + row*5 + col.

    Returns:
        Dataset: In-memory three-band dataset on a unit grid.
    """
    bands = np.stack(
        [np.arange(4 * 5, dtype="float32").reshape(4, 5) + b * 1000 for b in range(3)],
        axis=0,
    )
    return Dataset.create_from_array(
        bands, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326
    )


class TestReadStrategyABC:
    """The abstract base class contract."""

    def test_cannot_instantiate_abstract_base(self):
        """``ReadStrategy`` is abstract and cannot be instantiated directly.

        Test scenario:
            The ABC leaves ``matches``/``read`` abstract, so construction raises.
        """
        with pytest.raises(TypeError, match="abstract"):
            ReadStrategy()  # type: ignore[abstract]


class TestLazyRead:
    """``LazyRead`` (the ``chunks=`` path)."""

    def test_backend_is_dask(self):
        """The lazy path declares the ``dask`` backend."""
        assert LazyRead().backend == "dask", "lazy reads must report the dask backend"

    @pytest.mark.parametrize(
        "chunks, expected", [(4, True), ("auto", True), (None, False)]
    )
    def test_matches_on_chunks(self, chunks, expected):
        """``matches`` is true exactly when a chunk spec was given.

        Args:
            chunks: The request's ``chunks`` value.
            expected: Whether ``LazyRead`` should claim the request.
        """
        req = make_request(chunks=chunks)
        assert LazyRead().matches(req) is expected, f"matches({chunks}) should be {expected}"

    def test_delegates_to_lazy_read_array(self, mocker):
        """A valid lazy read forwards band/chunks/lock/threadsafe to the helper.

        Test scenario:
            ``LazyRead.read`` calls ``io._lazy_read_array`` with the request's
            options and returns its result unchanged.
        """
        io = mocker.Mock()
        io._lazy_read_array.return_value = "lazy-array"
        req = make_request(band=2, chunks=8, lock="LK", threadsafe=True)
        result = LazyRead().read(io, req, None)
        assert result == "lazy-array", "the helper's return must pass through"
        io._lazy_read_array.assert_called_once_with(
            band=2, chunks=8, lock="LK", threadsafe=True
        )

    def test_rejects_window(self, mocker):
        """A ``window`` with ``chunks`` raises ``ValueError``.

        Test scenario:
            Lazy reads cannot take a window; slice the dask array instead.
        """
        with pytest.raises(ValueError, match="chunks"):
            LazyRead().read(mocker.Mock(), make_request(chunks=4), Window(0, 0, 2, 2))

    def test_rejects_out_shape(self, mocker):
        """``out_shape`` with ``chunks`` raises ``NotImplementedError``."""
        req = make_request(chunks=4, out_shape=(2, 2))
        with pytest.raises(NotImplementedError, match="out_shape"):
            LazyRead().read(mocker.Mock(), req, None)

    def test_rejects_masked(self, mocker):
        """``masked`` with ``chunks`` raises ``NotImplementedError``."""
        req = make_request(chunks=4, masked=True)
        with pytest.raises(NotImplementedError, match="masked=True"):
            LazyRead().read(mocker.Mock(), req, None)


class TestDecimatedRead:
    """``DecimatedRead`` (the ``out_shape=`` path)."""

    def test_backend_is_numpy(self):
        """The decimated path declares the ``numpy`` backend."""
        assert DecimatedRead().backend == "numpy", "decimated reads are numpy-backed"

    @pytest.mark.parametrize("out_shape, expected", [((2, 2), True), (None, False)])
    def test_matches_on_out_shape(self, out_shape, expected):
        """``matches`` is true exactly when an ``out_shape`` was given."""
        req = make_request(out_shape=out_shape)
        assert DecimatedRead().matches(req) is expected, "matches must track out_shape"

    def test_delegates_to_decimated_read(self, mocker):
        """Forwards band/window/out_shape/resampling to the decimation helper.

        Test scenario:
            ``DecimatedRead.read`` calls ``io._decimated_read`` positionally.
        """
        io = mocker.Mock()
        io._decimated_read.return_value = "decimated"
        window = Window(0, 0, 4, 4)
        req = make_request(band=1, out_shape=(2, 3), resampling="bilinear")
        result = DecimatedRead().read(io, req, window)
        assert result == "decimated", "the helper's return must pass through"
        io._decimated_read.assert_called_once_with(1, window, (2, 3), "bilinear")

    def test_rejects_masked(self, mocker):
        """``masked`` with ``out_shape`` raises ``NotImplementedError``."""
        req = make_request(out_shape=(2, 2), masked=True)
        with pytest.raises(NotImplementedError, match="masked=True"):
            DecimatedRead().read(mocker.Mock(), req, None)


class TestBoundlessRead:
    """``BoundlessRead`` (the ``boundless=True`` path)."""

    def test_backend_is_numpy(self):
        """The boundless path declares the ``numpy`` backend."""
        assert BoundlessRead().backend == "numpy", "boundless reads are numpy-backed"

    @pytest.mark.parametrize("boundless, expected", [(True, True), (False, False)])
    def test_matches_on_boundless(self, boundless, expected):
        """``matches`` is true exactly when ``boundless`` is set."""
        req = make_request(boundless=boundless)
        assert BoundlessRead().matches(req) is expected, "matches must track boundless"

    def test_delegates_to_boundless_read(self, mocker):
        """Forwards band/window/fill_value to the boundless helper.

        Test scenario:
            ``BoundlessRead.read`` calls ``io._boundless_read`` positionally.
        """
        io = mocker.Mock()
        io._boundless_read.return_value = "padded"
        window = Window(-1, -1, 4, 4)
        req = make_request(band=0, boundless=True, fill_value=7.0)
        result = BoundlessRead().read(io, req, window)
        assert result == "padded", "the helper's return must pass through"
        io._boundless_read.assert_called_once_with(0, window, 7.0)

    def test_rejects_masked(self, mocker):
        """``masked`` with ``boundless`` raises ``NotImplementedError``."""
        req = make_request(boundless=True, masked=True)
        with pytest.raises(NotImplementedError, match="masked=True"):
            BoundlessRead().read(mocker.Mock(), req, Window(0, 0, 2, 2))

    def test_requires_a_window(self, mocker):
        """A boundless read without a window raises ``ValueError``."""
        with pytest.raises(ValueError, match="requires a window"):
            BoundlessRead().read(mocker.Mock(), make_request(boundless=True), None)

    def test_rejects_geometry_window(self, mocker):
        """A geometry (``GeoDataFrame``) window with boundless raises ``ValueError``.

        Test scenario:
            Boundless needs a pixel window; a FeatureCollection is clipped by
            definition and must be rejected.
        """
        geom_window = FeatureCollection.from_bbox((0, 0, 1, 1), epsg=4326)
        with pytest.raises(ValueError, match="pixel window"):
            BoundlessRead().read(mocker.Mock(), make_request(boundless=True), geom_window)


class TestThreadsafeRead:
    """``ThreadsafeRead`` (the ``threadsafe=True`` path)."""

    def test_backend_is_numpy(self):
        """The thread-safe path declares the ``numpy`` backend."""
        assert ThreadsafeRead().backend == "numpy", "thread-safe reads are numpy-backed"

    @pytest.mark.parametrize("threadsafe, expected", [(True, True), (False, False)])
    def test_matches_on_threadsafe(self, threadsafe, expected):
        """``matches`` is true exactly when ``threadsafe`` is set."""
        req = make_request(threadsafe=threadsafe)
        assert ThreadsafeRead().matches(req) is expected, "matches must track threadsafe"

    def test_delegates_to_threadsafe_eager_read(self, mocker):
        """Forwards band/window (by keyword) to the per-thread read helper."""
        io = mocker.Mock()
        io._threadsafe_eager_read.return_value = "ts-array"
        window = Window(0, 0, 3, 3)
        req = make_request(band=1, threadsafe=True)
        result = ThreadsafeRead().read(io, req, window)
        assert result == "ts-array", "the helper's return must pass through"
        io._threadsafe_eager_read.assert_called_once_with(band=1, window=window)

    def test_rejects_masked(self, mocker):
        """``masked`` with ``threadsafe`` raises ``NotImplementedError``."""
        req = make_request(threadsafe=True, masked=True)
        with pytest.raises(NotImplementedError, match="masked=True"):
            ThreadsafeRead().read(mocker.Mock(), req, None)


class TestEagerRead:
    """``EagerRead`` (the default shared-handle path) against a live IO engine."""

    def test_backend_is_numpy(self):
        """The eager path declares the ``numpy`` backend."""
        assert EagerRead().backend == "numpy", "eager reads are numpy-backed"

    def test_matches_always(self):
        """``EagerRead`` is the fallback and matches any request."""
        assert EagerRead().matches(make_request()) is True, "eager must always match"
        assert EagerRead().matches(make_request(chunks=4)) is True, "eager is the fallback"

    def test_single_band_full_read(self, single_band):
        """A full single-band read returns the raster's array.

        Test scenario:
            ``band=0``, no window → the whole 6x6 band.
        """
        result = EagerRead().read(single_band.io, make_request(band=0), None)
        assert result.shape == (6, 6), f"expected the full 6x6 band, got {result.shape}"
        assert result[1, 0] == 6.0, f"value at (1, 0) should be 6.0, got {result[1, 0]}"

    def test_single_band_windowed_read(self, single_band):
        """A windowed single-band read returns the window's block.

        Test scenario:
            ``Window(1, 1, 2, 2)`` returns the 2x2 sub-block at (row 1, col 1).
        """
        result = EagerRead().read(single_band.io, make_request(band=0), Window(1, 1, 2, 2))
        assert result.shape == (2, 2), f"windowed read should be 2x2, got {result.shape}"
        expected = np.array([[7.0, 8.0], [13.0, 14.0]], dtype="float32")
        np.testing.assert_array_equal(result, expected, err_msg="window block mismatch")

    def test_single_band_none_reads_first_band(self, single_band):
        """``band=None`` on a single-band raster reads band 0 in full.

        Test scenario:
            With one band the multiband branch is skipped, so ``band=None`` falls
            through to ``band = 0`` and reads the whole band.
        """
        result = EagerRead().read(single_band.io, make_request(band=None), None)
        assert result.shape == (6, 6), f"expected the full 6x6 band, got {result.shape}"
        assert result[1, 0] == 6.0, f"value at (1, 0) should be 6.0, got {result[1, 0]}"

    def test_multiband_full_read_stacks_bands(self, multi_band):
        """A full read with ``band=None`` keeps the band axis.

        Test scenario:
            ``band=None`` on a 3-band raster returns ``(3, rows, cols)``.
        """
        result = EagerRead().read(multi_band.io, make_request(band=None), None)
        assert result.shape == (3, 4, 5), f"expected (3, 4, 5), got {result.shape}"
        assert result[2, 0, 0] == 2000.0, "band 2 origin should be 2000.0"

    def test_multiband_windowed_read_stacks_bands(self, multi_band):
        """A windowed read with ``band=None`` stacks the per-band blocks.

        Test scenario:
            ``band=None`` + ``Window(1, 1, 2, 2)`` returns ``(3, 2, 2)`` sliced
            identically per band.
        """
        window = Window(1, 1, 2, 2)
        result = EagerRead().read(multi_band.io, make_request(band=None), window)
        full = EagerRead().read(multi_band.io, make_request(band=None), None)
        assert result.shape == (3, 2, 2), f"expected (3, 2, 2), got {result.shape}"
        np.testing.assert_array_equal(
            result, full[:, 1:3, 1:3], err_msg="per-band window slice mismatch"
        )

    def test_masked_wrap_returns_masked_array(self, single_band):
        """``masked=True`` wraps the eager result as a masked array.

        Test scenario:
            The nodata cell (0, 0) is masked; a data cell is not.
        """
        result = EagerRead().read(single_band.io, make_request(band=0, masked=True), None)
        assert isinstance(result, np.ma.MaskedArray), "masked read must return a MaskedArray"
        assert result.mask[0, 0], "the nodata cell must be masked"
        assert not result.mask[1, 1], "a real-data cell must not be masked"

    def test_multiband_masked_wrap_passes_band_none(self):
        """``band=None`` + ``masked=True`` masks the stacked multiband read.

        Test scenario:
            The multiband branch feeds ``band=None`` (not ``0``) to ``_to_masked``;
            the result is a masked array with the band axis preserved and the nodata
            cell masked — pinning the ``band=None``-to-``_to_masked`` contract at the
            unit boundary (also covered at the integration level).
        """
        bands = np.stack(
            [np.arange(4 * 5, dtype="float32").reshape(4, 5) + b * 1000 for b in range(2)],
            axis=0,
        )
        bands[0, 0, 0] = -9999.0
        ds = Dataset.create_from_array(
            bands, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326, no_data_value=-9999.0
        )
        result = EagerRead().read(ds.io, make_request(band=None, masked=True), None)
        assert isinstance(result, np.ma.MaskedArray), "multiband masked read must return a MaskedArray"
        assert result.shape == (2, 4, 5), f"expected (2, 4, 5), got {result.shape}"
        assert result.mask[0, 0, 0], "the nodata cell must be masked in the multiband result"


class TestStrategySelection:
    """``READ_STRATEGIES`` order reproduces the original if/elif ladder."""

    def test_selects_lazy_for_chunks(self):
        """A ``chunks`` request selects ``LazyRead``."""
        assert isinstance(select(make_request(chunks=4)), LazyRead), "chunks → LazyRead"

    def test_selects_decimated_for_out_shape(self):
        """An ``out_shape`` request (no chunks) selects ``DecimatedRead``."""
        req = make_request(out_shape=(2, 2))
        assert isinstance(select(req), DecimatedRead), "out_shape → DecimatedRead"

    def test_selects_boundless(self):
        """A ``boundless`` request selects ``BoundlessRead``."""
        assert isinstance(select(make_request(boundless=True)), BoundlessRead), (
            "boundless → BoundlessRead"
        )

    def test_selects_threadsafe(self):
        """A ``threadsafe`` request selects ``ThreadsafeRead``."""
        req = make_request(threadsafe=True)
        assert isinstance(select(req), ThreadsafeRead), "threadsafe → ThreadsafeRead"

    def test_selects_eager_by_default(self):
        """A plain request falls through to ``EagerRead``."""
        assert isinstance(select(make_request()), EagerRead), "default → EagerRead"

    def test_chunks_precedence_over_out_shape(self):
        """When both ``chunks`` and ``out_shape`` are set, ``LazyRead`` wins.

        Test scenario:
            The ladder checks ``chunks`` first, so the lazy path is selected and
            its own guard (not the decimated one) governs — matching the original.
        """
        req = make_request(chunks=4, out_shape=(2, 2))
        assert isinstance(select(req), LazyRead), "chunks must take precedence over out_shape"
