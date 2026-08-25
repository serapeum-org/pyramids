"""Unit tests for the ``NetCDF._read_variable`` helper decomposition (ARC-150).

``_read_variable`` was split into a dispatcher plus five private helpers:

- ``_read_mdim_variable`` — MDIM orchestration (guarded MDArray read + fallback),
- ``_read_mdarray`` — windowed/full MDArray read (static),
- ``_normalize_mdarray_axes`` — axis-flip normalization (static),
- ``_read_indexing_variable`` — dimension indexing-variable fallback,
- ``_read_classic_variable`` — classic ``NETCDF:file:var`` subdataset read.

These tests isolate each helper with mocks so the branch behaviour (windowed vs
full, flip vs no-flip, MDArray vs fallback vs classic, and — critically — the
try/except *scoping* that must not swallow fallback failures) is pinned
independently of any real NetCDF file. The end-to-end behaviour on real fixtures
is covered by ``test_windowed_reads.py`` and ``unit/test_netcdf_unit_read.py``.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyramids.netcdf.netcdf import NetCDF


class TestReadMdarray:
    """Tests for the static ``NetCDF._read_mdarray``."""

    def test_full_read_calls_readasarray_without_args(self):
        """A ``None`` window triggers a bare full ``ReadAsArray()``.

        Test scenario:
            ``window=None`` must call ``md_arr.ReadAsArray()`` with no index
            arguments and return its result verbatim.
        """
        md_arr = Mock()
        sentinel = np.arange(6).reshape(2, 3)
        md_arr.ReadAsArray.return_value = sentinel

        result = NetCDF._read_mdarray(md_arr, None)

        md_arr.ReadAsArray.assert_called_once_with()
        assert result is sentinel, "full read must return ReadAsArray() verbatim"

    def test_windowed_read_passes_start_and_count(self):
        """A window becomes ``array_start_idx`` / ``count`` lists in storage order.

        Test scenario:
            ``window=[(0, 1), (100, 256)]`` must call
            ``ReadAsArray(array_start_idx=[0, 100], count=[1, 256])`` and return
            its result (no flipping — that is the caller's job).
        """
        md_arr = Mock()
        sentinel = np.zeros((1, 256))
        md_arr.ReadAsArray.return_value = sentinel

        result = NetCDF._read_mdarray(md_arr, [(0, 1), (100, 256)])

        md_arr.ReadAsArray.assert_called_once_with(
            array_start_idx=[0, 100], count=[1, 256]
        )
        assert result is sentinel, "windowed read must return ReadAsArray() verbatim"

    def test_returns_none_when_gdal_yields_none(self):
        """A ``None`` from GDAL propagates as ``None``.

        Test scenario:
            If ``ReadAsArray`` returns ``None`` the helper returns ``None``.
        """
        md_arr = Mock()
        md_arr.ReadAsArray.return_value = None

        assert NetCDF._read_mdarray(md_arr, None) is None, "None must propagate"


class TestNormalizeMdarrayAxes:
    """Tests for the static ``NetCDF._normalize_mdarray_axes``."""

    @pytest.fixture
    def arr(self):
        """Return a distinct 3x4 array so flips are observable.

        Returns:
            np.ndarray: ``arange(12).reshape(3, 4)``.
        """
        return np.arange(12).reshape(3, 4)

    def test_no_flip_returns_array_unchanged(self, arr):
        """``(False, False)`` leaves the array untouched.

        Args:
            arr: The 3x4 fixture array.

        Test scenario:
            When ``axis_flips`` reports neither axis needs flipping, the input is
            returned unchanged.
        """
        with patch(
            "pyramids.netcdf.netcdf.axis_flips", return_value=(False, False)
        ) as af:
            result = NetCDF._normalize_mdarray_axes(Mock(), Mock(), arr)
        af.assert_called_once()
        assert np.array_equal(result, arr), "no-flip must return the array unchanged"

    def test_forwards_rg_and_mdarray_to_axis_flips(self, arr):
        """``rg`` and ``md_arr`` are forwarded, in order, to ``axis_flips``.

        Test scenario:
            The helper takes ``rg`` (not just ``md_arr``) so the root group is held
            alive across the probe and SWIG cannot GC it. A regression that dropped
            ``rg`` and called ``axis_flips(md_arr)`` must be caught here, so assert
            the exact positional forwarding rather than a bare call count.
        """
        rg = Mock(name="rg")
        md_arr = Mock(name="md_arr")
        with patch(
            "pyramids.netcdf.netcdf.axis_flips", return_value=(False, False)
        ) as af:
            NetCDF._normalize_mdarray_axes(rg, md_arr, arr)
        af.assert_called_once_with(rg, md_arr)

    def test_flip_y_only(self, arr):
        """``(True, False)`` flips the row axis (``ndim - 2``) only.

        Test scenario:
            Y-flip must equal ``np.flip(arr, axis=0)`` for a 2-D array.
        """
        with patch("pyramids.netcdf.netcdf.axis_flips", return_value=(True, False)):
            result = NetCDF._normalize_mdarray_axes(Mock(), Mock(), arr)
        assert np.array_equal(result, np.flip(arr, axis=0)), "Y axis must be flipped"

    def test_flip_x_only(self, arr):
        """``(False, True)`` flips the column axis (``ndim - 1``) only.

        Test scenario:
            X-flip must equal ``np.flip(arr, axis=1)`` for a 2-D array.
        """
        with patch("pyramids.netcdf.netcdf.axis_flips", return_value=(False, True)):
            result = NetCDF._normalize_mdarray_axes(Mock(), Mock(), arr)
        assert np.array_equal(result, np.flip(arr, axis=1)), "X axis must be flipped"

    def test_flip_both_axes(self, arr):
        """``(True, True)`` flips both axes.

        Test scenario:
            Both-flip must equal ``np.flip(np.flip(arr, axis=0), axis=1)``.
        """
        with patch("pyramids.netcdf.netcdf.axis_flips", return_value=(True, True)):
            result = NetCDF._normalize_mdarray_axes(Mock(), Mock(), arr)
        expected = np.flip(np.flip(arr, axis=0), axis=1)
        assert np.array_equal(result, expected), "both axes must be flipped"

    def test_uses_trailing_axes_for_3d(self):
        """Flips target the trailing two axes for a >2-D array.

        Test scenario:
            For a 3-D array the Y/X flips must apply to axes ``ndim-2`` / ``ndim-1``
            (i.e. 1 and 2), leaving the leading band axis intact.
        """
        arr3d = np.arange(24).reshape(2, 3, 4)
        with patch("pyramids.netcdf.netcdf.axis_flips", return_value=(True, True)):
            result = NetCDF._normalize_mdarray_axes(Mock(), Mock(), arr3d)
        expected = np.flip(np.flip(arr3d, axis=1), axis=2)
        assert np.array_equal(result, expected), "3-D flips must target trailing axes"


class TestReadIndexingVariable:
    """Tests for ``NetCDF._read_indexing_variable`` (the fallback)."""

    def test_returns_none_when_no_dimension(self):
        """A missing dimension yields ``None``.

        Test scenario:
            ``_get_dimension`` returning ``None`` short-circuits to ``None``.
        """
        mock_self = Mock()
        mock_self._get_dimension.return_value = None

        result = NetCDF._read_indexing_variable(mock_self, "x", None)

        assert result is None, "no dimension must yield None"

    def test_returns_none_when_no_indexing_variable(self):
        """A dimension without an indexing variable yields ``None``.

        Test scenario:
            ``dim.GetIndexingVariable()`` returning ``None`` yields ``None``.
        """
        mock_self = Mock()
        dim = Mock()
        dim.GetIndexingVariable.return_value = None
        mock_self._get_dimension.return_value = dim

        result = NetCDF._read_indexing_variable(mock_self, "x", None)

        assert result is None, "no indexing variable must yield None"

    def test_full_read_when_window_is_none(self):
        """``window=None`` reads the whole indexing variable.

        Test scenario:
            The indexing variable's ``ReadAsArray()`` is called with no args.
        """
        mock_self = Mock()
        iv = Mock()
        coords = np.array([0.0, 1.0, 2.0])
        iv.ReadAsArray.return_value = coords
        dim = Mock()
        dim.GetIndexingVariable.return_value = iv
        mock_self._get_dimension.return_value = dim

        result = NetCDF._read_indexing_variable(mock_self, "time", None)

        iv.ReadAsArray.assert_called_once_with()
        assert result is coords, "full 1-D read must return ReadAsArray() verbatim"

    def test_windowed_read_when_window_length_one(self):
        """A length-1 window slices the 1-D coordinate.

        Test scenario:
            ``window=[(5, 3)]`` must call
            ``ReadAsArray(array_start_idx=[5], count=[3])``.
        """
        mock_self = Mock()
        iv = Mock()
        iv.ReadAsArray.return_value = np.array([5.0, 6.0, 7.0])
        dim = Mock()
        dim.GetIndexingVariable.return_value = iv
        mock_self._get_dimension.return_value = dim

        NetCDF._read_indexing_variable(mock_self, "time", [(5, 3)])

        iv.ReadAsArray.assert_called_once_with(array_start_idx=[5], count=[3])

    def test_multi_dim_window_falls_back_to_full_read(self):
        """A window with length != 1 is ignored (full 1-D read).

        Test scenario:
            The indexing variable is 1-D, so a multi-entry window cannot apply;
            ``ReadAsArray()`` is called with no args.
        """
        mock_self = Mock()
        iv = Mock()
        iv.ReadAsArray.return_value = np.array([0.0, 1.0])
        dim = Mock()
        dim.GetIndexingVariable.return_value = iv
        mock_self._get_dimension.return_value = dim

        NetCDF._read_indexing_variable(mock_self, "x", [(0, 1), (2, 3)])

        iv.ReadAsArray.assert_called_once_with()


class TestReadClassicVariable:
    """Tests for ``NetCDF._read_classic_variable`` (classic subdataset read)."""

    def test_reads_full_variable(self):
        """A successful open reads the subdataset in full.

        Test scenario:
            ``gdal.Open("NETCDF:<file>:<var>")`` returning a dataset yields its
            ``ReadAsArray()`` result.
        """
        mock_self = Mock()
        mock_self.file_name = "file.nc"
        mock_self._vsimem_path = None  # a real on-disk container has no /vsimem backing
        ds = Mock()
        arr = np.arange(4).reshape(2, 2)
        ds.ReadAsArray.return_value = arr

        with patch("pyramids.netcdf.netcdf.gdal.Open", return_value=ds) as gopen:
            result = NetCDF._read_classic_variable(mock_self, "salt")

        gopen.assert_called_once_with("NETCDF:file.nc:salt")
        assert result is arr, "classic read must return ReadAsArray() verbatim"

    def test_prefers_vsimem_path_over_cosmetic_file_name(self):
        """A ``from_bytes`` backing path is used even when a cosmetic name shadows ``file_name``.

        Test scenario:
            When ``_vsimem_path`` is set (a ``from_bytes`` container) and ``file_name`` is a
            cosmetic ``name=`` label, the classic open must target the real ``/vsimem`` path,
            not the label (#1057).
        """
        mock_self = Mock()
        mock_self.file_name = "downloaded.nc"
        mock_self._vsimem_path = "/vsimem/abc.nc"
        ds = Mock()
        ds.ReadAsArray.return_value = np.zeros((2, 2))

        with patch("pyramids.netcdf.netcdf.gdal.Open", return_value=ds) as gopen:
            NetCDF._read_classic_variable(mock_self, "salt")

        gopen.assert_called_once_with("NETCDF:/vsimem/abc.nc:salt")

    def test_returns_none_when_open_returns_none(self):
        """A ``None`` from ``gdal.Open`` yields ``None``.

        Test scenario:
            ``gdal.Open`` returning ``None`` (variable/file not openable) yields
            ``None`` without calling ``ReadAsArray``.
        """
        mock_self = Mock()
        mock_self.file_name = "file.nc"

        with patch("pyramids.netcdf.netcdf.gdal.Open", return_value=None):
            result = NetCDF._read_classic_variable(mock_self, "nope")

        assert result is None, "open returning None must yield None"

    @pytest.mark.parametrize("exc", [RuntimeError, AttributeError])
    def test_swallows_expected_open_errors(self, exc):
        """RuntimeError/AttributeError from ``gdal.Open`` are swallowed to ``None``.

        Args:
            exc: The exception type ``gdal.Open`` raises.

        Test scenario:
            Both guarded exception types must be caught and produce ``None``.
        """
        mock_self = Mock()
        mock_self.file_name = "file.nc"

        with patch("pyramids.netcdf.netcdf.gdal.Open", side_effect=exc("boom")):
            result = NetCDF._read_classic_variable(mock_self, "salt")

        assert result is None, f"{exc.__name__} must be swallowed to None"

    @pytest.mark.parametrize("exc", [RuntimeError, AttributeError])
    def test_swallows_read_errors_after_successful_open(self, exc):
        """RuntimeError/AttributeError from ``ReadAsArray`` (post-open) yield ``None``.

        Args:
            exc: The exception type ``ReadAsArray`` raises.

        Test scenario:
            ``gdal.Open`` returns a dataset, but its ``ReadAsArray`` then raises a
            guarded error; the ``except`` must swallow it and produce ``None`` -- a
            distinct sub-branch from ``gdal.Open`` itself raising.
        """
        mock_self = Mock()
        mock_self.file_name = "file.nc"
        ds = Mock()
        ds.ReadAsArray.side_effect = exc("boom")

        with patch("pyramids.netcdf.netcdf.gdal.Open", return_value=ds):
            result = NetCDF._read_classic_variable(mock_self, "salt")

        assert result is None, f"{exc.__name__} from ReadAsArray must yield None"


class TestReadMdimVariable:
    """Tests for ``NetCDF._read_mdim_variable`` orchestration and scoping."""

    def _mock_self(self):
        """Build a mock ``self`` with the leaf read helpers stubbed.

        Returns:
            Mock: a stand-in ``NetCDF`` whose ``_read_mdarray`` /
            ``_normalize_mdarray_axes`` / ``_read_indexing_variable`` are Mocks.
        """
        mock_self = Mock()
        return mock_self

    def test_full_2d_read_is_normalized(self):
        """A full >=2-D MDArray read is passed through normalization.

        Test scenario:
            ``window=None`` and a 2-D read => ``_normalize_mdarray_axes`` is called
            and its return value is used; the fallback is not consulted.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.return_value = Mock()
        mock_self._read_mdarray.return_value = np.zeros((3, 4))
        normalized = np.ones((3, 4))
        mock_self._normalize_mdarray_axes.return_value = normalized

        result = NetCDF._read_mdim_variable(mock_self, rg, "t2m", None)

        mock_self._read_mdarray.assert_called_once_with(
            rg.OpenMDArray.return_value, None
        )
        norm_args = mock_self._normalize_mdarray_axes.call_args.args
        assert norm_args[0] is rg, "rg must be forwarded to _normalize_mdarray_axes"
        assert norm_args[1] is rg.OpenMDArray.return_value, "md_arr must be forwarded"
        assert norm_args[2] is mock_self._read_mdarray.return_value, (
            "the read result must be forwarded to _normalize_mdarray_axes"
        )
        mock_self._read_indexing_variable.assert_not_called()
        assert result is normalized, "full 2-D read must be normalized"

    def test_1d_read_is_not_normalized(self):
        """A 1-D read skips normalization (and the fallback).

        Test scenario:
            ``ndim < 2`` => no ``_normalize_mdarray_axes``, no fallback, result is
            the raw read.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.return_value = Mock()
        one_d = np.arange(5)
        mock_self._read_mdarray.return_value = one_d

        result = NetCDF._read_mdim_variable(mock_self, rg, "x", None)

        mock_self._normalize_mdarray_axes.assert_not_called()
        mock_self._read_indexing_variable.assert_not_called()
        assert result is one_d, "1-D read must be returned un-normalized"

    def test_windowed_2d_read_is_not_normalized(self):
        """A windowed read stays in storage order (no flip) even if 2-D.

        Test scenario:
            ``window`` is not None => ``_normalize_mdarray_axes`` must not be
            called; the raw windowed read is returned.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.return_value = Mock()
        windowed = np.zeros((2, 2))
        mock_self._read_mdarray.return_value = windowed

        result = NetCDF._read_mdim_variable(mock_self, rg, "t2m", [(0, 2), (0, 2)])

        mock_self._read_mdarray.assert_called_once_with(
            rg.OpenMDArray.return_value, [(0, 2), (0, 2)]
        )
        mock_self._normalize_mdarray_axes.assert_not_called()
        assert result is windowed, "windowed read must not be flipped"

    def test_normalization_error_returns_unnormalized_without_fallback(self):
        """A guarded error DURING normalization returns the raw read, no fallback.

        Test scenario:
            A non-None 2-D read succeeds, then ``_normalize_mdarray_axes`` raises a
            guarded ``RuntimeError``. Because ``result`` already holds the raw read
            (the failed assignment does not rebind it), the ``except`` swallows the
            error and the ``if result is None`` fallback is skipped -- the
            un-normalized, storage-order array is returned. This pins the subtle
            branch the original preserved (the flips lived inside the same ``try``),
            distinct from the ``OpenMDArray``-raises-before-read path.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.return_value = Mock()
        raw = np.zeros((3, 4))
        mock_self._read_mdarray.return_value = raw
        mock_self._normalize_mdarray_axes.side_effect = RuntimeError("flip boom")

        result = NetCDF._read_mdim_variable(mock_self, rg, "t2m", None)

        assert result is raw, "a normalization error must return the raw read"
        mock_self._read_indexing_variable.assert_not_called()

    def test_missing_mdarray_falls_back_to_indexing_variable(self):
        """A ``None`` MDArray triggers the indexing-variable fallback.

        Test scenario:
            ``OpenMDArray`` returning ``None`` => ``_read_mdarray`` is not called,
            the fallback runs and its value is returned.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.return_value = None
        coords = np.array([0.0, 1.0])
        mock_self._read_indexing_variable.return_value = coords

        result = NetCDF._read_mdim_variable(mock_self, rg, "time", None)

        mock_self._read_mdarray.assert_not_called()
        mock_self._read_indexing_variable.assert_called_once_with("time", None)
        assert result is coords, "missing MDArray must fall back to indexing var"

    def test_none_mdarray_read_falls_back(self):
        """A present MDArray whose read yields ``None`` still falls back.

        Test scenario:
            ``OpenMDArray`` returns an object but ``_read_mdarray`` returns
            ``None`` (e.g. GDAL yielded nothing) => normalization is skipped and
            the indexing-variable fallback supplies the result.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.return_value = Mock()
        mock_self._read_mdarray.return_value = None
        fallback = np.array([9.0])
        mock_self._read_indexing_variable.return_value = fallback

        result = NetCDF._read_mdim_variable(mock_self, rg, "x", None)

        mock_self._normalize_mdarray_axes.assert_not_called()
        mock_self._read_indexing_variable.assert_called_once_with("x", None)
        assert result is fallback, "a None read must fall back to the indexing var"

    @pytest.mark.parametrize("exc", [RuntimeError, ValueError])
    def test_openmdarray_error_falls_back(self, exc):
        """A guarded error from ``OpenMDArray`` is swallowed and falls back.

        Args:
            exc: The guarded exception ``OpenMDArray`` raises.

        Test scenario:
            RuntimeError/ValueError from the MDArray branch => caught, then the
            fallback runs and supplies the result.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.side_effect = exc("boom")
        fallback = np.array([1.0])
        mock_self._read_indexing_variable.return_value = fallback

        result = NetCDF._read_mdim_variable(mock_self, rg, "x", None)

        mock_self._read_indexing_variable.assert_called_once_with("x", None)
        assert result is fallback, f"{exc.__name__} must be swallowed then fall back"

    @pytest.mark.parametrize("exc", [RuntimeError, ValueError])
    def test_read_error_after_open_falls_back(self, exc):
        """A guarded error from ``_read_mdarray`` (post-open) is swallowed → fallback.

        Args:
            exc: The guarded exception ``_read_mdarray`` raises.

        Test scenario:
            ``OpenMDArray`` returns a non-None array, but ``_read_mdarray`` then
            raises a guarded ``RuntimeError`` / ``ValueError``. The ``except``
            swallows it, ``result`` stays ``None``, and the indexing-variable
            fallback supplies the result -- a distinct raising call site from
            ``OpenMDArray`` itself raising.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.return_value = Mock()
        mock_self._read_mdarray.side_effect = exc("read boom")
        fallback = np.array([2.0])
        mock_self._read_indexing_variable.return_value = fallback

        result = NetCDF._read_mdim_variable(mock_self, rg, "x", None)

        mock_self._read_indexing_variable.assert_called_once_with("x", None)
        assert result is fallback, f"{exc.__name__} from the read must fall back"

    def test_fallback_failure_is_not_swallowed(self):
        """The indexing-variable fallback runs OUTSIDE the guard (R3).

        Test scenario:
            After the MDArray branch fails, a failure in the fallback itself must
            propagate (it is deliberately not inside the try/except). This pins the
            scoping invariant the refactor had to preserve.
        """
        mock_self = self._mock_self()
        rg = Mock()
        rg.OpenMDArray.side_effect = RuntimeError("mdarray gone")
        mock_self._read_indexing_variable.side_effect = ValueError("fallback bug")

        with pytest.raises(ValueError, match="fallback bug"):
            NetCDF._read_mdim_variable(mock_self, rg, "x", None)


class TestReadVariableDispatcher:
    """Tests for the ``NetCDF._read_variable`` dispatcher branch selection."""

    def test_mdim_branch_when_working_group_present(self):
        """A non-None working group routes to ``_read_mdim_variable``.

        Test scenario:
            ``_working_group()`` returning a group => MDIM path is taken with the
            group forwarded; the classic path is not used.
        """
        mock_self = Mock()
        rg = Mock()
        mock_self._working_group.return_value = rg
        sentinel = np.zeros((2, 2))
        mock_self._read_mdim_variable.return_value = sentinel

        result = NetCDF._read_variable(mock_self, "t2m", None)

        mock_self._read_mdim_variable.assert_called_once_with(rg, "t2m", None)
        mock_self._read_classic_variable.assert_not_called()
        assert result is sentinel, "present working group must take the MDIM path"

    def test_classic_branch_when_no_working_group(self):
        """A ``None`` working group routes to ``_read_classic_variable``.

        Test scenario:
            ``_working_group()`` returning ``None`` => classic path is taken; the
            MDIM path is not used.
        """
        mock_self = Mock()
        mock_self._working_group.return_value = None
        sentinel = np.zeros((2, 2))
        mock_self._read_classic_variable.return_value = sentinel

        result = NetCDF._read_variable(mock_self, "salt", None)

        mock_self._read_classic_variable.assert_called_once_with("salt")
        mock_self._read_mdim_variable.assert_not_called()
        assert result is sentinel, "absent working group must take the classic path"
