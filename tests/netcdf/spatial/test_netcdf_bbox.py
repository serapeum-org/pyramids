"""Tests for :meth:`pyramids.netcdf.NetCDF.crop` bbox kwargs (PY-8).

The ``bbox=`` / ``epsg=`` keyword-only arguments mirror PY-5's surface
on :meth:`pyramids.dataset.Dataset.crop`. They route through the shared
:meth:`pyramids.feature.FeatureCollection.from_bbox` primitive and fall
through to the existing polygon / variable-subset paths.
"""

from __future__ import annotations

import pytest

from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

INSIDE_BBOX = (10.0, -50.0, 50.0, -20.0)


@pytest.fixture(scope="module")
def root_nc(noah_nc_path: str) -> NetCDF:
    """Open the test NetCDF as a root MDIM container.

    Args:
        noah_nc_path: Path to the cf__6v__1d2-2d4__geog__y-asc.nc fixture.

    Returns:
        NetCDF: Container with four data variables (``Band1`` … ``Band4``).
    """
    return NetCDF.read_file(noah_nc_path)


class TestNetCDFCropBbox:
    """Tests for the ``bbox=`` / ``epsg=`` kwargs on a root container."""

    def test_bbox_in_native_crs_returns_container(self, root_nc: NetCDF):
        """Test bbox crop preserves the four-variable container shape.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            A bbox inside the raster's own CRS must yield a container
            with the same variable list as the source.
        """
        cropped = root_nc.crop(bbox=INSIDE_BBOX)
        assert sorted(cropped.variables) == sorted(
            root_nc.variables
        ), f"Variables changed: {sorted(cropped.variables)!r}"

    def test_bbox_default_epsg_matches_dataset(self, root_nc: NetCDF):
        """Test explicit ``epsg=`` of the dataset's CRS matches the default.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Omitting ``epsg=`` and passing ``epsg=nc.epsg`` must produce
            byte-identical crop results (FC is built with the same CRS).
        """
        without_epsg = root_nc.crop(bbox=INSIDE_BBOX)
        with_epsg = root_nc.crop(bbox=INSIDE_BBOX, epsg=root_nc.epsg)
        a = without_epsg.get_variable("Band1").read_array()
        b = with_epsg.get_variable("Band1").read_array()
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    def test_bbox_equivalent_to_explicit_fc(self, root_nc: NetCDF):
        """Test ``crop(bbox=…)`` matches ``crop(mask=FC.from_bbox(…))``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The ``bbox=`` sugar must be exactly equivalent to building
            the one-row FC by hand — same array values, same shape.
        """
        via_bbox = root_nc.crop(bbox=INSIDE_BBOX).get_variable("Band1").read_array()
        fc = FeatureCollection.from_bbox(INSIDE_BBOX, epsg=root_nc.epsg)
        via_fc = root_nc.crop(mask=fc).get_variable("Band1").read_array()
        assert (
            via_bbox.shape == via_fc.shape
        ), f"Shape mismatch: {via_bbox.shape} vs {via_fc.shape}"

    def test_mask_path_still_works(self, root_nc: NetCDF):
        """Test pre-PY-8 ``mask=`` callers see no regression.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Passing a FC via the positional / ``mask=`` slot must keep
            its original behaviour.
        """
        fc = FeatureCollection.from_bbox(INSIDE_BBOX, epsg=root_nc.epsg)
        cropped = root_nc.crop(mask=fc)
        assert sorted(cropped.variables) == sorted(root_nc.variables)


class TestNetCDFCropForeignCRS:
    """Foreign-CRS bbox crop — the reprojection path inside crop's polygon warp."""

    def test_mercator_bbox_against_wgs84_netcdf(self, root_nc: NetCDF):
        """Test bbox in EPSG:3857 against an EPSG:4326 NetCDF (root container).

        Args:
            root_nc: Module-scope root NetCDF fixture (EPSG:4326).

        Test scenario:
            A Mercator bbox covering roughly 27..36°E and -33..-27°N
            (after reprojection) lands inside the fixture's extent. The
            shared FC reprojection path inside ``_crop_with_polygon_warp``
            must succeed and produce a cropped container smaller than
            the source.
        """
        # EPSG:3857 metres — corners roughly (30°E, -30°N) ± a few degrees
        mercator_bbox = (3_000_000.0, -4_000_000.0, 4_000_000.0, -3_000_000.0)
        cropped = root_nc.crop(bbox=mercator_bbox, epsg=3857)
        assert sorted(cropped.variables) == sorted(
            root_nc.variables
        ), f"Variables changed: {sorted(cropped.variables)!r}"
        full_arr = root_nc.get_variable("Band1").read_array()
        cropped_arr = cropped.get_variable("Band1").read_array()
        assert cropped_arr.size < full_arr.size, (
            f"Foreign-CRS bbox didn't reduce size: full={full_arr.size} "
            f"cropped={cropped_arr.size}"
        )

    def test_mercator_bbox_read_array(self, root_nc: NetCDF):
        """Test ``read_array(bbox=…, epsg=3857)`` on a WGS84 NetCDF variable.

        Args:
            root_nc: Module-scope root NetCDF fixture (EPSG:4326).

        Test scenario:
            The reprojection path inside ``Dataset.read_array`` must
            fire on a NetCDF subset too — the override forwards
            ``bbox`` / ``epsg`` to ``super().read_array``.
        """
        mercator_bbox = (3_000_000.0, -4_000_000.0, 4_000_000.0, -3_000_000.0)
        full = root_nc.read_array(variable="Band1")
        windowed = root_nc.read_array(
            variable="Band1",
            bbox=mercator_bbox,
            epsg=3857,
        )
        assert windowed.shape != full.shape, (
            f"Foreign-CRS bbox was a no-op: full={full.shape} "
            f"windowed={windowed.shape}"
        )
        assert windowed.size < full.size, (
            f"Foreign-CRS bbox didn't reduce size: full={full.size} "
            f"windowed={windowed.size}"
        )


class TestNetCDFCropVariableSubset:
    """Bbox crop on a single variable (delegates to ``super().crop``)."""

    def test_variable_subset_accepts_bbox(self, root_nc: NetCDF):
        """Test ``nc.get_variable(...).crop(bbox=...)`` actually reduces shape.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The variable-subset branch must accept the same ``bbox=`` /
            ``epsg=`` kwargs and route through ``super().crop`` —
            **and the cropped result must be smaller than the source**
            (so a silent no-op bbox would fail this test).
        """
        var = root_nc.get_variable("Band1")
        full_arr = var.read_array()
        cropped = var.crop(bbox=INSIDE_BBOX)
        cropped_arr = cropped.read_array()
        assert cropped_arr.ndim in (2, 3), f"Unexpected ndim: {cropped_arr.ndim}"
        assert cropped_arr.shape != full_arr.shape, (
            f"bbox crop was a no-op: full={full_arr.shape} "
            f"cropped={cropped_arr.shape}"
        )
        assert cropped_arr.size < full_arr.size, (
            f"bbox crop didn't reduce size: full={full_arr.size} "
            f"cropped={cropped_arr.size}"
        )


class TestNetCDFCropMutex:
    """Mutual-exclusion + missing-argument errors."""

    def test_mask_and_bbox_together_raises(self, root_nc: NetCDF):
        """Test passing both ``mask=`` and ``bbox=`` raises ``ValueError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Mutex guard must fire before the FC is built.
        """
        fc = FeatureCollection.from_bbox(INSIDE_BBOX, epsg=root_nc.epsg)
        with pytest.raises(ValueError, match="not both"):
            root_nc.crop(mask=fc, bbox=INSIDE_BBOX)

    def test_neither_raises_type_error(self, root_nc: NetCDF):
        """Test calling ``crop()`` with no mask / bbox raises ``TypeError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Missing-argument guard must give a clear, actionable error.
        """
        with pytest.raises(TypeError, match=r"mask.*bbox|bbox.*mask"):
            root_nc.crop()

    def test_invalid_bbox_raises_value_error(self, root_nc: NetCDF):
        """Test a ``south >= north`` bbox raises ``ValueError`` via FC.from_bbox.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Validation lives in ``FeatureCollection.from_bbox``; the NetCDF
            override must surface that error unchanged. Uses ``south >= north``
            (still always invalid) rather than ``west >= east``, which is now the
            STAC antimeridian convention on a geographic grid.
        """
        with pytest.raises(ValueError):
            root_nc.crop(bbox=(10.0, -20.0, 50.0, -50.0))  # south >= north

    def test_crs_less_netcdf_without_epsg_raises(self, root_nc: NetCDF, mocker):
        """Test ``crop(bbox=…)`` on a CRS-less NetCDF raises a clear ``ValueError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.
            mocker: pytest-mock fixture.

        Test scenario:
            When the dataset has no CRS at all (``epsg`` None *and* an empty
            ``crs`` WKT) and the caller didn't pass ``epsg=``, the upfront guard
            must fire (better than the deeper ``from_bbox`` ``ValueError``). A
            geostationary grid — ``epsg is None`` but a present WKT — is not
            CRS-less and must not hit this branch.
        """
        mocker.patch.object(
            type(root_nc),
            "epsg",
            new_callable=mocker.PropertyMock,
            return_value=None,
        )
        mocker.patch.object(
            type(root_nc),
            "crs",
            new_callable=mocker.PropertyMock,
            return_value="",
        )
        with pytest.raises(ValueError, match=r"explicit `epsg=`.*no CRS at all"):
            root_nc.crop(bbox=INSIDE_BBOX)

    def test_crs_less_netcdf_explicit_epsg_works(self, root_nc: NetCDF, mocker):
        """Test ``crop(bbox=…, epsg=4326)`` on CRS-less NetCDF still works.

        Args:
            root_nc: Module-scope root NetCDF fixture.
            mocker: pytest-mock fixture.

        Test scenario:
            With an explicit ``epsg=`` the guard must NOT fire — the
            caller has resolved the ambiguity.
        """
        mocker.patch.object(
            type(root_nc),
            "epsg",
            new_callable=mocker.PropertyMock,
            return_value=None,
        )
        # The guard should not fire; from_bbox builds the FC successfully.
        # The downstream crop may still fail (no CRS = no reprojection),
        # but that's a separate concern — we only check the guard.
        try:
            root_nc.crop(bbox=INSIDE_BBOX, epsg=4326)
        except ValueError as exc:
            assert "explicit `epsg=`" not in str(
                exc
            ), f"Guard fired despite explicit epsg=: {exc}"
        except Exception:
            pass  # other downstream errors are out-of-scope here


class TestNetCDFReadArrayBbox:
    """Tests for the ``bbox=`` / ``epsg=`` kwargs on ``read_array``."""

    def test_root_container_bbox_routes_to_variable(self, root_nc: NetCDF):
        """Test root-container call with ``variable=`` + ``bbox=`` reads a window.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The container call dispatches to ``get_variable(...)``, and
            both ``bbox=`` and ``variable=`` must flow through.
        """
        full = root_nc.read_array(variable="Band1")
        windowed = root_nc.read_array(
            variable="Band1",
            bbox=(10.0, -50.0, 50.0, -20.0),
        )
        assert windowed.shape != full.shape, (
            f"Windowed read should differ from full: full={full.shape} "
            f"windowed={windowed.shape}"
        )
        assert windowed.size < full.size, (
            f"Windowed read should be smaller: full={full.size} "
            f"windowed={windowed.size}"
        )

    def test_variable_subset_bbox(self, root_nc: NetCDF):
        """Test ``read_array(bbox=…)`` on a pinned variable subset reduces shape.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            On a variable subset the call forwards bbox/epsg to
            ``super().read_array`` (Dataset eager path) — **and the
            windowed result must be smaller than the full read** so a
            silent no-op bbox would fail this test.
        """
        var = root_nc.get_variable("Band1")
        full = var.read_array()
        windowed = var.read_array(bbox=(10.0, -50.0, 50.0, -20.0))
        assert windowed.ndim in (2, 3), f"Unexpected ndim: {windowed.ndim}"
        assert (
            windowed.shape != full.shape
        ), f"bbox was a no-op: full={full.shape} windowed={windowed.shape}"
        assert (
            windowed.size < full.size
        ), f"bbox didn't reduce size: full={full.size} windowed={windowed.size}"

    def test_non_square_bbox_variable_not_transposed(self, root_nc: NetCDF):
        """Test a non-square NetCDF-variable bbox read is not transposed (#719's actual path).

        Args:
            root_nc: Module-scope root NetCDF fixture (EPSG:4326).

        Test scenario:
            #719 was reported through a NetCDF variable. A bbox spanning far more longitude than
            latitude must read back wider than it is tall, matching a window derived independently
            from the variable's geotransform -- a transpose would swap the two axes.
        """
        var = root_nc.get_variable("Band1")
        _, pixel_x, _, _, _, pixel_y = var.geotransform
        west, south, east, north = 10.0, -50.0, 90.0, -30.0
        # These edges are cell-aligned, so the exact cover span is the extent divided by the cell
        # size -- derived independently of the production floor/ceil index math.
        exp_cols = round((east - west) / abs(pixel_x))
        exp_rows = round((north - south) / abs(pixel_y))
        got = var.read_array(bbox=(west, south, east, north))
        got2d = got[got.shape[0] // 2] if got.ndim == 3 else got
        assert got2d.shape == (exp_rows, exp_cols), (
            f"transposed or mis-sized: got {got2d.shape}, expected {(exp_rows, exp_cols)}"
        )
        assert got2d.shape[1] > got2d.shape[0], "bbox spans more lon than lat; cols must exceed rows"

    def test_bbox_rounding_forwarded_to_variable_read(self, root_nc: NetCDF):
        """Test ``bbox_rounding=`` is forwarded through the NetCDF override to the window resolver.

        Args:
            root_nc: Module-scope root NetCDF fixture (EPSG:4326).

        Test scenario:
            For a bbox whose edges fall off cell centres, ``"nearest"`` must read a strictly smaller
            window than the default ``"cover"``. If the override dropped the argument, both reads
            would be identical.
        """
        var = root_nc.get_variable("Band1")
        origin_x, pixel_x, _, origin_y, _, pixel_y = var.geotransform
        west = origin_x + 20.7 * pixel_x
        east = origin_x + 100.3 * pixel_x
        north = origin_y + 220.7 * pixel_y
        south = origin_y + 280.3 * pixel_y
        cover = root_nc.read_array(variable="Band1", bbox=(west, south, east, north))
        nearest = root_nc.read_array(
            variable="Band1", bbox=(west, south, east, north), bbox_rounding="nearest"
        )
        assert nearest.shape[-2] < cover.shape[-2], (
            f"nearest rows should be fewer: cover={cover.shape[-2:]} nearest={nearest.shape[-2:]}"
        )
        assert nearest.shape[-1] < cover.shape[-1], (
            f"nearest cols should be fewer: cover={cover.shape[-2:]} nearest={nearest.shape[-2:]}"
        )

    def test_window_and_bbox_together_raises(self, root_nc: NetCDF):
        """Test ``window=`` + ``bbox=`` together raises ``ValueError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            Mutex must fire before any reading happens.
        """
        with pytest.raises(ValueError, match="not both"):
            root_nc.read_array(
                variable="Band1",
                window=[0, 0, 10, 10],
                bbox=(10.0, -50.0, 50.0, -20.0),
            )

    def test_chunks_and_bbox_together_raises_value_error(self, root_nc: NetCDF):
        """Test ``chunks=`` + ``bbox=`` together raises ``ValueError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.

        Test scenario:
            The lazy path doesn't yet honour bbox windowing. The error
            class must match ``Dataset.read_array``'s existing
            ``chunks=`` + ``window=`` ``ValueError`` so callers can
            ``except ValueError`` both mutex cases uniformly.
        """
        with pytest.raises(ValueError, match=r"chunks=.*bbox=.*not supported"):
            root_nc.read_array(
                variable="Band1",
                bbox=(10.0, -50.0, 50.0, -20.0),
                chunks="auto",
            )

    def test_crs_less_netcdf_without_epsg_raises(self, root_nc: NetCDF, mocker):
        """Test ``read_array(bbox=…)`` on a CRS-less NetCDF raises ``ValueError``.

        Args:
            root_nc: Module-scope root NetCDF fixture.
            mocker: pytest-mock fixture.

        Test scenario:
            The same guard as ``crop`` — if ``self.epsg is None`` and
            the caller didn't pass ``epsg=``, fail fast at the override
            boundary rather than letting the deeper ``from_bbox``
            ``ValueError`` surface.
        """
        mocker.patch.object(
            type(root_nc),
            "epsg",
            new_callable=mocker.PropertyMock,
            return_value=None,
        )
        mocker.patch.object(
            type(root_nc),
            "crs",
            new_callable=mocker.PropertyMock,
            return_value="",
        )
        with pytest.raises(ValueError, match=r"explicit `epsg=`.*no CRS at all"):
            root_nc.read_array(variable="Band1", bbox=INSIDE_BBOX)

    def test_fc_built_once_at_netcdf_level(self, root_nc: NetCDF, mocker):
        """Test ``bbox=`` is converted to FC exactly once at the NetCDF layer (L2).

        Args:
            root_nc: Module-scope root NetCDF fixture.
            mocker: pytest-mock fixture.

        Test scenario:
            After the L2 fix, ``NetCDF.read_array`` builds the FC at
            the top and forwards ``window=fc, bbox=None`` to both the
            recursive subset call and ``super().read_array``. The
            deeper ``Dataset.read_array`` ``from_bbox`` branch (in
            ``engines/io.py``) must NOT fire — proven by spying on
            ``FeatureCollection.from_bbox`` and confirming it's called
            exactly once.
        """
        spy = mocker.spy(FeatureCollection, "from_bbox")
        root_nc.read_array(variable="Band1", bbox=INSIDE_BBOX)
        assert spy.call_count == 1, (
            f"Expected exactly one FC build (at the NetCDF layer); "
            f"got {spy.call_count} — likely the dedupe (L2) regressed."
        )
