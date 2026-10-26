"""`UgridDataset.plot_vector_field` (issue #1373).

A hydrodynamic mesh's velocity field is the canonical results visual — D-Flow FM writes
`mesh2d_ucx` / `mesh2d_ucy` per face — and `UgridDataset` had no way to draw it. The facade
forwards to cleopatra's `MeshGlyph.plot_vector_field` (available from cleopatra 0.44.0),
keeping the `kind` / `thin` vocabulary of the raster `Dataset.plot_vector_field` so the two
read the same.

Rendering is asserted through the matplotlib artist cleopatra returns rather than by
comparing pixels: a `Quiver` for `kind="quiver"` and a `Barbs` for `kind="barbs"`, with the
arrow count showing what `thin=` dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = [pytest.mark.plot, pytest.mark.core]


def _mesh() -> Mesh2d:
    """Return the shared 4-node / 2-face triangle mesh.

    Returns:
        Mesh2d: Two triangles sharing an edge.
    """
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5, 1.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=fnc,
    )


def _variable(name: str, values, location: str = "face") -> MeshVariable:
    """Wrap an array as a MeshVariable on the given location.

    Args:
        name: Variable name.
        values: The data, converted to float64.
        location: Where the values live. Defaults to `"face"`.

    Returns:
        MeshVariable: The loaded variable.
    """
    array = np.asarray(values, dtype="float64")
    return MeshVariable(
        name=name,
        location=location,
        mesh_name="mesh2d",
        shape=array.shape,
        _data=array,
    )


def _dataset(**variables) -> UgridDataset:
    """Build a UgridDataset over `_mesh()` carrying the given variables.

    Args:
        **variables: Name to MeshVariable.

    Returns:
        UgridDataset: The dataset.
    """
    return UgridDataset(
        mesh=_mesh(), data_variables=dict(variables), global_attributes={}
    )


@pytest.fixture
def velocity() -> UgridDataset:
    """A 2-face dataset with a face-located velocity field.

    Returns:
        UgridDataset: `ucx` pointing east on one face, `ucy` north on the other.
    """
    return _dataset(ucx=_variable("ucx", [1.0, 0.0]), ucy=_variable("ucy", [0.0, 1.0]))


@pytest.fixture
def temporal_velocity() -> UgridDataset:
    """A 3-step face-located velocity field, each step an order of magnitude apart.

    Returns:
        UgridDataset: `ucx` of shape (3, 2) and a matching zero `ucy`.
    """
    return _dataset(
        ucx=_variable("ucx", [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]),
        ucy=_variable("ucy", np.zeros((3, 2))),
    )


class TestRendering:
    """What the facade draws, and what it returns."""

    def test_the_method_exists_on_the_class(self):
        """`UgridDataset` advertises the member the issue reported missing."""
        assert hasattr(UgridDataset, "plot_vector_field"), (
            "UgridDataset should expose plot_vector_field (issue #1373)"
        )

    def test_a_face_field_renders_as_a_quiver(self, velocity):
        """The default `kind` draws arrows and returns the raster method's triple.

        Test scenario:
            `Dataset.plot_vector_field` returns `(fig, ax, im)` from cleopatra, and the mesh
            facade matches that contract so the two are interchangeable in a caller's hands.
        """
        from matplotlib.quiver import Quiver

        fig, ax, im = velocity.plot_vector_field("ucx", "ucy")
        assert fig is not None, "a figure must come back even when no ax was supplied"
        assert ax is not None, "the axes the field was drawn on must come back"
        assert isinstance(im, Quiver), f"expected a Quiver, got {type(im).__name__}"
        assert im.N == 2, f"one arrow per face, got {im.N}"

    def test_barbs_are_available_through_kind(self, velocity):
        """`kind="barbs"` picks the wind-barb renderer, as on the raster method."""
        from matplotlib.quiver import Barbs

        _fig, _ax, im = velocity.plot_vector_field("ucx", "ucy", kind="barbs")
        assert isinstance(im, Barbs), f"expected Barbs, got {type(im).__name__}"

    def test_thin_reduces_the_arrow_count(self, velocity):
        """`thin=n` draws every nth element, which is what keeps a large mesh legible."""
        _fig, _ax, dense = velocity.plot_vector_field("ucx", "ucy")
        _fig, _ax, thinned = velocity.plot_vector_field("ucx", "ucy", thin=2)
        assert thinned.N < dense.N, (
            f"thin=2 should drop arrows: {thinned.N} vs {dense.N}"
        )

    def test_the_default_title_names_both_components(self, velocity):
        """With no `title` the plot is labelled from the two variable names."""
        _fig, ax, _im = velocity.plot_vector_field("ucx", "ucy")
        assert ax.get_title() == "ucx / ucy"

    def test_an_explicit_title_wins(self, velocity):
        """A caller's `title` is not overwritten by the default."""
        _fig, ax, _im = velocity.plot_vector_field(
            "ucx", "ucy", title="Depth-averaged velocity"
        )
        assert ax.get_title() == "Depth-averaged velocity"

    def test_node_located_components_render(self):
        """Node-located components anchor at the nodes, the other supported location."""
        dataset = _dataset(
            un=_variable("un", [1.0, 1.0, 0.0, 0.0], location="node"),
            vn=_variable("vn", [0.0, 0.0, 1.0, 1.0], location="node"),
        )
        _fig, _ax, im = dataset.plot_vector_field("un", "vn")
        assert im.N == 4, f"one arrow per node, got {im.N}"

    def test_colorbar_is_forwarded_when_set(self, velocity):
        """An explicit `colorbar` reaches cleopatra; `None` leaves its default alone.

        Test scenario:
            The renderer forwards `colorbar` only when the caller sets it, so cleopatra's own
            default survives otherwise. Both arms are exercised: the figure drawn with
            `colorbar=True` carries an extra axes for the bar, the one with `colorbar=False`
            does not.
        """
        with_bar, _ax, _im = velocity.plot_vector_field("ucx", "ucy", colorbar=True)
        without_bar, _ax, _im = velocity.plot_vector_field("ucx", "ucy", colorbar=False)
        assert len(with_bar.axes) > len(without_bar.axes), (
            f"colorbar=True should add an axes: {len(with_bar.axes)} vs "
            f"{len(without_bar.axes)}"
        )


class TestTimeStep:
    """How a temporal component is reduced to the single step that gets drawn."""

    @pytest.mark.parametrize(
        "step", [0, 1, 2, -1], ids=["first", "middle", "last", "negative"]
    )
    def test_time_index_selects_a_step(self, temporal_velocity, step):
        """Each in-range step draws, which is the single-step contract `plot` applies."""
        _fig, _ax, im = temporal_velocity.plot_vector_field(
            "ucx", "ucy", time_index=step
        )
        assert im.N == 2, f"one arrow per face at step {step}, got {im.N}"

    def test_the_step_actually_changes_the_data_drawn(self, temporal_velocity):
        """A different `time_index` draws different vectors, not just a different call.

        Test scenario:
            The steps are an order of magnitude apart, so the magnitudes the renderer is
            handed must differ. Without this, a `time_index` that was accepted and ignored
            would pass every other test in this class.
        """
        _fig, _ax, first = temporal_velocity.plot_vector_field(
            "ucx", "ucy", time_index=0
        )
        _fig, _ax, last = temporal_velocity.plot_vector_field(
            "ucx", "ucy", time_index=2
        )
        assert not np.allclose(first.U, last.U), (
            f"step 0 and step 2 should differ, got {first.U} and {last.U}"
        )

    @pytest.mark.parametrize(
        "step", [3, 9, -4], ids=["one-past", "far", "negative-past"]
    )
    def test_an_out_of_range_step_is_refused(self, temporal_velocity, step):
        """A step outside the time axis is named rather than left to numpy."""
        with pytest.raises(ValueError, match=r"time_index .* is outside"):
            temporal_velocity.plot_vector_field("ucx", "ucy", time_index=step)

    def test_a_time_index_past_the_data_rank_is_named(self):
        """A declared time axis the data does not have is refused, not leaked (M4).

        Test scenario:
            `time_index` comes from `MeshVariable.dimensions` by name, so a variable whose
            declared dimensions name more axes than its data has yields an axis past the
            array's rank. Indexing `shape` with it raised a bare `IndexError: tuple index
            out of range` -- a type the documented contract does not promise, naming neither
            the method nor the variable, from the pre-check that exists to report this well.
        """
        skewed = MeshVariable(
            name="ux",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([1.0, 2.0]),
            dimensions=["nFaces", "time"],
        )
        other = _variable("uy", [0.0, 1.0])
        dataset = _dataset(ux=skewed, uy=other)
        with pytest.raises(ValueError) as caught:
            dataset.plot_vector_field("ux", "uy")
        message = str(caught.value)
        assert "plot_vector_field" in message, (
            f"the refusal must be branded, got: {message}"
        )
        assert "ux" in message, f"the refusal must name the variable, got: {message}"


class TestRefusals:
    """The inputs the facade refuses instead of drawing arrows in the wrong place."""

    def test_an_unknown_variable_is_named(self, velocity):
        """A typo reports the available names, as the rest of the dataset API does."""
        with pytest.raises(KeyError, match="nope"):
            velocity.plot_vector_field("nope", "ucy")

    @pytest.mark.parametrize("role", ["u", "v"], ids=["u-component", "v-component"])
    def test_a_component_with_no_loaded_data_is_refused(self, role):
        """Either side being unloaded is refused, and the message says which.

        Test scenario:
            Checking one component and trusting the other is how a half-loaded pair reaches
            the renderer, so both are validated the same way.
        """
        empty = MeshVariable(
            name="empty", location="face", mesh_name="mesh2d", shape=(2,)
        )
        loaded = _variable("loaded", [1.0, 0.0])
        names = ("empty", "loaded") if role == "u" else ("loaded", "empty")
        dataset = _dataset(empty=empty, loaded=loaded)
        with pytest.raises(ValueError, match=f"{role} component 'empty' has no loaded"):
            dataset.plot_vector_field(*names)

    def test_an_edge_located_component_is_refused_with_the_supported_locations(self):
        """D-Flow FM's `mesh2d_u1` is edge-located, and is named rather than mis-drawn.

        Test scenario:
            A vector can only be anchored at a face centroid or a node. Silently treating an
            edge-located component as face-located would place every arrow wrong, so the
            refusal names both supported locations.
        """
        dataset = _dataset(
            u1=_variable("u1", [1.0, 0.0, 1.0, 0.0, 1.0], location="edge"),
            ucy=_variable("ucy", [0.0, 1.0]),
        )
        with pytest.raises(ValueError) as caught:
            dataset.plot_vector_field("u1", "ucy")
        message = str(caught.value)
        assert "edge-located" in message
        assert "'face', 'node'" in message, (
            f"the refusal should name the supported locations, got:\n{message}"
        )

    def test_components_on_different_locations_are_refused(self):
        """A node-located u with a face-located v is not a vector field."""
        dataset = _dataset(
            un=_variable("un", [1.0, 0.0, 1.0, 0.0], location="node"),
            ucy=_variable("ucy", [0.0, 1.0]),
        )
        with pytest.raises(ValueError, match="same place"):
            dataset.plot_vector_field("un", "ucy")

    def test_components_of_different_element_counts_are_refused(self):
        """Matching locations but different element counts is still not a pair.

        Test scenario:
            Both variables claim `location="face"`, so the location check passes and only the
            shape comparison catches it. Reaching cleopatra would raise a broadcasting error
            naming neither variable. Note that a 2-D face variable is *not* this case: its
            leading axis reads as time and is reduced away, so the pair has to differ in its
            element count to be mismatched at the renderer.
        """
        dataset = _dataset(
            ucx=_variable("ucx", [1.0, 0.0, 1.0]),
            ucy=_variable("ucy", [0.0, 1.0]),
        )
        with pytest.raises(ValueError, match="the two components must match"):
            dataset.plot_vector_field("ucx", "ucy")

    def test_missing_cleopatra_wins_over_a_colour_bar_typo(self, velocity, monkeypatch):
        """With both problems present, the more actionable error must win (round 2 M4).

        Test scenario:
            Round 1 moved the loose-cbar rejection ahead of `_mesh_to_glyph` to save a glyph
            construction on a typo. `require_cleopatra` lives inside `_mesh_to_glyph`, so
            that made a caller with no usable cleopatra *and* a `cbar_label` typo hear about
            the typo -- and it left this function ordering the two checks the opposite way
            from its sibling `plot_mesh_data`.
        """
        from pyramids.base._errors import OptionalPackageDoesNotExist
        from pyramids.netcdf.ugrid import plot as plot_module

        class GlyphWithoutVectorField:
            """Stands in for a cleopatra older than 0.44.0."""

        monkeypatch.setattr(
            plot_module,
            "_mesh_to_glyph",
            lambda mesh, **kwargs: GlyphWithoutVectorField(),
        )
        with pytest.raises(OptionalPackageDoesNotExist):
            velocity.plot_vector_field("ucx", "ucy", cbar_label="x")


class TestNoDataAndShape:
    """Gaps and ranks the components are reduced through before they are drawn."""

    def test_a_no_data_component_is_blanked_to_nan(self):
        """A declared fill must not reach the renderer as a real vector (H2).

        Test scenario:
            A D-Flow FM map file writes its `_FillValue` into `mesh2d_ucx` / `mesh2d_ucy` for
            dry cells, which is the feature's stated use case. Handing `-999` to cleopatra
            made the colour scale run to 1413 while the real arrows were 1.4 units long, so
            they were invisible and the colour bar meaningless.

            Asserted on the magnitude array rather than on `im.U`: cleopatra fills a blanked
            component back in for the arrow geometry, so `im.U` is not NaN -- what carries the
            gap is the masked scalar array the colour mapping reads.
        """
        dry = MeshVariable(
            name="ux",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([1.0, -999.0]),
            nodata=-999.0,
        )
        wet = MeshVariable(
            name="uy",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([1.0, -999.0]),
            nodata=-999.0,
        )
        dataset = _dataset(ux=dry, uy=wet)
        _fig, _ax, im = dataset.plot_vector_field("ux", "uy")
        magnitudes = im.get_array()
        assert magnitudes.mask[1], (
            f"the dry face must be masked out of the colour mapping, got {magnitudes!r}"
        )
        assert not magnitudes.mask[0], (
            f"the wet face must still be coloured, got {magnitudes!r}"
        )
        assert magnitudes[0] == pytest.approx(np.hypot(1.0, 1.0)), (
            f"the wet face keeps its own magnitude, got {magnitudes[0]}"
        )

    def test_the_colour_scale_is_not_dominated_by_the_fill(self):
        """With the fill blanked, the magnitude range reflects the real vectors only (H2)."""

        def component(name, values):
            return MeshVariable(
                name=name,
                location="face",
                mesh_name="mesh2d",
                shape=(2,),
                _data=np.asarray(values, dtype="float64"),
                nodata=-999.0,
            )

        dataset = _dataset(
            ux=component("ux", [1.0, -999.0]), uy=component("uy", [1.0, -999.0])
        )
        _fig, _ax, im = dataset.plot_vector_field("ux", "uy")
        low, high = im.get_clim()
        assert high < 10.0, (
            f"the fill magnitude (1412.8) must not set the upper limit, got {high}"
        )

    def test_a_layered_component_is_refused_by_pyramids(self):
        """A non-temporal layered variable is named here, not left to cleopatra (L5).

        Test scenario:
            A `(n_layers, n_face)` variable reports `has_time` False, so no reduction happens
            and a 2-D array reached the renderer, which answered `u and v must be 1D, got 2D
            and 2D` -- naming neither the variable nor pyramids. The shape-equality check
            passed it through because both components had the same wrong rank.
        """
        layered = MeshVariable(
            name="ux",
            location="face",
            mesh_name="mesh2d",
            shape=(3, 2),
            _data=np.ones((3, 2)),
            dimensions=["nLayers", "nFaces"],
        )
        other = MeshVariable(
            name="uy",
            location="face",
            mesh_name="mesh2d",
            shape=(3, 2),
            _data=np.ones((3, 2)),
            dimensions=["nLayers", "nFaces"],
        )
        dataset = _dataset(ux=layered, uy=other)
        with pytest.raises(ValueError) as caught:
            dataset.plot_vector_field("ux", "uy")
        message = str(caught.value)
        assert "plot_vector_field" in message, (
            f"the refusal must be branded by pyramids, got: {message}"
        )
        assert "ux" in message, f"the refusal must name the variable, got: {message}"

    def test_a_non_numeric_component_is_not_misdiagnosed_as_a_rank_problem(self):
        """A dtype the reducer cannot float must not be branded a rank problem (L1).

        Test scenario:
            The rebranding wrapper caught every `ValueError` the shared reducer could raise
            and called it "not a single value per element". `_element_values` also funnels
            `gaps_as_nan`, which refuses a non-float-convertible dtype -- so a 1-D,
            per-element string component was diagnosed as having the wrong rank, which is
            flatly untrue. The rank is now checked directly, so this error arrives as
            itself.
        """
        text = MeshVariable(
            name="ux",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array(["a", "b"]),
        )
        dataset = _dataset(ux=text, uy=_variable("uy", [0.0, 1.0]))
        with pytest.raises(ValueError) as caught:
            dataset.plot_vector_field("ux", "uy")
        message = str(caught.value)
        assert "not a single value per element" not in message, (
            f"a 1-D component does not have the wrong rank; got: {message}"
        )
        assert "convert string to float" in message, (
            f"the real cause is the dtype, and it must survive; got: {message}"
        )


class TestCleopatraFloor:
    """The 0.44.0 floor the vector field needs is enforced, not just declared."""

    def test_a_cleopatra_without_the_capability_is_branded(self, velocity, monkeypatch):
        """A glyph with no `plot_vector_field` raises the branded upgrade hint (L1).

        Test scenario:
            `require_cleopatra` only checks that cleopatra is importable, and the guarded
            import in `_plot_helpers` brands only the 0.38-era names -- so a cleopatra 0.43
            satisfied every guard and the call died on a bare
            `AttributeError: 'MeshGlyph' object has no attribute 'plot_vector_field'`.
        """
        from pyramids.base._errors import OptionalPackageDoesNotExist
        from pyramids.netcdf.ugrid import plot as plot_module

        class GlyphWithoutVectorField:
            """Stands in for a cleopatra older than 0.44.0."""

        monkeypatch.setattr(
            plot_module,
            "_mesh_to_glyph",
            lambda mesh, **kwargs: GlyphWithoutVectorField(),
        )
        with pytest.raises(OptionalPackageDoesNotExist) as caught:
            velocity.plot_vector_field("ucx", "ucy")
        message = str(caught.value)
        assert "0.44.0" in message, (
            f"the message must name the floor that provides it, got: {message}"
        )
        assert "plot_vector_field" in message, (
            f"the message must name the missing capability, got: {message}"
        )
