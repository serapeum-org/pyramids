"""Every optional-extra guard composes its hint, none writes one out.

Eleven guards across ten modules had each typed the same two install commands
under their own lead sentence, so adding an extra or changing a command meant
finding all of them. They now call `extra_hint`, and this pins that: the shipped
source may not contain a hand-written install block for a pyramids extra.

The messages themselves are asserted here too, because collapsing them is only
safe if what users see did not move.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import pyramids
from pyramids.base._utils import extra_hint

pytestmark = pytest.mark.core

SRC = pathlib.Path(pyramids.__file__).parent

# `extra_hint` itself, and the one guard whose PyPI line carries an annotation
# ("… (which pins `cleopatra[tiles]`)") the composer cannot produce.
EXEMPT = {"base/_utils.py", "basemap/basemap.py"}


def _runtime_strings(source: str) -> list[str]:
    """Every string literal in `source` that is not a docstring.

    Docstrings are prose: they document the extra in RST for the rendered API
    reference and cannot call a helper. Only the strings a guard actually
    raises are in scope here.
    """
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _offenders(fragment: str) -> list[str]:
    """Shipped modules whose runtime strings contain `fragment`."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        name = path.relative_to(SRC).as_posix()
        if name in EXEMPT:
            continue
        source = path.read_text(encoding="utf-8")
        if fragment in source and any(fragment in s for s in _runtime_strings(source)):
            found.append(name)
    return found


class TestNoHandWrittenInstallBlocks:
    """The composer is the only place the commands appear."""

    def test_only_the_helper_spells_out_the_pypi_command(self):
        """A guard writing the command itself would drift from the rest."""
        offenders = _offenders("pip install 'pyramids-gis[")

        assert offenders == [], (
            "these modules hand-write the PyPI install command instead of "
            f"calling extra_hint(): {offenders}"
        )

    def test_the_conda_command_is_equally_centralised(self):
        """Both halves of the block move together or neither does."""
        offenders = _offenders("conda install -c conda-forge pyramids-")

        assert offenders == [], f"hand-written conda command in: {offenders}"


class TestTheCollapsedMessagesAreUnchanged:
    """What each collapsed guard now raises, spelled out."""

    @pytest.mark.parametrize(
        ("lead", "extra"),
        [
            (
                "GeoParquet support requires the optional 'pyarrow' dependency.",
                "parquet",
            ),
            (
                "backend='dask' requires the optional 'dask-geopandas' dependency.",
                "parquet",
            ),
            ("search requires the optional 'pystac-client' dependency.", "stac"),
        ],
    )
    def test_a_collapsed_guard_still_names_its_extra(self, lead: str, extra: str):
        """The lead is the guard's; the two commands track its extra."""
        hint = extra_hint(lead, extra)

        assert hint.startswith(lead)
        assert f"pip install 'pyramids-gis[{extra}]'" in hint
        assert f"conda install -c conda-forge pyramids-{extra}" in hint

    def test_the_reprojection_guard_message_is_verbatim(self):
        """Pinned in full: this is the string users read on a missing dask."""
        from pyramids.dataset.ops.reproject import _LAZY_IMPORT_ERROR

        assert _LAZY_IMPORT_ERROR == (
            "Lazy reprojection (compute=False) requires the optional 'dask' "
            "dependency. Install with one of:\n"
            "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
            "  - conda-forge: conda install -c conda-forge pyramids-lazy"
        )

    def test_the_stac_guard_message_is_verbatim(self):
        """Same, for the extra whose name differs from its import."""
        from pyramids.stac.search import _STAC_INSTALL_HINT

        assert _STAC_INSTALL_HINT == (
            "search requires the optional 'pystac-client' dependency. "
            "Install with one of:\n"
            "  - PyPI:        pip install 'pyramids-gis[stac]'\n"
            "  - conda-forge: conda install -c conda-forge pyramids-stac"
        )


class TestGuardsThatAppendTheirOwnLine:
    """Two guards add a line the composer cannot know about."""

    def test_the_stac_asset_caveat_survives_the_block(self):
        """`[stac]` carries stac-asset on PyPI, but conda-forge cannot."""
        from pyramids.stac.download import _STAC_ASSET_INSTALL_HINT

        assert _STAC_ASSET_INSTALL_HINT == (
            "download_item requires the optional 'stac-asset' dependency. "
            "Install with one of:\n"
            "  - PyPI:        pip install 'pyramids-gis[stac]'\n"
            "  - conda-forge: conda install -c conda-forge pyramids-stac\n"
            "                 (stac-asset is not on conda-forge; install it "
            "alone: pip install stac-asset)"
        )

    def test_the_mesh_plot_hint_keeps_its_upstream_link(self):
        """A third route -- cleopatra direct -- is this guard's own."""
        from pyramids.netcdf.ugrid.plot import _CLEOPATRA_MSG

        assert _CLEOPATRA_MSG == (
            "Mesh plotting requires the cleopatra package. Install with one of:\n"
            "  - PyPI:        pip install 'pyramids-gis[viz]'\n"
            "  - conda-forge: conda install -c conda-forge pyramids-viz\n"
            "  - or see https://github.com/serapeum-org/cleopatra"
        )


class TestTwoDriftedGuardsNowMatchTheRest:
    """Two guards had drifted, and converging them changed what users read.

    Both changes are improvements rather than regressions, which is why they
    were made rather than exempted -- but they are the only two messages in
    this commit whose text moved, so both are pinned here.
    """

    def test_the_parquet_hint_now_names_the_pyramids_extra_for_conda(self):
        """It used to send conda users to bare pyarrow, unlike its three peers."""
        from pyramids.netcdf.labeled import _PARQUET_INSTALL_HINT

        assert "conda install -c conda-forge pyramids-parquet" in _PARQUET_INSTALL_HINT
        assert "conda install -c conda-forge pyarrow" not in _PARQUET_INSTALL_HINT

    def test_the_viz_hint_now_offers_conda_at_all(self):
        """It used to give a one-line pip-only instruction."""
        from pyramids.plot import _VIZ_HINT

        assert _VIZ_HINT.startswith(
            "The pyramids plotting specs require cleopatra (the [viz] extra)."
        )
        assert "conda install -c conda-forge pyramids-viz" in _VIZ_HINT
