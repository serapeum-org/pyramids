"""Guard meta-test: no `core`/unmarked module pulls an optional dep at module scope.

Prevents the #638 / #645 class of bug — a test module that ``import``s or
``pytest.importorskip``s an optional dependency at *module* scope while still being
selected by the extras-free ``-m core`` pure-wheel job. Such a module either aborts
collection (an unguarded ``import``) or silently skips (a module-level
``importorskip``), so its coverage is lost in the bare-wheel build and reads as green.

Rule: if a test module pulls an optional dependency at MODULE scope, it MUST carry a
matching extras marker (``lazy`` / ``xarray`` / ``netcdf_lazy`` / ``parquet`` /
``parquet_lazy`` / ``plot`` / ``stac`` / ``vfs``) so it is routed off the core
selection. Collection-safe forms are intentionally NOT flagged:

* a guarded import — ``try: import zarr`` / ``except ImportError: zarr = None`` — the
  module still imports without the dep, so per-test ``@pytest.mark.<extra>`` is enough;
* a function-body ``pytest.importorskip(...)`` inside an individual test.

This is a static AST check; it runs in the ``core`` suite and needs no optional deps.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._marks import EXTRA_MARKERS

pytestmark = pytest.mark.core

_TESTS_ROOT = Path(__file__).parent
_EXTRAS = set(EXTRA_MARKERS)

# Top-level module names of the optional dependencies pyramids gates behind extras.
_OPTIONAL_DEPS = {
    "dask",
    "dask_geopandas",
    "distributed",
    "flox",
    "zarr",
    "xarray",
    "kerchunk",
    "h5py",
    "h5netcdf",
    "netCDF4",
    # NB: cftime is intentionally absent — it is a hard `[project]` dependency
    # (production `pyramids.netcdf.utils` imports it unguarded), always present.
    "fsspec",
    "s3fs",
    "pyarrow",
    "cleopatra",
    "pystac",
    "pystac_client",
    "stac_asset",
}


def _marker_names(value: ast.expr) -> set[str]:
    """Marker names from a ``pytestmark = ...`` RHS (single mark or list/tuple of marks)."""
    nodes = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    names: set[str] = set()
    for node in nodes:
        # ``pytest.mark.<name>`` -> Attribute(attr=<name>, value=Attribute(attr="mark"))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            names.add(node.attr)
    return names


def _importorskip_dep(call: ast.Call) -> str | None:
    """Top-level module of a ``pytest.importorskip("x.y")`` / ``importorskip("x")`` call."""
    func = call.func
    is_importorskip = (isinstance(func, ast.Attribute) and func.attr == "importorskip") or (
        isinstance(func, ast.Name) and func.id == "importorskip"
    )
    if (
        is_importorskip
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ):
        return call.args[0].value.split(".")[0]
    return None


def _bare_import_deps(node: ast.stmt) -> list[str]:
    """Optional-dep top-level modules imported by a top-level ``import`` / ``from`` node."""
    if isinstance(node, ast.Import):
        tops = [alias.name.split(".")[0] for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        tops = [(node.module or "").split(".")[0]]
    else:
        return []
    return [top for top in tops if top in _OPTIONAL_DEPS]


def _node_importorskip_dep(node: ast.stmt) -> str | None:
    """Optional dep ``importorskip``ed by a top-level expr/assignment node, else ``None``."""
    # `pytest.importorskip(...)` appears either bare (`Expr`) or assigned (`da = ...`); both
    # expose the call as `node.value`, so they collapse into one branch.
    if isinstance(node, (ast.Expr, ast.Assign)) and isinstance(node.value, ast.Call):
        dep = _importorskip_dep(node.value)
        if dep in _OPTIONAL_DEPS:
            return dep
    return None


def _module_offenders(tree: ast.Module) -> tuple[list[str], list[str]]:
    """Optional deps pulled at MODULE scope (top-level only; guarded/inner are skipped).

    Returns ``(bare_imports, importorskips)``: bare ``import``/``from`` statements abort
    collection regardless of marker, while ``importorskip`` is collection-safe but needs
    the matching extras marker to avoid a silent skip in the core suite.
    """
    bare_imports: list[str] = []
    importorskips: list[str] = []
    for node in tree.body:
        bare_imports += _bare_import_deps(node)
        dep = _node_importorskip_dep(node)
        if dep is not None:
            importorskips.append(dep)
    return bare_imports, importorskips


def _module_markers(tree: ast.Module) -> set[str]:
    """Marker names declared via a module-level ``pytestmark = ...`` assignment."""
    markers: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            markers |= _marker_names(node.value)
    return markers


_TEST_FILES = sorted(
    p for p in _TESTS_ROOT.rglob("test_*.py") if p.name != Path(__file__).name
)


@pytest.mark.parametrize(
    "path", _TEST_FILES, ids=lambda p: str(p.relative_to(_TESTS_ROOT)).replace("\\", "/")
)
def test_module_optional_dep_requires_extras_marker(path: Path):
    """A module pulling an optional dep at module scope must be collection-safe + marked."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bare_imports, importorskips = _module_offenders(tree)
    rel = path.relative_to(_TESTS_ROOT)

    # A bare `import <opt>` runs during collection of EVERY module under `-m core`,
    # so it aborts the bare-wheel run regardless of the module's marker.
    assert not bare_imports, (
        f"{rel} has an UNGUARDED module-level import of optional dependency(ies) "
        f"{sorted(set(bare_imports))}. pytest imports every module during `-m core` "
        f"collection, so this aborts the extras-free pure-wheel run. Use "
        f"`pytest.importorskip(...)` or a guarded `try/except ImportError` instead."
    )

    # A module-level `importorskip` is collection-safe (it skips the module), but if the
    # module is `core`/unmarked it is *selected then silently skipped* in the core suite.
    if importorskips:
        markers = _module_markers(tree)
        assert markers & _EXTRAS, (
            f"{rel} module-level `importorskip`s optional dependency(ies) "
            f"{sorted(set(importorskips))}, but its module markers {sorted(markers) or '{}'} "
            f"include no extras marker ({sorted(_EXTRAS)}). It will silently skip in the "
            f"`-m core` pure-wheel job. Add the matching marker "
            f"(e.g. `pytestmark = pytest.mark.lazy`)."
        )
