# Testing in CI

This page describes how the pyramids test suite is sliced in CI, which optional
dependencies each slice exercises, and how to reproduce any CI job locally.

## One picture

```
              ┌────────────────────────────────────────────────────┐
              │              .github/workflows/tests.yml           │
              ├──────────────────────┬─────────────────────────────┤
              │   main-package       │   extras-package            │
              │                      │                             │
              │   9 matrix cells     │   6 matrix cells            │
              │   (OS × py-version)  │   (OS × extra)              │
              │                      │                             │
              │   env: py311/12/13   │   env: lazy /               │
              │                      │        parquet               │
              │   suite: full repo   │                             │
              │                      │                             │
              │   gate: cov ≥ 88%    │   suite: one sub-dir        │
              │                      │   gate: cov ≥ 0 %           │
              └──────────────────────┴─────────────────────────────┘
```

Everything converges on Codecov; Codecov merges every upload for a given commit
SHA and shows the combined coverage on the dashboard.

## The optional-dependency groups

pyramids ships a minimal core plus five opt-in extras declared in
[`pyproject.toml`](../../pyproject.toml) under
`[project.optional-dependencies]`. Each extra turns on a feature family:

| Extra | Headline dep(s) | Turns on |
| --- | --- | --- |
| `viz` | `cleopatra` | Plotting helpers, basemaps |
| `lazy` | `dask`, `distributed`, `zarr`, `fsspec`, `kerchunk`, `h5py` | Dask-backed lazy array paths + kerchunk NetCDF manifests |
| `parquet` | `pyarrow`, `dask-geopandas` (+ `[lazy]`) | Eager GeoParquet I/O **and** the lazy `LazyFeatureCollection` |

`xarray` is **not** an extra: pyramids is GDAL-backed, so xarray is a peer for the
`to_xarray` / `from_xarray` / `to_netcdf` interop helpers only — `pip install xarray`
directly. It is still wired as a pixi *feature* (`[tool.pixi.feature.xarray]`) so the
`dev` / CI environments have it to exercise those code paths.

End-users install exactly what they need:

```bash
pip install "pyramids-gis[lazy]"          # lazy raster + NetCDF + kerchunk
pip install "pyramids-gis[parquet]"       # eager + lazy GeoParquet (pyarrow + dask-geopandas)
```

## The pixi environments

Each extras group has a dedicated pixi env with the same name as the extra
(minus the `pyramids-gis[…]` wrapping). The envs share `solve-group = "default"`
so incremental solves stay cheap (~8–15 s when adding one extra on top of `dev`).

| Pixi env | What it contains |
| --- | --- |
| `dev` | Core + `viz` + `xarray` + `lazy`. The "everything except heavy natives" env. |
| `lazy` | `dev` + the Dask/Zarr lazy stack and the HDF5-linking bits from conda-forge (kerchunk, h5py — pinned to hdf5 1.14). |
| `parquet` | `dev` + `pyarrow` + `dask-geopandas` (conda-forge) + `[lazy]`. Runs both eager and lazy GeoParquet tests. |
| `py311` / `py312` / `py313` / `py314` | Same as `dev`, pinned Python. What `main-package` CI uses. |
| `docs` | Everything needed to build the MkDocs site. |
| `notebook` | Jupyter + viz only. Used to validate the example notebooks. |

Why some deps are duplicated between `[project.optional-dependencies]` and
`[tool.pixi.feature.*.dependencies]`: the PyPI extra controls what `pip install`
pulls in, while the pixi feature controls what conda-forge resolves for local
development. They coincide in name but they serve different audiences. We only
duplicate a dep between the two blocks when conda-forge needs a different pin
than PyPI — for example the lazy feature pins `hdf5 = "1.14.*"` and a
specific `netcdf4` build string so the HDF5 shared library matches
`libgdal-netcdf` on Windows. See [ADR-HDF5](../adr/) (or the inline comment in
`pyproject.toml`) for the full story.

## What tests run where

| Where in `tests/` | Requires | Main-package CI | Extras-package CI | Local env |
| --- | --- | --- | --- | --- |
| `tests/base/` | core | ✅ | — | `dev` |
| `tests/config/` | core | ✅ | — | `dev` |
| `tests/dataset/` (excl. sub-dirs below) | core | ✅ | — | `dev` |
| `tests/dataset/cog/` | core | ✅ | — | `dev` |
| `tests/dataset/ops/` (excl. lazy sub-dir) | core + parts of `lazy` | ✅ | — | `dev` |
| `tests/dataset/ops/lazy/` | `lazy` | ✅ | — | `dev` |
| `tests/dataset/collection/` | `lazy` (+ kerchunk seam, auto-skip) | ✅ | — | `dev` |
| `tests/dataset/ops/test_zonal_stats.py` | core | ✅ | — | `dev` |
| `tests/dataset/test_stac.py` | core (duck-typed, no pystac dep) | ✅ | — | `dev` |
| `tests/feature/` (excl. `lazy/`) | core + `parquet` (some) | ✅ | — | `dev` |
| `tests/feature/lazy/` | `parquet` | ✅ (skips) | ✅ `parquet` | `parquet` |
| `tests/netcdf/` (excl. `lazy/`) | `xarray` | ✅ (xarray-tests step) | — | `dev` |
| `tests/netcdf/lazy/` | `lazy` | ✅ (most skip without kerchunk) | ✅ `lazy` | `lazy` |
| `tests/ugrid/` | `xarray` | ✅ | — | `dev` |

"Skips" in the main-package column means the tests are tagged or guarded so
they self-skip when the extra's headline dep is missing. They *run* fully
(no skip) in the corresponding extras-package job.

## Running each slice locally

### Full suite in the minimal dev env

```bash
pixi run -e dev main                 # "main" task: pytest -m 'not plot and not xarray'
pixi run -e dev plot                 # plot tests only
pixi run -e dev xarray-tests         # xarray-marker tests only
pixi run -e dev test-all             # everything in one go
```

### Each extras env, scoped to the directory CI runs there

```bash
pixi run -e lazy         pytest -vvv tests/netcdf/lazy
pixi run -e parquet pytest -vvv tests/feature/lazy
```

These two commands mirror the two extras-package matrix entries. If they pass
locally and `dev` passes too, CI should be green.

### First-time install

```bash
pixi install -e dev
# Opt in to the extras you need to hack on:
pixi install -e lazy
pixi install -e parquet
```

Each extras env reuses the `default` solve group, so adding one on top of an
existing `dev` install is 8–15 s per env (not a full re-solve).

## Markers

Every optional-dependency group has a registered pytest marker of the same name
(underscored where the extra uses a hyphen — pytest markers must be valid Python
identifiers):

| Marker | Maps to extra |
| --- | --- |
| `@pytest.mark.plot` | `viz` (plotting / basemap tests) |
| `@pytest.mark.lazy` | `lazy` |
| `@pytest.mark.xarray` | `xarray` (peer dep / pixi feature, not an extra) |
| `@pytest.mark.netcdf_lazy` | `lazy` (kerchunk + h5py) |
| `@pytest.mark.parquet` | `parquet` |
| `@pytest.mark.parquet_lazy` | `parquet` (dask-geopandas) |

`tests/conftest.py` runs a `pytest_collection_modifyitems` hook that, for each
test tagged with one of these markers, auto-applies the matching
`pytest.mark.skipif` when the extra's headline dep isn't importable. This means
a test author writes:

```python
@pytest.mark.netcdf_lazy
def test_kerchunk_roundtrip(tmp_path):
    ...
```

and it runs wherever `[lazy]` is installed (the `lazy` env, and `dev` —
`lazy` now ships kerchunk + h5py), auto-skipping only where those are
absent. No manual `try/except ImportError` + `skipif` boilerplate required.

The skip-decorator aliases (`requires_netcdf_lazy`, `requires_parquet_lazy`, …)
live in `tests/_marks.py` and are still available for inline use on individual
test methods when a full class-level marker is overkill.

### Filtering

```bash
pytest -m netcdf_lazy                           # only netcdf-lazy-tagged tests
pytest -m "not (viz or lazy or xarray or ...)"  # drop every extras-gated test
pytest -m "parquet_lazy and not slow"           # compose with other markers
```

## Coverage

The `main-package` job enforces a **88 % line + branch gate** (`--cov-fail-under=88`
in `.github/workflows/tests.yml`). The `extras-package` jobs pass
`--cov-fail-under=0` because each one is intentionally narrow-scope and its
partial coverage would tank the gate on its own. Both job types upload their
`coverage.xml` to Codecov; Codecov merges them by commit SHA and reports one
combined number on the PR.

If you want a merged number gated in CI itself (rather than relying on
Codecov's dashboard), see the follow-up notes in the PR that added the
extras-package matrix — adding a `coverage-combine` job that downloads each
job's `.coverage` artifact, runs `coverage combine`, and enforces a single
`--fail-under` against the merged total is the standard pattern.

## Troubleshooting

### "Required test coverage of 88 % not reached"

Either real regression, or a test suite failure that dropped some lines out of
the numerator. Look at the failure list above the coverage summary — every
failed test contributes zero coverage. Fix the failures first, then re-check.

### "import file mismatch: imported module 'test_read' has this `__file__`"

Two test files share a basename across different directories, and the dirs are
missing `__init__.py`. Pytest's prepend import mode can't disambiguate.
Every directory under `tests/` needs an `__init__.py`; check that the new dir
has one.

### "DLL load failed while importing defs" (h5py / netcdf4 on Windows)

HDF5 ABI skew — something pulled in a second HDF5 shared library that doesn't
match the one `libgdal-netcdf` links against. In the `lazy` env the fix
is a pinned `hdf5 = "1.14.*"` and an `h5py < 3.16` pin in
`[tool.pixi.feature.lazy.dependencies]`. If you see this error in a
different context, run `pixi list -e <env> | grep -iE "hdf5|h5py|netcdf4"` and
check every row has the same major.minor HDF5 version.

### "plugin gdal_HDF5.so is not available"

The `GDAL_DRIVER_PATH` env variable points at a directory that doesn't exist
— usually a leak from a test that set a fake path and didn't restore. If you
see `/fake/conda/prefix/...` in the error, the culprit is
`tests/config/test_config.py::TestConfigMock`; that class snapshots and
restores the env in `setUp`/`tearDown` so the fake path can't outlive a
single test.

### "backend must be 'pandas' or 'dask', got '…'"

A reader call used a backend name that isn't supported. `FeatureCollection.read_file`
and `read_parquet` both accept `backend="pandas"` (the eager default) or
`backend="dask"` (requires the `parquet` extra).
