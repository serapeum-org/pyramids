# Pre-commit hooks

This page explains the pyramids [pre-commit](https://pre-commit.com/) setup: what the hooks are, how to install them,
how to trigger them (all, a subset, or one), and how to skip the slow ones. The hook definitions live in
[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml); the fast checks run in isolated tool environments while a few
heavier hooks (`mypy`, `pytest-check`, `doctest`, `notebook-check`, `pixi-lock-check`) shell out to the pixi **dev**
environment.

The same config is the local mirror of CI: the `pre-commit` job in
[`.github/workflows/lint.yml`](../../.github/workflows/lint.yml) runs `pre-commit run --all-files` with the slow test
hooks skipped, so getting a clean local run is the fastest way to keep that job green.

## Install (one-time)

Register the git hook so it fires on every `git commit`:

```bash
pixi run --frozen -e dev pre-commit install
```

`--frozen` makes pixi use `pixi.lock` as-is instead of re-solving every platform on each run (which is slow and can hang
on Windows). The first hook run also builds an isolated toolchain per hook repo (a few minutes, needs network) — that is
expected, not a failure.

## Trigger the hooks

### Automatically, on commit

Once installed, the hooks run against the **staged files** every time you commit:

```bash
git commit -m "feat(dataset): ..."
```

`fail_fast: true` is set, so the first hook that modifies a file (e.g. `ruff-format`) stops the run — re-stage and commit
again until it passes.

### Manually, on the whole tree

Run every hook against all files without committing (this is what CI does):

```bash
pixi run --frozen -e dev pre-commit run --all-files
```

### On specific files only

```bash
pixi run --frozen -e dev pre-commit run --files src/pyramids/dataset/collection.py tests/dataset/collection/test_meta.py
```

To scope the run to just what your branch changed:

```bash
pixi run --frozen -e dev pre-commit run --files $(git diff --name-only origin/main...HEAD)
```

### A single specific hook

Pass the hook **id** (the left column in the table below). Combine with `--all-files` or `--files` to choose the scope:

```bash
pixi run --frozen -e dev pre-commit run ruff-check --all-files       # lint only
pixi run --frozen -e dev pre-commit run ruff-format --all-files      # format only
pixi run --frozen -e dev pre-commit run mypy --all-files             # type-check only
pixi run --frozen -e dev pre-commit run pytest-check --all-files     # the full test suite
```

## The hooks

| id | What it does | Speed |
| --- | --- | --- |
| `check-toml` / `check-json` / `check-yaml` | Validate config file syntax | fast |
| `end-of-file-fixer` / `trailing-whitespace` / `mixed-line-ending` | Whitespace / newline normalisation | fast |
| `pretty-format-json` / `requirements-txt-fixer` | Canonicalise JSON and `requirements.txt` | fast |
| `check-added-large-files` | Block files over 2 MB | fast |
| `check-merge-conflict` / `detect-private-key` / `debug-statements` | Catch conflict markers, keys, stray `breakpoint()` | fast |
| `no-commit-to-branch` | Refuse a commit made directly on `main` | fast |
| `ruff-check` | Lint Python + notebooks (auto-fix) | fast |
| `ruff-format` | Format Python + notebooks | fast |
| `nbstripout` | Strip notebook outputs / execution counts | fast |
| `beautysh` / `shellcheck` | Format and lint shell scripts | fast |
| `check-summary-*` / `check-description-*` / `check-second-line-empty` | Commit-message conventional-commit checks | fast |
| `bandit` | Python security linter | medium |
| `gitleaks` | Scan staged changes for secrets | medium |
| `checkov` | Infrastructure-as-code security scan | medium |
| `mypy` | Static type-check (in the pixi `dev` env) | **slow** |
| `pytest-check` | Full test suite with coverage (`-m "not plot"`) | **slow** |
| `doctest` | `--doctest-modules src` under the Agg backend | **slow** |
| `notebook-check` | `nbval` execution of the example notebooks | **slow** |
| `pixi-lock-check` | Fail if `pixi.lock` is stale vs `pyproject.toml` | medium |

## Skip hooks

### Skip specific hooks with `SKIP`

pre-commit reads the `SKIP` environment variable (a comma-separated list of hook ids) at commit time and at
`pre-commit run` time. Everything else still runs. To skip the slow test trio on a commit:

```bash
# Git Bash — inline, scoped to the single command:
SKIP=pytest-check,doctest,notebook-check git commit -m "docs: ..."
```

```powershell
# PowerShell — persists for the rest of the session (clear with: Remove-Item Env:SKIP):
$env:SKIP = 'pytest-check,doctest,notebook-check'
git commit -m "docs: ..."
```

To make the skip permanent (also applies to commits from the PyCharm/JetBrains commit dialog, which inherits the
environment it was launched with):

```powershell
[Environment]::SetEnvironmentVariable('SKIP', 'pytest-check,doctest,notebook-check', 'User')
```

Restart your shells / IDE afterward. Note that a permanent `SKIP` also excludes those hooks from
`pre-commit run --all-files`, so they never run locally until you clear it — fine if you rely on CI as the backstop.

### The CI skip set

The `lint.yml` job runs everything **except** the hooks it can't or shouldn't run there. To reproduce that job exactly:

```bash
SKIP=no-commit-to-branch,gitleaks,pytest-check,doctest,notebook-check,pixi-lock-check \
  pixi run --frozen -e dev pre-commit run --all-files
```

### Skip every hook

`--no-verify` bypasses **all** hooks for a single commit — use it sparingly, since it also skips the fast formatting and
lint that keep the diff clean:

```bash
git commit --no-verify -m "wip"
```

## Gotchas

- **pixi startup cost.** `pixi run` has a noticeable cold-start on Windows (~15–25 s). That is env warm-up, not a hung
  hook.
- **`--frozen` always.** Without it, pixi re-solves every platform on each hook invocation, which is slow and can hang.
- **`-p no:cacheprovider`.** The pytest-based hooks disable the cache because writing `.pytest_cache` fails with
  `WinError 183` on the Google-Drive-synced tree.
- **Agg backend for doctests.** The `doctest` hook forces `matplotlib.use("Agg")` so no plotting doctest pops a GUI
  window that would block the commit.
- **`no-commit-to-branch` blocks `main`.** Commits must be made on a feature branch; this hook refuses a direct commit on
  `main`.

## See also

- [Testing & CI](testing.md) — how the test suite is sliced across CI jobs and reproduced locally.
- [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) — the source of truth for every hook.
