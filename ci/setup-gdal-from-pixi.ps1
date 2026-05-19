#Requires -Version 5.1
# Install GDAL + native dependencies via pixi (conda-forge), then extract
# DLLs, headers, and data files into $env:BUILD_PREFIX so downstream steps
# (install-and-vendor-osgeo.py, setuptools, delvewheel) can find them.
#
# Runs once per cibuildwheel Windows invocation (CIBW_BEFORE_ALL).
#
# build-wheels.yml invokes this via `powershell -File ...` which uses
# Windows PowerShell 5.1 on the windows-2022 runner — the `#Requires`
# above documents that minimum so any future move to `pwsh` (PS 7)
# still passes, but a regression to an older PowerShell would fail
# fast instead of silently miscompiling somewhere downstream.
#
# See docs/how-to/wheel-build-flow.md for the end-to-end pipeline.

$ErrorActionPreference = "Stop"

$BuildPrefix = if ($env:BUILD_PREFIX) { $env:BUILD_PREFIX } else { "C:\gdal-prefix" }
Write-Host "=== setup-gdal-from-pixi.ps1 ==="
Write-Host "BUILD_PREFIX=$BuildPrefix"

# 1. Install pixi.
#
# Pin the version via .pixi-version at the repo root so CI, local
# developers, and the cibuildwheel container all agree. PIXI_VERSION
# is consumed by pixi.sh/install.ps1 — the install script verifies
# the downloaded binary's SHA256 against its built-in checksum table,
# so pinning makes the installer reproducible. Override locally with
# `$env:PIXI_VERSION = "X.Y.Z"; .\ci\setup-gdal-from-pixi.ps1`.
if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    $PixiVersionFile = Join-Path (Split-Path -Parent $PSScriptRoot) ".pixi-version"
    if (-not $env:PIXI_VERSION -and (Test-Path $PixiVersionFile)) {
        $env:PIXI_VERSION = (Get-Content $PixiVersionFile -Raw).Trim()
    }
    if (-not $env:PIXI_VERSION) {
        throw "PIXI_VERSION not set and .pixi-version missing at $PixiVersionFile"
    }
    Write-Host "--- Installing pixi $($env:PIXI_VERSION) ---"
    $env:PIXI_HOME = $BuildPrefix
    $env:PIXI_NO_PATH_UPDATE = "1"
    Invoke-WebRequest -Uri "https://pixi.sh/install.ps1" -OutFile "$env:TEMP\install-pixi.ps1"
    & "$env:TEMP\install-pixi.ps1"
    $env:PATH = "$BuildPrefix\bin;$env:PATH"
}
pixi --version

# 2. Install the wheel-build environment using the committed pixi.lock.
Write-Host "--- Resolving wheel-build environment ---"
pixi install -e wheel-build --frozen
if ($LASTEXITCODE -ne 0) { throw "pixi install failed" }

$PixiEnv = Join-Path (Get-Location) ".pixi\envs\wheel-build"
if (-not (Test-Path $PixiEnv)) {
    throw "wheel-build env not found at $PixiEnv"
}
Write-Host "wheel-build env: $PixiEnv"

# Resolve the concrete GDAL version from the env we just materialized
# and persist it for install-and-vendor-osgeo.py. Single source of truth
# = pixi.lock — no more hardcoded duplicates in pyproject.toml or
# build-wheels.yml. conda-forge's Windows gdal package doesn't ship a
# usable gdal-config, so parse the version from conda-meta\gdal-X.Y.Z-*.json
# instead (the same file conda/pixi/micromamba all populate).
$GdalMeta = Get-ChildItem -Path (Join-Path $PixiEnv "conda-meta\gdal-*.json") -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $GdalMeta) {
    throw "gdal package not found in $PixiEnv\conda-meta"
}
if ($GdalMeta.BaseName -match '^gdal-(\d+\.\d+\.\d+)') {
    $GdalVersion = $Matches[1]
} else {
    throw "could not parse version from $($GdalMeta.Name)"
}
New-Item -ItemType Directory -Force -Path $BuildPrefix | Out-Null
[System.IO.File]::WriteAllText((Join-Path $BuildPrefix "GDAL_VERSION"), $GdalVersion)
Write-Host "resolved GDAL_VERSION=$GdalVersion"

# 3. Extract native artifacts into $BuildPrefix.
#
# Windows conda packages nest under Library/: Library/bin (DLLs),
# Library/include (headers), Library/lib (import .lib files),
# Library/share (data).
Write-Host "--- Extracting native artifacts into $BuildPrefix ---"

# Mirror the Library/ layout to keep paths predictable for downstream tools.
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\bin"     | Out-Null
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\include" | Out-Null
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\lib"     | Out-Null
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\share"   | Out-Null

# DLLs
Copy-Item -Path "$PixiEnv\Library\bin\*" -Destination "$BuildPrefix\Library\bin" -Recurse -Force

# Headers
Copy-Item -Path "$PixiEnv\Library\include\*" -Destination "$BuildPrefix\Library\include" -Recurse -Force

# Import libs
if (Test-Path "$PixiEnv\Library\lib") {
    Copy-Item -Path "$PixiEnv\Library\lib\*" -Destination "$BuildPrefix\Library\lib" -Recurse -Force
}

# GDAL_DATA + PROJ_DATA
Copy-Item -Path "$PixiEnv\Library\share\gdal" -Destination "$BuildPrefix\Library\share" -Recurse -Force
Copy-Item -Path "$PixiEnv\Library\share\proj" -Destination "$BuildPrefix\Library\share" -Recurse -Force

# GDAL plugins (NetCDF / HDF4 / HDF5 drivers)
if (Test-Path "$PixiEnv\Library\lib\gdalplugins") {
    New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\lib\gdalplugins" | Out-Null
    Copy-Item -Path "$PixiEnv\Library\lib\gdalplugins\*" -Destination "$BuildPrefix\Library\lib\gdalplugins" -Recurse -Force
}

# 4. Diagnostic output.
Write-Host "=== setup-gdal-from-pixi.ps1 complete ==="
& "$BuildPrefix\Library\bin\gdalinfo.exe" --version
Write-Host "DLLs bundled: $((Get-ChildItem "$BuildPrefix\Library\bin\*.dll").Count)"
