# Build the GDAL native stack from source via vcpkg and stage it in the
# SAME layout ci/setup-gdal-from-pixi.ps1 produces, so everything
# downstream (ci/install-and-vendor-osgeo.py, the cibuildwheel windows
# environment table, the delvewheel repair command) works unchanged:
#
#   $BuildPrefix/GDAL_VERSION        - version for `pip install GDAL==X.Y.Z`
#   $BuildPrefix/Library/bin         - DLLs (gdal.dll, geos_c.dll, ...)
#   $BuildPrefix/Library/include     - headers
#   $BuildPrefix/Library/lib         - MSVC import .lib files
#   $BuildPrefix/Library/share       - GDAL_DATA / PROJ data
#
# Used by the win_arm64 wheels (#334): conda-forge has no win-arm64 GDAL,
# so this is the from-source path — the rasterio model (their
# build-wheels.yaml builds ALL Windows wheels this way). The dependency
# set lives in ci/vcpkg.json (manifest mode, pinned builtin-baseline).
$ErrorActionPreference = "Stop"

$BuildPrefix = if ($env:BUILD_PREFIX) { $env:BUILD_PREFIX } else { "C:\gdal-prefix" }
$Triplet = if ($env:VCPKG_TRIPLET) { $env:VCPKG_TRIPLET } else { "arm64-windows" }
$VcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
if (-not $VcpkgRoot) {
    throw "VCPKG_INSTALLATION_ROOT is not set (expected preinstalled vcpkg on hosted runners)"
}
$InstallRoot = Join-Path $VcpkgRoot "installed"
$ManifestRoot = $PSScriptRoot   # this script lives in ci/, next to vcpkg.json

Write-Host "=== vcpkg GDAL stack: triplet=$Triplet -> $BuildPrefix ==="

# 1. Bootstrap vcpkg (fetches the vcpkg binary itself; ports come from the
#    manifest's pinned builtin-baseline, not from whatever the runner image
#    happens to ship).
& "$VcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
if ($LASTEXITCODE -ne 0) { throw "bootstrap-vcpkg failed ($LASTEXITCODE)" }

# 2. Install the manifest. --x-install-root keeps the tree in a stable,
#    cacheable location (the workflow caches it keyed on vcpkg.json + this
#    script + the triplet).
& "$VcpkgRoot\vcpkg.exe" install `
    --triplet $Triplet `
    --x-manifest-root=$ManifestRoot `
    --x-install-root=$InstallRoot `
    --feature-flags=versions,manifests
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed ($LASTEXITCODE)" }

# 3. Mirror the triplet tree into the conda-style Library/ layout the
#    vendor script and cibuildwheel environment expect.
$TripletDir = Join-Path $InstallRoot $Triplet
foreach ($d in @("bin", "include", "lib", "share")) {
    $src = Join-Path $TripletDir $d
    $dst = "$BuildPrefix\Library\$d"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    if (Test-Path $src) {
        Copy-Item -Path "$src\*" -Destination $dst -Recurse -Force
    }
}

# GDAL's Python-binding setup.py links `gdal_i.lib` on Windows (the
# historical GDAL import-library name); vcpkg names it `gdal.lib`.
# Provide both so `pip install GDAL==X.Y.Z` links either way.
$GdalLib = "$BuildPrefix\Library\lib\gdal.lib"
if (Test-Path $GdalLib) {
    Copy-Item -Path $GdalLib -Destination "$BuildPrefix\Library\lib\gdal_i.lib" -Force
}

# 4. Resolve the GDAL version vcpkg actually installed and write the
#    GDAL_VERSION file install-and-vendor-osgeo.py keys the pip-binding
#    install on. `vcpkg list gdal` prints e.g. "gdal:arm64-windows  3.12.4#1".
$ListOutput = & "$VcpkgRoot\vcpkg.exe" list --x-install-root=$InstallRoot "gdal"
$GdalVersion = $null
foreach ($line in $ListOutput) {
    if ($line -match "^gdal:$Triplet\s+(\d+\.\d+\.\d+)") {
        $GdalVersion = $Matches[1]
        break
    }
}
if (-not $GdalVersion) {
    throw "could not parse the installed GDAL version from 'vcpkg list gdal': $ListOutput"
}
New-Item -ItemType Directory -Force -Path $BuildPrefix | Out-Null
Set-Content -Path (Join-Path $BuildPrefix "GDAL_VERSION") -Value $GdalVersion -NoNewline
Write-Host "resolved GDAL_VERSION=$GdalVersion"

# Sanity: the DLL and data dirs the wheel build depends on must exist.
foreach ($required in @("$BuildPrefix\Library\bin\gdal.dll",
                        "$BuildPrefix\Library\share\gdal")) {
    if (-not (Test-Path $required)) {
        throw "expected artifact missing after vcpkg install: $required"
    }
}
Write-Host "=== vcpkg GDAL stack staged at $BuildPrefix ==="
