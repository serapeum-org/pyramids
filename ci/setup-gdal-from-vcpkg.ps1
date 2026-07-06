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
# so this is the from-source path. The dependency set lives in
# ci/vcpkg.json (manifest mode, pinned builtin-baseline).
$ErrorActionPreference = "Stop"

$BuildPrefix = if ($env:BUILD_PREFIX) { $env:BUILD_PREFIX } else { "C:\gdal-prefix" }
$Triplet = if ($env:VCPKG_TRIPLET) { $env:VCPKG_TRIPLET } else { "arm64-windows" }
$VcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
if (-not $VcpkgRoot) {
    throw "VCPKG_INSTALLATION_ROOT is not set (expected preinstalled vcpkg on hosted runners)"
}
if ($VcpkgRoot -ne "C:\vcpkg") {
    # The workflow caches C:\vcpkg\installed by literal path; a relocated
    # vcpkg would silently turn every build into a 40-70 min cache miss.
    throw "VCPKG_INSTALLATION_ROOT is '$VcpkgRoot' but the workflow cache step assumes C:\vcpkg - update both"
}
$InstallRoot = Join-Path $VcpkgRoot "installed"
$ManifestRoot = $PSScriptRoot   # this script lives in ci/, next to vcpkg.json

Write-Host "=== vcpkg GDAL stack: triplet=$Triplet -> $BuildPrefix ==="

# 1. Put the vcpkg tree EXACTLY at the manifest's pinned builtin-baseline.
#    The runner image ships an arbitrary-age vcpkg checkout, and manifest
#    mode reads the ports + version database from the WORKING TREE while
#    reading baseline.json from the pinned commit — a tree older than the
#    baseline fails with "no version database entry for <port> at <ver>"
#    (observed for curl/tiff/sqlite3/json-c/libjpeg-turbo), and a tree
#    newer than it would silently drift. Fetch the commit (reachable-SHA
#    fetch, full fetch as fallback) and check it out before bootstrapping,
#    so the vcpkg binary matches the tree it runs against.
$Baseline = (Get-Content (Join-Path $ManifestRoot "vcpkg.json") -Raw |
    ConvertFrom-Json)."builtin-baseline"
& git -C $VcpkgRoot fetch origin $Baseline
if ($LASTEXITCODE -ne 0) {
    Write-Host "direct baseline fetch failed; falling back to a full fetch"
    & git -C $VcpkgRoot fetch origin
}
& git -C $VcpkgRoot cat-file -e "$Baseline^{commit}"
if ($LASTEXITCODE -ne 0) {
    throw "vcpkg clone still lacks baseline commit $Baseline after fetch"
}
& git -C $VcpkgRoot -c advice.detachedHead=false checkout --force $Baseline
if ($LASTEXITCODE -ne 0) { throw "git checkout of vcpkg baseline failed" }

# 2. Bootstrap vcpkg (fetches the vcpkg binary matching the checked-out
#    tree).
& "$VcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
if ($LASTEXITCODE -ne 0) { throw "bootstrap-vcpkg failed ($LASTEXITCODE)" }

# 3. Install the manifest. --x-install-root keeps the tree in a stable,
#    cacheable location (the workflow caches it keyed on vcpkg.json + this
#    script + the triplet).
& "$VcpkgRoot\vcpkg.exe" install `
    --triplet $Triplet `
    --x-manifest-root=$ManifestRoot `
    --x-install-root=$InstallRoot `
    --feature-flags=versions,manifests
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed ($LASTEXITCODE)" }

# 4. Mirror the triplet tree into the conda-style Library/ layout the
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
if (-not (Test-Path $GdalLib)) {
    throw "gdal.lib missing from the vcpkg tree - the GDAL binding link (gdal_i.lib) would fail later"
}
Copy-Item -Path $GdalLib -Destination "$BuildPrefix\Library\lib\gdal_i.lib" -Force

# Third-party license collection: vcpkg installs every port's license text
# at share/<port>/copyright. Gather them under the directory
# install-and-vendor-osgeo.py's from-source branch mirrors into the wheel's
# _licenses/ — without this the wheel would vendor GDAL/PROJ/GEOS/curl/...
# binaries with no redistribution notices.
$LicenseRoot = "$BuildPrefix\Library\share\pyramids-bundled-licenses"
New-Item -ItemType Directory -Force -Path $LicenseRoot | Out-Null
Get-ChildItem -Directory (Join-Path $TripletDir "share") | ForEach-Object {
    $copyright = Join-Path $_.FullName "copyright"
    if (Test-Path $copyright) {
        $dst = Join-Path $LicenseRoot $_.Name
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item -Path $copyright -Destination (Join-Path $dst "copyright") -Force
    }
}
$Collected = (Get-ChildItem -Directory $LicenseRoot).Count
if ($Collected -lt 10) {
    throw "license collection suspiciously small (${Collected} ports) - vcpkg share layout changed?"
}
Write-Host "collected license texts for ${Collected} vcpkg ports"

# 5. Resolve the GDAL version vcpkg actually installed and write the
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
                        "$BuildPrefix\Library\share\gdal",
                        "$BuildPrefix\Library\share\proj\proj.db")) {
    if (-not (Test-Path $required)) {
        throw "expected artifact missing after vcpkg install: $required"
    }
}
Write-Host "=== vcpkg GDAL stack staged at $BuildPrefix ==="
