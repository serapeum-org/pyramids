# Third-Party Licenses

`pyramids-gis` itself is licensed under the **GNU General Public License v3 or
later** — see [LICENSE.md](../LICENSE.md).

The platform wheels published on PyPI **bundle** GDAL and its native dependencies
so that `pip install pyramids-gis` yields a self-contained install with no
out-of-band compiler or system GDAL. Each bundled library has its own copyright
and license, listed below. The full license text for every entry is shipped
inside the wheel under `pyramids/_licenses/<package>/`. The mirror source
depends on the build model: **Linux** wheels compile the stack from source
(`ci/source-build/`) and copy each dependency's `LICENSE`/`COPYING` from its
source tree; **macOS/Windows** wheels extract conda-forge packages and mirror
each package's `info/licenses/` directory. The exact bundled set therefore
differs per platform (the Linux wheel carries a leaner, curated stack — no
HDF4, SpatiaLite, Xerces-C, ICU, or JPEG-XL). Source-distribution (sdist)
users build their own GDAL out-of-band, so the sdist does not bundle
third-party binaries and the licenses below do not apply to it.

If you redistribute a `pyramids-gis` platform wheel — directly or as a component
of a larger product — the MIT, BSD, LGPL, and Apache notices below must travel
with it. The most practical way to satisfy this is to leave the
`pyramids/_licenses/` directory untouched inside the wheel.

## Bundled libraries

| Component | Role | License (SPDX) |
|---|---|---|
| **GDAL** | core C/C++ engine + `osgeo` / `osgeo_utils` SWIG bindings | MIT |
| **PROJ** | cartographic projections + coordinate transforms | MIT |
| **GEOS** | computational geometry (intersections, unions, buffers) | LGPL-2.1-or-later |
| **libtiff** | TIFF read/write (underlies GeoTIFF) | libtiff (BSD-style) |
| **libgeotiff** | GeoTIFF tags on top of libtiff | libgeotiff (BSD-style) |
| **NetCDF-C** | NetCDF-4 / classic reader + writer | NetCDF-C (BSD, UCAR/Unidata) |
| **HDF5** | Hierarchical Data Format v5 (NetCDF-4 storage layer) | BSD (HDF Group / NCSA) |
| **HDF4** | Hierarchical Data Format v4 (HDF-EOS, MODIS, …) | HDF4 (BSD, HDF Group / NCSA) |
| **libcurl** | HTTP / S3 / GS / Azure I/O for VSI cloud paths | curl (MIT-like) |
| **OpenSSL** | TLS for libcurl | Apache-2.0 (OpenSSL 3.x+) |
| **libxml2** | XML parser used by GDAL + PROJ | MIT |
| **libexpat** | XML pull parser used by some GDAL drivers | MIT |
| **libpng** | PNG read/write | libpng (zlib-like) |
| **libjpeg-turbo** | JPEG read/write | IJG (BSD-style) + zlib |
| **libwebp** | WebP read/write | BSD-3-Clause (Google) |
| **zlib** | generic DEFLATE compression | zlib |
| **libdeflate** | faster DEFLATE variant | MIT |
| **zstd** | Zstandard compression (drivers + COG) | BSD-3-Clause + GPL-2.0 (dual) |
| **lz4** | LZ4 compression (some GDAL drivers) | BSD-2-Clause + GPL-2.0 (dual) |
| **bzip2** | bzip2 compression | bzip2-1.0.6 (BSD-like) |
| **liblzma / xz** | LZMA / XZ compression | 0BSD / public domain |
| **libiconv** | charset conversion used by libxml2 / PROJ | LGPL-2.1-or-later |
| **libstdc++** | GCC 13 C++ runtime (Linux wheels) | GPL-3.0-or-later WITH GCC-exception-3.1 |
| **GDAL_DATA** | CRS / prime-meridian / datum tables shipped with libgdal | MIT (with GDAL) |
| **PROJ_DATA** | datum-transformation grids, ellipsoid definitions | MIT (with PROJ) |

Upstream project pages: [GDAL](https://gdal.org), [PROJ](https://proj.org),
[GEOS](https://libgeos.org), [libtiff](http://www.libtiff.org),
[libgeotiff](https://github.com/OSGeo/libgeotiff),
[NetCDF-C](https://www.unidata.ucar.edu/software/netcdf),
[HDF5](https://www.hdfgroup.org/solutions/hdf5),
[HDF4](https://www.hdfgroup.org/solutions/hdf4),
[libcurl](https://curl.se), [OpenSSL](https://www.openssl.org),
[libxml2](https://gitlab.gnome.org/GNOME/libxml2),
[libexpat](https://libexpat.github.io),
[libpng](http://www.libpng.org/pub/png/libpng.html),
[libjpeg-turbo](https://libjpeg-turbo.org),
[libwebp](https://chromium.googlesource.com/webm/libwebp),
[zlib](https://zlib.net), [libdeflate](https://github.com/ebiggers/libdeflate),
[zstd](https://github.com/facebook/zstd), [lz4](https://lz4.org),
[bzip2](https://sourceware.org/bzip2), [liblzma](https://tukaani.org/xz),
[libiconv](https://www.gnu.org/software/libiconv),
[libstdc++](https://gcc.gnu.org/onlinedocs/libstdc++).

The exact set of libraries in a given wheel depends on the platform (Linux ships
`libstdcxx-ng`; Windows ships extra DLLs from the conda-forge `Library/bin/`
layout; macOS ships `.dylib` equivalents). The single source of truth for what
shipped in any specific wheel is the `pyramids/_licenses/` directory inside that
wheel.

## How license texts are gathered

`ci/install-and-vendor-osgeo.py` runs in cibuildwheel's `CIBW_BEFORE_BUILD`
hook, after `ci/setup-gdal-from-pixi.{sh,ps1}` materialises the wheel-build
pixi environment. The script walks every `${PIXI_ENV}/conda-meta/*.json`,
reads each package's `extracted_package_dir`, and copies the contents of
`<extracted>/info/licenses/` into `src/pyramids/_licenses/<package-name>/`.
setuptools then packages everything under `src/pyramids/_licenses/` into the
wheel via the `"_licenses/**/*"` glob in `[tool.setuptools.package-data]`.

This guarantees that whichever conda-forge build of GDAL we link against in
a given release also contributes its current upstream LICENSE files into the
wheel — no hand-maintained list of license texts to keep in sync.

## License compatibility with GPLv3

All licenses listed above are compatible with the GNU General Public License
v3 (the license under which `pyramids-gis` itself is distributed):

- **MIT / BSD-2-Clause / BSD-3-Clause / zlib / Boost** — explicitly GPL-compatible
  permissive licenses.
- **LGPL-2.1-or-later** (GEOS, libiconv) — explicitly GPL-compatible.
- **Apache-2.0** (OpenSSL 3.x+) — compatible with GPLv3 (not with GPLv2-only).
- **libtiff, libpng, IJG, HDF4, HDF5, bzip2, NetCDF-C** licenses — all
  BSD-style permissive, GPL-compatible.
- **libstdc++** — GPL-3.0-or-later **with the GCC Runtime Library Exception**,
  which explicitly permits use in non-GPL programs and is GPLv3-compatible by
  construction.

If you redistribute a combined work containing `pyramids-gis` and any of these
libraries, the combined work as a whole is governed by GPLv3 (because
`pyramids-gis` is GPLv3 and absorbs the permissive licenses on combination),
**and** the original notices and license texts of the bundled libraries must
travel with the combined work.

## Citing GDAL

If you use `pyramids-gis` in academic or publication contexts, please also cite
GDAL per https://gdal.org/cite_gdal.html. Pyramids stands on GDAL's shoulders;
none of this would work without that project's two decades of work.

## Reporting issues

If you find a missing attribution, an incorrect license entry, or a bundled
binary whose license text isn't in `pyramids/_licenses/`, please open an issue
at https://github.com/serapeum-org/pyramids/issues — we want to get this right.
