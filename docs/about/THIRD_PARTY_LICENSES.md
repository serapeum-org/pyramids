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

| Component | Role | License (SPDX) | Wheels |
|---|---|---|---|
| **GDAL** | core C/C++ engine + `osgeo` / `osgeo_utils` SWIG bindings | MIT | all |
| **PROJ** | cartographic projections + coordinate transforms | MIT | all |
| **GEOS** | computational geometry (intersections, unions, buffers) | LGPL-2.1-or-later | all |
| **libtiff** | TIFF read/write (underlies GeoTIFF) | libtiff (BSD-style) | all |
| **libgeotiff** | GeoTIFF tags on top of libtiff | libgeotiff (BSD-style) | macOS + Windows¹ |
| **NetCDF-C** | NetCDF-4 / classic reader + writer | NetCDF-C (BSD, UCAR/Unidata) | all |
| **HDF5** | Hierarchical Data Format v5 (NetCDF-4 storage layer) | BSD (HDF Group / NCSA) | all |
| **HDF4** | Hierarchical Data Format v4 (HDF-EOS, MODIS, …) | HDF4 (BSD, HDF Group / NCSA) | macOS + Windows |
| **libcurl** | HTTP / S3 / GS / Azure I/O for VSI cloud paths | curl (MIT-like) | all |
| **OpenSSL** | TLS for libcurl | Apache-2.0 (OpenSSL 3.x+) | all |
| **nghttp2** | HTTP/2 support for libcurl | MIT | all |
| **libxml2** | XML parser used by GDAL + PROJ | MIT | macOS + Windows |
| **libexpat** | XML pull parser used by some GDAL drivers | MIT | all² |
| **OpenJPEG** | JPEG-2000 codec (`JP2OpenJPEG` driver) | BSD-2-Clause | all |
| **lcms2** | color management used by OpenJPEG | MIT | all |
| **libpng** | PNG read/write | libpng (zlib-like) | all |
| **libjpeg-turbo** | JPEG read/write | IJG (BSD-style) + zlib | all |
| **libwebp** | WebP read/write | BSD-3-Clause (Google) | all |
| **giflib** | GIF read/write | MIT | all |
| **LERC** | limited-error raster compression (TIFF/COG) | Apache-2.0 | all |
| **json-c** | JSON parsing inside GDAL | MIT | all |
| **PCRE2** | regular expressions (SQLite `REGEXP`) | BSD-3-Clause | all |
| **SQLite** | GPKG driver + PROJ's CRS database storage | Public domain³ | all |
| **c-blosc** | Blosc compression (Zarr driver) | BSD-3-Clause | all |
| **libaec** | szip-compatible compression for HDF5 | BSD-2-Clause | all |
| **zlib** | generic DEFLATE compression | zlib | all |
| **libdeflate** | faster DEFLATE variant | MIT | all |
| **zstd** | Zstandard compression (drivers + COG) | BSD-3-Clause + GPL-2.0 (dual) | all |
| **lz4** | LZ4 compression (some GDAL drivers) | BSD-2-Clause + GPL-2.0 (dual) | macOS + Windows |
| **bzip2** | bzip2 compression | bzip2-1.0.6 (BSD-like) | macOS + Windows |
| **liblzma / xz** | LZMA / XZ compression | 0BSD / public domain | all |
| **libiconv** | charset conversion used by libxml2 / PROJ | LGPL-2.1-or-later | macOS + Windows |
| **GDAL_DATA** | CRS / prime-meridian / datum tables shipped with libgdal | MIT (with GDAL) | all |
| **PROJ_DATA** | datum-transformation grids, ellipsoid definitions | MIT (with PROJ) | all |

¹ On Linux the GeoTIFF tag support is GDAL's internal libgeotiff copy
(`GDAL_USE_GEOTIFF_INTERNAL=ON`), covered by GDAL's own MIT license; macOS and
Windows bundle conda-forge's separate libgeotiff library.

² Linked statically into libgdal on Linux (no separate `.so` in the wheel);
a shared library on macOS/Windows. Its MIT notice ships either way.

³ SQLite is public domain — there is no license text to ship, which is why
`pyramids/_licenses/` contains no `sqlite` entry even though the library is
linked into every wheel. See https://sqlite.org/copyright.html.

Upstream project pages: [GDAL](https://gdal.org), [PROJ](https://proj.org),
[GEOS](https://libgeos.org), [libtiff](http://www.libtiff.org),
[libgeotiff](https://github.com/OSGeo/libgeotiff),
[NetCDF-C](https://www.unidata.ucar.edu/software/netcdf),
[HDF5](https://www.hdfgroup.org/solutions/hdf5),
[HDF4](https://www.hdfgroup.org/solutions/hdf4),
[libcurl](https://curl.se), [OpenSSL](https://www.openssl.org),
[nghttp2](https://nghttp2.org),
[libxml2](https://gitlab.gnome.org/GNOME/libxml2),
[libexpat](https://libexpat.github.io),
[OpenJPEG](https://www.openjpeg.org),
[lcms2](https://www.littlecms.com),
[libpng](http://www.libpng.org/pub/png/libpng.html),
[libjpeg-turbo](https://libjpeg-turbo.org),
[libwebp](https://chromium.googlesource.com/webm/libwebp),
[giflib](https://giflib.sourceforge.net),
[LERC](https://github.com/Esri/lerc),
[json-c](https://github.com/json-c/json-c),
[PCRE2](https://www.pcre.org), [SQLite](https://sqlite.org),
[c-blosc](https://github.com/Blosc/c-blosc),
[libaec](https://github.com/MathisRosenhauer/libaec),
[zlib](https://zlib.net), [libdeflate](https://github.com/ebiggers/libdeflate),
[zstd](https://github.com/facebook/zstd), [lz4](https://lz4.org),
[bzip2](https://sourceware.org/bzip2), [liblzma](https://tukaani.org/xz),
[libiconv](https://www.gnu.org/software/libiconv).

The exact set of libraries in a given wheel depends on the platform (the Linux
from-source stack is leaner; Windows ships extra DLLs from the conda-forge
`Library/bin/` layout; macOS ships `.dylib` equivalents). The single source of
truth for what shipped in any specific wheel is the `pyramids/_licenses/`
directory inside that wheel.

## How license texts are gathered

`ci/install-and-vendor-osgeo.py` runs in cibuildwheel's `CIBW_BEFORE_BUILD`
hook and fills `src/pyramids/_licenses/` from whichever license source the
build model provides; setuptools then packages the directory into the wheel
via the `"_licenses/**/*"` glob in `[tool.setuptools.package-data]`.

- **Linux (from-source)**: `ci/source-build/build-gdal-stack.sh` copies each
  dependency's `LICENSE`/`COPYING` file out of its extracted source tree into
  `share/pyramids-bundled-licenses/<dep>-<version>/` right after the compile,
  and a CI gate fails the build if any dependency in the pinned stack is
  missing its license dir (SQLite is the one deliberate absentee — public
  domain, no license file exists). The vendor script mirrors that directory
  into the wheel.
- **macOS / Windows (conda-extract)**: after `ci/setup-gdal-from-pixi.{sh,ps1}`
  materialises the wheel-build pixi environment, the vendor script walks every
  `${PIXI_ENV}/conda-meta/*.json`, reads each package's
  `extracted_package_dir`, and copies the contents of
  `<extracted>/info/licenses/` into `src/pyramids/_licenses/<package-name>/`.

Either way, whichever exact upstream versions we link against in a given
release contribute their own current LICENSE files into the wheel — no
hand-maintained list of license texts to keep in sync.

## License compatibility with GPLv3

All licenses listed above are compatible with the GNU General Public License
v3 (the license under which `pyramids-gis` itself is distributed):

- **MIT / BSD-2-Clause / BSD-3-Clause / zlib / Boost** — explicitly GPL-compatible
  permissive licenses.
- **LGPL-2.1-or-later** (GEOS, libiconv) — explicitly GPL-compatible.
- **Apache-2.0** (OpenSSL 3.x+, LERC) — compatible with GPLv3 (not with
  GPLv2-only).
- **libtiff, libpng, IJG, HDF4, HDF5, bzip2, NetCDF-C, libaec, PCRE2** licenses
  — all BSD-style permissive, GPL-compatible.
- **Public domain** (SQLite, parts of xz) — no restrictions, trivially
  GPL-compatible.

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
