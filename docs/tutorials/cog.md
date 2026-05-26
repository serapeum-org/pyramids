# Cloud Optimized GeoTIFF (COG) cookbook

A **Cloud Optimized GeoTIFF** is a regular GeoTIFF whose internal
layout is organized so that HTTP clients can efficiently read just
the pixels they need via HTTP range requests. pyramids provides:

- `Dataset.to_cog()` — write a COG with full control over the GDAL
  COG driver's option surface (band subset, dtype cast, NoData,
  named profiles, tags/colourmap).
- `Dataset.to_cog_bytes()` — encode a COG to an in-memory `bytes`
  buffer for direct object-store upload (no temp file).
- `Dataset.cog_info()` — structured inspection (compression,
  predictor, blocksize, the overview pyramid, tags).
- `Dataset.is_cog` / `Dataset.validate_cog()` — validate a file.
- `Dataset.read_part()` / `preview()` / `point()` / `read_tile()` —
  overview-decimated partial reads, ideal over `/vsicurl/`.
- `Dataset.read_file(url)` — transparently open COGs from S3, GCS,
  Azure, and HTTPS via GDAL's virtual filesystem.
- `DatasetCollection.to_cog_stack()` — export a temporal stack as a
  COG-per-slice directory.
- `pyramids cog create|validate|info` — a command-line interface.

> **Defaults.** `to_cog` resolves the two pixel-affecting options
> per source dtype: the **predictor** (`2` for integer rasters, `3`
> for float) and the **overview resampling** (`mode` for categorical
> sources — integer dtype or a colour table — and `average` for
> continuous float). Pass either explicitly to override. The same
> policy backs the `write_cog(...)` facade, so a direct `ds.to_cog(...)`
> and `write_cog(ds, ...)` produce identical output.

## Writing a COG

Minimal example:

```python
import numpy as np
from pyramids.dataset import Dataset

arr = np.random.rand(1, 512, 512).astype("float32")
ds = Dataset.create_from_array(
    arr, top_left_corner=(0, 0), cell_size=0.001, epsg=4326
)
ds.to_cog("scene.tif")
```

### Compression

```python
ds.to_cog("scene.tif", compress="ZSTD", level=18)
ds.to_cog("scene.tif", compress="LERC", extra={"MAX_Z_ERROR": 0.001})
ds.to_cog("scene.tif", compress="DEFLATE")   # predictor auto-resolves per dtype
ds.to_cog("scene.tif", compress="DEFLATE", predictor=3)   # force the float predictor
```

Supported `compress` values: `DEFLATE` (default), `LZW`, `ZSTD`,
`WEBP`, `JPEG`, `LERC`, `LERC_DEFLATE`, `LERC_ZSTD`, `NONE`.

By default the **predictor** is chosen from the source dtype (`2` for
integer, `3` for float); pass `predictor=` to override.

### Named profiles

Instead of remembering compression + level/quality combos, pick a named
profile (`deflate`, `zstd`, `lzw`, `packbits`, `jpeg`, `webp`, `lerc`,
`lerc_deflate`, `lerc_zstd`, `raw`):

```python
ds.to_cog("scene.tif", profile="zstd")     # COMPRESS=ZSTD, LEVEL=9
ds.to_cog("scene.tif", profile="lerc_deflate")

# Explicit kwargs and `extra` override a profile:
ds.to_cog("scene.tif", profile="zstd", level=22)
```

`jpeg` / `webp` enforce their dtype/band constraints (Byte; 1-3 / 3-4
bands) up-front with a clear error.

### Tile size and bigtiff

```python
ds.to_cog(
    "scene.tif",
    blocksize=256,        # power of 2 in [64, 4096]
    bigtiff="IF_SAFER",   # default
    num_threads="ALL_CPUS",
)
```

### Overviews

The COG driver generates overviews itself; `to_cog` exposes the knobs:

```python
ds.to_cog(
    "scene.tif",
    overview_resampling="bilinear",   # default: dtype-aware (see below)
    overview_count=5,                 # default: auto
    overview_compress="DEFLATE",
)
```

> **Categorical-safe by default.** When you don't pass
> `overview_resampling`, `to_cog` picks `mode` for categorical sources
> (integer dtype or a colour table) and `average` for continuous float
> — so the default never corrupts category labels (land cover, basin
> IDs, classification masks). pyramids only emits a `UserWarning` when
> **you** explicitly request an averaging method (`average`,
> `bilinear`, `cubic`, `cubicspline`, `lanczos`) on a categorical
> raster.

## Web-optimized COGs

For serving to map/tile consumers:

```python
ds.to_cog(
    "web.tif",
    tiling_scheme="GoogleMapsCompatible",
    resampling="bilinear",
)
# Output is in EPSG:3857 aligned to the web-Mercator zoom grid.
```

## Reading a COG from cloud storage

pyramids rewrites URL-scheme paths to GDAL's `/vsi*` form automatically:

| URL form | Rewritten internally to |
| --- | --- |
| `s3://bucket/key.tif` | `/vsis3/bucket/key.tif` |
| `gs://bucket/key.tif` | `/vsigs/bucket/key.tif` |
| `az://container/blob.tif` | `/vsiaz/container/blob.tif` |
| `abfs://container/blob.tif` | `/vsiaz/container/blob.tif` |
| `https://foo/x.tif` | `/vsicurl/https://foo/x.tif` |
| `file:///C:/path/x.tif` | `C:/path/x.tif` |

So you just:

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("s3://sentinel-cogs/tile/path/B04.tif")
ds = Dataset.read_file("https://example.com/scene.tif")
ds = Dataset.read_file("gs://bucket/key.tif")
```

Presigned URLs (with `?sig=...`) are preserved verbatim.

### Credentials via environment variables

GDAL reads the standard cloud env vars by default; pyramids does not
invent its own scheme. Before running your Python process:

```bash
# AWS
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# Google Cloud Storage
export GS_OAUTH2_REFRESH_TOKEN=...

# Azure
export AZURE_STORAGE_ACCOUNT=...
export AZURE_STORAGE_ACCESS_KEY=...
```

For public buckets:

```bash
export AWS_NO_SIGN_REQUEST=YES
```

### Credentials via `CloudConfig`

When you need to override per-operation:

```python
from pyramids.base.remote import CloudConfig

with CloudConfig(aws_region="us-east-1", aws_no_sign_request=True):
    ds = Dataset.read_file("s3://public-bucket/scene.tif")

with CloudConfig(
    aws_access_key_id="AK",
    aws_secret_access_key="SEC",
    aws_region="eu-west-1",
):
    ds = Dataset.read_file("s3://private-bucket/scene.tif")
```

`CloudConfig` is a context manager that applies options via
`gdal.config_options` and restores the previous values on exit.

## Validating a COG

```python
from pyramids.dataset.cog import validate

report = validate("scene.tif")
if report:
    print("valid COG; blocksize=", report.details.get("blocksize"))
else:
    for err in report.errors:
        print("ERROR:", err)
    for warn in report.warnings:
        print("WARN :", warn)

# Strict: warnings (e.g., "no overviews") become errors
strict = validate("scene.tif", strict=True)
```

Or from a `Dataset` instance:

```python
ds = Dataset.read_file("scene.tif")
if ds.is_cog:
    print("is a COG")

report = ds.validate_cog(strict=True)
```

## Exporting a `DatasetCollection` as a COG stack

One COG per time slice — the typical static-STAC pattern:

```python
from pyramids.dataset import DatasetCollection

dc = DatasetCollection.read_multiple_files("path/to/folder")
dc.open_multi_dataset(band=0)

paths = dc.to_cog_stack(
    "out_cogs/",
    pattern="B04_{i:04d}.tif",   # filename template
    name="B04",
    compress="ZSTD",
)
# -> [PosixPath('out_cogs/B04_0000.tif'), ..., PosixPath('out_cogs/B04_NNNN.tif')]
```

## Selecting bands, casting dtype, and NoData on write

`to_cog` can pre-process the source in one call — band subset/reorder
(`indexes`, 0-based), dtype cast (`out_dtype`), and an explicit
`nodata` — routed through an in-memory `gdal.Translate` so your open
dataset is never mutated:

```python
# RGB-from-multiband, cast to uint8, set NoData=0:
ds.to_cog("rgb.tif", indexes=[3, 2, 1], out_dtype="uint8", nodata=0)
```

The dtype-aware predictor is resolved from the **post-cast** dtype (so a
float→int16 cast correctly yields `PREDICTOR=2`).

## Attaching tags, a colour table, and metadata

Useful when the source is a bare array that carries no band
descriptions or palette:

```python
ds.to_cog(
    "landcover.tif",
    band_tags={0: {"name": "landcover", "units": "class"}},
    colormap={0: (0, 0, 0, 255), 1: (34, 139, 34, 255), 2: (70, 130, 180, 255)},
    metadata={"source": "my-pipeline", "version": "1.0"},
)
```

These are stamped onto an in-memory copy before the write, so the
original dataset is left untouched.

## Encoding to bytes (in-memory)

To upload a COG straight to an object store without a temp file, use
`to_cog_bytes` — it accepts the same keywords as `to_cog`:

```python
blob = ds.to_cog_bytes(compress="ZSTD")
# e.g. boto3:  s3.put_object(Bucket="b", Key="scene.tif", Body=blob)
```

## Inspecting a COG

`cog_info()` reads only headers/metadata (no pixels), so it is cheap
even for a large remote COG:

```python
info = ds.cog_info()              # or: from pyramids.dataset.cog import cog_info
info.compression                  # 'DEFLATE'
info.predictor                    # '3'
info.blocksize                    # (512, 512)
info.dtype, info.band_count       # ('Float32', 1)
info.crs_epsg, info.bounds        # (4326, (...))
[o.decimation for o in info.overviews]   # [2, 4, 8]
```

## Partial / overview-decimated reads

The whole point of a COG is reading only what you need. These methods
ask GDAL for a smaller output size, which serves the data from the
nearest overview — over `/vsicurl/` only the relevant byte ranges are
fetched:

```python
# Whole-image thumbnail (long edge <= 1024 px):
thumb = ds.preview(max_size=1024)

# A geographic window, decimated to an explicit output size:
part = ds.read_part((12.4, 41.8, 12.6, 42.0), dst_width=256, dst_height=256)

# A single coordinate value (reprojected from bbox_crs if needed):
value = ds.point(12.5, 41.9, point_crs=4326)

# A Web-Mercator XYZ/slippy-map tile (no morecantile dependency):
tile = ds.read_tile(z=12, x=2191, y=1539, tilesize=256)
```

`read_part` / `read_tile` accept a `resampling` argument (`nearest`,
`bilinear`, `cubic`, `cubicspline`, `lanczos`, `average`, `mode`) and a
`band` (0-based) to read a single band.

## Tuning remote reads from Python

`to_cog`, `validate`, and `cog_info` accept a `config` dict applied via
`gdal.config_options` for the duration of the call. Remote reads with no
explicit `config` automatically get cloud-friendly defaults
(`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`, curl range tuning):

```python
from pyramids.dataset.cog import validate, cog_info

report = validate("s3://bucket/scene.tif")          # remote defaults applied
info = cog_info("/vsicurl/https://host/scene.tif",  # or override explicitly
                config={"GDAL_HTTP_MAX_RETRY": "5"})
```

## Command line

The `pyramids cog` command group exposes the common workflow from the
shell:

```bash
pyramids cog create input.tif scene_cog.tif --profile zstd
pyramids cog validate scene_cog.tif --strict
pyramids cog info scene_cog.tif
```

`create` validates the output by default (`--no-validate` to skip);
`validate` exits non-zero for a non-COG; `info` prints the structured
metadata and the overview pyramid. The entry point is registered as the
`pyramids` console script (available after `pip install pyramids-gis`).

## Performance tuning

Recommended GDAL environment variables for cloud reads:

```bash
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.TIF,.vrt
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=26214400       # 25 MiB
export GDAL_CACHEMAX=512             # MiB
```

The first two are the biggest wins — they stop GDAL from probing for
sidecar files (.aux.xml, .ovr) that usually don't exist on public
buckets, saving one extra HTTP request per open.

## Troubleshooting

**"Range downloading not supported by this server!"**
The remote HTTP server doesn't support HTTP Range headers. Python's
stdlib `http.server` is one such case. Real cloud storage (S3, GCS,
Azure) always supports Range.

**`/vsis3/` returns opaque error for a private bucket**
Check that `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_REGION` are set, or use `CloudConfig(...)` explicitly. For
public buckets use `AWS_NO_SIGN_REQUEST=YES`.

**`GoogleMapsCompatible` output has unexpected bounds**
The tiling scheme reprojects to EPSG:3857 and aligns to a zoom grid.
Your output will always be larger than the input extent — that's by
design; it lets a tile server serve the file without further
reprojection.

**`overview_resampling="average"` warning on my categorical raster**
This is the documented safety check. Either switch to
`overview_resampling="nearest"` / `"mode"`, or suppress the warning
if you really want averaged overviews:

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    ds.to_cog("x.tif", overview_resampling="average")
```
