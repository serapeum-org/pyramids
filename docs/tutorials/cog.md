# Cloud Optimized GeoTIFF (COG) cookbook

A **Cloud Optimized GeoTIFF** is a regular GeoTIFF whose internal
layout is organized so that HTTP clients can efficiently read just
the pixels they need via HTTP range requests. pyramids provides:

- `Dataset.to_cog()` — write a COG with full control over the GDAL
  COG driver's option surface.
- `Dataset.is_cog` / `Dataset.validate_cog()` — validate a file.
- `Dataset.read_file(url)` — transparently open COGs from S3, GCS,
  Azure, and HTTPS via GDAL's virtual filesystem.
- `DatasetCollection.to_cog_stack()` — export a temporal stack as a
  COG-per-slice directory.

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
ds.to_cog("scene.tif", compress="DEFLATE", predictor=2)   # best for float
```

Supported `compress` values: `DEFLATE` (default), `LZW`, `ZSTD`,
`WEBP`, `JPEG`, `LERC`, `LERC_DEFLATE`, `LERC_ZSTD`, `NONE`.

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
    overview_resampling="bilinear",   # default "nearest"
    overview_count=5,                 # default: auto
    overview_compress="DEFLATE",
)
```

> For **categorical** rasters (land cover, basin IDs, classification
> masks) leave `overview_resampling="nearest"` or use `"mode"`.
> Averaging methods corrupt category labels. pyramids emits a
> `UserWarning` if it detects an integer dtype or color-table raster
> with an averaging resampler.

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
