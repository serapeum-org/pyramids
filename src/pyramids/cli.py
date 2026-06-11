"""Command-line interface for pyramids.

Thin, scriptable wrappers over the library's primitives, built on the
standard-library :mod:`argparse` (no extra dependency):

- `pyramids cog create|validate|info ...` — the Cloud Optimized GeoTIFF group
- `pyramids info FILE [--json]` — raster metadata at a glance
- `pyramids bounds FILE [--crs CRS] [--json]` — bounding box
- `pyramids clip SRC DST (--bbox MINX MINY MAXX MAXY | --vector PATH)` — crop
- `pyramids warp SRC DST --crs CRS [--resampling M]` — reproject
- `pyramids merge SRC... DST` — mosaic
- `pyramids overview FILE [--resampling M] [--levels N...]` — build overviews
- `pyramids sample FILE --points "x,y;x,y..." [--json]` — point sampling
- `pyramids convert SRC DST [--driver NAME]` — format conversion

Every command maps 1:1 onto an existing library call — no business logic lives
here. Expected user errors (missing file, bad CRS, unknown driver) exit
non-zero with a one-line message instead of a traceback. The entry point is
registered as the `pyramids` console script; the functions here are also
callable in-process (`main([...])`) for testing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Sequence

from osgeo import osr
from pandas import DataFrame

from pyramids.base._errors import _PyramidsError
from pyramids.base.crs import sr_from_user_input, sr_from_wkt
from pyramids.dataset import Dataset
from pyramids.dataset.cog import PROFILES, cog_info, validate
from pyramids.dataset.merge import merge_rasters
from pyramids.feature import FeatureCollection


def _json_safe(value: float | None) -> float | None:
    """Map non-finite floats to `None` so the JSON output stays parseable.

    `json.dumps` serializes NaN/Infinity as bare `NaN` / `Infinity`,
    which is not valid JSON and breaks strict consumers (e.g. `jq`).

    Args:
        value: A float (possibly NaN/infinite) or `None`.

    Returns:
        float | None: `value` unchanged, or `None` when it is not finite.
    """
    result: float | None = value
    if isinstance(value, float) and not math.isfinite(value):
        result = None
    return result


_HELP_SRC_RASTER = "source raster path"
_HELP_DST_RASTER = "destination raster path"
_HELP_OVERWRITE = "replace the output if it already exists"


def _refuse_existing(path: str, overwrite: bool) -> None:
    """Raise if ``path`` exists and ``--overwrite`` was not passed.

    Args:
        path: Destination path the command is about to write.
        overwrite: Whether the user passed ``--overwrite``.

    Raises:
        ValueError: ``path`` already exists and ``overwrite`` is ``False``.
    """
    if not overwrite and os.path.exists(path):
        raise ValueError(
            f"output {path!r} already exists; pass --overwrite to replace it."
        )
_HELP_INSPECT_RASTER = "raster path to inspect"
_HELP_JSON = "emit JSON"


def _cmd_create(args: argparse.Namespace) -> int:
    """Handle `pyramids cog create`.

    Args:
        args: Parsed arguments with `input`, `output`, `profile`,
            `compress`, `blocksize`, and `no_validate`.

    Returns:
        int: `0` on success, `1` when post-write validation fails.
    """
    ds = Dataset.read_file(args.input)
    kwargs: dict = {}
    if args.profile:
        kwargs["profile"] = args.profile
    if args.compress:
        kwargs["compress"] = args.compress
    if args.blocksize:
        kwargs["blocksize"] = args.blocksize
    out = ds.to_cog(args.output, **kwargs)
    print(f"wrote {out}")
    if args.no_validate:
        return 0
    report = validate(str(out))
    if report.is_valid:
        print("valid COG")
        return 0
    print("INVALID COG:", file=sys.stderr)
    for err in report.errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle `pyramids cog validate`.

    Args:
        args: Parsed arguments with `file` and `strict`.

    Returns:
        int: `0` when the file is a valid COG, `1` otherwise.
    """
    report = validate(args.file, strict=args.strict)
    if report.is_valid:
        print(f"{args.file}: valid COG")
        return 0
    print(f"{args.file}: NOT a valid COG", file=sys.stderr)
    for err in report.errors:
        print(f"  error: {err}", file=sys.stderr)
    for warn in report.warnings:
        print(f"  warning: {warn}", file=sys.stderr)
    return 1


def _cmd_info(args: argparse.Namespace) -> int:
    """Handle `pyramids cog info`.

    Args:
        args: Parsed arguments with `file`.

    Returns:
        int: Always `0` (raises if the file cannot be opened).
    """
    info = cog_info(args.file)
    print(f"file:        {args.file}")
    print(f"is_cog:      {info.is_cog}")
    print(f"driver:      {info.driver}")
    print(f"size:        {info.width} x {info.height} ({info.band_count} band(s))")
    print(f"dtype:       {info.dtype}")
    print(f"crs:         EPSG:{info.crs_epsg}")
    print(f"resolution:  {info.resolution}")
    print(f"bounds:      {info.bounds}")
    print(f"compression: {info.compression}")
    print(f"predictor:   {info.predictor}")
    print(f"blocksize:   {info.blocksize}")
    print(f"overviews:   {info.overview_count}")
    for ovr in info.overviews:
        print(f"  - level {ovr.index}: {ovr.width} x {ovr.height} (1/{ovr.decimation})")
    return 0


def _cmd_raster_info(args: argparse.Namespace) -> int:
    """Handle `pyramids info` — print raster metadata.

    Args:
        args: Parsed arguments with `file` and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    payload = {
        "path": args.file,
        "driver": ds.raster.GetDriver().ShortName,
        "epsg": ds.epsg,
        "bands": ds.band_count,
        "rows": ds.rows,
        "columns": ds.columns,
        "cell_size": ds.cell_size,
        "dtype": list(ds.dtype),
        "no_data_value": [
            None if value is None else _json_safe(float(value))
            for value in ds.no_data_value
        ],
        "bounds": [float(value) for value in ds.bbox],
    }
    if args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_bounds(args: argparse.Namespace) -> int:
    """Handle `pyramids bounds` — print the bounding box.

    With `--crs` the four corners are reprojected and the min/max taken —
    a corner-based approximation (no edge densification).

    Args:
        args: Parsed arguments with `file`, `crs`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    min_x, min_y, max_x, max_y = (float(value) for value in ds.bbox)
    if args.crs:
        if not ds.crs:
            raise ValueError(
                "bounds --crs needs a source CRS, but the raster has none; "
                "cannot reproject its bounds."
            )
        src_sr = sr_from_wkt(ds.crs)
        dst_sr = sr_from_user_input(args.crs)
        src_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transformer = osr.CoordinateTransformation(src_sr, dst_sr)
        corners = transformer.TransformPoints(
            [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
        )
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    payload = {"bounds": [min_x, min_y, max_x, max_y]}
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"{min_x} {min_y} {max_x} {max_y}")
    return 0


def _cmd_clip(args: argparse.Namespace) -> int:
    """Handle `pyramids clip` — crop a raster by bbox or vector mask.

    Args:
        args: Parsed arguments with `input`, `output`, `bbox`, and `vector`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.input)
    if args.vector:
        mask = FeatureCollection.read_file(args.vector)
    else:
        if not ds.crs:
            raise ValueError(
                "clip --bbox needs the raster to have a CRS to interpret the "
                "bbox, but it has none; clip with a --vector mask instead."
            )
        mask = FeatureCollection.from_bbox(tuple(args.bbox), epsg=ds.epsg)
    _refuse_existing(args.output, args.overwrite)
    clipped = ds.crop(mask)
    clipped.to_file(args.output)
    print(f"wrote {args.output}")
    return 0


def _cmd_warp(args: argparse.Namespace) -> int:
    """Handle `pyramids warp` — reproject a raster.

    Args:
        args: Parsed arguments with `input`, `output`, `crs`, `resampling`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    warped = ds.to_crs(args.crs, method=args.resampling)
    warped.to_file(args.output)
    print(f"wrote {args.output}")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    """Handle `pyramids merge` — mosaic rasters into one file.

    Args:
        args: Parsed arguments with `inputs` (>= 2 paths) and `output`.

    Returns:
        int: `0` on success.

    Raises:
        ValueError: Fewer than two input rasters are given.
    """
    if len(args.inputs) < 2:
        raise ValueError("merge needs at least two input rasters.")
    _refuse_existing(args.output, args.overwrite)
    merge_rasters(args.inputs, args.output)
    print(f"wrote {args.output}")
    return 0


def _cmd_overview(args: argparse.Namespace) -> int:
    """Handle `pyramids overview` — build image pyramids in place.

    Args:
        args: Parsed arguments with `file`, `resampling`, and `levels`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file, read_only=False)
    # create_overviews validates against GDAL's overview RESAMPLING_METHODS
    # ("nearest", "average", "gauss", "cubic", ...). Accept the warp-family
    # "nearest neighbor" spelling too, so the resampling vocabulary is
    # consistent with `warp` / `merge` across subcommands (L9).
    resampling = args.resampling
    if resampling.strip().lower() == "nearest neighbor":
        resampling = "nearest"
    ds.create_overviews(resampling_method=resampling, overview_levels=args.levels)
    counts = ds.overview_count
    print(f"built {counts[0] if counts else 0} overview level(s) for {args.file}")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    """Handle `pyramids sample` — read band values at points.

    Points outside the raster extent sample as the band's no-data fill
    (NaN when none is set) and are printed as `None` / JSON `null`.

    Args:
        args: Parsed arguments with `file`, `points` (``"x,y;x,y..."``),
            and `json`.

    Returns:
        int: `0` on success.

    Raises:
        ValueError: `points` is empty or a chunk is not a numeric `'x,y'` pair.
    """
    pairs = [chunk for chunk in args.points.split(";") if chunk.strip()]
    if not pairs:
        raise ValueError("--points must contain at least one 'x,y' pair.")
    xs, ys = [], []
    for chunk in pairs:
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"bad point {chunk.strip()!r}; expected 'x,y'.")
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            raise ValueError(
                f"bad point {chunk.strip()!r}; coordinates must be numeric."
            ) from None
    ds = Dataset.read_file(args.file)
    values = ds.sample(DataFrame({"x": xs, "y": ys}))
    # ds.sample returns (bands, points); transpose to one row per point.
    # Out-of-bounds points come back as NaN — emit them as None/null.
    per_point = [[_json_safe(value) for value in row] for row in values.T.tolist()]
    if args.json:
        print(json.dumps({"values": per_point}))
    else:
        for (x, y), point_values in zip(zip(xs, ys), per_point):
            print(f"{x},{y}: {point_values}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Handle `pyramids convert` — re-save a raster in another format.

    The driver is inferred from the output extension unless `--driver` is
    given — a pyramids catalog driver name (e.g. ``geotiff``, ``ascii``)
    or a GDAL driver name (e.g. ``GTiff``, ``COG``).

    Args:
        args: Parsed arguments with `input`, `output`, and `driver`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    ds.to_file(args.output, driver=args.driver)
    print(f"wrote {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser with the `cog`
        command group (`create` / `validate` / `info`) and the
        single-shot raster commands (`info`, `bounds`, `clip`, `warp`,
        `merge`, `overview`, `sample`, `convert`).
    """
    parser = argparse.ArgumentParser(
        prog="pyramids", description="pyramids GIS toolkit"
    )
    sub = parser.add_subparsers(dest="group", required=True)

    cog = sub.add_parser("cog", help="Cloud Optimized GeoTIFF commands")
    cog_sub = cog.add_subparsers(dest="command", required=True)

    create = cog_sub.add_parser("create", help="write a raster as a COG")
    create.add_argument("input", help=_HELP_SRC_RASTER)
    create.add_argument("output", help="destination COG path")
    create.add_argument(
        "--profile", choices=sorted(PROFILES), help="named compression profile"
    )
    create.add_argument("--compress", help="compression method (e.g. DEFLATE, ZSTD)")
    create.add_argument("--blocksize", type=int, help="internal tile size")
    create.add_argument(
        "--no-validate", action="store_true", help="skip post-write validation"
    )
    create.set_defaults(func=_cmd_create)

    val = cog_sub.add_parser("validate", help="validate a COG")
    val.add_argument("file", help="raster path to validate")
    val.add_argument("--strict", action="store_true", help="treat warnings as errors")
    val.set_defaults(func=_cmd_validate)

    info = cog_sub.add_parser("info", help="print structured COG metadata")
    info.add_argument("file", help=_HELP_INSPECT_RASTER)
    info.set_defaults(func=_cmd_info)

    raster_info = sub.add_parser("info", help="print raster metadata")
    raster_info.add_argument("file", help=_HELP_INSPECT_RASTER)
    raster_info.add_argument("--json", action="store_true", help=_HELP_JSON)
    raster_info.set_defaults(func=_cmd_raster_info)

    bounds = sub.add_parser("bounds", help="print the raster bounding box")
    bounds.add_argument("file", help=_HELP_INSPECT_RASTER)
    bounds.add_argument(
        "--crs", help="reproject the corners to this CRS (corner-based approximation)"
    )
    bounds.add_argument("--json", action="store_true", help=_HELP_JSON)
    bounds.set_defaults(func=_cmd_bounds)

    clip = sub.add_parser("clip", help="crop a raster by bbox or vector mask")
    clip.add_argument("input", help=_HELP_SRC_RASTER)
    clip.add_argument("output", help=_HELP_DST_RASTER)
    clip_how = clip.add_mutually_exclusive_group(required=True)
    clip_how.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="clip extent in the raster CRS",
    )
    clip_how.add_argument("--vector", help="vector file whose polygons clip the raster")
    clip.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    clip.set_defaults(func=_cmd_clip)

    warp = sub.add_parser("warp", help="reproject a raster")
    warp.add_argument("input", help=_HELP_SRC_RASTER)
    warp.add_argument("output", help=_HELP_DST_RASTER)
    warp.add_argument("--crs", required=True, help="target CRS (EPSG code, WKT, PROJ4)")
    warp.add_argument(
        "--resampling", default="nearest neighbor", help="resampling method"
    )
    warp.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    warp.set_defaults(func=_cmd_warp)

    merge = sub.add_parser("merge", help="mosaic rasters into one file")
    merge.add_argument("inputs", nargs="+", help="source raster paths (two or more)")
    merge.add_argument("output", help=_HELP_DST_RASTER)
    merge.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    merge.set_defaults(func=_cmd_merge)

    overview = sub.add_parser("overview", help="build image pyramids in place")
    overview.add_argument("file", help="raster path to build overviews for")
    overview.add_argument(
        "--resampling",
        default="nearest",
        help="overview resampling: nearest (alias 'nearest neighbor'), average, "
        "gauss, cubic, mode, ...",
    )
    overview.add_argument(
        "--levels", nargs="+", type=int, help="decimation levels (e.g. 2 4 8)"
    )
    overview.set_defaults(func=_cmd_overview)

    sample = sub.add_parser("sample", help="read band values at points")
    sample.add_argument("file", help="raster path to sample")
    sample.add_argument(
        "--points", required=True, help="semicolon-separated 'x,y' pairs"
    )
    sample.add_argument("--json", action="store_true", help=_HELP_JSON)
    sample.set_defaults(func=_cmd_sample)

    convert = sub.add_parser("convert", help="re-save a raster in another format")
    convert.add_argument("input", help=_HELP_SRC_RASTER)
    convert.add_argument("output", help=_HELP_DST_RASTER)
    convert.add_argument(
        "--driver",
        help="catalog (geotiff) or GDAL (GTiff) driver name (default: from extension)",
    )
    convert.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    convert.set_defaults(func=_cmd_convert)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `pyramids` console script.

    Args:
        argv: Argument list (excluding the program name). Defaults to
            :data:`sys.argv` when `None`.

    Returns:
        int: Process exit code (`0` success, non-zero failure).

    Examples:
        - Inspect a COG from the shell:
            ```python
            >>> main(["cog", "info", "scene.tif"])  # doctest: +SKIP
            0

            ```
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        _PyramidsError,
    ) as exc:
        # Expected user errors (missing file, bad CRS, unknown driver, ...)
        # exit non-zero with a one-line message instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        result = 1
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
