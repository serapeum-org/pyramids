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
- `pyramids georeference SRC DST --gcp PIXEL LINE X Y ... --gcp-crs CRS` —
  warp from ground-control points
- `pyramids orthorectify SRC DST [--dem PATH | --rpc-height H]` — RPC
  orthorectification
- `pyramids edit-info FILE [--crs CRS] [--nodata V] [--tag K=V...]` — edit
  CRS / nodata / tags in place
- `pyramids calc EXPR SRC... DST [--dtype T]` — evaluate a band expression
  (safe AST evaluator, no `eval`)
- `pyramids shapes SRC DST [--geometry polygon|point]` — vectorize a raster
- `pyramids rasterize SRC DST (--cell-size S | --like RASTER) [--column C]` —
  burn a vector into a raster

Every command maps 1:1 onto an existing library call — no business logic lives
here. Expected user errors (missing file, bad CRS, unknown driver) exit
non-zero with a one-line message instead of a traceback. The entry point is
registered as the `pyramids` console script; the functions here are also
callable in-process (`main([...])`) for testing.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import sys
from collections.abc import Sequence

import numpy as np
from osgeo import osr
from pandas import DataFrame

from pyramids.base._errors import _PyramidsError
from pyramids.base._utils import DEFAULT_RESAMPLING
from pyramids.base.crs import sr_from_user_input, sr_from_wkt
from pyramids.dataset import Dataset
from pyramids.dataset._gcp import GroundControlPoint
from pyramids.dataset.abstract_dataset import OVERVIEW_LEVELS
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
_HELP_RESAMPLING = "resampling method"


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


def _bbox_disjoint(a, b) -> bool:
    """Return True when two ``(minx, miny, maxx, maxy)`` extents do not overlap.

    Args:
        a: First extent ``(minx, miny, maxx, maxy)``.
        b: Second extent ``(minx, miny, maxx, maxy)``.

    Returns:
        bool: ``True`` when the rectangles share no area (edge-touching counts
        as disjoint, since a zero-area clip yields an empty grid).
    """
    aminx, aminy, amaxx, amaxy = a
    bminx, bminy, bmaxx, bmaxy = b
    return aminx >= bmaxx or amaxx <= bminx or aminy >= bmaxy or amaxy <= bminy


_HELP_INSPECT_RASTER = "raster path to inspect"
_HELP_JSON = "emit JSON"


def _cmd_create(args: argparse.Namespace) -> int:
    """Handle `pyramids cog create`.

    Args:
        args: Parsed arguments with `input`, `output`, `profile`,
            `compress`, `blocksize`, `no_validate`, and `overwrite`.

    Returns:
        int: `0` on success, `1` when post-write validation fails.
    """
    _refuse_existing(args.output, args.overwrite)
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
    # A mask/bbox disjoint from the raster crops to an empty grid, which surfaces
    # deep inside crop as an opaque IndexError; intersect-check up front for both
    # the --vector and --bbox paths and raise a clear message instead.
    if args.vector:
        mask = FeatureCollection.read_file(args.vector)
        mask_extent = mask.total_bounds
        if mask.crs is not None and ds.crs and mask.epsg != ds.epsg:
            mask_extent = mask.to_crs(ds.epsg).total_bounds
        if _bbox_disjoint(mask_extent, ds.bbox):
            raise ValueError(
                f"clip --vector mask extent {tuple(mask_extent)} does not "
                f"intersect the raster extent {tuple(ds.bbox)}; nothing to clip."
            )
        _refuse_existing(args.output, args.overwrite)
        clipped = ds.crop(mask)
    else:
        if not ds.crs:
            raise ValueError(
                "clip --bbox needs the raster to have a CRS to interpret the "
                "bbox, but it has none; clip with a --vector mask instead."
            )
        clip_bbox = tuple(args.bbox)
        if _bbox_disjoint(clip_bbox, ds.bbox):
            raise ValueError(
                f"clip --bbox {clip_bbox} does not intersect the raster extent "
                f"{tuple(ds.bbox)}; nothing to clip."
            )
        _refuse_existing(args.output, args.overwrite)
        # Use crop's native bbox path (epsg defaults to the dataset CRS) rather
        # than hand-wrapping the bbox in a FeatureCollection.
        clipped = ds.crop(bbox=clip_bbox)
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


def _cmd_georeference(args: argparse.Namespace) -> int:
    """Handle `pyramids georeference` — warp a raster from ground-control points.

    Args:
        args: Parsed args with `input`, `output`, `gcp` (list of [pixel, line,
            x, y]), `gcp_crs`, `transform`, `order`, `to_crs`, `resampling`,
            `overwrite`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    points = [
        GroundControlPoint(col=pixel, row=line, x=x, y=y)
        for pixel, line, x, y in args.gcp
    ]
    source = Dataset.read_file(args.input)
    # Reconstruct in a writable MEM dataset so attaching GCPs does not mutate the
    # input file; the source geotransform is irrelevant — GCPs replace it.
    working = Dataset.create_from_array(
        source.read_array(),
        top_left_corner=source.top_left_corner,
        cell_size=source.cell_size,
        epsg=source.epsg or 4326,
        no_data_value=source.no_data_value,
    )
    working.set_gcps(points, args.gcp_crs)
    out = working.georeference(
        to_epsg=args.to_crs,
        method=args.resampling,
        transform=args.transform,
        order=args.order,
    )
    out.to_file(args.output)
    print(f"wrote {args.output}")
    return 0


def _cmd_orthorectify(args: argparse.Namespace) -> int:
    """Handle `pyramids orthorectify` — orthorectify a raster from its RPCs.

    Args:
        args: Parsed args with `input`, `output`, `dem`, `rpc_height`, `to_crs`,
            `resampling`, `overwrite`. The input must already carry RPC metadata.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    out = ds.orthorectify(
        dem=args.dem,
        rpc_height=args.rpc_height,
        to_epsg=args.to_crs,
        method=args.resampling,
    )
    out.to_file(args.output)
    print(f"wrote {args.output}")
    return 0


_CALC_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_CALC_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CALC_COMPARE = {
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
# The only function calls a `calc` expression may use, all from numpy.
_CALC_NP_FUNCS = frozenset(
    {
        "where",
        "clip",
        "log",
        "log10",
        "exp",
        "sqrt",
        "abs",
        "minimum",
        "maximum",
        "power",
    }
)


def _safe_calc_eval(node: ast.AST, variables: dict) -> object:
    """Evaluate one node of a `calc` expression AST against a whitelist.

    This is a deliberately small interpreter — it never uses ``eval``/``exec``,
    so a hostile expression (``__import__('os')``, attribute access, comprehensions,
    ...) is rejected with a ``ValueError`` rather than executed.

    Args:
        node: The AST node to evaluate.
        variables: Band arrays bound to their names (``A``, ``B``, ...).

    Returns:
        The numeric / ndarray value of the node.

    Raises:
        ValueError: The node is not in the allowed grammar.
    """
    if isinstance(node, ast.Expression):
        result = _safe_calc_eval(node.body, variables)
    elif isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINOPS:
        result = _CALC_BINOPS[type(node.op)](
            _safe_calc_eval(node.left, variables),
            _safe_calc_eval(node.right, variables),
        )
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARYOPS:
        result = _CALC_UNARYOPS[type(node.op)](_safe_calc_eval(node.operand, variables))
    elif (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and type(node.ops[0]) in _CALC_COMPARE
    ):
        result = _CALC_COMPARE[type(node.ops[0])](
            _safe_calc_eval(node.left, variables),
            _safe_calc_eval(node.comparators[0], variables),
        )
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        result = node.value
    elif isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"unknown name in calc expression: {node.id!r}")
        result = variables[node.id]
    elif (
        isinstance(node, ast.Call)
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in _CALC_NP_FUNCS
    ):
        result = getattr(np, node.func.attr)(
            *[_safe_calc_eval(arg, variables) for arg in node.args]
        )
    else:
        raise ValueError(
            "disallowed element in calc expression; only arithmetic over the "
            "input bands (A, B, ...), comparisons, and a small set of np.<func> "
            "calls are permitted."
        )
    return result


def _cmd_calc(args: argparse.Namespace) -> int:
    """Handle `pyramids calc` — evaluate a band expression into a new raster.

    The expression operates on the input rasters bound to ``A``, ``B``, ... in
    order; it is evaluated by a small AST whitelist, never ``eval``.

    Args:
        args: Parsed args with `expr`, `operands` (inputs... + output), `dtype`,
            `overwrite`.

    Returns:
        int: `0` on success.

    Raises:
        ValueError: Fewer than one input + output, or a disallowed expression.
    """
    if len(args.operands) < 2:
        raise ValueError("calc needs at least one input raster and an output path.")
    *inputs, output = args.operands
    if len(inputs) > 26:
        raise ValueError("calc supports at most 26 input rasters (bound A..Z).")
    _refuse_existing(output, args.overwrite)
    datasets = [Dataset.read_file(path) for path in inputs]
    names = [chr(ord("A") + index) for index in range(len(datasets))]
    variables = {name: np.asarray(ds.read_array()) for name, ds in zip(names, datasets)}
    result = np.asarray(_safe_calc_eval(ast.parse(args.expr, mode="eval"), variables))
    if args.dtype:
        result = result.astype(args.dtype)
    template = datasets[0]
    Dataset.create_from_array(
        result,
        top_left_corner=template.top_left_corner,
        cell_size=template.cell_size,
        epsg=template.epsg or 4326,
    ).to_file(output)
    print(f"wrote {output}")
    return 0


# `shapes` emits one feature per cell, so it does not scale like a region-
# dissolving polygonizer; refuse above this cell count unless --allow-large.
_SHAPES_MAX_CELLS = 4_000_000


def _cmd_shapes(args: argparse.Namespace) -> int:
    """Handle `pyramids shapes` — vectorize a raster to a vector file.

    Emits **one feature per cell** (a square polygon, or a centre point with
    ``--geometry point``) carrying the band value — it is *not* a region-
    dissolving polygonizer. Above `_SHAPES_MAX_CELLS` cells it refuses unless
    `--allow-large` is given, since one feature per cell can exhaust memory.

    Args:
        args: Parsed args with `input`, `output`, `geometry`, `driver`,
            `allow_large`, `overwrite`.

    Returns:
        int: `0` on success.

    Raises:
        ValueError: The raster exceeds `_SHAPES_MAX_CELLS` and `--allow-large`
            was not passed.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    cells = ds.rows * ds.columns
    if cells > _SHAPES_MAX_CELLS and not args.allow_large:
        raise ValueError(
            f"shapes emits one feature per cell ({cells:,} cells here), which can "
            f"exhaust memory; crop/downsample first, or pass --allow-large to proceed."
        )
    gdf = ds.to_feature_collection(add_geometry=args.geometry)
    FeatureCollection(gdf).to_file(args.output, driver=args.driver)
    print(f"wrote {args.output}")
    return 0


def _cmd_rasterize(args: argparse.Namespace) -> int:
    """Handle `pyramids rasterize` — burn a vector into a new raster.

    Args:
        args: Parsed args with `input` (vector), `output`, and one of
            `cell_size` / `like` (template raster), plus optional `column`.

    Returns:
        int: `0` on success.

    Raises:
        ValueError: Neither `--cell-size` nor `--like` is given.
    """
    _refuse_existing(args.output, args.overwrite)
    if args.cell_size is None and args.like is None:
        raise ValueError("rasterize needs --cell-size or --like (a template raster).")
    if args.cell_size is not None and args.like is not None:
        print(
            "note: --cell-size is ignored because --like sets the output grid",
            file=sys.stderr,
        )
    features = FeatureCollection.read_file(args.input)
    template = Dataset.read_file(args.like) if args.like else None
    out = Dataset.from_features(
        features,
        cell_size=args.cell_size,
        template=template,
        column_name=args.column,
    )
    out.to_file(args.output)
    print(f"wrote {args.output}")
    return 0


def _cmd_edit_info(args: argparse.Namespace) -> int:
    """Handle `pyramids edit-info` — edit a raster's CRS / nodata / tags in place.

    Args:
        args: Parsed args with `input` and any of `crs`, `nodata`, `tag`.

    Returns:
        int: `0` on success (a no-edit call prints a notice and still exits 0).
    """
    ds = Dataset.read_file(args.input, read_only=False)
    edited = False
    if args.crs is not None:
        # Resolve any accepted CRS form (EPSG int, "EPSG:3857", WKT, PROJ4) to WKT
        # *before* touching the file, so an invalid CRS fails cleanly with nothing
        # half-applied, and `set_crs` (which expects WKT) gets what it wants.
        wkt = sr_from_user_input(args.crs).ExportToWkt()
        ds.set_crs(crs=wkt)
        edited = True
    if args.nodata is not None:
        ds.no_data_value = args.nodata
        edited = True
    for tag in args.tag or []:
        key, _, value = tag.partition("=")
        ds.raster.SetMetadataItem(key, value)
        edited = True
    if edited:
        ds.raster.FlushCache()
        print(f"edited {args.input}")
    else:
        print("edit-info: no edits requested (pass --crs / --nodata / --tag)")
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
    if args.levels is not None:
        # Validate up front, before opening the file in update mode, so a bad
        # level fails with a clear message instead of after the write handle opens.
        invalid = [lvl for lvl in args.levels if lvl not in OVERVIEW_LEVELS]
        if invalid:
            raise ValueError(
                f"overview --levels must be power-of-two reduction factors "
                f"{OVERVIEW_LEVELS}; got invalid {invalid}."
            )
    ds = Dataset.read_file(args.file, read_only=False)
    # create_overviews validates against GDAL's overview RESAMPLING_METHODS
    # ("nearest", "average", "gauss", "cubic", ...). Accept the warp-family
    # "nearest neighbor" spelling too, so the resampling vocabulary is
    # consistent with `warp` / `merge` across subcommands (L9).
    resampling = args.resampling
    if resampling.strip().lower() == DEFAULT_RESAMPLING:
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
    create.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
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
    warp.add_argument("--resampling", default=DEFAULT_RESAMPLING, help=_HELP_RESAMPLING)
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
        "--levels",
        nargs="+",
        type=int,
        help="power-of-two decimation levels (2, 4, 8, ... 2048); e.g. 2 4 8",
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

    georeference = sub.add_parser(
        "georeference", help="warp a raster from ground-control points"
    )
    georeference.add_argument("input", help=_HELP_SRC_RASTER)
    georeference.add_argument("output", help=_HELP_DST_RASTER)
    georeference.add_argument(
        "--gcp",
        nargs=4,
        type=float,
        action="append",
        required=True,
        metavar=("PIXEL", "LINE", "X", "Y"),
        help="a ground-control point 'PIXEL LINE X Y'; repeat for each point",
    )
    georeference.add_argument(
        "--gcp-crs", required=True, help="CRS of the GCP map coordinates"
    )
    georeference.add_argument(
        "--transform",
        default="polynomial",
        choices=["polynomial", "tps"],
        help="transform fitted through the GCPs (default: polynomial)",
    )
    georeference.add_argument(
        "--order", type=int, default=1, help="polynomial order 1-3 (default: 1)"
    )
    georeference.add_argument("--to-crs", help="reproject the result to this CRS")
    georeference.add_argument(
        "--resampling", default=DEFAULT_RESAMPLING, help=_HELP_RESAMPLING
    )
    georeference.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    georeference.set_defaults(func=_cmd_georeference)

    orthorectify = sub.add_parser(
        "orthorectify", help="orthorectify a raster from its RPC sensor model"
    )
    orthorectify.add_argument("input", help=_HELP_SRC_RASTER)
    orthorectify.add_argument("output", help=_HELP_DST_RASTER)
    orthorectify.add_argument("--dem", help="elevation model raster path")
    orthorectify.add_argument(
        "--rpc-height",
        type=float,
        help="constant elevation (map units) to use when no --dem is given",
    )
    orthorectify.add_argument("--to-crs", help="reproject the result to this CRS")
    orthorectify.add_argument("--resampling", default="bilinear", help=_HELP_RESAMPLING)
    orthorectify.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    orthorectify.set_defaults(func=_cmd_orthorectify)

    edit_info = sub.add_parser(
        "edit-info", help="edit a raster's CRS / nodata / tags in place"
    )
    edit_info.add_argument("input", help="raster path to edit in place")
    edit_info.add_argument("--crs", help="set the CRS (EPSG code, WKT, PROJ4)")
    edit_info.add_argument("--nodata", type=float, help="set the no-data value")
    edit_info.add_argument(
        "--tag",
        action="append",
        metavar="KEY=VALUE",
        help="set a metadata tag; repeat for each tag",
    )
    edit_info.set_defaults(func=_cmd_edit_info)

    calc = sub.add_parser("calc", help="evaluate a band expression into a new raster")
    calc.add_argument(
        "expr",
        help="expression over inputs A, B, ... e.g. '(A - B) / (A + B)'",
    )
    calc.add_argument(
        "operands",
        nargs="+",
        help="one or more input rasters followed by the output path",
    )
    calc.add_argument("--dtype", help="output numpy dtype (e.g. float32)")
    calc.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    calc.set_defaults(func=_cmd_calc)

    shapes = sub.add_parser(
        "shapes", help="vectorize a raster to a vector file (one feature per cell)"
    )
    shapes.add_argument("input", help=_HELP_SRC_RASTER)
    shapes.add_argument("output", help="destination vector path")
    shapes.add_argument(
        "--geometry",
        choices=["polygon", "point"],
        default="polygon",
        help="per-cell geometry to emit (default: polygon)",
    )
    shapes.add_argument(
        "--driver", default="geojson", help="OGR vector driver (default: geojson)"
    )
    shapes.add_argument(
        "--allow-large",
        action="store_true",
        help="proceed even when the raster has more than ~4M cells "
        "(one feature per cell can exhaust memory)",
    )
    shapes.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    shapes.set_defaults(func=_cmd_shapes)

    rasterize = sub.add_parser("rasterize", help="burn a vector into a new raster")
    rasterize.add_argument("input", help="source vector path")
    rasterize.add_argument("output", help=_HELP_DST_RASTER)
    rasterize.add_argument(
        "--cell-size",
        type=float,
        help="output cell size (required unless --like; ignored when --like is given)",
    )
    rasterize.add_argument(
        "--like", help="template raster whose grid/CRS the output adopts"
    )
    rasterize.add_argument(
        "--column", help="attribute column to burn (default: all non-geometry columns)"
    )
    rasterize.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    rasterize.set_defaults(func=_cmd_rasterize)

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
    except Exception as exc:  # noqa: BLE001
        # An unexpected internal error (a bug, or a GDAL error type not listed
        # above) still gets a one-line message for the CLI user rather than a raw
        # traceback — unless PYRAMIDS_DEBUG is set, which re-raises the full stack.
        if os.environ.get("PYRAMIDS_DEBUG"):
            raise
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        result = 1
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
