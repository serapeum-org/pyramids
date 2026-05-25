"""Command-line interface for pyramids — currently the `cog` command group.

Exposes the common Cloud Optimized GeoTIFF workflow from the shell, mirroring
`rio cogeo create|validate|info` but built on the pyramids `COG` engine and
the standard-library :mod:`argparse` (no extra dependency):

- `pyramids cog create IN OUT [--profile P] [--compress C] [--blocksize N]`
- `pyramids cog validate FILE [--strict]`
- `pyramids cog info FILE`

The entry point is registered as the `pyramids` console script; the functions
here are also callable in-process (`main([...])`) for testing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from pyramids.dataset import Dataset
from pyramids.dataset.cog import PROFILES, cog_info, validate


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


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser with the `cog`
        command group and its `create` / `validate` / `info` subcommands.
    """
    parser = argparse.ArgumentParser(prog="pyramids", description="pyramids GIS toolkit")
    sub = parser.add_subparsers(dest="group", required=True)

    cog = sub.add_parser("cog", help="Cloud Optimized GeoTIFF commands")
    cog_sub = cog.add_subparsers(dest="command", required=True)

    create = cog_sub.add_parser("create", help="write a raster as a COG")
    create.add_argument("input", help="source raster path")
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
    val.add_argument(
        "--strict", action="store_true", help="treat warnings as errors"
    )
    val.set_defaults(func=_cmd_validate)

    info = cog_sub.add_parser("info", help="print structured COG metadata")
    info.add_argument("file", help="raster path to inspect")
    info.set_defaults(func=_cmd_info)

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
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
