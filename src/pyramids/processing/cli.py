"""CLI subcommands for the processing pipeline layer.

Registers three geoprocessing subcommands on the top-level ``pyramids``
parser: ``run`` (execute a pipeline YAML over inputs), ``tools`` (list the
registered tools), and ``tool <name>`` (print a tool's parameter schema). Help
text is generated from the registry, never hand-written.
"""

from __future__ import annotations

import argparse

from pyramids.processing.pipeline import Pipeline
from pyramids.processing.registry import resolve, tool_names
from pyramids.processing.runner import run


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a pipeline YAML over ``--inputs`` and report the outcome."""
    pipeline = Pipeline.from_yaml(args.pipeline)
    result = run(
        pipeline,
        args.inputs,
        on_error=args.on_error,
        out=args.out,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
    print(f"{len(result.outputs)} succeeded, {len(result.failures)} failed")
    for source, exc in result.failures:
        print(f"  FAILED {source}: {exc}")
    return 0 if result.ok else 1


def _cmd_tools(args: argparse.Namespace) -> int:
    """List every registered tool with its input/output types."""
    for name in tool_names():
        tool = resolve(name)
        print(f"{name:24} {tool.input_type} -> {tool.output_type}   {tool.description}")
    return 0


def _cmd_tool(args: argparse.Namespace) -> int:
    """Print one tool's parameter schema (its help block).

    An unknown name raises ``ValueError`` from ``resolve``; it propagates to
    ``main`` which reports it on stderr and exits non-zero, like every other
    subcommand.
    """
    print(resolve(args.name).help())
    return 0


def add_processing_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``run`` / ``tools`` / ``tool`` subcommands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    run_p = sub.add_parser("run", help="run a pipeline YAML over inputs")
    run_p.add_argument("pipeline", help="pipeline YAML path")
    run_p.add_argument(
        "--inputs", required=True, help="input path or glob (quote globs)"
    )
    run_p.add_argument("--out", required=True, help="output directory")
    run_p.add_argument(
        "--on-error",
        dest="on_error",
        choices=("skip", "raise"),
        default="skip",
        help="error policy (default: skip)",
    )
    run_p.add_argument(
        "--parallel",
        action="store_true",
        help="run the batch across a process pool (path inputs only)",
    )
    run_p.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=None,
        help="worker-process count for --parallel (default: ~CPU count)",
    )
    run_p.set_defaults(func=_cmd_run)

    tools_p = sub.add_parser("tools", help="list registered processing tools")
    tools_p.set_defaults(func=_cmd_tools)

    tool_p = sub.add_parser("tool", help="print a tool's parameter schema")
    tool_p.add_argument("name", help="tool name (see 'pyramids tools')")
    tool_p.set_defaults(func=_cmd_tool)
