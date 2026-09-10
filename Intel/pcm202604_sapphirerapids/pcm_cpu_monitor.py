# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Collect CPU, memory, cache, power, and UPI metrics with Intel PCM."""

from __future__ import annotations

import argparse

from pcm_common import add_common_arguments, run_monitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Intel PCM system, socket, and core metrics as native CSV. "
            "On supported multi-socket servers this includes per-link UPI traffic."
        )
    )
    add_common_arguments(
        parser,
        tool="pcm",
        environment_variable="PCM_BIN",
        output_name="pcm_cpu",
    )
    parser.add_argument("--no-cores", action="store_true", help="omit per-core columns")
    parser.add_argument(
        "--no-sockets", action="store_true", help="omit socket aggregate columns"
    )
    parser.add_argument(
        "--no-system", action="store_true", help="omit system aggregate columns"
    )
    parser.add_argument("--pid", type=int, help="collect core metrics for this process ID")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    native_arguments: list[str] = []
    if args.no_cores:
        native_arguments.append("-nc")
    if args.no_sockets:
        native_arguments.append("-ns")
    if args.no_system:
        native_arguments.append("-nsys")
    if args.pid is not None:
        if args.pid <= 0:
            build_parser().error("--pid must be greater than zero")
        native_arguments.extend(["-pid", str(args.pid)])
    return run_monitor(args, native_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
