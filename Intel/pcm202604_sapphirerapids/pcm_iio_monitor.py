# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Collect per-IIO-stack and per-PCIe-device bandwidth with pcm-iio."""

from __future__ import annotations

import argparse

from pcm_common import add_common_arguments, run_monitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect timestamped PCIe bandwidth per Intel IIO stack and PCIe "
            "device. Root-port rows are enabled by default for GPU attribution."
        )
    )
    add_common_arguments(
        parser,
        tool="pcm-iio",
        environment_variable="PCM_IIO_BIN",
        output_name="pcm_iio",
    )
    parser.add_argument(
        "--no-root-ports",
        action="store_true",
        help="do not add PCIe root-port devices to the CSV",
    )
    parser.add_argument(
        "--human-readable",
        action="store_true",
        help="ask pcm-iio to scale values and add unit suffixes",
    )
    parser.add_argument(
        "--list-topology",
        action="store_true",
        help="write the detected socket/IIO/PCIe mapping once and exit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    native_arguments: list[str] = []
    if not args.no_root_ports:
        native_arguments.append("-root-port")
    if args.human_readable:
        native_arguments.append("-human-readable")
    if args.list_topology:
        native_arguments.append("-list")
    return run_monitor(args, native_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
