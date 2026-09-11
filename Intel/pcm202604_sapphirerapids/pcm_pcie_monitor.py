# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Collect aggregate per-socket PCIe transaction metrics with pcm-pcie."""

from __future__ import annotations

import argparse

from pcm_common import add_common_arguments, run_monitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect aggregate per-socket PCIe transactions with pcm-pcie. "
            "Byte volume is estimated as 64 bytes per counted transfer."
        )
    )
    add_common_arguments(
        parser,
        tool="pcm-pcie",
        environment_variable="PCM_PCIE_BIN",
        output_name="pcm_pcie",
    )
    parser.add_argument(
        "--no-bandwidth",
        action="store_true",
        help="omit pcm-pcie's 64-byte-per-transfer byte estimate",
    )
    parser.add_argument(
        "--no-llc-breakdown",
        action="store_true",
        help="omit the additional LLC hit/miss rows",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    native_arguments: list[str] = []
    if not args.no_bandwidth:
        native_arguments.append("-B")
    if not args.no_llc_breakdown:
        native_arguments.append("-e")
    return run_monitor(
        args,
        native_arguments,
        csv_header_prefixes=("Skt,PCIRdCur,",),
    )


if __name__ == "__main__":
    raise SystemExit(main())
