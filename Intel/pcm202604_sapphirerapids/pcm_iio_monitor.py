# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Collect per-IIO-stack and per-PCIe-device bandwidth with pcm-iio."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pcm_common import add_common_arguments, run_monitor


MCFG_PATHS = (
    Path("/sys/firmware/acpi/tables/MCFG"),
    Path("/sys/firmware/acpi/tables/MCFG1"),
)

KNOWN_TOPOLOGY_WARNINGS = (
    "Cannot map CPU bus ",
    "IIO PMU unit (stack) 10 is not found",
    "IIO PMU unit (stack) 11 is not found",
)


def needs_sudo(mode: str) -> bool:
    """Decide whether pcm-iio should be launched through sudo."""

    if os.geteuid() == 0 or mode == "never":
        return False
    if mode == "always":
        return True
    return not any(path.is_file() and os.access(path, os.R_OK) for path in MCFG_PATHS)


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
    privilege = parser.add_mutually_exclusive_group()
    privilege.add_argument(
        "--sudo",
        dest="sudo_mode",
        action="store_const",
        const="always",
        help="always launch pcm-iio through sudo",
    )
    privilege.add_argument(
        "--no-sudo",
        dest="sudo_mode",
        action="store_const",
        const="never",
        help="never launch pcm-iio through sudo, even if MCFG is unreadable",
    )
    parser.set_defaults(sudo_mode="auto")
    parser.add_argument(
        "--show-topology-warnings",
        action="store_true",
        help="retain repetitive unmapped-bus and absent-stack 10/11 warnings",
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
    return run_monitor(
        args,
        native_arguments,
        use_sudo=needs_sudo(args.sudo_mode),
        suppressed_diagnostics=(
            () if args.show_topology_warnings else KNOWN_TOPOLOGY_WARNINGS
        ),
        csv_header_prefixes=(
            () if args.list_topology else ("Date,Time,Socket,",)
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
