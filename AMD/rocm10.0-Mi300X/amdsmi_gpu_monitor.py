# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
# This Python script is used to collect and monitor GPU Temperature, Power, Clock Frequency, Memory Utilization, and PCIe Bandwidth Utilization of AMD Mi300X GPUs. 
# It is a telemetry collection script that utilizes the AMD-SMI command-line tool to gather GPU metrics and write them to a CSV file for further analysis.
 
# The basic command is `amd-smi dmon`.
#
# This script is tested on ROCm 10.0 with AMD Mi300X GPUs, and it may not work properly on other ROCm versions or GPU models.
# +------------------------------------------------------------------------------+
# | AMD-SMI            27.0.0+6b0e43f3                                           |
# | amdgpu Version:    7.1.3.31500000                                            |
# | ROCm Version:      10.0.0                                                    |
# | VBIOS Version:     00182096                                                  |
# | FW PLDM:           01.25.06.10                                               |
# | Platform:          Linux Baremetal                                           |
# |-------------------------------------+----------------------------------------| 


from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from amdsmi_common import open_text_file, parse_gpu_ids, positive_int, resolve_output_path


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="This script is used to collect and monitor Temperature, Power, Clock Frequency, Memory Utilization, and PCIe Bandwidth Utilization of AMD Mi300X GPUs.\n(C) 2026 Bagus Hanindhito, Dell Technologies Inc.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-g",
        "--gpus",
        nargs="+",
        metavar="GPU",
        help="GPU indexes, space- or comma-separated (default: all)",
    )
    parser.add_argument("-w", "--interval", type=positive_int, default=1)
    parser.add_argument("-W", "--duration", type=positive_int)
    parser.add_argument(
        "--output-mode",
        choices=("stdout", "file", "both"),
        default="file",
        help="where CSV rows are written (default: file)",
    )
    shortcuts = parser.add_mutually_exclusive_group()
    shortcuts.add_argument("--stdout", action="store_true", help="shortcut for stdout mode")
    shortcuts.add_argument("--both", action="store_true", help="shortcut for both mode")
    parser.add_argument("-o", "--output", type=Path, help="complete output CSV path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        #default=project_root / "runs" / "telemetry",
        default="./",
        help="directory used when --output is omitted",
    )
    parser.add_argument(
        "--filename",
        default=f"{timestamp}_amdsmi_monitor.csv",
        help="filename within --output-dir",
    )
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument("--overwrite", action="store_true")
    write_group.add_argument("--append", action="store_true")
    parser.add_argument("--ecc", action="store_true", help="include ECC/replay counters")
    parser.add_argument(
        "--violation",
        action="store_true",
        help="include MI300 power and thermal violation status",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args: argparse.Namespace, gpu_ids: list[int]) -> list[str]:
    """Build arguments without a shell, preserving each selected GPU separately."""

    # Board-temperature groups are unsupported on the tested MI300X host.
    # Encoder is absent in MI300X, and its video decoder is unrelated to LLM decode.
    command = [
        "amd-smi",
        "monitor",
        "--power-usage",
        "--temperature",
        "--gfx",
        "--mem",
        "--vram-usage",
        "--pcie",
    ]
    if args.ecc:
        command.append("--ecc")
    if args.violation:
        command.append("--violation")
    command.extend(["--gpu", *(str(gpu) for gpu in gpu_ids or ["all"])])
    command.extend(["--csv", "--watch", str(args.interval)])
    if args.duration is not None:
        command.extend(["--watch_time", str(args.duration)])
    return command


def printable_command(command: list[str]) -> str:
    """Return a copy/pasteable representation for logs and dry runs."""

    import shlex

    return shlex.join(command)


def normalize_csv_header(header: str) -> str:
    """Make the unit of AMD-SMI's raw PCIe bandwidth field explicit."""

    columns = header.split(",")
    return ",".join(
        "pcie_bw_mbps" if column == "pcie_bw" else column for column in columns
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = build_parser(project_root)
    args = parser.parse_args()

    if args.stdout:
        args.output_mode = "stdout"
    elif args.both:
        args.output_mode = "both"
    if args.output_mode == "stdout" and (args.overwrite or args.append):
        parser.error("--overwrite and --append require file or both output mode")
    if shutil.which("amd-smi") is None:
        parser.error("amd-smi was not found on PATH")

    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as error:
        parser.error(str(error))

    command = build_command(args, gpu_ids)
    print(f"AMD-SMI telemetry command: {printable_command(command)}", file=sys.stderr)
    if args.dry_run:
        return 0

    outputs: list[tuple[TextIO, bool]] = []
    output_path: Path | None = None
    if args.output_mode in ("stdout", "both"):
        outputs.append((sys.stdout, False))
    if args.output_mode in ("file", "both"):
        try:
            output_path = resolve_output_path(
                args.output, args.output_dir, args.filename, ".csv"
            )
            file_handle, has_header = open_text_file(
                output_path, overwrite=args.overwrite, append=args.append
            )
        except ValueError as error:
            parser.error(str(error))
        outputs.append((file_handle, has_header))

    print(f"GPUs: {','.join(map(str, gpu_ids)) if gpu_ids else 'all'}", file=sys.stderr)
    print(f"Interval: {args.interval}s", file=sys.stderr)
    print(f"Duration: {args.duration if args.duration else 'until interrupted'}", file=sys.stderr)
    if output_path is not None:
        print(f"Output CSV: {output_path}", file=sys.stderr)

    # Force unbuffered Python output inside AMD-SMI so every one-second sample
    # reaches this process immediately even though stdout is a pipe.
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
        start_new_session=True,
    )
    received_signal: int | None = None

    def stop(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    raw_header: str | None = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if not line or line == "'CTRL' + 'C' to stop watching output:":
                continue
            if raw_header is None:
                raw_header = line
                output_header = normalize_csv_header(raw_header)
                for handle, already_has_header in outputs:
                    if not already_has_header:
                        handle.write(f"{output_header}\n")
                        handle.flush()
                continue
            if line == raw_header:
                continue
            for handle, _already_has_header in outputs:
                handle.write(f"{line}\n")
                handle.flush()
    finally:
        if process.poll() is None:
            process.terminate()
        return_code = process.wait()
        for handle, _already_has_header in outputs:
            if handle is not sys.stdout:
                handle.close()

    if received_signal is not None:
        print("Telemetry stopped.", file=sys.stderr)
        return 128 + received_signal
    if return_code != 0:
        print(f"Error: amd-smi monitor exited with status {return_code}", file=sys.stderr)
        return return_code
    print("Telemetry complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
