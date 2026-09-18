# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Collect H200 telemetry from ``nvidia-smi dmon`` as analysis-ready CSV."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from nvsmi_common import (
    open_text_file,
    parse_gpu_ids,
    positive_int,
    resolve_output_path,
    utc_timestamp,
)


HEADER_NAMES = {
    "gpu": "gpu",
    "pwr": "power_w",
    "gtemp": "gpu_temperature_c",
    "mtemp": "memory_temperature_c",
    "sm": "sm_utilization_pct",
    "mem": "memory_utilization_pct",
    "enc": "encoder_utilization_pct",
    "dec": "decoder_utilization_pct",
    "jpg": "jpeg_utilization_pct",
    "ofa": "ofa_utilization_pct",
    "mclk": "memory_clock_mhz",
    "pclk": "processor_clock_mhz",
    "fb": "framebuffer_memory_used_mb",
    "bar1": "bar1_memory_used_mb",
    "ccpm": "confidential_compute_memory_used_mb",
    "sbecc": "ecc_single_bit_errors",
    "dbecc": "ecc_double_bit_errors",
    "pci": "pcie_replay_errors",
    "rxpci": "pcie_rx_mb_s",
    "txpci": "pcie_tx_mb_s",
}


def normalize_header(raw: list[str]) -> list[str]:
    fields = [field.strip().lstrip("#").strip().lower() for field in raw]
    if "gpu" not in fields:
        raise ValueError("nvidia-smi dmon header has no GPU column")
    normalized: list[str] = []
    used: dict[str, int] = {}
    for field in fields:
        name = HEADER_NAMES.get(field, field.replace(" ", "_") or "unnamed")
        used[name] = used.get(name, 0) + 1
        normalized.append(name if used[name] == 1 else f"{name}_{used[name]}")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Monitor NVIDIA H200 telemetry with nvidia-smi dmon."
    )
    parser.add_argument("-g", "--gpus", nargs="+", metavar="GPU")
    parser.add_argument("-w", "--interval", type=positive_int, default=1)
    parser.add_argument("-W", "--duration", type=positive_int)
    parser.add_argument(
        "--metric-groups",
        default="pucmet",
        help="dmon metric groups (default: pucmet)",
    )
    parser.add_argument("--output-mode", choices=("stdout", "file", "both"), default="file")
    shortcuts = parser.add_mutually_exclusive_group()
    shortcuts.add_argument("--stdout", action="store_true")
    shortcuts.add_argument("--both", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--filename", default=f"{stamp}_nvsmi_dmon.csv")
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument("--overwrite", action="store_true")
    writes.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args: argparse.Namespace, gpu_ids: list[int]) -> list[str]:
    command = [
        "nvidia-smi", "dmon", "-s", args.metric_groups,
        "-d", str(args.interval), "--format", "csv,nounit",
    ]
    if gpu_ids:
        command.extend(["-i", ",".join(map(str, gpu_ids))])
    if args.duration is not None:
        command.extend(["-c", str(math.ceil(args.duration / args.interval))])
    return command


def configure_outputs(args: argparse.Namespace) -> tuple[list[tuple[TextIO, bool]], Path | None]:
    if args.stdout:
        args.output_mode = "stdout"
    elif args.both:
        args.output_mode = "both"
    if args.output_mode == "stdout":
        if args.overwrite or args.append:
            raise ValueError("--overwrite and --append require file or both mode")
        return [(sys.stdout, False)], None
    output = resolve_output_path(args.output, args.output_dir, args.filename)
    handle, had_content = open_text_file(
        output, overwrite=args.overwrite, append=args.append
    )
    destinations: list[tuple[TextIO, bool]] = [(handle, had_content)]
    if args.output_mode == "both":
        destinations.insert(0, (sys.stdout, False))
    return destinations, output


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi was not found on PATH")
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    command = build_command(args, gpu_ids)
    import shlex
    print(f"NVIDIA-SMI dmon command: {shlex.join(command)}", file=sys.stderr)
    if args.dry_run:
        return 0
    try:
        outputs, output_path = configure_outputs(args)
    except ValueError as error:
        parser.error(str(error))
    if output_path:
        print(f"Output CSV: {output_path}", file=sys.stderr)

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True
    )
    received_signal: int | None = None

    def stop(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    header: list[str] | None = None
    writers: list[csv.writer] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            values = next(csv.reader([line]))
            if not values or not any(value.strip() for value in values):
                continue
            if header is None:
                if values[0].strip().lower().lstrip("#").strip() != "gpu":
                    continue
                header = normalize_header(values)
                writers = [csv.writer(handle) for handle, _ in outputs]
                for writer, (handle, has_header) in zip(writers, outputs):
                    if not has_header:
                        writer.writerow(["timestamp_epoch", "timestamp_utc", *header])
                        handle.flush()
                continue
            if values[0].strip().startswith("#"):
                continue
            if len(values) != len(header):
                print(f"Warning: skipped malformed dmon row: {line.rstrip()}", file=sys.stderr)
                continue
            epoch = time.time()
            row = [f"{epoch:.6f}", utc_timestamp(epoch), *(value.strip() for value in values)]
            for writer, (handle, _) in zip(writers, outputs):
                writer.writerow(row)
                handle.flush()
    finally:
        if process.poll() is None:
            process.terminate()
        return_code = process.wait()
        for handle, _ in outputs:
            if handle is not sys.stdout:
                handle.close()
    if received_signal is not None:
        return 128 + received_signal
    if return_code != 0:
        print(f"Error: nvidia-smi dmon exited with status {return_code}", file=sys.stderr)
        return return_code
    if header is None:
        print("Error: nvidia-smi dmon produced no recognizable header", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
