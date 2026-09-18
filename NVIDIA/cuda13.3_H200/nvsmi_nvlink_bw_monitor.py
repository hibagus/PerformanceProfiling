# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Derive H200 per-link NVLink bandwidth from NVIDIA-SMI counters."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from nvsmi_common import (
    open_text_file, parse_gpu_ids, positive_float, positive_int,
    resolve_output_path, utc_timestamp,
)

KIB_TO_GB = 1024.0 / 1_000_000_000.0
H200_NVLINK_CAPACITY_GB_S = 50.0
GPU_RE = re.compile(r"^GPU\s+(\d+)\s*:", re.IGNORECASE)
LINK_RE = re.compile(
    r"^\s*Link\s+(\d+)\s*:\s*(Tx|Rx)\d*\s*:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*KiB\s*$", re.IGNORECASE
)

CSV_FIELDS = (
    "timestamp_epoch", "timestamp_utc", "interval_seconds", "gpu", "link",
    "rx_counter_kib", "tx_counter_kib", "rx_delta_kib", "tx_delta_kib",
    "rx_gb_s", "tx_gb_s", "total_gb_s", "per_direction_capacity_gb_s",
    "rx_utilization_pct", "tx_utilization_pct",
    "bidirectional_utilization_pct", "counter_status",
)


@dataclass(frozen=True)
class LinkCounter:
    gpu: int
    link: int
    rx_kib: float
    tx_kib: float

    @property
    def key(self) -> tuple[int, int]:
        return self.gpu, self.link


@dataclass(frozen=True)
class Sample:
    monotonic_time: float
    epoch_time: float
    links: dict[tuple[int, int], LinkCounter]


def parse_nvlink_counters(output: str) -> dict[tuple[int, int], LinkCounter]:
    """Parse ``nvidia-smi nvlink --getthroughput d`` without fixed line offsets."""

    current_gpu: int | None = None
    partial: dict[tuple[int, int], dict[str, float]] = {}
    for line in output.splitlines():
        gpu_match = GPU_RE.match(line.strip())
        if gpu_match:
            current_gpu = int(gpu_match.group(1))
            continue
        link_match = LINK_RE.match(line)
        if link_match and current_gpu is not None:
            link = int(link_match.group(1))
            direction = link_match.group(2).lower()
            partial.setdefault((current_gpu, link), {})[direction] = float(
                link_match.group(3)
            )
    return {
        key: LinkCounter(key[0], key[1], values["rx"], values["tx"])
        for key, values in partial.items()
        if "rx" in values and "tx" in values
    }


def query_nvlink(gpu_ids: list[int], timeout: float) -> Sample:
    command = ["nvidia-smi", "nvlink", "--getthroughput", "d"]
    if gpu_ids:
        command.extend(["-i", ",".join(map(str, gpu_ids))])
    before_mono, before_epoch = time.monotonic(), time.time()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    after_mono, after_epoch = time.monotonic(), time.time()
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error text"
        raise RuntimeError(f"NVIDIA-SMI exited with status {result.returncode}: {detail}")
    links = parse_nvlink_counters(result.stdout)
    if not links:
        raise RuntimeError("NVIDIA-SMI returned no complete NVLink Tx/Rx counters")
    return Sample((before_mono + after_mono) / 2, (before_epoch + after_epoch) / 2, links)


def fmt(value: int | float) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.6f}"


def bandwidth_rows(previous: Sample, current: Sample, capacity: float) -> list[dict[str, object]]:
    elapsed = current.monotonic_time - previous.monotonic_time
    if elapsed <= 0:
        raise RuntimeError("non-positive elapsed time between NVLink samples")
    rows: list[dict[str, object]] = []
    for key in sorted(current.links):
        now, before = current.links[key], previous.links.get(key)
        if before is None:
            continue
        rx_delta, tx_delta = now.rx_kib - before.rx_kib, now.tx_kib - before.tx_kib
        reset = rx_delta < 0 or tx_delta < 0
        row: dict[str, object] = {
            "timestamp_epoch": f"{current.epoch_time:.6f}",
            "timestamp_utc": utc_timestamp(current.epoch_time),
            "interval_seconds": f"{elapsed:.6f}", "gpu": now.gpu, "link": now.link,
            "rx_counter_kib": fmt(now.rx_kib), "tx_counter_kib": fmt(now.tx_kib),
            "per_direction_capacity_gb_s": f"{capacity:.6f}",
            "counter_status": "reset_or_wrap" if reset else "ok",
        }
        if reset:
            row.update({field: "" for field in CSV_FIELDS[7:16] if field not in row})
        else:
            rx_rate = rx_delta * KIB_TO_GB / elapsed
            tx_rate = tx_delta * KIB_TO_GB / elapsed
            row.update(
                rx_delta_kib=fmt(rx_delta), tx_delta_kib=fmt(tx_delta),
                rx_gb_s=f"{rx_rate:.6f}", tx_gb_s=f"{tx_rate:.6f}",
                total_gb_s=f"{rx_rate + tx_rate:.6f}",
                rx_utilization_pct=f"{100 * rx_rate / capacity:.6f}",
                tx_utilization_pct=f"{100 * tx_rate / capacity:.6f}",
                bidirectional_utilization_pct=f"{100 * (rx_rate + tx_rate) / (2 * capacity):.6f}",
            )
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Monitor NVIDIA H200 NVLink bandwidth.")
    parser.add_argument("-g", "--gpus", nargs="+", metavar="GPU")
    parser.add_argument("-w", "--interval", type=positive_float, default=1.0)
    parser.add_argument("-W", "--duration", type=positive_float)
    parser.add_argument("--link-capacity-gb-s", type=positive_float, default=H200_NVLINK_CAPACITY_GB_S)
    parser.add_argument("--query-timeout", type=positive_float, default=10.0)
    parser.add_argument("--max-errors", type=positive_int, default=3)
    parser.add_argument("--output-mode", choices=("stdout", "file", "both"), default="file")
    shortcuts = parser.add_mutually_exclusive_group()
    shortcuts.add_argument("--stdout", action="store_true")
    shortcuts.add_argument("--both", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--filename", default=f"{stamp}_nvsmi_nvlink_bandwidth.csv")
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument("--overwrite", action="store_true")
    writes.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


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
    handle, had_content = open_text_file(output, overwrite=args.overwrite, append=args.append)
    result: list[tuple[TextIO, bool]] = [(handle, had_content)]
    if args.output_mode == "both":
        result.insert(0, (sys.stdout, False))
    return result, output


def main() -> int:
    parser, args = build_parser(), None
    args = parser.parse_args()
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi was not found on PATH")
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
        outputs, output_path = configure_outputs(args) if not args.dry_run else ([], None)
    except ValueError as error:
        parser.error(str(error))
    command = ["nvidia-smi", "nvlink", "--getthroughput", "d"]
    if gpu_ids:
        command.extend(["-i", ",".join(map(str, gpu_ids))])
    import shlex
    print(f"NVIDIA-SMI NVLink command: {shlex.join(command)}", file=sys.stderr)
    if args.dry_run:
        return 0
    if output_path:
        print(f"Output CSV: {output_path}", file=sys.stderr)
    writers = [csv.DictWriter(handle, fieldnames=CSV_FIELDS) for handle, _ in outputs]
    for writer, (handle, had_header) in zip(writers, outputs):
        if not had_header:
            writer.writeheader()
            handle.flush()
    stopped = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    previous: Sample | None = None
    started = time.monotonic()
    errors = 0
    try:
        while not stopped:
            iteration = time.monotonic()
            try:
                current = query_nvlink(gpu_ids, args.query_timeout)
                errors = 0
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                errors += 1
                print(f"Warning: NVLink query failed ({errors}/{args.max_errors}): {error}", file=sys.stderr)
                if errors >= args.max_errors:
                    raise RuntimeError("too many consecutive NVIDIA-SMI query failures") from error
            else:
                if previous is not None:
                    for row in bandwidth_rows(previous, current, args.link_capacity_gb_s):
                        for writer in writers:
                            writer.writerow(row)
                    for handle, _ in outputs:
                        handle.flush()
                previous = current
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            remaining = args.interval - (time.monotonic() - iteration)
            if remaining > 0:
                time.sleep(remaining)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        for handle, _ in outputs:
            if handle is not sys.stdout:
                handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
