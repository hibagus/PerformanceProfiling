# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
# This Python script is used to collect and monitor xGMI bandwidth utilization between AMD Mi300X GPUs. 
# 
# The basic command is `amd-smi xgmi`, which unfortunately only exposes cumulative data transferred in and out between GPUs.
# Therefore, extra processing is needed to obtain instantaneous rate (i.e., bandwidth utilization) between GPUs.
# The script samples the cumulative data repeatedly at specific interval, then the instantaneous rate is calculated by taking 
# the difference between two consecutive samples and dividing it by the time interval.
#
# bandwidth = (current_counter - previous_counter) / elapsed_time
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

#%% Import Required External Libraries
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

#%% Import Required Internal Libraries
from amdsmi_common import (
    open_text_file,
    parse_gpu_ids,
    positive_float,
    positive_int,
    resolve_output_path,
)

#%% Constants
# AMD-SMI labels its counters as KB. 
# Use decimal SI conversions so the emitted GB/s values match AMD's stated 
# MI300X link capacity:
# 1 KB = 1,000 bytes, 1 GB = 1,000,000,000 bytes.
KB_TO_GB = 1_000.0 / 1_000_000_000.0

# This is specific to Mi300X GPUs. Each MI300X has 8 xGMI links, one to each of the other seven GPUs on the UBB, plus one to the CPU.
# The CPU link is the PCIe Express 5.0 x16; use the `amdsmi_gpu_monitor.py` to collect the CPU link bandwidth utilization if needed.
# Every link carries 64 GB/s in each direction, or 128 GB/s full duplex.
# This can be adjusted through "--link-capacity-gb-s" if the user knows the actual link capacity of their system.
MAX_XGMI_BANDWIDTH_GB_S = 64.0

#%% CSV_Fields
# This is the CSV header produced by this script.
CSV_FIELDS = (
    "timestamp_epoch",
    "timestamp_utc",
    "interval_seconds",
    "source_gpu",
    "source_bdf",
    "peer_gpu",
    "peer_bdf",
    "read_counter_kb",
    "write_counter_kb",
    "read_delta_kb",
    "write_delta_kb",
    "read_gb_s",
    "write_gb_s",
    "total_gb_s",
    "unidirectional_capacity_gb_s",
    "read_utilization_pct",
    "write_utilization_pct",
    "bidirectional_utilization_pct",
    "counter_status",
)

#%% Dataclass
@dataclass(frozen=True)
class LinkCounter:
    """One source GPU's cumulative counters for one peer GPU."""

    source_gpu: int
    source_bdf: str
    peer_gpu: int
    peer_bdf: str
    read_kb: float
    write_kb: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.source_gpu, self.peer_gpu)


@dataclass(frozen=True)
class Sample:
    """All valid directed link counters returned by one AMD-SMI query."""

    monotonic_time: float
    epoch_time: float
    links: dict[tuple[int, int], LinkCounter]


#%% JSON Dictionary Handling and Parsing
def nested_dicts(value: Any) -> Iterable[dict[str, Any]]:
    """Yield dictionaries from AMD-SMI's occasionally nested metric arrays."""

    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from nested_dicts(item)


def counter_as_kb(value: Any) -> float | None:
    """Return a numeric AMD-SMI counter normalized to KB, or None for N/A.

    Current MI300X output uses ``{"value": N, "unit": "KB"}``, but handling a
    few adjacent units makes the parser safer across AMD-SMI revisions.
    """

    if not isinstance(value, dict):
        return None
    raw = value.get("value")
    unit = str(value.get("unit", "KB")).upper()
    if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        return None

    factors = {
        "B": 1.0 / 1_000.0,
        "KB": 1.0,
        "MB": 1_000.0,
        "GB": 1_000_000.0,
    }
    factor = factors.get(unit)
    return float(raw) * factor if factor is not None else None


def parse_xgmi_links(payload: dict[str, Any]) -> dict[tuple[int, int], LinkCounter]:
    """Extract valid directed per-peer counters from an AMD-SMI JSON payload."""

    links: dict[tuple[int, int], LinkCounter] = {}
    for source in nested_dicts(payload.get("xgmi_metric", [])):
        link_metrics = source.get("link_metrics")
        if not isinstance(link_metrics, dict) or not isinstance(link_metrics.get("links"), list):
            continue

        source_gpu = source.get("gpu")
        if not isinstance(source_gpu, int):
            continue

        for peer in link_metrics["links"]:
            if not isinstance(peer, dict):
                continue
            peer_gpu = peer.get("gpu")
            read_kb = counter_as_kb(peer.get("read"))
            write_kb = counter_as_kb(peer.get("write"))

            # Self-links and unsupported links are represented as N/A. A rate
            # requires both directional counters, so omit incomplete entries.
            if not isinstance(peer_gpu, int) or read_kb is None or write_kb is None:
                continue

            counter = LinkCounter(
                source_gpu=source_gpu,
                source_bdf=str(source.get("bdf", "")),
                peer_gpu=peer_gpu,
                peer_bdf=str(peer.get("bdf", "")),
                read_kb=read_kb,
                write_kb=write_kb,
            )
            links[counter.key] = counter
    return links


def query_xgmi(gpu_ids: list[int], timeout: float) -> Sample:
    """Run one AMD-SMI query and timestamp the midpoint of the call.

    Querying takes nonzero time. The midpoint is a closer estimate of when the
    returned counters were observed than timestamping only before or after it.
    """

    # AMD-SMI 27.0 only exposes the complete per-peer counter matrix when all
    # GPUs are queried. Asking it for one GPU returns an N/A self-link instead
    # of that GPU's links to its peers. Query the full matrix here and apply
    # the user's source-GPU selection after parsing it.
    command = ["amd-smi", "xgmi", "--metric", "--gpu", "all", "--json"]

    monotonic_before = time.monotonic()
    epoch_before = time.time()
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    monotonic_after = time.monotonic()
    epoch_after = time.time()

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error text"
        raise RuntimeError(f"AMD-SMI exited with status {result.returncode}: {detail}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"AMD-SMI returned invalid JSON: {error}") from error

    links = parse_xgmi_links(payload)
    if gpu_ids:
        selected_gpus = set(gpu_ids)
        links = {
            key: counter
            for key, counter in links.items()
            if counter.source_gpu in selected_gpus
        }
    if not links:
        selection = ", ".join(map(str, gpu_ids)) if gpu_ids else "all GPUs"
        raise RuntimeError(
            "AMD-SMI returned no usable per-peer XGMI counters for " + selection
        )

    return Sample(
        monotonic_time=(monotonic_before + monotonic_after) / 2.0,
        epoch_time=(epoch_before + epoch_after) / 2.0,
        links=links,
    )


def format_number(value: int | float, decimal_places: int = 6) -> str:
    """Keep integral counters readable while formatting calculated rates."""

    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.{decimal_places}f}"


def bandwidth_rows(
    previous: Sample,
    current: Sample,
    capacity_gb_s: float,
) -> list[dict[str, object]]:
    """Calculate rows for links present in consecutive samples.

    If a counter decreases, its device was reset or its unknown-width counter
    wrapped. Guessing the wrap width could create a false bandwidth spike, so the
    interval is marked ``reset_or_wrap`` and its rates are left blank. The current
    value still becomes the baseline for the next interval.
    """

    elapsed = current.monotonic_time - previous.monotonic_time
    if elapsed <= 0:
        raise RuntimeError("non-positive elapsed time between XGMI samples")

    timestamp_utc = datetime.fromtimestamp(current.epoch_time, timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    rows: list[dict[str, object]] = []

    for key in sorted(current.links):
        now = current.links[key]
        before = previous.links.get(key)
        if before is None:
            continue

        read_delta = now.read_kb - before.read_kb
        write_delta = now.write_kb - before.write_kb
        reset = read_delta < 0 or write_delta < 0

        row: dict[str, object] = {
            "timestamp_epoch": f"{current.epoch_time:.6f}",
            "timestamp_utc": timestamp_utc,
            "interval_seconds": f"{elapsed:.6f}",
            "source_gpu": now.source_gpu,
            "source_bdf": now.source_bdf,
            "peer_gpu": now.peer_gpu,
            "peer_bdf": now.peer_bdf,
            "read_counter_kb": format_number(now.read_kb),
            "write_counter_kb": format_number(now.write_kb),
            "unidirectional_capacity_gb_s": f"{capacity_gb_s:.6f}",
            "counter_status": "reset_or_wrap" if reset else "ok",
        }

        if reset:
            row.update(
                read_delta_kb="",
                write_delta_kb="",
                read_gb_s="",
                write_gb_s="",
                total_gb_s="",
                read_utilization_pct="",
                write_utilization_pct="",
                bidirectional_utilization_pct="",
            )
        else:
            read_gb_s = read_delta * KB_TO_GB / elapsed
            write_gb_s = write_delta * KB_TO_GB / elapsed
            row.update(
                read_delta_kb=format_number(read_delta),
                write_delta_kb=format_number(write_delta),
                read_gb_s=f"{read_gb_s:.6f}",
                write_gb_s=f"{write_gb_s:.6f}",
                total_gb_s=f"{read_gb_s + write_gb_s:.6f}",
                read_utilization_pct=f"{100.0 * read_gb_s / capacity_gb_s:.6f}",
                write_utilization_pct=f"{100.0 * write_gb_s / capacity_gb_s:.6f}",
                # A full-duplex link can simultaneously read and write at 64
                # GB/s, so combined utilization uses the 128 GB/s denominator.
                bidirectional_utilization_pct=(
                    f"{100.0 * (read_gb_s + write_gb_s) / (2.0 * capacity_gb_s):.6f}"
                ),
            )
        rows.append(row)
    return rows


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    """Create the CLI separately to keep main() focused on collection."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="This script is used to collect and monitor xGMI bandwidth utilization between AMD Mi300X GPUs.\n(C) 2026 Bagus Hanindhito, Dell Technologies Inc.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-g",
        "--gpus",
        nargs="+",
        metavar="GPU",
        help="source GPU indexes, space- or comma-separated (default: all)",
    )
    parser.add_argument("-w", "--interval", type=positive_float, default=1.0)
    parser.add_argument("-W", "--duration", type=positive_float)
    parser.add_argument(
        "--link-capacity-gb-s",
        type=positive_float,
        default=MAX_XGMI_BANDWIDTH_GB_S,
        help="per-direction capacity of one peer link in GB/s (default: 64)",
    )
    parser.add_argument(
        "--query-timeout",
        type=positive_float,
        default=10.0,
        help="seconds allowed for each AMD-SMI query (default: 10)",
    )
    parser.add_argument(
        "--max-errors",
        type=positive_int,
        default=3,
        help="stop after this many consecutive query failures (default: 3)",
    )
    parser.add_argument(
        "--output-mode",
        choices=("stdout", "file", "both"),
        default="file",
        help="where CSV rows are written (default: file)",
    )
    parser.add_argument("--stdout", action="store_true", help="shortcut for --output-mode stdout")
    parser.add_argument("--both", action="store_true", help="shortcut for --output-mode both")
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
        default=f"{timestamp}_amdsmi_xgmi_bandwidth.csv",
        help="filename within --output-dir",
    )
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument("--overwrite", action="store_true")
    write_group.add_argument("--append", action="store_true")
    return parser


def configure_outputs(args: argparse.Namespace) -> tuple[list[TextIO], Path | None]:
    """Open requested CSV destinations without mixing diagnostics into stdout."""

    if args.stdout and args.both:
        raise ValueError("--stdout and --both are mutually exclusive")
    if args.stdout:
        args.output_mode = "stdout"
    elif args.both:
        args.output_mode = "both"

    if args.output_mode == "stdout":
        if args.overwrite or args.append:
            raise ValueError("--overwrite and --append require file or both output mode")
        return [sys.stdout], None

    output = resolve_output_path(args.output, args.output_dir, args.filename, ".csv")
    file_handle, append_to_existing = open_text_file(
        output, overwrite=args.overwrite, append=args.append
    )
    # Store this on the handle for header selection without inventing a wrapper type.
    setattr(file_handle, "_xgmi_skip_header", append_to_existing)

    handles: list[TextIO] = [file_handle]
    if args.output_mode == "both":
        handles.insert(0, sys.stdout)
    return handles, output


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    parser = build_parser(project_root)
    args = parser.parse_args()

    if shutil.which("amd-smi") is None:
        parser.error("amd-smi was not found on PATH")
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
        handles, output_path = configure_outputs(args)
    except ValueError as error:
        parser.error(str(error))

    writers = [csv.DictWriter(handle, fieldnames=CSV_FIELDS) for handle in handles]
    for writer, handle in zip(writers, handles):
        if not getattr(handle, "_xgmi_skip_header", False):
            writer.writeheader()
            handle.flush()

    print(
        f"Collecting XGMI counters every {args.interval:g}s for GPUs "
        f"{','.join(map(str, gpu_ids)) if gpu_ids else 'all'}",
        file=sys.stderr,
    )
    if output_path is not None:
        print(f"Output CSV: {output_path}", file=sys.stderr)
    print("The first query establishes the counter baseline; Ctrl+C stops collection.", file=sys.stderr)

    previous: Sample | None = None
    started = time.monotonic()
    consecutive_errors = 0

    # SIGTERM is common when a benchmark wrapper stops a background collector.
    # Converting it to KeyboardInterrupt provides the same clean flush path as Ctrl+C.
    def stop_on_signal(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)

    try:
        while True:
            iteration_started = time.monotonic()
            try:
                current = query_xgmi(gpu_ids, args.query_timeout)
                consecutive_errors = 0
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                consecutive_errors += 1
                print(
                    f"Warning: XGMI query failed ({consecutive_errors}/{args.max_errors}): {error}",
                    file=sys.stderr,
                )
                if consecutive_errors >= args.max_errors:
                    raise RuntimeError("too many consecutive AMD-SMI query failures") from error
            else:
                if previous is not None:
                    for row in bandwidth_rows(previous, current, args.link_capacity_gb_s):
                        for writer in writers:
                            writer.writerow(row)
                    for handle in handles:
                        handle.flush()
                previous = current

            if args.duration is not None and time.monotonic() - started >= args.duration:
                break

            # Compensate for AMD-SMI query time so query starts remain roughly one
            # interval apart. If a query itself exceeds the interval, start the next
            # one immediately; reported rates still use the measured elapsed time.
            remaining = args.interval - (time.monotonic() - iteration_started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("XGMI collection stopped.", file=sys.stderr)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        for handle in handles:
            if handle is not sys.stdout:
                handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
