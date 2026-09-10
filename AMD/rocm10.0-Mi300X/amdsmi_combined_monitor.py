# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Run both AMD-SMI monitors and merge their timestamped CSV output."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from amdsmi_common import parse_gpu_ids, positive_float, positive_int


def nonnegative_float(value: str) -> float:
    """Parse a finite floating-point value greater than or equal to zero."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Run the GPU and xGMI monitors concurrently, then merge xGMI "
            "utilization into the GPU telemetry CSV."
        )
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
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--run-id",
        default=timestamp,
        help="prefix shared by all three output files (default: current UTC time)",
    )
    parser.add_argument("--gpu-output", type=Path, help="raw GPU-monitor CSV path")
    parser.add_argument("--xgmi-output", type=Path, help="raw xGMI CSV path")
    parser.add_argument("-o", "--output", type=Path, help="merged CSV path")
    parser.add_argument(
        "--match-tolerance",
        type=nonnegative_float,
        help="maximum timestamp difference in seconds (default: interval / 2)",
    )
    parser.add_argument(
        "--link-capacity-gb-s",
        type=positive_float,
        default=64.0,
        help="xGMI capacity in each direction in GB/s (default: 64)",
    )
    parser.add_argument("--query-timeout", type=positive_float, default=10.0)
    parser.add_argument("--max-errors", type=positive_int, default=3)
    parser.add_argument("--ecc", action="store_true")
    parser.add_argument("--violation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve the raw and merged output paths from one run identifier."""

    if not args.run_id or args.run_id in {".", ".."} or "/" in args.run_id:
        raise ValueError("--run-id must be non-empty and must not contain '/'")
    output_dir = args.output_dir
    return (
        args.gpu_output or output_dir / f"{args.run_id}_amdsmi_monitor.csv",
        args.xgmi_output
        or output_dir / f"{args.run_id}_amdsmi_xgmi_bandwidth.csv",
        args.output or output_dir / f"{args.run_id}_amdsmi_consolidated.csv",
    )


def monitor_commands(
    args: argparse.Namespace,
    gpu_ids: list[int],
    gpu_output: Path,
    xgmi_output: Path,
) -> tuple[list[str], list[str]]:
    """Build both child commands with the shared selection and timing options."""

    script_dir = Path(__file__).resolve().parent
    common = ["-w", str(args.interval)]
    if args.duration is not None:
        common.extend(["-W", str(args.duration)])
    if gpu_ids:
        common.extend(["-g", *(str(gpu) for gpu in gpu_ids)])

    gpu_command = [
        sys.executable,
        str(script_dir / "amdsmi_gpu_monitor.py"),
        *common,
        "-o",
        str(gpu_output),
    ]
    if args.ecc:
        gpu_command.append("--ecc")
    if args.violation:
        gpu_command.append("--violation")

    xgmi_command = [
        sys.executable,
        str(script_dir / "amdsmi_xgmi_bw_monitor.py"),
        *common,
        "--link-capacity-gb-s",
        str(args.link_capacity_gb_s),
        "--query-timeout",
        str(args.query_timeout),
        "--max-errors",
        str(args.max_errors),
        "-o",
        str(xgmi_output),
    ]
    if args.overwrite:
        gpu_command.append("--overwrite")
        xgmi_command.append("--overwrite")
    return gpu_command, xgmi_command


def printable_command(command: list[str]) -> str:
    import shlex

    return shlex.join(command)


def terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    """Ask each monitor process group to stop and flush its output."""

    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def run_monitors(commands: tuple[list[str], list[str]]) -> tuple[list[int], int | None]:
    """Run both collectors concurrently and stop the peer if either one fails."""

    processes = [subprocess.Popen(command, start_new_session=True) for command in commands]
    received_signal: int | None = None

    def stop(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        terminate_processes(processes)

    previous_handlers = {
        signum: signal.signal(signum, stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while any(process.poll() is None for process in processes):
            if received_signal is not None:
                break
            failed = next(
                (
                    process
                    for process in processes
                    if process.poll() not in (None, 0)
                ),
                None,
            )
            if failed is not None:
                terminate_processes(processes)
                break
            time.sleep(0.1)
    finally:
        return_codes = [process.wait() for process in processes]
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return return_codes, received_signal


def read_xgmi_samples(
    path: Path,
) -> tuple[dict[int, list[tuple[float, dict[int, str]]]], list[int]]:
    """Pivot directed xGMI rows into one peer-value mapping per source sample."""

    grouped: dict[tuple[int, float], dict[int, str]] = defaultdict(dict)
    peers: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "timestamp_epoch",
            "source_gpu",
            "peer_gpu",
            "bidirectional_utilization_pct",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"xGMI CSV is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp_epoch"])
                source_gpu = int(row["source_gpu"])
                peer_gpu = int(row["peer_gpu"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid xGMI identifiers on CSV row {row_number}") from error
            grouped[(source_gpu, timestamp)][peer_gpu] = row[
                "bidirectional_utilization_pct"
            ]
            peers.add(peer_gpu)

    by_source: dict[int, list[tuple[float, dict[int, str]]]] = defaultdict(list)
    for (source_gpu, timestamp), values in grouped.items():
        by_source[source_gpu].append((timestamp, values))
    for samples in by_source.values():
        samples.sort(key=lambda sample: sample[0])
    return dict(by_source), sorted(peers)


def nearest_sample(
    samples: list[tuple[float, dict[int, str]]],
    timestamp: float,
    tolerance: float,
) -> dict[int, str] | None:
    """Return the closest xGMI sample when it is within the allowed skew."""

    position = bisect.bisect_left(samples, timestamp, key=lambda sample: sample[0])
    candidates = samples[max(0, position - 1) : min(len(samples), position + 1)]
    if not candidates:
        return None
    closest = min(candidates, key=lambda sample: abs(sample[0] - timestamp))
    return closest[1] if abs(closest[0] - timestamp) <= tolerance else None


def merge_csv_files(
    gpu_path: Path,
    xgmi_path: Path,
    output_path: Path,
    tolerance: float,
    overwrite: bool,
) -> tuple[int, int, list[str]]:
    """Append nearest per-peer xGMI utilization values to every GPU row."""

    samples_by_source, peer_ids = read_xgmi_samples(xgmi_path)
    xgmi_fields = [
        f"xgmi_to_gpu_{peer}_bidirectional_utilization_pct" for peer in peer_ids
    ]
    mode = "w" if overwrite else "x"
    matched = 0
    total = 0

    with gpu_path.open(newline="", encoding="utf-8") as gpu_handle:
        reader = csv.DictReader(gpu_handle)
        if reader.fieldnames is None:
            raise ValueError("GPU CSV has no header")
        required = {"timestamp", "gpu"}
        if not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames))
            raise ValueError(f"GPU CSV is missing columns: {', '.join(missing)}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode, newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=[*reader.fieldnames, *xgmi_fields],
            )
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                total += 1
                try:
                    timestamp = float(row["timestamp"])
                    source_gpu = int(row["gpu"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid GPU timestamp or index on CSV row {row_number}"
                    ) from error
                sample = nearest_sample(
                    samples_by_source.get(source_gpu, []), timestamp, tolerance
                )
                if sample is not None:
                    matched += 1
                    for peer_gpu, value in sample.items():
                        row[
                            f"xgmi_to_gpu_{peer_gpu}_bidirectional_utilization_pct"
                        ] = value
                writer.writerow(row)
    return total, matched, xgmi_fields


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
        gpu_output, xgmi_output, merged_output = output_paths(args)
    except ValueError as error:
        parser.error(str(error))

    paths = (gpu_output, xgmi_output, merged_output)
    if len(set(path.resolve() for path in paths)) != len(paths):
        parser.error("raw and merged output paths must be different")
    collisions = [path for path in paths if path.exists()]
    if collisions and not args.overwrite:
        parser.error(
            "output already exists: "
            + ", ".join(str(path) for path in collisions)
            + " (use --overwrite)"
        )

    commands = monitor_commands(args, gpu_ids, gpu_output, xgmi_output)
    print(f"Run ID: {args.run_id}", file=sys.stderr)
    print(f"GPU command: {printable_command(commands[0])}", file=sys.stderr)
    print(f"xGMI command: {printable_command(commands[1])}", file=sys.stderr)
    print(f"Merged CSV: {merged_output}", file=sys.stderr)
    if args.dry_run:
        return 0

    return_codes, received_signal = run_monitors(commands)
    if any(code != 0 for code in return_codes) and received_signal is None:
        print(
            f"Error: monitor exit statuses were GPU={return_codes[0]}, "
            f"xGMI={return_codes[1]}; no merged CSV was created.",
            file=sys.stderr,
        )
        return next(code for code in return_codes if code != 0)

    tolerance = (
        args.match_tolerance
        if args.match_tolerance is not None
        else args.interval / 2.0
    )
    try:
        total, matched, fields = merge_csv_files(
            gpu_output,
            xgmi_output,
            merged_output,
            tolerance,
            args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"Error: could not merge monitor output: {error}", file=sys.stderr)
        return 1

    print(
        f"Merged {matched}/{total} GPU rows within {tolerance:g}s; "
        f"added {len(fields)} peer columns.",
        file=sys.stderr,
    )
    if received_signal is not None:
        return 128 + received_signal
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
