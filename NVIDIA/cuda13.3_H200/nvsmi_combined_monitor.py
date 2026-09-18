# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Run NVIDIA telemetry and NVLink monitors together, then merge their CSVs."""

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

from nvsmi_common import parse_gpu_ids, positive_float, positive_int

NVLINK_FIELDS = (
    "nvlink_rx_gb_s",
    "nvlink_tx_gb_s",
    "nvlink_total_gb_s",
    "nvlink_bidirectional_utilization_pct",
)


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run H200 dmon and NVLink monitors and merge their CSV output."
    )
    parser.add_argument("-g", "--gpus", nargs="+", metavar="GPU")
    parser.add_argument("-w", "--interval", type=positive_int, default=1)
    parser.add_argument("-W", "--duration", type=positive_int)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--run-id", default=stamp)
    parser.add_argument("--gpu-output", type=Path)
    parser.add_argument("--nvlink-output", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--match-tolerance", type=nonnegative_float)
    parser.add_argument("--link-capacity-gb-s", type=positive_float, default=50.0)
    parser.add_argument("--query-timeout", type=positive_float, default=10.0)
    parser.add_argument("--max-errors", type=positive_int, default=3)
    parser.add_argument("--metric-groups", default="pucmet")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if not args.run_id or args.run_id in {".", ".."} or "/" in args.run_id:
        raise ValueError("--run-id must be non-empty and must not contain '/'")
    return (
        args.gpu_output or args.output_dir / f"{args.run_id}_nvsmi_dmon.csv",
        args.nvlink_output or args.output_dir / f"{args.run_id}_nvsmi_nvlink_bandwidth.csv",
        args.output or args.output_dir / f"{args.run_id}_nvsmi_consolidated.csv",
    )


def monitor_commands(
    args: argparse.Namespace, gpu_ids: list[int], gpu_output: Path, nvlink_output: Path
) -> tuple[list[str], list[str]]:
    script_dir = Path(__file__).resolve().parent
    common = ["-w", str(args.interval)]
    if args.duration is not None:
        common.extend(["-W", str(args.duration)])
    if gpu_ids:
        common.extend(["-g", *(str(gpu) for gpu in gpu_ids)])
    gpu_command = [
        sys.executable, str(script_dir / "nvsmi_gpu_monitor.py"), *common,
        "--metric-groups", args.metric_groups, "-o", str(gpu_output),
    ]
    nvlink_command = [
        sys.executable, str(script_dir / "nvsmi_nvlink_bw_monitor.py"), *common,
        "--link-capacity-gb-s", str(args.link_capacity_gb_s),
        "--query-timeout", str(args.query_timeout), "--max-errors", str(args.max_errors),
        "-o", str(nvlink_output),
    ]
    if args.overwrite:
        gpu_command.append("--overwrite")
        nvlink_command.append("--overwrite")
    return gpu_command, nvlink_command


def run_monitors(commands: tuple[list[str], list[str]]) -> tuple[list[int], int | None]:
    processes = [subprocess.Popen(command, start_new_session=True) for command in commands]
    received_signal: int | None = None

    def terminate() -> None:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def stop(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        terminate()

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while any(process.poll() is None for process in processes):
            failure = next((p for p in processes if p.poll() not in (None, 0)), None)
            if failure is not None or received_signal is not None:
                terminate()
                break
            time.sleep(0.1)
    finally:
        statuses = [process.wait() for process in processes]
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return statuses, received_signal


def read_nvlink_samples(path: Path) -> dict[int, list[tuple[float, dict[str, str]]]]:
    """Aggregate physical-link rows into one GPU-level sample."""

    grouped: dict[tuple[int, float], dict[str, float]] = defaultdict(
        lambda: {"rx": 0.0, "tx": 0.0, "capacity": 0.0, "valid": 0.0}
    )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "timestamp_epoch", "gpu", "rx_gb_s", "tx_gb_s",
            "per_direction_capacity_gb_s", "counter_status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"NVLink CSV is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, 2):
            try:
                key = int(row["gpu"]), float(row["timestamp_epoch"])
                if row["counter_status"] != "ok":
                    continue
                grouped[key]["rx"] += float(row["rx_gb_s"])
                grouped[key]["tx"] += float(row["tx_gb_s"])
                grouped[key]["capacity"] += float(row["per_direction_capacity_gb_s"])
                grouped[key]["valid"] += 1
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid NVLink values on CSV row {row_number}") from error
    result: dict[int, list[tuple[float, dict[str, str]]]] = defaultdict(list)
    for (gpu, timestamp), values in grouped.items():
        if not values["valid"] or not values["capacity"]:
            continue
        total = values["rx"] + values["tx"]
        result[gpu].append(
            (timestamp, {
                NVLINK_FIELDS[0]: f"{values['rx']:.6f}",
                NVLINK_FIELDS[1]: f"{values['tx']:.6f}",
                NVLINK_FIELDS[2]: f"{total:.6f}",
                NVLINK_FIELDS[3]: f"{100 * total / (2 * values['capacity']):.6f}",
            })
        )
    for samples in result.values():
        samples.sort(key=lambda sample: sample[0])
    return dict(result)


def nearest_sample(
    samples: list[tuple[float, dict[str, str]]], timestamp: float, tolerance: float
) -> dict[str, str] | None:
    position = bisect.bisect_left(samples, timestamp, key=lambda item: item[0])
    candidates = samples[max(0, position - 1): min(len(samples), position + 1)]
    if not candidates:
        return None
    closest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return closest[1] if abs(closest[0] - timestamp) <= tolerance else None


def merge_csv_files(
    gpu_path: Path, nvlink_path: Path, output_path: Path, tolerance: float, overwrite: bool
) -> tuple[int, int]:
    samples = read_nvlink_samples(nvlink_path)
    mode, total, matched = ("w" if overwrite else "x"), 0, 0
    with gpu_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"timestamp_epoch", "gpu"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"GPU CSV is missing columns: {', '.join(missing)}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode, newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=[*reader.fieldnames, *NVLINK_FIELDS])
            writer.writeheader()
            for row_number, row in enumerate(reader, 2):
                total += 1
                try:
                    timestamp, gpu = float(row["timestamp_epoch"]), int(row["gpu"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid GPU identifiers on CSV row {row_number}") from error
                addition = nearest_sample(samples.get(gpu, []), timestamp, tolerance)
                if addition is not None:
                    row.update(addition)
                    matched += 1
                writer.writerow(row)
    return total, matched


def main() -> int:
    parser, args = build_parser(), None
    args = parser.parse_args()
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
        gpu_path, nvlink_path, merged_path = output_paths(args)
    except ValueError as error:
        parser.error(str(error))
    paths = (gpu_path, nvlink_path, merged_path)
    if len({path.resolve() for path in paths}) != 3:
        parser.error("raw and merged output paths must be different")
    collisions = [path for path in paths if path.exists()]
    if collisions and not args.overwrite:
        parser.error("output already exists: " + ", ".join(map(str, collisions)) + " (use --overwrite)")
    commands = monitor_commands(args, gpu_ids, gpu_path, nvlink_path)
    import shlex
    print(f"GPU command: {shlex.join(commands[0])}", file=sys.stderr)
    print(f"NVLink command: {shlex.join(commands[1])}", file=sys.stderr)
    print(f"Merged CSV: {merged_path}", file=sys.stderr)
    if args.dry_run:
        return 0
    statuses, received_signal = run_monitors(commands)
    if any(status != 0 for status in statuses) and received_signal is None:
        print(f"Error: monitor exit statuses were GPU={statuses[0]}, NVLink={statuses[1]}", file=sys.stderr)
        return next(status for status in statuses if status != 0)
    tolerance = args.match_tolerance if args.match_tolerance is not None else args.interval / 2
    try:
        total, matched = merge_csv_files(gpu_path, nvlink_path, merged_path, tolerance, args.overwrite)
    except (OSError, ValueError) as error:
        print(f"Error: could not merge monitor output: {error}", file=sys.stderr)
        return 1
    print(f"Merged {matched}/{total} GPU rows within {tolerance:g}s.", file=sys.stderr)
    return 128 + received_signal if received_signal is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
