# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Derive per-port InfiniBand bandwidth from the Linux RDMA sysfs counters."""

from __future__ import annotations

import argparse
import csv
import math
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

# port_xmit_data / port_rcv_data count data octets divided by 4 (the IB lane count),
# so one counter unit is 4 bytes; mlx5 exposes them as 64-bit counters.
COUNTER_UNIT_BYTES = 4
RATE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*Gb/sec", re.IGNORECASE)

CSV_FIELDS = (
    "timestamp_epoch", "timestamp_utc", "interval_seconds", "device", "port",
    "link_layer", "state", "rate_gb_s_nominal",
    "rx_counter", "tx_counter", "rx_delta_bytes", "tx_delta_bytes",
    "rx_gb_s", "tx_gb_s", "total_gb_s", "per_direction_capacity_gb_s",
    "rx_utilization_pct", "tx_utilization_pct", "bidirectional_utilization_pct",
    "counter_status",
)


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def utc_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


@dataclass(frozen=True)
class Port:
    device: str
    port: int
    path: Path
    link_layer: str
    state: str
    rate_gb_s: float | None

    @property
    def key(self) -> tuple[str, int]:
        return self.device, self.port

    @property
    def capacity_gb_s(self) -> float | None:
        """Nominal signalling rate per direction in decimal GB/s (Gb/s divided by 8)."""

        return None if self.rate_gb_s is None else self.rate_gb_s / 8.0


@dataclass(frozen=True)
class Sample:
    monotonic_time: float
    epoch_time: float
    counters: dict[tuple[str, int], tuple[int, int]]  # (rx, tx) in counter units


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def parse_rate(text: str) -> float | None:
    """``400 Gb/sec (4X NDR)`` -> 400.0 (Gb/s); None when unreadable."""

    match = RATE_RE.search(text)
    return float(match.group(1)) if match else None


def discover_ports(
    root: Path, devices: list[str] | None, include_inactive: bool, include_ethernet: bool
) -> list[Port]:
    """List the ports under ``root`` (normally /sys/class/infiniband)."""

    ports: list[Port] = []
    if not root.is_dir():
        return ports
    for device_dir in sorted(root.iterdir()):
        if devices and device_dir.name not in devices:
            continue
        ports_dir = device_dir / "ports"
        if not ports_dir.is_dir():
            continue
        for port_dir in sorted(ports_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if not port_dir.name.isdigit():
                continue
            counters = port_dir / "counters"
            if not (counters / "port_xmit_data").exists() or not (counters / "port_rcv_data").exists():
                continue
            link_layer = read_text(port_dir / "link_layer") or "unknown"
            state = read_text(port_dir / "state") or "unknown"
            if not include_ethernet and link_layer.lower() == "ethernet":
                continue
            if not include_inactive and "ACTIVE" not in state.upper():
                continue
            ports.append(Port(device_dir.name, int(port_dir.name), port_dir, link_layer, state,
                              parse_rate(read_text(port_dir / "rate"))))
    return ports


def read_counters(ports: list[Port]) -> Sample:
    before_mono, before_epoch = time.monotonic(), time.time()
    counters: dict[tuple[str, int], tuple[int, int]] = {}
    for port in ports:
        rx = read_text(port.path / "counters" / "port_rcv_data")
        tx = read_text(port.path / "counters" / "port_xmit_data")
        if rx.isdigit() and tx.isdigit():
            counters[port.key] = (int(rx), int(tx))
    after_mono, after_epoch = time.monotonic(), time.time()
    if not counters:
        raise RuntimeError("no readable port_rcv_data / port_xmit_data counters")
    return Sample((before_mono + after_mono) / 2, (before_epoch + after_epoch) / 2, counters)


def bandwidth_rows(
    previous: Sample, current: Sample, ports: list[Port], capacity_override: float | None
) -> list[dict[str, object]]:
    elapsed = current.monotonic_time - previous.monotonic_time
    if elapsed <= 0:
        raise RuntimeError("non-positive elapsed time between InfiniBand samples")
    rows: list[dict[str, object]] = []
    for port in ports:
        now, before = current.counters.get(port.key), previous.counters.get(port.key)
        if now is None or before is None:
            continue
        capacity = capacity_override if capacity_override is not None else port.capacity_gb_s
        rx_delta, tx_delta = now[0] - before[0], now[1] - before[1]
        reset = rx_delta < 0 or tx_delta < 0
        row: dict[str, object] = {
            "timestamp_epoch": f"{current.epoch_time:.6f}",
            "timestamp_utc": utc_timestamp(current.epoch_time),
            "interval_seconds": f"{elapsed:.6f}", "device": port.device, "port": port.port,
            "link_layer": port.link_layer, "state": port.state,
            "rate_gb_s_nominal": "" if port.rate_gb_s is None else f"{port.rate_gb_s:g}",
            "rx_counter": now[0], "tx_counter": now[1],
            "per_direction_capacity_gb_s": "" if capacity is None else f"{capacity:.6f}",
            "counter_status": "reset_or_wrap" if reset else "ok",
        }
        if not reset:
            rx_bytes, tx_bytes = rx_delta * COUNTER_UNIT_BYTES, tx_delta * COUNTER_UNIT_BYTES
            rx_rate, tx_rate = rx_bytes / elapsed / 1e9, tx_bytes / elapsed / 1e9
            row.update(
                rx_delta_bytes=rx_bytes, tx_delta_bytes=tx_bytes,
                rx_gb_s=f"{rx_rate:.6f}", tx_gb_s=f"{tx_rate:.6f}", total_gb_s=f"{rx_rate + tx_rate:.6f}",
            )
            if capacity:
                row.update(
                    rx_utilization_pct=f"{100 * rx_rate / capacity:.6f}",
                    tx_utilization_pct=f"{100 * tx_rate / capacity:.6f}",
                    bidirectional_utilization_pct=f"{100 * (rx_rate + tx_rate) / (2 * capacity):.6f}",
                )
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Monitor InfiniBand port bandwidth from /sys/class/infiniband counters (no root needed)."
    )
    parser.add_argument("-d", "--devices", nargs="+", metavar="DEVICE", help="e.g. mlx5_0 (default: all)")
    parser.add_argument("-w", "--interval", type=positive_float, default=0.1)
    parser.add_argument("-W", "--duration", type=positive_float)
    parser.add_argument("--capacity-gb-s", type=positive_float,
                        help="per-direction capacity for utilization (default: the port's rate / 8)")
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--include-ethernet", action="store_true", help="also RoCE / Ethernet ports")
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys/class/infiniband"))
    parser.add_argument("--stop-file", type=Path, help="stop when this file exists")
    parser.add_argument("--output-mode", choices=("stdout", "file", "both"), default="file")
    shortcuts = parser.add_mutually_exclusive_group()
    shortcuts.add_argument("--stdout", action="store_true")
    shortcuts.add_argument("--both", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--filename", default=f"{stamp}_ib_port_bandwidth.csv")
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument("--overwrite", action="store_true")
    writes.add_argument("--append", action="store_true")
    parser.add_argument("--list", action="store_true", help="list the ports found and exit")
    return parser


def open_outputs(args: argparse.Namespace) -> tuple[list[tuple[TextIO, bool]], Path | None]:
    if args.stdout:
        args.output_mode = "stdout"
    elif args.both:
        args.output_mode = "both"
    if args.output_mode == "stdout":
        if args.overwrite or args.append:
            raise ValueError("--overwrite and --append require file or both mode")
        return [(sys.stdout, False)], None
    path = args.output
    if path is None:
        name = args.filename if args.filename.endswith(".csv") else args.filename + ".csv"
        if not name or "/" in name:
            raise ValueError("--filename must be non-empty and must not contain '/'")
        path = args.output_dir / name
    if path.exists() and not args.overwrite and not args.append:
        raise ValueError(f"output already exists: {path} (use --overwrite or --append)")
    had_content = args.append and path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if args.append else "w", newline="", encoding="utf-8")
    outputs: list[tuple[TextIO, bool]] = [(handle, had_content)]
    if args.output_mode == "both":
        outputs.insert(0, (sys.stdout, False))
    return outputs, path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    ports = discover_ports(args.sysfs_root, args.devices, args.include_inactive, args.include_ethernet)
    if args.list:
        for port in ports:
            capacity = "?" if port.capacity_gb_s is None else f"{port.capacity_gb_s:g} GB/s per direction"
            print(f"{port.device} port {port.port}: {port.link_layer}, {port.state}, {capacity}")
        return 0 if ports else 1
    if not ports:
        print(f"Error: no InfiniBand ports with counters under {args.sysfs_root}"
              " (try --include-inactive or --include-ethernet)", file=sys.stderr)
        return 1
    try:
        outputs, output_path = open_outputs(args)
    except ValueError as error:
        parser.error(str(error))
    print("Ports: " + ", ".join(f"{p.device}/{p.port}" for p in ports), file=sys.stderr)
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
    started = time.monotonic()
    previous: Sample | None = None
    try:
        while not stopped:
            iteration = time.monotonic()
            current = read_counters(ports)
            if previous is not None:
                for row in bandwidth_rows(previous, current, ports, args.capacity_gb_s):
                    for writer in writers:
                        writer.writerow(row)
                for handle, _ in outputs:
                    handle.flush()
            previous = current
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            if args.stop_file is not None and args.stop_file.exists():
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
