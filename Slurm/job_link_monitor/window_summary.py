# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Cut per-node link telemetry to each marked run and summarize it per link.

    window_summary.py --markers MARKERS.csv --data-dir TELEMETRY_DIR -o summary.csv

Reads every ``<host>_ib.csv`` (Mellanox/sysfs_mlx5), ``<host>_nvlink.csv`` and ``<host>_dmon.csv``
(NVIDIA toolkit) in the directory. A sample stamped t with interval dt covers (t - dt, t]; its share of
a run window is the overlap of that span with [start, end]. Per window, host, link and direction the
summary gives the bytes moved, the mean rate over the window (bytes / window length), the peak
sample rate, utilization against the link capacity, and the coverage (sampled share of the window).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

OUT_FIELDS = (
    "label", "start_epoch", "end_epoch", "window_seconds", "host", "source", "entity", "direction",
    "gbytes", "mean_gb_s", "peak_gb_s", "capacity_gb_s", "mean_utilization_pct", "peak_utilization_pct",
    "coverage_pct", "samples",
)
FILE_RE = re.compile(r"^(?P<host>.+)_(?P<source>ib|nvlink|dmon)\.csv$")


@dataclass(frozen=True)
class Window:
    label: str
    start: float
    end: float
    meta: str


@dataclass(frozen=True)
class Point:
    t: float          # end of the sampled span
    dt: float         # span length
    rate: float       # GB/s (decimal)
    capacity: float | None


def read_windows(path: Path) -> list[Window]:
    """Pair each end with the latest unmatched start of the same label."""

    open_starts: dict[str, list[tuple[float, str]]] = defaultdict(list)
    windows: list[Window] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t, label = float(row["timestamp_epoch"]), row["label"]
            if row["event"] == "start":
                open_starts[label].append((t, row.get("meta", "")))
            elif row["event"] == "end" and open_starts[label]:
                start, meta = open_starts[label].pop()
                windows.append(Window(label, start, t, row.get("meta", "") or meta))
    unmatched = sum(len(v) for v in open_starts.values())
    if unmatched:
        print(f"Warning: {unmatched} start marker(s) without an end", file=sys.stderr)
    return sorted(windows, key=lambda w: w.start)


def fnum(text: str | None) -> float | None:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


def read_ib(path: Path) -> dict[tuple[str, str], list[Point]]:
    series: dict[tuple[str, str], list[Point]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("counter_status") != "ok":
                continue
            t, dt = fnum(row["timestamp_epoch"]), fnum(row["interval_seconds"])
            if t is None or dt is None:
                continue
            entity, cap = f"{row['device']}/{row['port']}", fnum(row.get("per_direction_capacity_gb_s"))
            for direction in ("rx", "tx"):
                rate = fnum(row.get(f"{direction}_gb_s"))
                if rate is not None:
                    series[(entity, direction)].append(Point(t, dt, rate, cap))
    return series


def read_nvlink(path: Path) -> dict[tuple[str, str], list[Point]]:
    """Sum the physical links of each GPU per sample."""

    grouped: dict[tuple[int, float], dict[str, float]] = defaultdict(lambda: {"rx": 0.0, "tx": 0.0, "cap": 0.0, "dt": 0.0})
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("counter_status") != "ok":
                continue
            t, dt = fnum(row["timestamp_epoch"]), fnum(row["interval_seconds"])
            rx, tx, cap = fnum(row.get("rx_gb_s")), fnum(row.get("tx_gb_s")), fnum(row.get("per_direction_capacity_gb_s"))
            if None in (t, dt, rx, tx):
                continue
            g = grouped[(int(row["gpu"]), t)]
            g["rx"] += rx
            g["tx"] += tx
            g["cap"] += cap or 0.0
            g["dt"] = dt
    series: dict[tuple[str, str], list[Point]] = defaultdict(list)
    for (gpu, t), g in grouped.items():
        for direction in ("rx", "tx"):
            series[(f"gpu{gpu}", direction)].append(Point(t, g["dt"], g[direction], g["cap"] or None))
    return series


def read_dmon(path: Path, pcie_capacity: float | None) -> dict[tuple[str, str], list[Point]]:
    """nvidia-smi dmon PCIe columns (MB/s); the span is the gap to the GPU's previous row."""

    rows: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "pcie_rx_mb_s" not in reader.fieldnames:
            return {}
        for row in reader:
            t, rx, tx = fnum(row["timestamp_epoch"]), fnum(row.get("pcie_rx_mb_s")), fnum(row.get("pcie_tx_mb_s"))
            if None in (t, rx, tx):
                continue
            rows[int(row["gpu"])].append((t, rx / 1000.0, tx / 1000.0))
    series: dict[tuple[str, str], list[Point]] = defaultdict(list)
    for gpu, samples in rows.items():
        samples.sort()
        gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:]) if b[0] > a[0]]
        default_dt = sorted(gaps)[len(gaps) // 2] if gaps else 1.0
        previous = None
        for t, rx, tx in samples:
            dt = t - previous if previous is not None and t > previous else default_dt
            previous = t
            series[(f"gpu{gpu}", "rx")].append(Point(t, dt, rx, pcie_capacity))
            series[(f"gpu{gpu}", "tx")].append(Point(t, dt, tx, pcie_capacity))
    return series


def summarize(window: Window, points: list[Point]) -> dict[str, object] | None:
    length = window.end - window.start
    if length <= 0:
        return None
    gbytes = covered = 0.0
    peak, used, cap = 0.0, 0, None
    for p in points:
        overlap = min(p.t, window.end) - max(p.t - p.dt, window.start)
        if overlap <= 0:
            continue
        gbytes += p.rate * overlap
        covered += overlap
        peak = max(peak, p.rate)
        cap = p.capacity if p.capacity else cap
        used += 1
    if not used:
        return None
    mean = gbytes / length
    return {
        "gbytes": f"{gbytes:.6f}", "mean_gb_s": f"{mean:.6f}", "peak_gb_s": f"{peak:.6f}",
        "capacity_gb_s": "" if not cap else f"{cap:.3f}",
        "mean_utilization_pct": "" if not cap else f"{100 * mean / cap:.3f}",
        "peak_utilization_pct": "" if not cap else f"{100 * peak / cap:.3f}",
        "coverage_pct": f"{100 * min(covered, length) / length:.1f}", "samples": used,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize link telemetry per marked run window.")
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, help="CSV (default: standard output)")
    parser.add_argument("--pcie-capacity-gb-s", type=float,
                        help="per-direction PCIe capacity for utilization (e.g. 64 for Gen5 x16; default: none)")
    parser.add_argument("--labels", nargs="+", help="only these run labels")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    windows = read_windows(args.markers)
    if args.labels:
        windows = [w for w in windows if w.label in set(args.labels)]
    if not windows:
        print("Error: no complete start/end windows in the markers file", file=sys.stderr)
        return 1
    data: list[tuple[str, str, dict[tuple[str, str], list[Point]]]] = []
    for path in sorted(args.data_dir.glob("*.csv")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        host, source = match.group("host"), match.group("source")
        reader = {"ib": read_ib, "nvlink": read_nvlink}.get(source)
        series = reader(path) if reader else read_dmon(path, args.pcie_capacity_gb_s)
        data.append((host, {"dmon": "pcie"}.get(source, source), series))
    if not data:
        print(f"Error: no <host>_ib.csv / _nvlink.csv / _dmon.csv files in {args.data_dir}", file=sys.stderr)
        return 1
    handle = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for window in windows:
            for host, source, series in data:
                for (entity, direction), points in sorted(series.items()):
                    summary = summarize(window, points)
                    if summary is None:
                        continue
                    writer.writerow({"label": window.label, "start_epoch": f"{window.start:.6f}",
                                     "end_epoch": f"{window.end:.6f}",
                                     "window_seconds": f"{window.end - window.start:.3f}", "host": host,
                                     "source": source, "entity": entity, "direction": direction, **summary})
    finally:
        if handle is not sys.stdout:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
