# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Run the InfiniBand, NVLink and GPU/PCIe monitors on one node until told to stop.

Start one copy per node inside a Slurm job, for example::

    srun --ntasks-per-node=1 --overlap python3 node_link_monitor.py \\
        --output-dir "$OUT/telemetry" --stop-file "$OUT/telemetry/STOP" &

and stop all of them with ``touch "$OUT/telemetry/STOP"`` (or scancel / SIGTERM). Each node writes
``<host>_ib.csv``, ``<host>_nvlink.csv``, ``<host>_dmon.csv`` and ``<host>_node_info.txt``: one file
per node and source, never one per rank.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NVIDIA_TOOLKIT = REPO_ROOT / "NVIDIA" / "cuda13.3_H200"
DEFAULT_IB_TOOLKIT = REPO_ROOT / "Mellanox" / "sysfs_mlx5"


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-node link telemetry for a Slurm job.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, help="stop when this file exists (checked every 0.2 s)")
    parser.add_argument("-W", "--duration", type=positive_float, help="stop after this many seconds")
    parser.add_argument("--host", default=socket.gethostname().split(".")[0])
    parser.add_argument("--ib-interval", type=positive_float, default=0.1)
    parser.add_argument("--nvlink-interval", type=positive_float, default=0.5)
    parser.add_argument("--dmon-interval", type=int, default=1, help="whole seconds (nvidia-smi dmon)")
    parser.add_argument("--nvlink-capacity-gb-s", type=positive_float, default=50.0,
                        help="per link and direction (Hopper and Blackwell: 50)")
    parser.add_argument("--no-ib", action="store_true")
    parser.add_argument("--no-nvlink", action="store_true")
    parser.add_argument("--no-dmon", action="store_true")
    parser.add_argument("--no-info", action="store_true")
    parser.add_argument("--nvidia-toolkit", type=Path, default=DEFAULT_NVIDIA_TOOLKIT)
    parser.add_argument("--ib-toolkit", type=Path, default=DEFAULT_IB_TOOLKIT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print what would run and exit")
    return parser


def run_quiet(command: list[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, f"{type(error).__name__}: {error}"
    return result.returncode, result.stdout + result.stderr


def has_nvlink_counters() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    code, text = run_quiet(["nvidia-smi", "nvlink", "--getthroughput", "d"])
    return code == 0 and ("KiB" in text)


def has_ib_ports(ib_toolkit: Path) -> bool:
    code, _ = run_quiet([sys.executable, str(ib_toolkit / "ib_port_bw_monitor.py"), "--list"])
    return code == 0


def write_node_info(path: Path, ib_toolkit: Path) -> None:
    """One text file per node: what the links are, for reading the numbers later."""

    sections = [
        ("hostname", ["hostname", "-f"]),
        ("nvidia-smi -L", ["nvidia-smi", "-L"]),
        ("nvidia-smi topo -m", ["nvidia-smi", "topo", "-m"]),
        ("nvidia-smi nvlink --status", ["nvidia-smi", "nvlink", "--status"]),
        ("nvidia-smi -q (fabric, PCIe, C2C sections)", ["nvidia-smi", "-q"]),
        ("InfiniBand ports (sysfs)", [sys.executable, str(ib_toolkit / "ib_port_bw_monitor.py"), "--list",
                                      "--include-inactive", "--include-ethernet"]),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for title, command in sections:
            if shutil.which(command[0]) is None and command[0] != sys.executable:
                handle.write(f"## {title}\n(not available)\n\n")
                continue
            code, text = run_quiet(command)
            if command[-1] == "-q":
                keep, lines = False, []
                for line in text.splitlines():
                    if line and not line.startswith(" "):
                        keep = True  # GPU header lines
                    elif line.startswith("    ") and not line.startswith("        "):
                        name = line.strip().lower()
                        keep = any(k in name for k in ("fabric", "pci", "c2c", "product", "bus id", "numa"))
                    if keep:
                        lines.append(line)
                text = "\n".join(lines)
            handle.write(f"## {title} (exit {code})\n{text.rstrip()}\n\n")


def monitor_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    out, host = args.output_dir, args.host
    common: list[str] = ["--overwrite"] if args.overwrite else []
    commands: list[tuple[str, list[str]]] = []
    if not args.no_ib and (args.dry_run or has_ib_ports(args.ib_toolkit)):
        commands.append(("ib", [sys.executable, str(args.ib_toolkit / "ib_port_bw_monitor.py"),
                                "-w", f"{args.ib_interval:g}", "-o", str(out / f"{host}_ib.csv"), *common]))
    if not args.no_nvlink and (args.dry_run or has_nvlink_counters()):
        commands.append(("nvlink", [sys.executable, str(args.nvidia_toolkit / "nvsmi_nvlink_bw_monitor.py"),
                                    "-w", f"{args.nvlink_interval:g}",
                                    "--link-capacity-gb-s", f"{args.nvlink_capacity_gb_s:g}",
                                    "-o", str(out / f"{host}_nvlink.csv"), *common]))
    if not args.no_dmon and (args.dry_run or shutil.which("nvidia-smi") is not None):
        commands.append(("dmon", [sys.executable, str(args.nvidia_toolkit / "nvsmi_gpu_monitor.py"),
                                  "-w", str(args.dmon_interval), "-o", str(out / f"{host}_dmon.csv"), *common]))
    return commands


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = monitor_commands(args)
    import shlex
    for name, command in commands:
        print(f"[{args.host}] {name}: {shlex.join(command)}", file=sys.stderr)
    if not commands:
        print(f"[{args.host}] Error: no InfiniBand ports, NVLink counters or nvidia-smi found", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    if not args.no_info:
        write_node_info(args.output_dir / f"{args.host}_node_info.txt", args.ib_toolkit)
    processes = [(name, subprocess.Popen(command, start_new_session=True)) for name, command in commands]
    received: int | None = None

    def stop(signum: int, _frame: object) -> None:
        nonlocal received
        received = signum

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    failed: list[str] = []
    try:
        while received is None:
            if args.stop_file is not None and args.stop_file.exists():
                break
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            for name, process in processes:
                if process.poll() not in (None, 0) and name not in failed:
                    failed.append(name)
                    print(f"[{args.host}] Warning: {name} monitor exited with {process.returncode}", file=sys.stderr)
            if len(failed) == len(processes):
                break
            time.sleep(0.2)
    finally:
        # SIGTERM to the monitor only, so it finishes its in-flight query and closes its CSV; the whole
        # process group only if it is still running after the grace period.
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 15
        for _, process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    print(f"[{args.host}] stopped after {time.monotonic() - started:.1f} s"
          + (f"; failed: {', '.join(failed)}" if failed else ""), file=sys.stderr)
    return 1 if failed and len(failed) == len(processes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
