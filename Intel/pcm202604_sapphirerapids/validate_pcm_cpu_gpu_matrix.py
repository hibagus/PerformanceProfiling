# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Run the complete CPU-NUMA/GPU transfer matrix under PCM IIO and UPI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pcm_common import positive_float, positive_int, resolve_binary, utc_run_id
from validate_pcm_transferbench import (
    DEFAULT_ROCM_LIB,
    DEFAULT_TRANSFERBENCH,
    SCRIPT_DIR,
    authenticate_iio_sudo,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test H2D and D2H between CPU NUMA nodes 0/1 and GPUs 0-7, "
            "collecting isolated pcm-iio PCIe and pcm UPI captures."
        )
    )
    parser.add_argument("--transferbench", type=Path, default=DEFAULT_TRANSFERBENCH)
    parser.add_argument("--rocm-lib", type=Path, default=DEFAULT_ROCM_LIB)
    parser.add_argument("--cpu-nodes", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--size", default="256M")
    parser.add_argument("--duration", type=positive_int, default=10)
    parser.add_argument("--interval", type=positive_float, default=1.0)
    parser.add_argument("--lead-seconds", type=positive_float, default=2.0)
    parser.add_argument("--tail-seconds", type=positive_float, default=2.0)
    parser.add_argument("--startup-timeout", type=positive_float, default=30.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.cpu_nodes or not args.gpus:
        parser.error("--cpu-nodes and --gpus cannot be empty")
    if any(value < 0 for value in (*args.cpu_nodes, *args.gpus)):
        parser.error("CPU NUMA node and GPU indices cannot be negative")
    if len(set(args.cpu_nodes)) != len(args.cpu_nodes):
        parser.error("--cpu-nodes contains a duplicate")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus contains a duplicate")
    if not re.fullmatch(r"[1-9][0-9]*[KMG]?", args.size, re.IGNORECASE):
        parser.error("--size must be a positive byte count with optional K/M/G suffix")


def child_command(
    args: argparse.Namespace, cpu: int, gpu: int, output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "validate_pcm_transferbench.py"),
        "--transferbench", str(args.transferbench.expanduser()),
        "--gpu", str(gpu),
        "--cpu-node", str(cpu),
        "--size", args.size,
        "--duration", str(args.duration),
        "--interval", str(args.interval),
        "--lead-seconds", str(args.lead_seconds),
        "--tail-seconds", str(args.tail_seconds),
        "--startup-timeout", str(args.startup_timeout),
        "--cases", "iio_h2d", "iio_d2h", "upi_h2d", "upi_d2h",
        "--output-dir", str(output_dir),
    ]
    if args.rocm_lib is not None:
        command[4:4] = ["--rocm-lib", str(args.rocm_lib.expanduser())]
    return command


def topology_probe(args: argparse.Namespace) -> str:
    try:
        binary = resolve_binary(str(args.transferbench))
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    args.transferbench = binary
    environment = os.environ.copy()
    previous = environment.get("LD_LIBRARY_PATH")
    if args.rocm_lib is not None:
        environment["LD_LIBRARY_PATH"] = str(args.rocm_lib) + (
            f":{previous}" if previous else ""
        )
    result = subprocess.run(
        [str(binary)], env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )
    match = re.search(r"(\d+) GPU device\(s\)", result.stdout)
    detected = int(match.group(1)) if match else 0
    if detected <= max(args.gpus):
        raise RuntimeError(
            f"requested GPU {max(args.gpus)}, but TransferBench detected {detected} GPU(s)"
        )
    return result.stdout


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    run_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir
        else SCRIPT_DIR / "runs" / f"{utc_run_id()}_cpu_gpu_matrix"
    )
    pairs = [(cpu, gpu) for cpu in args.cpu_nodes for gpu in args.gpus]

    if args.dry_run:
        print(f"result directory: {run_dir}")
        print(f"{len(pairs)} CPU/GPU pairs, 4 isolated captures per pair")
        for cpu, gpu in pairs:
            print(shlex.join(child_command(args, cpu, gpu, run_dir / f"cpu{cpu}_gpu{gpu}")))
        return 0

    try:
        topology = topology_probe(args)
        if run_dir.exists():
            raise RuntimeError(f"result directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        (run_dir / "transferbench_topology.txt").write_text(topology, encoding="utf-8")
        authenticate_iio_sudo()
        manifest: dict[str, object] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": os.uname().nodename,
            "cpu_nodes": args.cpu_nodes,
            "gpus": args.gpus,
            "size": args.size,
            "duration_seconds": args.duration,
            "runs": [],
        }
        manifest_path = run_dir / "manifest.json"
        for index, (cpu, gpu) in enumerate(pairs, 1):
            pair_dir = run_dir / f"cpu{cpu}_gpu{gpu}"
            print(f"\n=== pair {index}/{len(pairs)}: CPU{cpu} <-> GPU{gpu} ===", flush=True)
            command = child_command(args, cpu, gpu, pair_dir)
            status = subprocess.run(command, check=False).returncode
            entry = {
                "cpu_node": cpu, "gpu": gpu, "directory": str(pair_dir),
                "status": status,
            }
            manifest["runs"].append(entry)  # type: ignore[union-attr]
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            if status != 0:
                raise RuntimeError(f"CPU{cpu}/GPU{gpu} validation exited with status {status}")
        print(f"\nMatrix complete: {run_dir}")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        if run_dir.exists():
            print(f"partial artifacts: {run_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
