# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Validate Intel PCM PCIe/IIO/UPI counters with controlled TransferBench traffic."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pcm_common import positive_float, positive_int, utc_run_id


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRANSFERBENCH = Path(
    "/home/bagus/Dissagregated_PD/TransferBench_Source/TransferBench/TransferBench"
)
DEFAULT_ROCM_LIB = Path("/opt/rocm/core-10.0/lib")
ALL_CASES = (
    "pcie_h2d",
    "pcie_d2h",
    "iio_h2d",
    "iio_d2h",
    "upi_h2d",
    "upi_d2h",
    "upi_local",
    "upi_cross",
)
DEFAULT_CASES = (
    "pcie_h2d",
    "pcie_d2h",
    "iio_h2d",
    "iio_d2h",
    "upi_local",
    "upi_cross",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled TransferBench copies while collecting PCM. Cases run "
            "sequentially to avoid uncore PMU contention."
        )
    )
    parser.add_argument(
        "--transferbench",
        type=Path,
        default=DEFAULT_TRANSFERBENCH,
        help=f"TransferBench executable (default: {DEFAULT_TRANSFERBENCH})",
    )
    parser.add_argument(
        "--rocm-lib",
        type=Path,
        default=DEFAULT_ROCM_LIB,
        help=f"directory prepended to LD_LIBRARY_PATH (default: {DEFAULT_ROCM_LIB})",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument(
        "--cpu-node", type=int, default=0, help="source/local CPU NUMA node (default: 0)"
    )
    parser.add_argument(
        "--remote-cpu-node",
        type=int,
        default=1,
        help="destination CPU NUMA node for UPI traffic (default: 1)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=positive_int,
        default=40,
        help="CPU executors for local and cross-socket copies (default: 40)",
    )
    parser.add_argument("--size", default="256M", help="bytes per transfer (default: 256M)")
    parser.add_argument(
        "--duration",
        type=positive_int,
        default=10,
        help="TransferBench seconds per case (default: 10)",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=1.0,
        help="PCM sampling interval in seconds (default: 1)",
    )
    parser.add_argument(
        "--lead-seconds",
        type=positive_float,
        default=2.0,
        help="idle PCM collection before traffic (default: 2)",
    )
    parser.add_argument(
        "--tail-seconds",
        type=positive_float,
        default=2.0,
        help="idle PCM collection after traffic (default: 2)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=30.0,
        help="seconds to wait for each monitor's first CSV data (default: 30)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=ALL_CASES,
        default=list(DEFAULT_CASES),
        help="cases to run in the specified order (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory (default: runs/<UTC>_transferbench_pcm_validation)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("gpu", "cpu_node", "remote_cpu_node"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.cpu_node == args.remote_cpu_node and "upi_cross" in args.cases:
        parser.error("--cpu-node and --remote-cpu-node must differ for upi_cross")
    if not re.fullmatch(r"[1-9][0-9]*[KMG]?", args.size, re.IGNORECASE):
        parser.error("--size must be a positive byte count with optional K, M, or G suffix")


def transferbench_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    old_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(args.rocm_lib) + (
        f":{old_library_path}" if old_library_path else ""
    )
    environment["NUM_ITERATIONS"] = f"-{args.duration}"
    environment["NUM_WARMUPS"] = "1"
    environment["HIDE_ENV"] = "1"
    return environment


def case_spec(args: argparse.Namespace, case: str) -> tuple[str, list[str], str]:
    cpu = args.cpu_node
    gpu = args.gpu
    if case.endswith("h2d"):
        expression = f"1 1 (C{cpu}->D{gpu}->G{gpu})"
    elif case.endswith("d2h"):
        expression = f"1 1 (G{gpu}->D{gpu}->C{cpu})"
    elif case == "upi_local":
        expression = f"1 {args.cpu_threads} (C{cpu}->C{cpu}->C{cpu})"
    else:
        expression = (
            f"1 {args.cpu_threads} "
            f"(C{cpu}->C{cpu}->C{args.remote_cpu_node})"
        )

    if case.startswith("pcie_"):
        script = SCRIPT_DIR / "pcm_pcie_monitor.py"
        csv_name = "pcm_pcie.csv"
        extras: list[str] = []
    elif case.startswith("iio_"):
        script = SCRIPT_DIR / "pcm_iio_monitor.py"
        csv_name = "pcm_iio.csv"
        extras = []
    else:
        script = SCRIPT_DIR / "pcm_cpu_monitor.py"
        csv_name = "pcm_cpu.csv"
        extras = ["--no-cores"]
    return expression, [sys.executable, str(script), *extras], csv_name


def print_command(environment: dict[str, str], command: list[str]) -> None:
    selected = ("LD_LIBRARY_PATH", "NUM_ITERATIONS", "NUM_WARMUPS", "HIDE_ENV")
    prefix = " ".join(
        f"{name}={shlex.quote(environment[name])}" for name in selected if name in environment
    )
    print(f"  {prefix} {shlex.join(command)}" if prefix else f"  {shlex.join(command)}")


def preflight(args: argparse.Namespace, environment: dict[str, str]) -> None:
    binary = args.transferbench.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"TransferBench is not executable: {binary}")
    if not args.rocm_lib.expanduser().is_dir():
        raise RuntimeError(f"ROCm library directory does not exist: {args.rocm_lib}")

    result = subprocess.run(
        [str(binary)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if "0 GPU device(s)" in result.stdout and any(
        case.endswith(("h2d", "d2h")) for case in args.cases
    ):
        raise RuntimeError(
            "TransferBench detected zero GPUs. Run this command in a shell with "
            "access to /dev/kfd and the render devices."
        )
    if "Detected Topology:" not in result.stdout:
        raise RuntimeError("TransferBench topology probe failed:\n" + result.stdout[-2000:])


def authenticate_iio_sudo() -> None:
    if os.geteuid() == 0:
        return
    sudo = shutil.which("sudo")
    if sudo is None:
        raise RuntimeError("sudo is required for pcm-iio but was not found")
    print("Authenticating sudo once for pcm-iio cases ...", flush=True)
    result = subprocess.run([sudo, "-v"], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"sudo authentication failed with status {result.returncode}")


def wait_for_monitor(
    process: subprocess.Popen[bytes],
    csv_path: Path,
    diagnostics_path: Path,
    timeout: float,
) -> None:
    """Wait until PCM reaches its collection loop.

    Native PCM uses a buffered C++ file stream for CSV output. In particular,
    pcm-pcie may collect many samples while the visible file size remains zero,
    then flush everything only when stopped. Its diagnostic stream is
    unbuffered, so use its post-initialization messages as the readiness signal.
    """

    ready_markers = (
        "Update every ",
        "Successfully programmed on-core PMU",
        "Detected Intel(R)",
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if csv_path.is_file() and csv_path.stat().st_size > 0:
            return
        status = process.poll()
        if status is not None:
            raise RuntimeError(f"monitor exited before producing CSV data (status {status})")
        if diagnostics_path.is_file():
            diagnostics = diagnostics_path.read_text(encoding="utf-8", errors="replace")
            if any(marker in diagnostics for marker in ready_markers):
                return
        time.sleep(0.2)
    raise RuntimeError(f"monitor did not finish initialization within {timeout:g} seconds")


def stop_monitor(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def parse_bandwidth(output: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(
            r"Transfer\s+\d+\s*(?:\||│)\s*([0-9.]+)\s+GB/s", output
        )
    ]


def run_case(
    args: argparse.Namespace,
    case: str,
    run_dir: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    expression, monitor_base, csv_name = case_spec(args, case)
    case_dir = run_dir / case
    case_dir.mkdir()
    csv_path = case_dir / csv_name
    pcm_stderr = case_dir / f"{Path(csv_name).stem}_stderr.log"
    wrapper_log = case_dir / "monitor_wrapper.log"
    benchmark_log = case_dir / "transferbench.log"
    # Let PCM stop itself. A synthetic SIGINT from this unprivileged runner
    # cannot reliably pass through sudo to a root-owned pcm-iio process.
    monitor_duration = (
        args.lead_seconds
        + args.duration
        + args.tail_seconds
        + 2 * args.interval
    )
    monitor_command = [
        *monitor_base,
        "--interval",
        str(args.interval),
        "--duration",
        str(monitor_duration),
        "--output",
        str(csv_path),
        "--stderr-log",
        str(pcm_stderr),
    ]
    benchmark_command = [
        str(args.transferbench.expanduser().resolve()),
        "cmdline",
        args.size,
        expression,
    ]

    print(f"[{case}] {expression}", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    monitor: subprocess.Popen[bytes] | None = None
    monitor_status: int | None = None
    try:
        with wrapper_log.open("wb") as monitor_output:
            monitor = subprocess.Popen(
                monitor_command,
                stdout=monitor_output,
                stderr=subprocess.STDOUT,
            )
            wait_for_monitor(monitor, csv_path, pcm_stderr, args.startup_timeout)
            time.sleep(args.lead_seconds)
            benchmark = subprocess.run(
                benchmark_command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            benchmark_log.write_text(benchmark.stdout, encoding="utf-8")
            time.sleep(args.tail_seconds)
            try:
                monitor_status = monitor.wait(timeout=max(15.0, args.startup_timeout))
            except subprocess.TimeoutExpired as error:
                stop_monitor(monitor)
                raise RuntimeError(
                    f"monitor did not finish its {monitor_duration:g}-second capture"
                ) from error
    finally:
        if monitor is not None and monitor.poll() is None:
            stop_monitor(monitor)

    failed_text = "[ERROR]" in benchmark.stdout
    rates = parse_bandwidth(benchmark.stdout)
    result: dict[str, object] = {
        "case": case,
        "expression": expression,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "monitor_command": monitor_command,
        "transferbench_command": benchmark_command,
        "monitor_status": monitor_status,
        "transferbench_status": benchmark.returncode,
        "transferbench_reported_error": failed_text,
        "transferbench_gbps": rates,
        "measurement_csv": str(csv_path),
        "pcm_stderr_log": str(pcm_stderr),
        "monitor_wrapper_log": str(wrapper_log),
        "transferbench_log": str(benchmark_log),
    }
    if benchmark.returncode != 0 or failed_text:
        raise RuntimeError(
            f"TransferBench failed in {case}; inspect {benchmark_log}"
        )
    if monitor_status not in (0, 130):
        raise RuntimeError(
            f"monitor failed in {case} with status {monitor_status}; inspect {wrapper_log}"
        )
    shown_rate = f", peak {max(rates):.3f} GB/s" if rates else ""
    print(f"[{case}] complete{shown_rate}; artifacts: {case_dir}", flush=True)
    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    environment = transferbench_environment(args)
    run_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else SCRIPT_DIR / "runs" / f"{utc_run_id()}_transferbench_pcm_validation"
    )

    if args.dry_run:
        print(f"result directory: {run_dir}")
        iio_authenticated = False
        for case in args.cases:
            if case.startswith("iio_") and not iio_authenticated:
                print("  sudo -v")
                iio_authenticated = True
            expression, monitor_base, csv_name = case_spec(args, case)
            case_dir = run_dir / case
            monitor = [
                *monitor_base,
                "--interval",
                str(args.interval),
                "--duration",
                str(
                    args.lead_seconds
                    + args.duration
                    + args.tail_seconds
                    + 2 * args.interval
                ),
                "--output",
                str(case_dir / csv_name),
                "--stderr-log",
                str(case_dir / f"{Path(csv_name).stem}_stderr.log"),
            ]
            print(f"[{case}] {expression}")
            print_command({}, monitor)
            print_command(
                environment,
                [str(args.transferbench.expanduser().resolve()), "cmdline", args.size, expression],
            )
        return 0

    try:
        preflight(args, environment)
        if run_dir.exists():
            raise RuntimeError(f"result directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        manifest: dict[str, object] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": os.uname().nodename,
            "arguments": vars(args) | {"transferbench": str(args.transferbench), "rocm_lib": str(args.rocm_lib), "output_dir": str(run_dir)},
            "results": [],
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        iio_authenticated = False
        for case in args.cases:
            if case.startswith("iio_") and not iio_authenticated:
                authenticate_iio_sudo()
                iio_authenticated = True
            result = run_case(args, case, run_dir, environment)
            manifest["results"].append(result)  # type: ignore[union-attr]
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Validation complete: {run_dir}")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        if run_dir.exists():
            print(f"partial artifacts: {run_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
