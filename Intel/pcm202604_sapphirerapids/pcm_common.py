# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Shared process and argument helpers for the Intel PCM monitor wrappers."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def positive_float(value: str) -> float:
    """Parse a finite floating-point value greater than zero."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    """Parse an integer greater than zero."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_binary(tool: str, environment_variable: str) -> str:
    """Prefer an explicit environment setting, then PATH."""

    configured = os.environ.get(environment_variable)
    if configured:
        return configured
    return shutil.which(tool) or tool


def resolve_binary(value: str) -> Path:
    """Resolve and validate either an executable path or a PATH command."""

    if "/" in value:
        path = Path(value).expanduser().resolve()
    else:
        found = shutil.which(value)
        if found is None:
            raise ValueError(f"executable not found on PATH: {value}")
        path = Path(found).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"not an executable file: {path}")
    return path


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    tool: str,
    environment_variable: str,
    output_name: str,
) -> None:
    parser.add_argument(
        "-w",
        "--interval",
        type=positive_float,
        default=1.0,
        help="sampling interval in seconds (default: 1)",
    )
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument(
        "-W",
        "--duration",
        type=positive_float,
        help="approximate collection duration in seconds",
    )
    limit.add_argument(
        "-i",
        "--iterations",
        type=positive_int,
        help="number of samples to collect (default: until interrupted)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(f"{utc_run_id()}_{output_name}.csv"),
        help=f"measurement CSV path (default: UTC timestamp + _{output_name}.csv)",
    )
    parser.add_argument(
        "--stderr-log",
        type=Path,
        help="PCM diagnostic log path (default: derived from --output)",
    )
    parser.add_argument(
        "--binary",
        default=default_binary(tool, environment_variable),
        help=(
            f"{tool} executable (default: ${environment_variable}, then PATH)"
        ),
    )
    access = parser.add_mutually_exclusive_group()
    access.add_argument(
        "--no-msr",
        dest="no_msr",
        action="store_true",
        default=True,
        help="request PCM's Linux perf_event mode (default)",
    )
    access.add_argument(
        "--direct-msr",
        dest="no_msr",
        action="store_false",
        help="allow direct MSR/PCI access; normally requires elevated privileges",
    )
    parser.add_argument(
        "--disable-nmi-watchdog",
        action="store_true",
        help="let PCM disable the NMI watchdog while it runs",
    )
    parser.add_argument(
        "--pcm-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra native PCM argument; repeat and use --pcm-arg=-option",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def sample_iterations(args: argparse.Namespace) -> int | None:
    if args.iterations is not None:
        return args.iterations
    if args.duration is not None:
        return max(1, math.ceil(args.duration / args.interval))
    return None


def diagnostic_path(args: argparse.Namespace) -> Path:
    if args.stderr_log is not None:
        return args.stderr_log
    return args.output.with_name(f"{args.output.stem}_stderr.log")


def prepare_paths(output: Path, diagnostics: Path, overwrite: bool) -> None:
    if output.resolve() == diagnostics.resolve():
        raise ValueError("--output and --stderr-log must be different paths")
    for path in (output, diagnostics):
        if path.exists() and not overwrite:
            raise ValueError(f"output already exists: {path} (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)


def build_command(
    binary: Path,
    args: argparse.Namespace,
    native_arguments: Sequence[str],
) -> list[str]:
    command = [str(binary), str(args.interval), *native_arguments]
    iterations = sample_iterations(args)
    if iterations is not None:
        command.append(f"-i={iterations}")
    command.extend(args.pcm_arg)
    command.append(f"-csv={args.output.resolve()}")
    return command


def run_monitor(
    args: argparse.Namespace,
    native_arguments: Sequence[str] = (),
    *,
    use_sudo: bool = False,
    suppressed_diagnostics: Sequence[str] = (),
) -> int:
    """Execute one PCM utility and preserve its native, version-specific CSV."""

    try:
        binary = resolve_binary(args.binary)
        diagnostics = diagnostic_path(args)
        command = build_command(binary, args, native_arguments)
        environment_values = [
            f"PCM_NO_MSR={'1' if args.no_msr else '0'}",
            f"PCM_KEEP_NMI_WATCHDOG={'0' if args.disable_nmi_watchdog else '1'}",
        ]
        if args.dry_run:
            printable_command = command
            if use_sudo and os.geteuid() != 0:
                printable_command = [
                    "sudo",
                    "--prompt",
                    "[sudo] password for %u:\n",
                    "--",
                    shutil.which("env") or "/usr/bin/env",
                    *environment_values,
                    *command,
                ]
                print(shlex.join(printable_command))
            else:
                print(" ".join(environment_values + [shlex.join(command)]))
            print(f"diagnostics: {diagnostics.resolve()}")
            return 0
        prepare_paths(args.output, diagnostics, args.overwrite)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment["PCM_NO_MSR"] = "1" if args.no_msr else "0"
    environment["PCM_KEEP_NMI_WATCHDOG"] = (
        "0" if args.disable_nmi_watchdog else "1"
    )

    elevated = use_sudo and os.geteuid() != 0
    if elevated:
        sudo = shutil.which("sudo")
        env = shutil.which("env") or "/usr/bin/env"
        if sudo is None:
            print("error: sudo is required but was not found on PATH", file=sys.stderr)
            return 2
        print("pcm-iio needs privileged PCI topology access; invoking sudo.", file=sys.stderr)
        # Pre-create a new artifact as the calling user. pcm-iio truncates the
        # existing inode, so sudo does not make ordinary outputs root-owned.
        if not args.output.exists():
            args.output.touch()
        command = [
            sudo,
            "--prompt",
            "[sudo] password for %u:\n",
            "--",
            env,
            *environment_values,
            *command,
        ]

    iterations = sample_iterations(args)
    limit_description = (
        f"{iterations} sample(s)"
        if iterations is not None
        else "continuously until Ctrl+C"
    )
    print(
        f"collecting with {binary.name} to {args.output.resolve()} "
        f"({limit_description})",
        file=sys.stderr,
    )
    if binary.name == "pcm-iio":
        print(
            "pcm-iio initialization and the first CSV rows may take several seconds.",
            file=sys.stderr,
        )

    # pcm-iio loads its opCode-<family>-<model>.txt beside the installed binary.
    # Using that directory is harmless for the other PCM utilities as well.
    with diagnostics.open("w", encoding="utf-8") as error_stream:
        capture_diagnostics = elevated or bool(suppressed_diagnostics)
        process = subprocess.Popen(
            command,
            cwd=binary.parent,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_diagnostics else error_stream,
            text=capture_diagnostics,
            start_new_session=not elevated,
        )

        relay_thread: threading.Thread | None = None
        if capture_diagnostics:
            assert process.stderr is not None

            def relay_stderr() -> None:
                """Retain useful diagnostics without known repetitive noise."""

                suppressed_count = 0
                for line in process.stderr:
                    if any(pattern in line for pattern in suppressed_diagnostics):
                        suppressed_count += 1
                        continue
                    error_stream.write(line)
                    error_stream.flush()
                    if elevated:
                        sys.stderr.write(line)
                        sys.stderr.flush()
                if suppressed_count:
                    summary = (
                        "[pcm wrapper] suppressed "
                        f"{suppressed_count} known non-fatal topology warning line(s).\n"
                    )
                    error_stream.write(summary)
                    error_stream.flush()
                    if elevated:
                        sys.stderr.write(summary)
                        sys.stderr.flush()

            relay_thread = threading.Thread(target=relay_stderr, daemon=True)
            relay_thread.start()

        received_signal: int | None = None

        def forward_signal(signum: int, _frame: object) -> None:
            nonlocal received_signal
            received_signal = signum
            try:
                if elevated:
                    process.send_signal(signum)
                else:
                    os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

        previous_handlers = {
            signum: signal.signal(signum, forward_signal)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            return_code = process.wait()
        finally:
            if relay_thread is not None:
                relay_thread.join()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    if received_signal is not None:
        if args.output.is_file() and args.output.stat().st_size > 0:
            print(f"stopped; partial CSV saved to {args.output}", file=sys.stderr)
        else:
            print("stopped before PCM wrote its first CSV sample", file=sys.stderr)
        return 128 + received_signal
    if return_code != 0:
        print(
            f"error: {binary.name} exited with status {return_code}; "
            f"see {diagnostics}",
            file=sys.stderr,
        )
        return return_code
    if not args.output.is_file() or args.output.stat().st_size == 0:
        print(
            f"error: {binary.name} completed without writing CSV data; "
            f"see {diagnostics}",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {args.output}", file=sys.stderr)
    print(f"diagnostics: {diagnostics}", file=sys.stderr)
    return 0
