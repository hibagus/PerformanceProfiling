# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Shared process and argument helpers for the Intel PCM monitor wrappers."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
        help=(
            "measurement CSV path for file/both modes "
            f"(default: UTC timestamp + _{output_name}.csv)"
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("stdout", "file", "both"),
        default="file",
        help="where CSV rows are written (default: file)",
    )
    output_shortcut = parser.add_mutually_exclusive_group()
    output_shortcut.add_argument(
        "--stdout",
        dest="output_mode",
        action="store_const",
        const="stdout",
        help="shortcut for --output-mode stdout",
    )
    output_shortcut.add_argument(
        "--both",
        dest="output_mode",
        action="store_const",
        const="both",
        help="shortcut for --output-mode both",
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
    rdt = parser.add_mutually_exclusive_group()
    rdt.add_argument(
        "--no-rdt",
        dest="no_rdt",
        action="store_true",
        help="disable PCM RDT metrics (sets PCM_NO_RDT=1)",
    )
    rdt.add_argument(
        "--rdt",
        dest="no_rdt",
        action="store_false",
        help="enable PCM RDT metrics (sets PCM_NO_RDT=0)",
    )
    parser.set_defaults(no_rdt=None)
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


def prepare_paths(
    outputs: Sequence[Path], diagnostics: Path, overwrite: bool
) -> None:
    paths = [*outputs, diagnostics]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("measurement and diagnostic paths must be different")
    for path in paths:
        if path.exists() and not overwrite:
            raise ValueError(f"output already exists: {path} (use --overwrite)")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


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
    if args.output_mode == "file":
        command.append(f"-csv={args.output.resolve()}")
    else:
        command.extend(("-silent", "-csv"))
    return command


def run_monitor(
    args: argparse.Namespace,
    native_arguments: Sequence[str] = (),
    *,
    use_sudo: bool = False,
    suppressed_diagnostics: Sequence[str] = (),
    csv_header_prefixes: Sequence[str] = (),
) -> int:
    """Execute one PCM utility and preserve its native, version-specific CSV."""

    try:
        binary = resolve_binary(args.binary)
        diagnostics = diagnostic_path(args)
        command = build_command(binary, args, native_arguments)
        writes_file = args.output_mode in {"file", "both"}
        environment_values = [
            f"PCM_NO_MSR={'1' if args.no_msr else '0'}",
            f"PCM_KEEP_NMI_WATCHDOG={'0' if args.disable_nmi_watchdog else '1'}",
        ]
        if args.no_rdt is not None:
            environment_values.append(f"PCM_NO_RDT={'1' if args.no_rdt else '0'}")
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
            if args.output_mode == "both":
                print(f"CSV tee: stdout and {args.output.resolve()}")
            print(f"diagnostics: {diagnostics.resolve()}")
            return 0
        prepare_paths(
            ([args.output] if writes_file else []), diagnostics, args.overwrite
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment["PCM_NO_MSR"] = "1" if args.no_msr else "0"
    environment["PCM_KEEP_NMI_WATCHDOG"] = (
        "0" if args.disable_nmi_watchdog else "1"
    )
    if args.no_rdt is not None:
        environment["PCM_NO_RDT"] = "1" if args.no_rdt else "0"

    elevated = use_sudo and os.geteuid() != 0
    if elevated:
        sudo = shutil.which("sudo")
        env = shutil.which("env") or "/usr/bin/env"
        if sudo is None:
            print("error: sudo is required but was not found on PATH", file=sys.stderr)
            return 2
        print(
            "pcm-iio needs privileged PCI topology access; invoking sudo.",
            file=sys.stderr,
        )
        # Pre-create a new artifact as the calling user. pcm-iio truncates the
        # existing inode, so sudo does not make ordinary outputs root-owned.
        if writes_file and not args.output.exists():
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
    destination = {
        "file": str(args.output.resolve()),
        "stdout": "stdout",
        "both": f"stdout and {args.output.resolve()}",
    }[args.output_mode]
    print(
        f"collecting with {binary.name} to {destination} ({limit_description})",
        file=sys.stderr,
    )
    if binary.name == "pcm-iio":
        print(
            "pcm-iio initialization and the first CSV rows may take several seconds.",
            file=sys.stderr,
        )

    # pcm-iio loads its opCode-<family>-<model>.txt beside the installed binary.
    # Using that directory is harmless for the other PCM utilities as well.
    with ExitStack() as stack:
        error_stream = stack.enter_context(
            diagnostics.open("w", encoding="utf-8")
        )
        output_stream = (
            stack.enter_context(args.output.open("w", encoding="utf-8"))
            if args.output_mode == "both"
            else None
        )
        capture_diagnostics = elevated or bool(suppressed_diagnostics)
        capture_stdout = args.output_mode in {"stdout", "both"}
        saw_csv_output = threading.Event()
        process = subprocess.Popen(
            command,
            cwd=binary.parent,
            env=environment,
            stdout=(
                subprocess.PIPE
                if capture_stdout
                else subprocess.DEVNULL
            ),
            stderr=subprocess.PIPE if capture_diagnostics else error_stream,
            text=capture_diagnostics or capture_stdout,
            start_new_session=not elevated,
            # Keep sudo and its privileged child in a group that belongs only
            # to this collector while preserving access to the controlling
            # terminal for an initial password prompt.
            preexec_fn=os.setpgrp if elevated else None,
        )

        stdout_thread: threading.Thread | None = None
        if capture_stdout:
            assert process.stdout is not None

            def relay_stdout() -> None:
                """Tee the native CSV stream to the terminal and output file."""

                stdout_open = True
                csv_started = not csv_header_prefixes
                for line in process.stdout:
                    if not csv_started:
                        csv_started = any(
                            line.startswith(prefix) for prefix in csv_header_prefixes
                        )
                        if not csv_started:
                            sys.stderr.write(line)
                            sys.stderr.flush()
                            continue
                    saw_csv_output.set()
                    if output_stream is not None:
                        output_stream.write(line)
                        output_stream.flush()
                    if stdout_open:
                        try:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                        except BrokenPipeError:
                            stdout_open = False

            stdout_thread = threading.Thread(target=relay_stdout, daemon=True)
            stdout_thread.start()

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
                        f"{suppressed_count} known non-fatal topology "
                        "warning line(s).\n"
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
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
            except PermissionError:
                process.send_signal(signum)
            if elevated:
                # The caller cannot directly signal the root-owned pcm-iio
                # member of the group.  Ask cached, non-interactive sudo to
                # signal the complete group so no collector is orphaned.
                signal_name = signal.Signals(signum).name.removeprefix("SIG")
                kill = shutil.which("kill") or "/bin/kill"
                try:
                    subprocess.Popen(
                        [
                            sudo,
                            "-n",
                            "--",
                            kill,
                            f"-{signal_name}",
                            "--",
                            f"-{process.pid}",
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except OSError:
                    pass

        previous_handlers = {
            signum: signal.signal(signum, forward_signal)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            return_code = process.wait()
        finally:
            if stdout_thread is not None:
                stdout_thread.join()
            if relay_thread is not None:
                relay_thread.join()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    if received_signal is not None:
        if writes_file and args.output.is_file() and args.output.stat().st_size > 0:
            print(f"stopped; partial CSV saved to {args.output}", file=sys.stderr)
        else:
            print("monitor stopped", file=sys.stderr)
        return 128 + received_signal
    if return_code != 0:
        print(
            f"error: {binary.name} exited with status {return_code}; "
            f"see {diagnostics}",
            file=sys.stderr,
        )
        return return_code
    if capture_stdout and not saw_csv_output.is_set():
        print(
            f"error: {binary.name} completed without writing recognizable CSV data; "
            f"see {diagnostics}",
            file=sys.stderr,
        )
        return 1
    if writes_file and (
        not args.output.is_file() or args.output.stat().st_size == 0
    ):
        print(
            f"error: {binary.name} completed without writing CSV data; "
            f"see {diagnostics}",
            file=sys.stderr,
        )
        return 1
    if writes_file:
        print(f"wrote {args.output}", file=sys.stderr)
    print(f"diagnostics: {diagnostics}", file=sys.stderr)
    return 0
