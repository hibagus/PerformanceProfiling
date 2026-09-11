# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Collect CPU and IIO telemetry concurrently and merge samples by timestamp."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from pcm_common import default_binary, positive_float, resolve_binary, utc_run_id
from pcm_iio_monitor import needs_sudo


SCRIPT_DIR = Path(__file__).resolve().parent
IIO_BANDWIDTH_FIELDS = ("IB write", "IB read", "OB read", "OB write")


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
    parser = argparse.ArgumentParser(
        description=(
            "Run pcm CPU and IIO monitors concurrently, then merge every IIO "
            "port's bandwidth into the native two-row PCM CPU CSV schema."
        )
    )
    parser.add_argument("-w", "--interval", type=positive_float, default=1.0)
    parser.add_argument("-W", "--duration", type=positive_float)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--run-id",
        default=utc_run_id(),
        help="prefix shared by all output files (default: current UTC time)",
    )
    parser.add_argument("--cpu-output", type=Path, help="raw pcm CPU CSV path")
    parser.add_argument("--iio-output", type=Path, help="raw pcm-iio CSV path")
    parser.add_argument("-o", "--output", type=Path, help="consolidated CSV path")
    parser.add_argument(
        "--output-mode",
        choices=("stdout", "file", "both"),
        default="file",
        help="where the consolidated CSV is written (default: file)",
    )
    output_shortcut = parser.add_mutually_exclusive_group()
    output_shortcut.add_argument(
        "--stdout", dest="output_mode", action="store_const", const="stdout"
    )
    output_shortcut.add_argument(
        "--both", dest="output_mode", action="store_const", const="both"
    )
    parser.add_argument(
        "--match-tolerance",
        type=nonnegative_float,
        help="maximum CPU/IIO timestamp difference (default: interval / 2)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=30.0,
        help="seconds to wait for each monitor's first valid sample (default: 30)",
    )
    parser.add_argument(
        "--cpu-binary",
        default=default_binary("pcm", "PCM_BIN"),
        help="pcm executable (default: $PCM_BIN, then PATH)",
    )
    parser.add_argument(
        "--iio-binary",
        default=default_binary("pcm-iio", "PCM_IIO_BIN"),
        help="pcm-iio executable (default: $PCM_IIO_BIN, then PATH)",
    )
    cores = parser.add_mutually_exclusive_group()
    cores.add_argument(
        "--no-cores",
        dest="no_cores",
        action="store_true",
        help="omit per-core metrics (default; recommended with concurrent pcm-iio)",
    )
    cores.add_argument(
        "--with-cores",
        dest="no_cores",
        action="store_false",
        help="include per-core metrics (may be unstable with concurrent pcm-iio)",
    )
    parser.add_argument(
        "--cpu-rdt",
        dest="no_cpu_rdt",
        action="store_false",
        help=(
            "enable CPU RDT metrics; disabled by default because resctrl access "
            "can prevent pcm from producing samples"
        ),
    )
    parser.add_argument("--no-sockets", action="store_true")
    parser.add_argument("--no-system", action="store_true")
    parser.add_argument("--no-root-ports", action="store_true")
    parser.add_argument("--direct-msr", action="store_true")
    parser.add_argument("--disable-nmi-watchdog", action="store_true")
    parser.add_argument("--cpu-pcm-arg", action="append", default=[], metavar="ARG")
    parser.add_argument("--iio-pcm-arg", action="append", default=[], metavar="ARG")
    privilege = parser.add_mutually_exclusive_group()
    privilege.add_argument(
        "--sudo", dest="sudo_mode", action="store_const", const="always"
    )
    privilege.add_argument(
        "--no-sudo", dest="sudo_mode", action="store_const", const="never"
    )
    parser.set_defaults(sudo_mode="auto", no_cores=True, no_cpu_rdt=True)
    parser.add_argument("--show-topology-warnings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Return raw CPU, raw IIO, and consolidated output paths."""

    if not args.run_id or args.run_id in {".", ".."} or "/" in args.run_id:
        raise ValueError("--run-id must be non-empty and must not contain '/'")
    return (
        args.cpu_output
        or args.output_dir / f"{args.run_id}_pcm_cpu.csv",
        args.iio_output
        or args.output_dir / f"{args.run_id}_pcm_iio.csv",
        args.output
        or args.output_dir / f"{args.run_id}_pcm_cpu_iio_consolidated.csv",
    )


def child_commands(
    args: argparse.Namespace, cpu_output: Path, iio_output: Path
) -> tuple[list[str], list[str]]:
    """Build child commands with one interval and a coordinated overlap window."""

    timing = ["--interval", str(args.interval)]
    # Make the privileged collector self-terminating.  Signalling the outer
    # sudo process is not sufficient on every sudo configuration, and can
    # otherwise leave pcm-iio collecting after this wrapper has exited.
    iio_timing = [*timing]
    if args.duration is not None:
        iio_timing.extend(("--duration", str(args.duration)))
    common = ["--output-mode", "file"]
    if args.direct_msr:
        common.append("--direct-msr")
    if args.disable_nmi_watchdog:
        common.append("--disable-nmi-watchdog")
    if args.overwrite:
        common.append("--overwrite")

    cpu_command = [
        sys.executable,
        str(SCRIPT_DIR / "pcm_cpu_monitor.py"),
        *timing,
        *common,
        "--binary",
        args.cpu_binary,
        "--output",
        str(cpu_output),
        "--stderr-log",
        str(cpu_output.with_name(f"{cpu_output.stem}_stderr.log")),
    ]
    if args.no_cores:
        cpu_command.append("--no-cores")
    if args.no_cpu_rdt:
        cpu_command.append("--no-rdt")
    else:
        cpu_command.append("--rdt")
    if args.no_sockets:
        cpu_command.append("--no-sockets")
    if args.no_system:
        cpu_command.append("--no-system")
    for value in args.cpu_pcm_arg:
        cpu_command.append(f"--pcm-arg={value}")

    iio_command = [
        sys.executable,
        str(SCRIPT_DIR / "pcm_iio_monitor.py"),
        *iio_timing,
        *common,
        "--binary",
        args.iio_binary,
        "--output",
        str(iio_output),
        "--stderr-log",
        str(iio_output.with_name(f"{iio_output.stem}_stderr.log")),
        "--sudo" if needs_sudo(args.sudo_mode) else "--no-sudo",
    ]
    if args.no_root_ports:
        iio_command.append("--no-root-ports")
    if args.show_topology_warnings:
        iio_command.append("--show-topology-warnings")
    for value in args.iio_pcm_arg:
        iio_command.append(f"--pcm-arg={value}")
    return cpu_command, iio_command


def authenticate_sudo(args: argparse.Namespace) -> None:
    """Populate sudo's credential cache before starting synchronized children."""

    if not needs_sudo(args.sudo_mode):
        return
    sudo = shutil.which("sudo")
    if sudo is None:
        raise RuntimeError("sudo is required for pcm-iio but was not found on PATH")
    print("Authenticating sudo once for pcm-iio ...", file=sys.stderr)
    result = subprocess.run([sudo, "-v"], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"sudo authentication failed with status {result.returncode}"
        )


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass


def iio_has_sample(path: Path) -> bool:
    """Return true after pcm-iio has flushed at least one measurement row."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return any(
                len(row) >= 2 and row[0] != "Date" and row[0][:1].isdigit()
                for row in reader
            )
    except OSError:
        return False


def cpu_has_sample(path: Path) -> bool:
    """Return true after pcm has flushed both headers and one measurement row."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            categories = next(reader, [])
            metrics = next(reader, [])
            sample = next(reader, [])
            return bool(
                categories
                and metrics[:2] == ["Date", "Time"]
                and len(categories) == len(metrics) == len(sample)
                and sample[0][:1].isdigit()
            )
    except OSError:
        return False


def run_children(
    commands: tuple[list[str], list[str]],
    cpu_output: Path,
    iio_output: Path,
    startup_timeout: float,
    duration: float | None,
) -> tuple[list[int], int | None, bool]:
    """Initialize CPU then IIO and forward termination to both collectors."""

    processes: list[subprocess.Popen[bytes]] = []
    received_signal: int | None = None
    duration_elapsed = False

    def stop(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        stop_processes(processes)

    previous_handlers = {
        signum: signal.signal(signum, stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # Establish pcm's native schema before pcm-iio programs or discovers
        # any uncore resources. Starting them in the opposite order can make
        # pcm emit duplicated header groups on some Sapphire Rapids systems.
        cpu_process = subprocess.Popen(commands[0], start_new_session=True)
        processes.append(cpu_process)
        deadline = time.monotonic() + startup_timeout
        while not cpu_has_sample(cpu_output):
            status = cpu_process.poll()
            if status is not None:
                raise RuntimeError(
                    f"pcm exited before its first valid sample (status {status})"
                )
            if received_signal is not None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"pcm did not produce a valid sample within {startup_timeout:g}s"
                )
            time.sleep(0.1)

        if received_signal is None:
            # Keep the IIO wrapper attached to this controlling terminal. Its
            # nested sudo process may need the terminal even after `sudo -v`.
            iio_process = subprocess.Popen(commands[1])
            processes.append(iio_process)
            deadline = time.monotonic() + startup_timeout
            while not iio_has_sample(iio_output):
                if cpu_process.poll() is not None:
                    raise RuntimeError(
                        "pcm exited while pcm-iio was initializing "
                        f"(status {cpu_process.returncode})"
                    )
                status = iio_process.poll()
                if status is not None:
                    raise RuntimeError(
                        f"pcm-iio exited before its first sample (status {status})"
                    )
                if received_signal is not None:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "pcm-iio did not produce a sample within "
                        f"{startup_timeout:g}s"
                    )
                time.sleep(0.1)

        if received_signal is None and len(processes) == 2:
            # A duration-limited pcm-iio child stops itself after the requested
            # number of samples.  It is the overlap clock because it starts
            # only after the CPU collector has produced a valid sample.
            while cpu_process.poll() is None and iio_process.poll() is None:
                if received_signal is not None:
                    break
                time.sleep(0.1)

            if received_signal is None:
                if iio_process.poll() is not None and cpu_process.poll() is None:
                    duration_elapsed = (
                        duration is not None and iio_process.returncode == 0
                    )
                    stop_processes([cpu_process])
                elif cpu_process.poll() is not None and iio_process.poll() is None:
                    stop_processes([iio_process])
    except RuntimeError:
        stop_processes(processes)
        raise
    finally:
        return_codes = [process.wait() for process in processes]
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if len(return_codes) == 1:
        return_codes.append(
            128 + received_signal if received_signal is not None else 1
        )
    return return_codes, received_signal, duration_elapsed


def parse_timestamp(date: str, time_value: str) -> datetime:
    try:
        return datetime.fromisoformat(f"{date.strip()}T{time_value.strip()}")
    except ValueError as error:
        raise ValueError(f"invalid PCM timestamp: {date!r}, {time_value!r}") from error


def slug(value: str, fallback: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return result or fallback


def decimal_text(value: Decimal) -> str:
    """Render a summed counter without introducing binary floating-point noise."""

    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def read_iio_samples(
    path: Path,
) -> tuple[list[datetime], list[dict[str, str]], list[str]]:
    """Pivot long-form IIO rows into one wide mapping per timestamp."""

    grouped: OrderedDict[datetime, dict[str, str]] = OrderedDict()
    columns: list[str] = []
    seen_columns: set[str] = set()
    identity_columns: dict[tuple[str, str, str, str], dict[str, str]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("IIO CSV is empty") from error
        required = {"Date", "Time", "Socket", "Root Port", "Name", "Part"}
        if not required.issubset(header) or not set(IIO_BANDWIDTH_FIELDS).issubset(
            header
        ):
            missing = sorted(required.union(IIO_BANDWIDTH_FIELDS).difference(header))
            raise ValueError(f"IIO CSV is missing columns: {', '.join(missing)}")
        index = {
            name: header.index(name)
            for name in required.union(IIO_BANDWIDTH_FIELDS)
        }

        for row_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            row.extend([""] * (len(header) - len(row)))
            if (
                row[index["Date"]].strip() == "Date"
                and row[index["Time"]].strip() == "Time"
            ):
                continue
            try:
                timestamp = parse_timestamp(row[index["Date"]], row[index["Time"]])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"invalid IIO timestamp on row {row_number}: {error}"
                ) from error
            identity = tuple(
                row[index[name]].strip()
                for name in ("Socket", "Root Port", "Name", "Part")
            )
            if identity not in identity_columns:
                socket, root_port, name, part = identity
                prefix = "pcm_iio__" + "__".join(
                    (
                        slug(socket, "unknown_socket"),
                        slug(name, "unknown_device"),
                        slug(part, "unknown_part"),
                        slug(root_port, "no_root_port"),
                    )
                )
                mapping = {
                    metric: (
                        f"{prefix}__{slug(metric, 'metric')}_bytes_per_second"
                    )
                    for metric in IIO_BANDWIDTH_FIELDS
                }
                if any(column in seen_columns for column in mapping.values()):
                    raise ValueError(
                        f"IIO column-name collision for identity {identity!r}"
                    )
                identity_columns[identity] = mapping
                for column in mapping.values():
                    seen_columns.add(column)
                    columns.append(column)
            sample = grouped.setdefault(timestamp, {})
            for metric, column in identity_columns[identity].items():
                sample[column] = row[index[metric]].strip()

    if not grouped:
        raise ValueError("IIO CSV contains no samples")

    # PCM models each IIO stack as eight selectable PMON channel masks named
    # Part0 through Part7. Preserve every per-part value above, and add one
    # stack-level total for each bandwidth direction below. Root Port is not
    # part of the aggregate identity because different parts can map to
    # different root ports within the same IIO stack.
    stack_sources: OrderedDict[
        tuple[str, str], dict[str, list[str]]
    ] = OrderedDict()
    for (socket, _root_port, name, part), mapping in identity_columns.items():
        if re.fullmatch(r"Part[0-7]", part) is None:
            continue
        sources = stack_sources.setdefault(
            (socket, name), {metric: [] for metric in IIO_BANDWIDTH_FIELDS}
        )
        for metric, column in mapping.items():
            sources[metric].append(column)

    for (socket, name), sources in stack_sources.items():
        prefix = "pcm_iio__" + "__".join(
            (
                slug(socket, "unknown_socket"),
                slug(name, "unknown_device"),
                "Part0_to_Part7_total",
            )
        )
        aggregate_columns = {
            metric: f"{prefix}__{slug(metric, 'metric')}_bytes_per_second"
            for metric in IIO_BANDWIDTH_FIELDS
        }
        if any(column in seen_columns for column in aggregate_columns.values()):
            raise ValueError(
                f"IIO aggregate column-name collision for {(socket, name)!r}"
            )
        for column in aggregate_columns.values():
            seen_columns.add(column)
            columns.append(column)

        for sample in grouped.values():
            for metric, aggregate_column in aggregate_columns.items():
                total = Decimal(0)
                found = False
                for source_column in sources[metric]:
                    raw_value = sample.get(source_column, "").strip()
                    if not raw_value:
                        continue
                    try:
                        total += Decimal(raw_value)
                    except InvalidOperation:
                        continue
                    found = True
                if found:
                    sample[aggregate_column] = decimal_text(total)

    ordered = sorted(grouped.items())
    return (
        [timestamp for timestamp, _sample in ordered],
        [sample for _timestamp, sample in ordered],
        columns,
    )


def output_handles(
    path: Path, mode: str, overwrite: bool
) -> tuple[list[TextIO], list[TextIO]]:
    """Return CSV destinations and handles that the caller must close."""

    destinations: list[TextIO] = []
    opened: list[TextIO] = []
    if mode in {"file", "both"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w" if overwrite else "x", newline="", encoding="utf-8")
        destinations.append(handle)
        opened.append(handle)
    if mode in {"stdout", "both"}:
        destinations.append(sys.stdout)
    return destinations, opened


def merge_csv_files(
    cpu_path: Path,
    iio_path: Path,
    output_path: Path,
    tolerance: float,
    output_mode: str,
    overwrite: bool,
) -> tuple[int, int, int]:
    """Append the nearest wide IIO sample to each native PCM CPU row."""

    timestamps, samples, iio_columns = read_iio_samples(iio_path)
    metadata_columns = ("pcm_iio_sample_timestamp", "pcm_iio_time_delta_seconds")
    destinations, opened = output_handles(output_path, output_mode, overwrite)
    total = 0
    matched = 0
    try:
        writers = [csv.writer(handle, lineterminator="\n") for handle in destinations]
        with cpu_path.open(newline="", encoding="utf-8") as cpu_handle:
            reader = csv.reader(cpu_handle)
            try:
                categories = next(reader)
                metrics = next(reader)
            except StopIteration as error:
                raise ValueError(
                    "CPU CSV does not contain its two header rows"
                ) from error
            if metrics[:2] != ["Date", "Time"]:
                raise ValueError(
                    "CPU CSV metric header does not begin with Date,Time"
                )
            if len(categories) != len(metrics):
                raise ValueError(
                    "CPU CSV header width changed during PCM initialization "
                    f"(category columns={len(categories)}, metric columns={len(metrics)}). "
                    "This can occur when per-core/resctrl discovery races with another "
                    "PCM process; use the combined monitor's default no-core mode "
                    "(or pass --no-cores explicitly)."
                )
            extension_count = len(metadata_columns) + len(iio_columns)
            for writer in writers:
                writer.writerow([*categories, *(["PCM IIO"] * extension_count)])
                writer.writerow([*metrics, *metadata_columns, *iio_columns])

            for row_number, row in enumerate(reader, start=3):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) < 2:
                    raise ValueError(f"CPU CSV row {row_number} has no timestamp")
                if len(row) != len(metrics):
                    raise ValueError(
                        f"CPU CSV row {row_number} has {len(row)} columns; "
                        f"the header has {len(metrics)}. Native PCM changed its "
                        "schema while sampling; rerun in the default no-core mode."
                    )
                total += 1
                timestamp = parse_timestamp(row[0], row[1])
                position = bisect.bisect_left(timestamps, timestamp)
                candidates = range(
                    max(0, position - 1), min(len(timestamps), position + 1)
                )
                nearest = min(
                    candidates,
                    key=lambda index: abs(
                        (timestamps[index] - timestamp).total_seconds()
                    ),
                    default=None,
                )
                extension = [""] * extension_count
                if nearest is not None:
                    delta = (timestamps[nearest] - timestamp).total_seconds()
                    if abs(delta) <= tolerance:
                        matched += 1
                        extension[0] = timestamps[nearest].isoformat(sep=" ")
                        extension[1] = f"{delta:.6f}"
                        sample = samples[nearest]
                        extension[2:] = [
                            sample.get(column, "") for column in iio_columns
                        ]
                merged = [*row, *extension]
                for writer in writers:
                    writer.writerow(merged)
    finally:
        for handle in opened:
            handle.close()
    return total, matched, len(iio_columns)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        cpu_output, iio_output, merged_output = output_paths(args)
    except ValueError as error:
        parser.error(str(error))

    file_paths = [cpu_output, iio_output]
    if args.output_mode in {"file", "both"}:
        file_paths.append(merged_output)
    diagnostic_paths = [
        cpu_output.with_name(f"{cpu_output.stem}_stderr.log"),
        iio_output.with_name(f"{iio_output.stem}_stderr.log"),
    ]
    all_paths = [*file_paths, *diagnostic_paths]
    if len({path.resolve() for path in all_paths}) != len(all_paths):
        parser.error("raw, consolidated, and diagnostic paths must be different")
    collisions = [path for path in all_paths if path.exists()]
    if collisions and not args.overwrite:
        parser.error(
            "output already exists: "
            + ", ".join(str(path) for path in collisions)
            + " (use --overwrite)"
        )
    if collisions and args.overwrite and not args.dry_run:
        invalid = [
            path for path in collisions if not path.is_file() and not path.is_symlink()
        ]
        if invalid:
            parser.error(
                "cannot overwrite non-file path: "
                + ", ".join(str(path) for path in invalid)
            )
        for path in collisions:
            path.unlink()

    if not args.dry_run:
        try:
            args.cpu_binary = str(resolve_binary(args.cpu_binary))
            args.iio_binary = str(resolve_binary(args.iio_binary))
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    commands = child_commands(args, cpu_output, iio_output)
    print(f"Run ID: {args.run_id}", file=sys.stderr)
    print(f"CPU command: {shlex.join(commands[0])}", file=sys.stderr)
    print(f"IIO command: {shlex.join(commands[1])}", file=sys.stderr)
    destination = (
        "stdout"
        if args.output_mode == "stdout"
        else str(merged_output)
        if args.output_mode == "file"
        else f"stdout and {merged_output}"
    )
    print(f"Consolidated CSV: {destination}", file=sys.stderr)
    if args.dry_run:
        return 0

    try:
        authenticate_sudo(args)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        return_codes, received_signal, duration_elapsed = run_children(
            commands,
            cpu_output,
            iio_output,
            args.startup_timeout,
            args.duration,
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    cpu_status, iio_status = return_codes
    signal_stops = {
        -signal.SIGINT,
        128 + signal.SIGINT,
    }
    expected_timed_stop = duration_elapsed and all(
        status in signal_stops.union({0}) for status in return_codes
    )
    expected_iio_stop = cpu_status == 0 and iio_status in signal_stops
    if (
        (
            not expected_timed_stop
            and (cpu_status != 0 or (iio_status != 0 and not expected_iio_stop))
        )
        and received_signal is None
    ):
        print(
            f"error: monitor statuses were CPU={cpu_status}, "
            f"IIO={iio_status}; no consolidated CSV was created",
            file=sys.stderr,
        )
        return cpu_status if cpu_status != 0 else iio_status
    if received_signal is not None and not (
        cpu_output.is_file() and iio_output.is_file()
    ):
        return 128 + received_signal

    tolerance = (
        args.match_tolerance
        if args.match_tolerance is not None
        else args.interval / 2.0
    )
    try:
        total, matched, added = merge_csv_files(
            cpu_output,
            iio_output,
            merged_output,
            tolerance,
            args.output_mode,
            args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"error: could not merge monitor output: {error}", file=sys.stderr)
        return 1
    print(
        f"merged {matched}/{total} CPU samples within {tolerance:g}s; "
        f"added {added} IIO bandwidth columns",
        file=sys.stderr,
    )
    if received_signal is not None:
        return 128 + received_signal
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
