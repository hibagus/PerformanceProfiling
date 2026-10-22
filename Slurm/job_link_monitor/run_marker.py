# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Record when each run of a job starts and ends, so link telemetry can be cut to it.

    run_marker.py MARKERS.csv start LABEL [--meta key=value ...]
    run_marker.py MARKERS.csv end LABEL
    run_marker.py MARKERS.csv wrap LABEL [--meta key=value ...] -- COMMAND [ARGS ...]

``wrap`` writes the start marker, runs the command, writes the end marker (with its exit status) and
returns the command's exit status. Each marker is one line appended with a single write, so markers
from concurrent runs (packed jobs) do not interleave.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("timestamp_epoch", "timestamp_utc", "event", "label", "host", "meta")


def utc_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def parse_meta(items: list[str] | None) -> dict[str, str]:
    meta: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--meta expects key=value, got {item!r}")
        meta[key] = value
    return meta


def append_marker(path: Path, event: str, label: str, meta: dict[str, str], epoch: float | None = None) -> None:
    epoch = time.time() if epoch is None else epoch
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([f"{epoch:.6f}", utc_timestamp(epoch), event, label, socket.gethostname().split(".")[0],
                     json.dumps(meta, sort_keys=True) if meta else ""])
    line = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, (",".join(FIELDS) + "\n").encode("utf-8"))
        os.write(fd, line)
    finally:
        os.close(fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append run start/end markers to a CSV.")
    parser.add_argument("markers", type=Path)
    parser.add_argument("event", choices=("start", "end", "wrap"))
    parser.add_argument("label")
    parser.add_argument("--meta", nargs="+", metavar="KEY=VALUE")
    parser.epilog = "wrap: put the command after a lone --, e.g. ... wrap run1 -- mpiexec -n 8 ./app input.json"
    return parser


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    if "," in args.label or "\n" in args.label:
        parser.error("LABEL must not contain commas or newlines")
    try:
        meta = parse_meta(args.meta)
    except ValueError as error:
        parser.error(str(error))
    if args.event != "wrap":
        if command:
            parser.error("a command is only accepted with wrap")
        append_marker(args.markers, args.event, args.label, meta)
        return 0
    if not command:
        parser.error("wrap needs -- COMMAND")
    append_marker(args.markers, "start", args.label, meta)
    try:
        status = subprocess.call(command)
    except OSError as error:
        print(f"Error: could not run {command[0]!r}: {error}", file=sys.stderr)
        status = 127
    append_marker(args.markers, "end", args.label, {**meta, "exit_status": str(status)})
    return status


if __name__ == "__main__":
    raise SystemExit(main())
