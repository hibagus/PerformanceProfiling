# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
# This Python script contains common functions and classes used by the xGMI bandwidth monitoring script (amdsmi_xgmi_bw_monitor.py) 
# and the AMD-SMI telemetry collection script (amdsmi_gpu_monitor.py).
#
# This script is tested on ROCm 7.2.0 with AMD MI300X GPUs, and it may not work properly on other ROCm versions or GPU models.
# +------------------------------------------------------------------------------+
# | AMD-SMI            26.2.1+fc0010cf6a                                         |
# | ROCm Version:      7.2.0                                                     |
# | Platform:          Linux Baremetal                                           |
# |-------------------------------------+----------------------------------------| 

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TextIO


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


def nonnegative_int(value: str) -> int:
    """Parse an integer greater than or equal to zero."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def parse_gpu_ids(values: list[str] | None) -> list[int]:
    """Accept GPU indexes in space-separated, comma-separated, or mixed form."""

    if not values:
        return []

    result: list[int] = []
    for value in values:
        for item in value.split(","):
            if not item.isdigit():
                raise ValueError(
                    f"invalid GPU index {item!r}; indexes must be non-negative integers"
                )
            gpu_id = int(item)
            if gpu_id not in result:
                result.append(gpu_id)
    return result


def resolve_output_path(
    explicit_path: Path | None,
    output_dir: Path,
    filename: str,
    suffix: str,
) -> Path:
    """Resolve either a complete path or a directory plus simple filename."""

    if explicit_path is not None:
        return explicit_path
    if not filename or "/" in filename:
        raise ValueError("--filename must be non-empty and must not contain '/'")
    if not filename.endswith(suffix):
        filename += suffix
    return output_dir / filename


def open_text_file(path: Path, *, overwrite: bool, append: bool) -> tuple[TextIO, bool]:
    """Safely open a text artifact and report whether it already had content.

    The boolean return value is useful when deciding whether an appended CSV
    needs a header. Parent directories are created only after collision checks.
    """

    if path.exists() and not overwrite and not append:
        raise ValueError(f"output already exists: {path} (use --overwrite or --append)")
    had_content = append and path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if append else "w", newline="", encoding="utf-8")
    return handle, had_content
