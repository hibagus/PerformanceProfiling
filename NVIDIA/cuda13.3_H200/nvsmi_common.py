# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Shared helpers for the NVIDIA H200 monitoring scripts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TextIO


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


def parse_gpu_ids(values: list[str] | None) -> list[int]:
    """Accept space-separated, comma-separated, or mixed GPU indexes."""

    if not values:
        return []
    result: list[int] = []
    for value in values:
        for item in value.split(","):
            if not item.isdigit():
                raise ValueError(
                    f"invalid GPU index {item!r}; indexes must be non-negative integers"
                )
            gpu = int(item)
            if gpu not in result:
                result.append(gpu)
    return result


def resolve_output_path(
    explicit_path: Path | None, output_dir: Path, filename: str
) -> Path:
    if explicit_path is not None:
        return explicit_path
    if not filename or "/" in filename:
        raise ValueError("--filename must be non-empty and must not contain '/'")
    if not filename.endswith(".csv"):
        filename += ".csv"
    return output_dir / filename


def open_text_file(path: Path, *, overwrite: bool, append: bool) -> tuple[TextIO, bool]:
    if path.exists() and not overwrite and not append:
        raise ValueError(f"output already exists: {path} (use --overwrite or --append)")
    had_content = append and path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a" if append else "w", newline="", encoding="utf-8"), had_content


def utc_timestamp(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
