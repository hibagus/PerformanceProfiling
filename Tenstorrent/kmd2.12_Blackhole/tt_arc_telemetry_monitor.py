#!/usr/bin/env python3
# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""The full firmware telemetry table of Tenstorrent Blackhole cards (about 80 tags), sampled over time.

sysfs shows nine of the firmware's telemetry tags. The rest (board input power, which clock arbiter limits AICLK,
GDDR temperatures and IO power, Ethernet link and heartbeat masks, kernel-throttler counters, ...) sit in the
same table in ARC memory. This monitor reads that table the way tt-kmd does for sysfs, through the driver's
NOC_READ ioctl on a power-neutral file descriptor (see tt_common.KmdNoc): no firmware message, no power request,
only the ARC tile is read. Needs tt-kmd >= 2.12.0.

The firmware refreshes the table every ``telem_update_ms`` (100 ms on 19.x); ``fresh`` is 1 when the table's
timer heartbeat advanced since the previous sample. Power and current tags are whole watts and amperes.
"""

from __future__ import annotations

import argparse
import errno
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tt_common import (TAGS, ArcTelemetry, CsvSink, KmdNoc, KmdNocError, add_common_args,  # noqa: E402
                       card_alive, discover_cards, open_outputs, run_loop, utc_timestamp)

# Default columns: the dynamic tags plus the limits; --all-tags adds every published tag (versions, IDs, ...).
DEFAULT_TAGS = ["aiclk_mhz", "axiclk_mhz", "arcclk_mhz", "l2cpuclk0_mhz", "vcore_mv", "power_w", "current_a",
                "input_power_w", "asic_temp_c", "max_gddr_temp_c", "gddr_west_io_power_w", "gddr_east_io_power_w",
                "aiclk_arb_min", "aiclk_arb_max", "enabled_min_arb", "enabled_max_arb", "aiclk_ppm_info",
                "host_aiclk_limit_mhz", "nop_start_count", "nop_on_ms", "kernel_throttler", "fw_active_config_0",
                "eth_live_status", "gddr_status", "gddr_uncorr_errs", "therm_trip_count", "timer_heartbeat",
                "tdp_limit_w", "tdc_limit_a", "thm_limit_throttle_c", "board_power_limit_w", "aiclk_limit_max_mhz"]
DERIVED = ["eth_heartbeat_mask", "eth_link_mask", "eth_links_up", "limiter",
           *[f"gddr{i}_temp_c" for i in range(8)], "gddr_corr_errs"]
BASE = ["timestamp_epoch", "timestamp_utc", "interval_seconds", "card", "bdf", "fresh", "status"]


def derive(raw: dict[int, int], decoded: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    eth = raw.get(21)
    if eth is not None:
        out.update(eth_heartbeat_mask=hex(eth & 0xFFFF), eth_link_mask=hex(eth >> 16),
                   eth_links_up=bin(eth >> 16).count("1"))
    for pair in range(4):  # each tag packs two GDDR modules, two dies each: [31:24] [23:16] = odd, [15:8] [7:0]
        v = raw.get(42 + pair)
        if v is not None:
            out[f"gddr{2 * pair}_temp_c"] = max(v & 0xFF, (v >> 8) & 0xFF)
            out[f"gddr{2 * pair + 1}_temp_c"] = max((v >> 16) & 0xFF, (v >> 24) & 0xFF)
    corr = [raw.get(46 + i) for i in range(4)]
    if all(c is not None for c in corr):
        out["gddr_corr_errs"] = sum((c >> s) & 0xFF for c in corr for s in (0, 8, 16, 24))
    # What sets AICLK now: the PPM's own statement ("max_arb:tdc" = a maximum arbiter holds the clock down).
    out["limiter"] = decoded.get("aiclk_ppm_info", "")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser, 0.1, "tt_arc_telemetry")
    parser.add_argument("--all-tags", action="store_true", help="one column per published tag")
    parser.add_argument("--dump", action="store_true", help="print every decoded tag of each card once and exit")
    args = parser.parse_args()
    try:
        cards = discover_cards(args.sysfs_root, args.cards)
    except ValueError as error:
        parser.error(str(error))
    if args.list or not cards:
        for c in cards:
            print(f"{c.label}  {c.bdf}  {c.card_type}  board {c.serial}  fw {c.fw_bundle}")
        return 0 if cards else 1

    readers: dict[int, tuple[KmdNoc, ArcTelemetry] | None] = {}

    def reader(card) -> tuple[KmdNoc, ArcTelemetry] | None:
        if readers.get(card.index) is None:
            if not card_alive(card):
                return None
            noc = KmdNoc(card.index)
            try:
                readers[card.index] = (noc, ArcTelemetry(noc))
            except Exception:
                noc.close()
                raise
        return readers[card.index]

    if args.dump:
        for card in cards:
            r = reader(card)
            if r is None:
                print(f"{card.label}: not alive in sysfs; skipped")
                continue
            raw = r[1].read_raw()
            print(f"== {card.label} {card.bdf}: telemetry version {r[1].version:#x}, {len(raw)} tags")
            for name, value in {**ArcTelemetry.decode(raw), **derive(raw, ArcTelemetry.decode(raw))}.items():
                print(f"  {name:28s} {value}")
        return 0

    tag_columns = [name for _, (name, _) in sorted(TAGS.items())] if args.all_tags else DEFAULT_TAGS
    try:
        outputs, path = open_outputs(args)
    except ValueError as error:
        parser.error(str(error))
    print("Cards: " + ", ".join(f"{c.label} ({c.bdf})" for c in cards), file=sys.stderr)
    if path:
        print(f"Output CSV: {path}", file=sys.stderr)
    sink = CsvSink(outputs, BASE + tag_columns + DERIVED)
    last: dict[int, tuple[float, int | None]] = {}

    def tick() -> None:
        rows = []
        for card in cards:
            t0, e0 = time.monotonic(), time.time()
            row: dict[str, object] = {"card": card.label, "bdf": card.bdf}
            try:
                r = reader(card)
                if r is None:
                    row["status"] = "card_dead"
                    raw = None
                else:
                    raw = r[1].read_raw()
                    row["status"] = "ok"
            except (OSError, KmdNocError) as error:
                raw = None
                row["status"] = f"error:{errno.errorcode.get(getattr(error, 'errno', 0) or 0, type(error).__name__)}"
                old = readers.pop(card.index, None)  # reopen next tick (e.g. -ENODEV after a card reset)
                if old is not None:
                    old[0].close()
            mono, epoch = (t0 + time.monotonic()) / 2, (e0 + time.time()) / 2
            row.update(timestamp_epoch=f"{epoch:.6f}", timestamp_utc=utc_timestamp(epoch))
            if raw is not None:
                decoded = ArcTelemetry.decode(raw)
                row.update(decoded)
                row.update(derive(raw, decoded))
                beat = raw.get(32)
                prev = last.get(card.index)
                if prev is not None:
                    row["interval_seconds"] = f"{mono - prev[0]:.6f}"
                    row["fresh"] = int(beat != prev[1])
                last[card.index] = (mono, beat)
            rows.append(row)
        sink.write(rows)

    try:
        run_loop(args, tick)
    finally:
        sink.close()
        for r in readers.values():
            if r is not None:
                r[0].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
