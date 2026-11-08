#!/usr/bin/env python3
# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Per-card power, clocks, temperature and PCIe traffic of Tenstorrent Blackhole cards, from sysfs only.

Reads the files tt-kmd publishes under /sys/class/tenstorrent/tenstorrent!N (hwmon, tt_aiclk, ...,
pcie_perf_counters/). It never opens /dev/tenstorrent, so it cannot change a card's power state, and it can run
next to any workload. One CSV row per card and sample.

PCIe traffic comes from the PCIe tile's NIU counters (tt-kmd 2.12 ``pcie_perf_counters``): 32-bit cumulative
counts of NoC data flits per NoC, wrapping at 2^32. A flit carries up to 64 bytes on Blackhole; measured on p150b
(validate_pcie_counters.py), card-initiated DMA fills them (64 B per count) and host MMIO through an uncached TLB
window puts 32 B in each (one count per 32-byte access), so ``--mst-flit-bytes`` (default 32) is an assumption
about the host's access size. At 64 B per count the counters wrap every 256 GiB (5 s at 50 GB/s): sample at
least every ~2 s (default 0.1 s). From the NoC's side of the PCIe tile:
  mst_*   requests the host started (MMIO through a TLB window): writes into the card, reads from the card;
  slv_*   requests a core on the card started towards host memory: DMA reads of host memory (e.g. tt-metal's
          prefetcher pulling commands and data), DMA writes into host memory (completion queue, readback).
  h2d = mst writes + slv reads, d2h = mst reads + slv writes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tt_common import (DEAD, CsvSink, SysfsCard, add_common_args, discover_cards, open_outputs,  # noqa: E402
                       read_int, run_loop, utc_timestamp)

SLV_FLIT_BYTES = 64     # card-initiated DMA: full 64-byte NoC flits (measured)
MST_FLIT_BYTES = 32     # host MMIO, uncached 32-byte accesses (measured with tt-kmd wc_mappings=0)
WRAP = 1 << 32
PCIE_COUNTERS = ["mst_posted_wr_data_word_sent", "mst_nonposted_wr_data_word_sent", "mst_rd_data_word_received",
                 "slv_rd_data_word_sent", "slv_posted_wr_data_word_received", "slv_nonposted_wr_data_word_received"]
PCIE_RAW = [f"{name}{noc}" for name in PCIE_COUNTERS for noc in (0, 1)]
FAMILIES = {  # rate column -> counters summed over both NoCs
    "mst_wr_gb_s": ["mst_posted_wr_data_word_sent", "mst_nonposted_wr_data_word_sent"],
    "mst_rd_gb_s": ["mst_rd_data_word_received"],
    "slv_rd_gb_s": ["slv_rd_data_word_sent"],
    "slv_wr_gb_s": ["slv_posted_wr_data_word_received", "slv_nonposted_wr_data_word_received"],
}
CSV_FIELDS = ["timestamp_epoch", "timestamp_utc", "interval_seconds", "card", "bdf", "alive", "holders",
              "aiclk_mhz", "axiclk_mhz", "arcclk_mhz", "power_w", "power_limit_w", "current_a", "current_limit_a",
              "vcore_mv", "vcore_max_mv", "asic_temp_c", "temp_throttle_c", "fan_rpm", "heartbeat",
              "therm_trip_count", *FAMILIES, "h2d_gb_s", "d2h_gb_s", "pcie_status", *PCIE_RAW]
HWMON = {"power_w": ("power1_input", 1e-6), "power_limit_w": ("power1_max", 1e-6),
         "current_a": ("curr1_input", 1e-3), "current_limit_a": ("curr1_max", 1e-3),
         "vcore_mv": ("in0_input", 1), "vcore_max_mv": ("in0_max", 1),
         "asic_temp_c": ("temp1_input", 1e-3), "temp_throttle_c": ("temp1_max", 1e-3), "fan_rpm": ("fan1_input", 1)}
CLOCKS = {"aiclk_mhz": "tt_aiclk", "axiclk_mhz": "tt_axiclk", "arcclk_mhz": "tt_arcclk",
          "heartbeat": "tt_heartbeat", "therm_trip_count": "tt_therm_trip_count"}


def holders_of(indices: list[int]) -> dict[int, str]:
    """``pid:comm`` of every process holding /dev/tenstorrent/N open (cheap enough once per second)."""

    targets = {f"/dev/tenstorrent/{i}": i for i in indices}
    found: dict[int, set[str]] = {i: set() for i in indices}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if link in targets:
                try:
                    comm = Path(f"/proc/{pid}/comm").read_text().strip()
                except OSError:
                    comm = "?"
                found[targets[link]].add(f"{pid}:{comm}")
    return {i: " ".join(sorted(v)) for i, v in found.items()}


def sample_card(card: SysfsCard) -> dict[str, object]:
    row: dict[str, object] = {"card": card.label, "bdf": card.bdf}
    for column, name in CLOCKS.items():
        row[column] = read_int(card.path / name)
    row["alive"] = int(row["aiclk_mhz"] not in (None, DEAD))
    if card.hwmon is not None:
        for column, (name, scale) in HWMON.items():
            value = read_int(card.hwmon / name)
            row[column] = None if value is None else round(value * scale, 3)
    counters_dir = card.path / "pcie_perf_counters"
    for name in PCIE_RAW:
        row[name] = read_int(counters_dir / name)
    return row


def pcie_rates(previous: dict, current: dict, elapsed: float, mst_flit_bytes: float = MST_FLIT_BYTES
               ) -> dict[str, object]:
    out: dict[str, object] = {}
    raw = [(previous.get(n), current.get(n)) for n in PCIE_RAW]
    if any(a is None or b is None for a, b in raw):
        return {"pcie_status": "unavailable"}
    if any(b == DEAD for _, b in raw) or not current.get("alive"):
        return {"pcie_status": "card_dead"}
    if elapsed <= 0:
        return {"pcie_status": "bad_interval"}
    for column, names in FAMILIES.items():
        flits = sum((current[f"{n}{noc}"] - previous[f"{n}{noc}"]) % WRAP for n in names for noc in (0, 1))
        unit = mst_flit_bytes if column.startswith("mst") else SLV_FLIT_BYTES
        out[column] = flits * unit / elapsed / 1e9
    out["h2d_gb_s"] = out["mst_wr_gb_s"] + out["slv_rd_gb_s"]
    out["d2h_gb_s"] = out["mst_rd_gb_s"] + out["slv_wr_gb_s"]
    for key in list(FAMILIES) + ["h2d_gb_s", "d2h_gb_s"]:
        out[key] = f"{out[key]:.6f}"
    out["pcie_status"] = "ok" if elapsed < 2.0 else "ok_long_interval_may_miss_wraps"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser, 0.1, "tt_card")
    parser.add_argument("--mst-flit-bytes", type=float, default=MST_FLIT_BYTES,
                        help="bytes per host-MMIO (mst_*) count: 32 for uncached 32-byte accesses (default), "
                             "64 for write-combined 64-byte writes")
    parser.add_argument("--holders-interval", type=float, default=1.0,
                        help="seconds between /proc scans for processes holding a card (0 disables)")
    args = parser.parse_args()
    try:
        cards = discover_cards(args.sysfs_root, args.cards)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        for c in cards:
            print(f"{c.label}  {c.bdf}  {c.card_type}  board {c.serial}  fw {c.fw_bundle}  "
                  f"alive {sample_card(c)['alive']}")
        return 0 if cards else 1
    if not cards:
        print(f"Error: no Tenstorrent cards under {args.sysfs_root}", file=sys.stderr)
        return 1
    try:
        outputs, path = open_outputs(args)
    except ValueError as error:
        parser.error(str(error))
    print("Cards: " + ", ".join(f"{c.label} ({c.bdf})" for c in cards), file=sys.stderr)
    if path:
        print(f"Output CSV: {path}", file=sys.stderr)
    sink = CsvSink(outputs, CSV_FIELDS)
    state: dict[str, object] = {"previous": {}, "holders": {}, "holders_at": -1e9}

    def tick() -> None:
        now_mono = time.monotonic()
        if args.holders_interval > 0 and now_mono - state["holders_at"] >= args.holders_interval:
            state["holders"] = holders_of([c.index for c in cards])
            state["holders_at"] = now_mono
        rows = []
        for card in cards:
            t0, e0 = time.monotonic(), time.time()
            row = sample_card(card)
            mono, epoch = (t0 + time.monotonic()) / 2, (e0 + time.time()) / 2
            prev = state["previous"].get(card.index)
            row.update(timestamp_epoch=f"{epoch:.6f}", timestamp_utc=utc_timestamp(epoch),
                       holders=state["holders"].get(card.index, ""))
            if prev is not None:
                elapsed = mono - prev[0]
                row["interval_seconds"] = f"{elapsed:.6f}"
                row.update(pcie_rates(prev[1], row, elapsed, args.mst_flit_bytes))
            state["previous"][card.index] = (mono, row)
            rows.append(row)
        sink.write(rows)

    try:
        run_loop(args, tick)
    finally:
        sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
