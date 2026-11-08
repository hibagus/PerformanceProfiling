#!/usr/bin/env python3
# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Per-link Ethernet bandwidth, frame and FEC counters of Tenstorrent Blackhole cards, without touching the traffic.

Each Blackhole Ethernet tile runs Tenstorrent's base firmware, which copies the MAC's hardware counters (bytes and
frames sent and received, bad-FCS and dropped frames, corrected and uncorrected FEC codewords, queue resends and
drops) into ``eth_live_status_t`` in the tile's L1 about once a second (link status check, 1000 ms). tt-metal's
fabric router, when it owns the tile, keeps calling the same check (``risc_context_switch``). This monitor reads
those snapshots with the driver's NOC_READ ioctl on a power-neutral file descriptor (tt_common.KmdNoc; tt-kmd >=
2.12.0) and turns consecutive snapshots into rates. It reads only the ARC tile (to learn which Ethernet tiles are
enabled) and enabled Ethernet tiles, never Tensix, DRAM or L2CPU, and writes nothing.

Resolution is the snapshot period, not the poll interval: 2.0 s under the base firmware, about 0.75 s while
tt-metal's fabric owns the tile (measured). A row is written when a new snapshot appears, with rates over the time
between the two snapshots: from the base firmware's timestamp (50 MHz) when it advanced, otherwise (tt-metal's
check does not update it) from the host times the snapshots were first seen, which jitter by one poll interval;
sum bytes over a longer window for exact averages. A link whose snapshot stops changing for
--stale-after seconds gets a ``stale`` row: its owner no longer runs the link check, so no rate can be given.
Bytes are MAC frame bytes (headers and FCS included, preamble and inter-frame gap not); utilization is against
the trained link speed (``train_speed``, 400 Gb/s per tile on p150).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tt_common import (BOOT_RESULTS, DEBUG_BUF, TAG_BOARD_ID_HIGH, TAG_BOARD_ID_LOW, TAG_ENABLED_ETH,  # noqa: E402
                       ArcTelemetry, CsvSink, EthCore, KmdNoc, KmdNocError, SysfsCard, add_common_args,
                       card_alive, discover_cards, enabled_eth_channels, open_outputs, positive_float, run_loop,
                       utc_timestamp)

CSV_FIELDS = ["timestamp_epoch", "timestamp_utc", "card", "bdf", "channel", "tile", "port_status", "speed_gbps",
              "peer", "heartbeat", "owner", "fw_stamp", "fw_clock_hz", "dt_fw_s", "dt_host_s", "dt_s",
              "tx_bytes", "rx_bytes", "tx_gb_s", "rx_gb_s", "tx_util_pct", "rx_util_pct",
              "tx_frames", "rx_frames", "tx_badfcs_frames", "rx_badfcs_frames", "rx_dropped_frames",
              "corr_cw_per_s", "uncorr_cw", "retrains", "link_flaps", "txq_resends", "rxq_drops", "rx_link_up",
              "counter_status", "bytes_txd_total", "bytes_rxd_total", "corr_cw_total", "uncorr_cw_total"]
SIGNATURE_TX = BOOT_RESULTS + 4 * 136      # eth_live_status_t.frames_txd (u64)
STAMP = DEBUG_BUF + 4 * 466                 # debug_buf_t.eth_link_status_check_timestamp (u64), one per snapshot
HEARTBEAT = BOOT_RESULTS + 0x70             # eth_status.heartbeat[0..1]


@dataclass
class Link:
    card: SysfsCard
    core: EthCore
    peer: str = ""
    last: dict | None = None          # previous snapshot
    last_sig: tuple | None = None
    last_seen_mono: float = 0.0       # host time the previous snapshot was first seen
    last_stale_row: float = 0.0
    last_hb: int | None = None
    first_stamp: tuple[int, float] | None = None   # (firmware stamp, host time) of the first snapshot seen


def owner_of(previous: int | None, current: int) -> str:
    """Who runs the tile: base firmware advances heartbeat[0] (0xabcdNNNN); tt-metal's runtime firmware writes
    0xdcbaNNNN into heartbeat[1] (and leaves it there when it exits), so only an *advancing* word counts."""

    if previous is None:
        return ""
    hb0, hb1 = current & 0xFFFFFFFF, current >> 32
    if hb1 != previous >> 32 and hb1 >> 16 == 0xDCBA:
        return "metal"
    if hb0 != previous & 0xFFFFFFFF:
        return "base"
    return "stalled"


def discover_links(cards: list[SysfsCard], all_ports: bool, host_cards: list[SysfsCard] | None = None
                   ) -> tuple[dict[int, KmdNoc], list[Link], list[str]]:
    """The links of ``cards``; peers are named after any card of ``host_cards`` (default: ``cards``)."""

    by_board = {c.board_id: c for c in (host_cards or cards) if c.board_id is not None}
    nocs: dict[int, KmdNoc] = {}
    links: list[Link] = []
    notes: list[str] = []
    for card in cards:
        if not card_alive(card):
            notes.append(f"{card.label}: not alive in sysfs; skipped")
            continue
        noc = KmdNoc(card.index)
        nocs[card.index] = noc
        raw = ArcTelemetry(noc).read_raw()
        board = (raw.get(TAG_BOARD_ID_HIGH, 0) << 32) | raw.get(TAG_BOARD_ID_LOW, 0)
        mask = raw.get(TAG_ENABLED_ETH, 0)
        for channel, tile in enumerate(enabled_eth_channels(mask)):
            core = EthCore(channel, tile)
            core.read_static(noc)
            if core.local.get("board_id") != board or core.local.get("eth_id") != tile:
                notes.append(f"{card.label} ch{channel}: chip info does not match (board "
                             f"{core.local.get('board_id', 0):#x} vs {board:#x}, eth_id {core.local.get('eth_id')} "
                             f"vs tile {tile}); not monitored")
                continue
            if not all_ports and core.port_status != "UP":
                continue
            rb = core.remote.get("board_id", 0)
            peer_card = by_board.get(rb)
            peer = (f"{peer_card.label}/t{core.remote.get('eth_id')}" if peer_card else
                    f"board {rb:#x}/t{core.remote.get('eth_id')}" if rb else "")
            links.append(Link(card, core, peer))
    return nocs, links, notes


def rates(link: Link, live: dict, dt: float) -> dict[str, object]:
    prev = link.last
    d = {k: live[k] - prev[k] for k in live}
    if any(v < 0 for k, v in d.items() if k not in ("rx_link_up",)):
        return {"counter_status": "reset"}
    speed = link.core.speed_gbps or 0
    tx_gb_s, rx_gb_s = d["bytes_txd"] / dt / 1e9, d["bytes_rxd"] / dt / 1e9
    row: dict[str, object] = {
        "tx_bytes": d["bytes_txd"], "rx_bytes": d["bytes_rxd"],
        "tx_gb_s": f"{tx_gb_s:.6f}", "rx_gb_s": f"{rx_gb_s:.6f}",
        "tx_frames": d["frames_txd"], "rx_frames": d["frames_rxd"],
        "tx_badfcs_frames": d["frames_txd_badfcs"], "rx_badfcs_frames": d["frames_rxd_badfcs"],
        "rx_dropped_frames": d["frames_rxd_dropped"], "corr_cw_per_s": f"{d['corr_cw'] / dt:.1f}",
        "uncorr_cw": d["uncorr_cw"], "retrains": d["retrain_count"], "link_flaps": d["link_flap_count"],
        "txq_resends": d["txq0_resend_cnt"] + d["txq1_resend_cnt"] + d["txq2_resend_cnt"],
        "rxq_drops": d["rxq0_pkt_drop"] + d["rxq1_pkt_drop"] + d["rxq2_pkt_drop"], "counter_status": "ok",
    }
    if speed:
        row["tx_util_pct"] = f"{100 * tx_gb_s * 8 / speed:.4f}"
        row["rx_util_pct"] = f"{100 * rx_gb_s * 8 / speed:.4f}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser, 0.25, "tt_eth_bw")
    parser.add_argument("--all-ports", action="store_true", help="also links whose port is not UP")
    parser.add_argument("--stale-after", type=positive_float, default=3.0,
                        help="seconds without a new firmware snapshot before a link is reported stale")
    parser.add_argument("--eth-clock-hz", type=float, default=50e6,
                        help="clock of the base firmware's snapshot timestamp (50 MHz measured on ETH FW 1.13.1); "
                             "0 = calibrate against the host")
    parser.add_argument("--calibrate-after", type=positive_float, default=10.0,
                        help="seconds of snapshots before the calibrated firmware clock times the rates")
    parser.add_argument("--links", action="store_true", help="print the links found with totals and exit")
    args = parser.parse_args()
    try:
        cards = discover_cards(args.sysfs_root, args.cards)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        for c in cards:
            print(f"{c.label}  {c.bdf}  {c.card_type}  board {c.serial}")
        return 0 if cards else 1
    try:
        nocs, links, notes = discover_links(cards, args.all_ports, discover_cards(args.sysfs_root))
    except (OSError, KmdNocError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    for note in notes:
        print(note, file=sys.stderr)
    if args.links:
        for link in links:
            live, hb, _ = link.core.read_live(nocs[link.card.index])
            c = link.core
            print(f"{link.card.label} ch{c.channel:<2} tile {c.tile:<2} xy {c.xy}  {c.port_status:<7} "
                  f"{c.train_status:<18} {c.speed_gbps} Gb/s  peer {link.peer or '-':<14} fw {c.eth_fw}  "
                  f"tx {live['bytes_txd']} B rx {live['bytes_rxd']} B corr {live['corr_cw']} "
                  f"uncorr {live['uncorr_cw']} retrain {live['retrain_count']} hb {hb:#x}")
        for noc in nocs.values():
            noc.close()
        return 0 if links else 1
    if not links:
        print("Error: no Ethernet links to monitor (try --all-ports)", file=sys.stderr)
        return 1
    try:
        outputs, path = open_outputs(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Links: {len(links)} on cards " + ", ".join(sorted({l.card.label for l in links})), file=sys.stderr)
    if path:
        print(f"Output CSV: {path}", file=sys.stderr)
    sink = CsvSink(outputs, CSV_FIELDS)

    def tick() -> None:
        rows = []
        for link in links:
            noc = nocs[link.card.index]
            mono, epoch = time.monotonic(), time.time()
            base = {"timestamp_epoch": f"{epoch:.6f}", "timestamp_utc": utc_timestamp(epoch),
                    "card": link.card.label, "bdf": link.card.bdf, "channel": link.core.channel,
                    "tile": link.core.tile, "port_status": link.core.port_status,
                    "speed_gbps": link.core.speed_gbps, "peer": link.peer}
            try:
                # Cheap change detection: the firmware's link-check timestamp and the first live counter words.
                sig = (noc.read(*link.core.xy, STAMP, 8), noc.read(*link.core.xy, SIGNATURE_TX, 8))
                if sig == link.last_sig:
                    if link.last is not None and mono - link.last_seen_mono > args.stale_after \
                            and mono - link.last_stale_row > args.stale_after:
                        hb = noc.read(*link.core.xy, HEARTBEAT, 8)
                        rows.append({**base, "heartbeat": hex(hb), "owner": owner_of(link.last_hb, hb),
                                     "counter_status": "stale", "dt_host_s": f"{mono - link.last_seen_mono:.3f}"})
                        link.last_stale_row, link.last_hb = mono, hb
                    continue
                live, _, stamp = link.core.read_live(noc)
                hb = noc.read(*link.core.xy, HEARTBEAT, 8)
            except OSError as error:
                rows.append({**base, "counter_status": f"error:{error.errno}"})
                continue
            row = {**base, "heartbeat": hex(hb), "owner": owner_of(link.last_hb, hb),
                   "rx_link_up": live["rx_link_up"], "fw_stamp": stamp,
                   "bytes_txd_total": live["bytes_txd"], "bytes_rxd_total": live["bytes_rxd"],
                   "corr_cw_total": live["corr_cw"], "uncorr_cw_total": live["uncorr_cw"]}
            if link.first_stamp is None and link.last is not None:
                # Anchor the clock calibration at the first *change*: that snapshot is seen within one poll of
                # being written, while the snapshot found at start-up may already be a whole period old.
                link.first_stamp = (stamp, mono)
            if link.last is not None:
                dt_host = mono - link.last_seen_mono
                dt = dt_host
                # The firmware stamps each snapshot; once a long enough window calibrates that clock against the
                # host, its deltas time the snapshots exactly (host first-seen times jitter by one poll interval).
                span_stamp, span_host = stamp - link.first_stamp[0], mono - link.first_stamp[1]
                clock = args.eth_clock_hz or (span_stamp / span_host if span_host >= args.calibrate_after
                                              and span_stamp > 0 else 0)
                if clock:
                    dt_fw = (stamp - link.last_sig[0]) / clock
                    row.update(dt_fw_s=f"{dt_fw:.6f}", fw_clock_hz=f"{clock:.0f}")
                    if 0 < dt_fw < 4 * dt_host + 1:
                        dt = dt_fw
                row.update(dt_host_s=f"{dt_host:.6f}", dt_s=f"{dt:.6f}")
                row.update(rates(link, live, dt))
                rows.append(row)
            link.last, link.last_seen_mono, link.last_hb = live, mono, hb
            link.last_sig = (stamp, live["frames_txd"])
        sink.write(rows)

    try:
        run_loop(args, tick)
    finally:
        sink.close()
        for noc in nocs.values():
            noc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
