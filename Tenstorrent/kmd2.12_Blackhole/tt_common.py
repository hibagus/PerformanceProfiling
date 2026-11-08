# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Shared code for the Tenstorrent Blackhole monitors (tt-kmd 2.12, firmware 19.x).

Three ways to read a card, from least to most contact with the device:

* sysfs (``SysfsCard``): hwmon, clock and PCIe counter files the driver exposes. Never opens the device.
* ``KmdNoc``: the driver's ``TENSTORRENT_IOCTL_NOC_READ`` (tt-kmd >= 2.12.0), a single 1-8 byte NoC read done by
  the kernel through its own reserved window. The device file is opened power-aware (``O_APPEND``) and no power
  state is ever requested, so the descriptor adds nothing to the power the driver asks the firmware for, and no
  firmware (ARC) message is sent by opening, reading or closing it. Only the ARC tile and enabled Ethernet tiles
  are read: neither is ever clock-gated (the firmware gates only Tensix, GDDR/MRISC and L2CPU), so a read cannot
  hit a gated tile however the card's power state changes while the monitor runs.
* ``ArcTelemetry`` and ``EthCore`` are built on ``KmdNoc``.

Nothing here writes to the device.
"""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import math
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, TextIO

SYSFS_ROOT = Path("/sys/class/tenstorrent")
DEV_ROOT = Path("/dev/tenstorrent")
DEAD = 0xFFFFFFFF


# --------------------------------------------------------------------------------------------- CLI helpers
def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def utc_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def add_common_args(parser: argparse.ArgumentParser, default_interval: float, suffix: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("-c", "--cards", nargs="+", metavar="CARD",
                        help="card index (0), PCI bus (3f) or BDF (0000:3f:00.0); default: all")
    parser.add_argument("-w", "--interval", type=positive_float, default=default_interval,
                        help=f"sampling interval in seconds (default {default_interval})")
    parser.add_argument("-W", "--duration", type=positive_float, help="stop after this many seconds")
    parser.add_argument("--stop-file", type=Path, help="stop when this file exists")
    parser.add_argument("--sysfs-root", type=Path, default=SYSFS_ROOT)
    parser.add_argument("--output-mode", choices=("stdout", "file", "both"), default="file")
    shortcuts = parser.add_mutually_exclusive_group()
    shortcuts.add_argument("--stdout", action="store_true")
    shortcuts.add_argument("--both", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--filename", default=f"{stamp}_{suffix}.csv")
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument("--overwrite", action="store_true")
    writes.add_argument("--append", action="store_true")
    parser.add_argument("--list", action="store_true", help="list the cards found and exit")


def open_outputs(args: argparse.Namespace) -> tuple[list[tuple[TextIO, bool]], Path | None]:
    if args.stdout:
        args.output_mode = "stdout"
    elif args.both:
        args.output_mode = "both"
    if args.output_mode == "stdout":
        if args.overwrite or args.append:
            raise ValueError("--overwrite and --append require file or both mode")
        return [(sys.stdout, False)], None
    path = args.output
    if path is None:
        name = args.filename if args.filename.endswith(".csv") else args.filename + ".csv"
        if not name or "/" in name:
            raise ValueError("--filename must be non-empty and must not contain '/'")
        path = args.output_dir / name
    if path.exists() and not args.overwrite and not args.append:
        raise ValueError(f"output already exists: {path} (use --overwrite or --append)")
    had_content = args.append and path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if args.append else "w", newline="", encoding="utf-8")
    outputs: list[tuple[TextIO, bool]] = [(handle, had_content)]
    if args.output_mode == "both":
        outputs.insert(0, (sys.stdout, False))
    return outputs, path


class CsvSink:
    """CSV to one or more handles; the header is written once."""

    def __init__(self, outputs: list[tuple[TextIO, bool]], fields: Iterable[str]):
        self.outputs = outputs
        self.writers = [csv.DictWriter(h, fieldnames=list(fields), extrasaction="ignore") for h, _ in outputs]
        for writer, (handle, had_header) in zip(self.writers, outputs):
            if not had_header:
                writer.writeheader()
                handle.flush()

    def write(self, rows: Iterable[dict]) -> None:
        for row in rows:
            for writer in self.writers:
                writer.writerow(row)
        for handle, _ in self.outputs:
            handle.flush()

    def close(self) -> None:
        for handle, _ in self.outputs:
            if handle is not sys.stdout:
                handle.close()


def run_loop(args: argparse.Namespace, tick: Callable[[], None]) -> None:
    """Call ``tick`` every ``args.interval`` s until SIGINT/SIGTERM, ``--duration`` or ``--stop-file``."""

    stopped = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    next_due = started
    while not stopped:
        tick()
        if args.duration is not None and time.monotonic() - started >= args.duration:
            break
        if args.stop_file is not None and args.stop_file.exists():
            break
        next_due += args.interval
        remaining = next_due - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:  # fell behind: do not try to catch up with a burst
            next_due = time.monotonic()


# --------------------------------------------------------------------------------------------- sysfs
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def read_int(path: Path) -> int | None:
    text = read_text(path)
    try:
        return int(text, 0) if text else None
    except ValueError:
        return None


@dataclass
class SysfsCard:
    """One card as the driver shows it in sysfs (``/sys/class/tenstorrent/tenstorrent!N``)."""

    index: int
    path: Path
    bdf: str
    card_type: str
    serial: str          # board ID, hex (tt_serial)
    fw_bundle: str
    hwmon: Path | None

    @property
    def board_id(self) -> int | None:
        try:
            return int(self.serial, 16)
        except ValueError:
            return None

    @property
    def bus(self) -> str:
        return self.bdf.split(":")[1] if self.bdf.count(":") == 2 else self.bdf

    @property
    def label(self) -> str:
        return f"{self.index}:{self.bus}"


def discover_cards(root: Path = SYSFS_ROOT, wanted: list[str] | None = None) -> list[SysfsCard]:
    cards: list[SysfsCard] = []
    if not root.is_dir():
        return cards
    for entry in root.iterdir():
        name = entry.name
        if not name.startswith("tenstorrent!") or not name.split("!")[1].isdigit():
            continue
        index = int(name.split("!")[1])
        try:
            bdf = os.path.basename(os.path.realpath(entry / "device"))
        except OSError:
            bdf = ""
        hwmons = sorted((entry / "device" / "hwmon").glob("hwmon*")) if (entry / "device" / "hwmon").is_dir() else []
        cards.append(SysfsCard(index, entry, bdf, read_text(entry / "tt_card_type"), read_text(entry / "tt_serial"),
                               read_text(entry / "tt_fw_bundle_ver"), hwmons[0] if hwmons else None))
    cards.sort(key=lambda c: c.index)
    if wanted:
        chosen = []
        for want in wanted:
            w = want.lower()
            match = [c for c in cards if w in (str(c.index), c.bdf.lower(), c.bus.lower())]
            if not match:
                raise ValueError(f"no card matches {want!r} (found: {', '.join(c.label for c in cards)})")
            for c in match:
                if c not in chosen:
                    chosen.append(c)
        cards = chosen
    return cards


def card_alive(card: SysfsCard) -> bool:
    """A card that fell off the bus reads 0xFFFFFFFF in every telemetry file (tt_aiclk)."""

    value = read_int(card.path / "tt_aiclk")
    return value is not None and value != DEAD


# --------------------------------------------------------------------------------------------- NoC reads
def _io(nr: int) -> int:  # _IO(TENSTORRENT_IOCTL_MAGIC, nr): no size or direction bits
    return (0xFA << 8) | nr


TENSTORRENT_IOCTL_NOC_READ = _io(18)
# struct tenstorrent_noc_io {u32 argsz, flags; u16 x, y; u8 noc, width, reserved0[2]; u64 addr, value;}
NOC_IO = struct.Struct("<IIHHBB2xQQ")


class KmdNocError(RuntimeError):
    pass


class KmdNoc:
    """Kernel-mediated NoC reads of one card (tt-kmd >= 2.12.0), opened without any power contribution.

    ``O_APPEND`` makes the descriptor power-aware with no power flags; with no ``SET_POWER_STATE`` call it never
    changes the aggregated power request (tt-kmd chardev.c: open sends nothing for a power-aware fd, release
    returns early for an fd with no power contribution). ``O_NONBLOCK`` makes open fail (EAGAIN) instead of
    waiting when another process holds the card exclusively (``O_EXCL``, e.g. a reset tool).
    """

    def __init__(self, index: int, dev_root: Path = DEV_ROOT, ioctl: Callable = fcntl.ioctl):
        self.index = index
        self.path = dev_root / str(index)
        self._ioctl = ioctl
        self.fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_NONBLOCK | os.O_CLOEXEC)
        self.reads = 0

    def read(self, x: int, y: int, addr: int, width: int = 4, noc: int = 0) -> int:
        buf = bytearray(NOC_IO.pack(NOC_IO.size, 0, x, y, noc, width, addr, 0))
        try:
            self._ioctl(self.fd, TENSTORRENT_IOCTL_NOC_READ, buf, True)
        except OSError as error:
            if error.errno in (errno.ENOTTY, errno.EINVAL) and self.reads == 0:
                raise KmdNocError(f"{self.path}: NOC_READ ioctl not supported ({error.strerror}); "
                                  "needs tt-kmd >= 2.12.0") from error
            raise
        self.reads += 1
        return NOC_IO.unpack(buf)[-1]

    def read_words(self, x: int, y: int, addr: int, nwords: int) -> list[int]:
        """``nwords`` little-endian u32 words from an 8-byte-aligned ``addr`` (as 8-byte reads)."""

        if addr % 8:
            raise ValueError("addr must be 8-byte aligned")
        out: list[int] = []
        for i in range((nwords + 1) // 2):
            v = self.read(x, y, addr + 8 * i, 8)
            out.extend((v & 0xFFFFFFFF, v >> 32))
        return out[:nwords]

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "KmdNoc":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------------------------- ARC telemetry
ARC_X, ARC_Y = 8, 0
RESET_SCRATCH = 0x80030400
ARC_TELEMETRY_PTR = RESET_SCRATCH + 13 * 4
ARC_TELEMETRY_DATA = RESET_SCRATCH + 12 * 4
CSM_BASE, CSM_SIZE = 0x10000000, 1 << 19


def _in_csm(addr: int, length: int) -> bool:
    return CSM_BASE <= addr <= CSM_BASE + CSM_SIZE - length


def _fixed16(v: int) -> float:  # signed 16.16
    return (v - (1 << 32) if v & 0x80000000 else v) / 65536.0


def _lo16(v: int) -> int:
    return v & 0xFFFF


def _ver(v: int) -> str:
    return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"


ARB_MAX = ["fmax", "tdp", "fast_tdc", "tdc", "thm", "board_power", "voltage", "gddr_thm", "doppler_slow",
           "doppler_critical", "host_fmax"]
ARB_MIN = ["fmin", "busy"]
FREQ_REASON = ["min_arb", "max_arb", "fmin", "sweep", "forced"]


def _arb(names: list[str]) -> Callable[[int], str]:
    def decode(v: int) -> str:
        a = v >> 16
        return f"{v & 0xFFFF}:{names[a] if a < len(names) else a}"
    return decode


def _ppm(v: int) -> str:
    arb, reason = v & 0xFFFF, v >> 16
    r = FREQ_REASON[reason] if reason < len(FREQ_REASON) else str(reason)
    names = ARB_MAX if r == "max_arb" else ARB_MIN if r == "min_arb" else []
    return f"{r}:{names[arb] if arb < len(names) else arb}"


def _mask_names(names: list[str]) -> Callable[[int], str]:
    return lambda v: "|".join(n for i, n in enumerate(names) if v >> i & 1) or "none"


# tag -> (column name, decoder). From tt-system-firmware lib/tenstorrent/bh_arc/telemetry.h (19.x); tags the
# firmware does not publish are skipped. Integer-valued tags (power, current) are truncated by the firmware.
TAGS: dict[int, tuple[str, Callable[[int], object]]] = {
    1: ("board_id_high", hex), 2: ("board_id_low", hex), 3: ("asic_id", hex),
    5: ("telem_update_ms", int), 6: ("vcore_mv", int), 7: ("power_w", int), 8: ("current_a", int),
    9: ("vdd_limits_mv", lambda v: f"{v & 0xFFFF}-{v >> 16}"), 10: ("thm_limit_shutdown_c", int),
    11: ("asic_temp_c", lambda v: round(_fixed16(v), 3)), 12: ("vreg_temp_c", lambda v: round(_fixed16(v), 3)),
    13: ("board_temp_c", lambda v: round(_fixed16(v), 3)),
    14: ("aiclk_mhz", _lo16), 15: ("axiclk_mhz", _lo16), 16: ("arcclk_mhz", _lo16),
    17: ("l2cpuclk0_mhz", _lo16), 18: ("l2cpuclk1_mhz", _lo16), 19: ("l2cpuclk2_mhz", _lo16),
    20: ("l2cpuclk3_mhz", _lo16),
    21: ("eth_live_status", hex), 22: ("gddr_status", hex), 23: ("gddr_speed_mbps", int),
    24: ("eth_fw_version", hex), 25: ("gddr_fw_version", hex), 26: ("dm_app_fw_version", hex),
    27: ("dm_bl_fw_version", hex), 28: ("flash_bundle_version", _ver), 29: ("cm_fw_version", _ver),
    31: ("fan_speed_pct", int), 32: ("timer_heartbeat", int), 33: ("telem_enum_count", int),
    34: ("enabled_tensix_col", hex), 35: ("enabled_eth", hex), 36: ("enabled_gddr", hex),
    37: ("enabled_l2cpu", hex), 38: ("pcie_usage", hex), 39: ("input_current_a", int),
    40: ("noc_translation", int), 41: ("fan_rpm", int),
    42: ("gddr_0_1_temp", hex), 43: ("gddr_2_3_temp", hex), 44: ("gddr_4_5_temp", hex), 45: ("gddr_6_7_temp", hex),
    46: ("gddr_0_1_corr_errs", hex), 47: ("gddr_2_3_corr_errs", hex), 48: ("gddr_4_5_corr_errs", hex),
    49: ("gddr_6_7_corr_errs", hex), 50: ("gddr_uncorr_errs", hex), 51: ("max_gddr_temp_c", int),
    52: ("asic_location", int), 53: ("board_power_limit_w", int), 54: ("input_power_w", int),
    55: ("tdc_limit_a", int), 56: ("thm_limit_throttle_c", int), 57: ("fw_build_date", hex),
    58: ("tt_flash_version", hex), 59: ("enabled_tensix_row", hex), 60: ("therm_trip_count", int),
    61: ("asic_id_high", hex), 62: ("asic_id_low", hex), 63: ("aiclk_limit_max_mhz", int),
    64: ("tdp_limit_w", int), 65: ("aiclk_arb_min", _arb(ARB_MIN)), 66: ("aiclk_arb_max", _arb(ARB_MAX)),
    67: ("enabled_min_arb", _mask_names(ARB_MIN)), 68: ("enabled_max_arb", _mask_names(ARB_MAX)),
    69: ("aiclk_ppm_info", _ppm), 70: ("host_aiclk_limit_mhz", int), 71: ("smbus_errors", int),
    72: ("gddr_mrisc_noc2axi_port", hex), 73: ("gddr_west_io_power_w", int), 74: ("gddr_east_io_power_w", int),
    75: ("kernel_throttler", hex), 76: ("nop_start_count", int), 77: ("nop_on_ms", int),
    78: ("fw_capabilities_0", hex), 79: ("fw_active_config_0", hex), 80: ("flash_jedec_id", hex),
}
TAG_ENABLED_ETH = 35
TAG_BOARD_ID_HIGH, TAG_BOARD_ID_LOW = 1, 2


class ArcTelemetry:
    """The firmware's telemetry table in ARC CSM, located as tt-kmd does (blackhole.c) and read with ``KmdNoc``."""

    def __init__(self, noc: KmdNoc):
        self.noc = noc
        base = noc.read(ARC_X, ARC_Y, ARC_TELEMETRY_PTR)
        data = noc.read(ARC_X, ARC_Y, ARC_TELEMETRY_DATA)
        if not _in_csm(base, 8) or not _in_csm(data, 4):
            raise KmdNocError(f"telemetry not available (table {base:#x}, data {data:#x})")
        self.version = noc.read(ARC_X, ARC_Y, base)
        if (self.version >> 16) & 0xFF > 1:
            raise KmdNocError(f"unsupported telemetry version {self.version:#x}")
        count = noc.read(ARC_X, ARC_Y, base + 4)
        if count > 1024 or not _in_csm(base + 8, 4 * count):
            raise KmdNocError(f"bad telemetry table: {count} entries at {base + 8:#x}")
        self.offsets: dict[int, int] = {}
        for i in range(count):
            entry = noc.read(ARC_X, ARC_Y, base + 8 + 4 * i)
            tag, offset = entry & 0xFFFF, entry >> 16
            if _in_csm(data + 4 * offset, 4):
                self.offsets[tag] = offset
        self.data = data
        self.words = max(self.offsets.values(), default=-1) + 1

    def read_raw(self) -> dict[int, int]:
        """All published tags in one pass over the data array (8-byte reads where aligned)."""

        start = self.data - (self.data % 8)
        lead = (self.data - start) // 4
        words = self.noc.read_words(ARC_X, ARC_Y, start, lead + self.words)[lead:]
        return {tag: words[off] for tag, off in self.offsets.items()}

    @staticmethod
    def decode(raw: dict[int, int]) -> dict[str, object]:
        out: dict[str, object] = {}
        for tag, value in raw.items():
            name, decoder = TAGS.get(tag, (f"tag_{tag}", hex))
            try:
                out[name] = decoder(value)
            except Exception:  # noqa: BLE001 - a decoder never stops the monitor
                out[name] = hex(value)
        return out


# --------------------------------------------------------------------------------------------- Ethernet
# Blackhole ERISC base firmware (tt-metal tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h).
BOOT_RESULTS = 0x7CC00            # boot_results_t, 1 KiB: eth_status (w0), serdes (w32), macpcs (w96),
LIVE_WORD = 128                   # eth_live_status_t (w128..191), api table, versions (w238/239), chip info
DEBUG_BUF = 0x7C400               # debug_buf_t, 2 KiB; eth_link_status_check_timestamp at word 466
ETH_TRANSLATED_X0, ETH_TRANSLATED_Y = 20, 25   # channel c (enabled tiles in tile order) = (20 + c, 25)
PORT_STATUS = {0: "UNKNOWN", 1: "UP", 2: "DOWN", 3: "UNUSED"}
TRAIN_STATUS = ["TRAINING", "SKIP", "PASS", "INT_LB", "EXT_LB", "TIMEOUT_MANUAL_EQ", "TIMEOUT_ANLT",
                "TIMEOUT_CDR_LOCK", "TIMEOUT_BIST_LOCK", "TIMEOUT_LINK_UP", "TIMEOUT_CHIP_INFO"]
# eth_live_status_t: 4 u32 + 4 spare, then u64 counters from word 8 (offsets below are u64 indices from word 8)
LIVE_U32 = ["retrain_count", "rx_link_up", "link_flap_count", "link_poll_alive_count"]
LIVE_U64 = ["frames_txd", "frames_txd_ok", "frames_txd_badfcs", "bytes_txd", "bytes_txd_ok", "bytes_txd_badfcs",
            "frames_rxd", "frames_rxd_ok", "frames_rxd_badfcs", "frames_rxd_dropped", "bytes_rxd", "bytes_rxd_ok",
            "bytes_rxd_badfcs", "bytes_rxd_dropped", "corr_cw", "uncorr_cw", "txq0_resend_cnt", "txq1_resend_cnt",
            "txq2_resend_cnt", "rxq0_pkt_drop", "rxq1_pkt_drop", "rxq2_pkt_drop"]
LIVE_WORDS = 8 + 2 * len(LIVE_U64)  # 52


def enabled_eth_channels(enabled_eth_mask: int) -> list[int]:
    """Physical tile id of each translated channel: channel c is the c-th enabled tile in tile order."""

    return [tile for tile in range(14) if enabled_eth_mask >> tile & 1]


def decode_chip_info(w: list[int]) -> dict[str, int]:
    b = struct.pack("<I", w[0])
    return {"pcb_type": b[0], "asic_location": b[1], "eth_id": b[2], "logical_eth_id": b[3],
            "board_id": (w[1] << 32) | w[2], "asic_id": (w[5] << 32) | w[6]}


@dataclass
class EthCore:
    """One enabled Ethernet channel of a card (translated NoC coordinate)."""

    channel: int
    tile: int
    port_status: str = ""
    train_status: str = ""
    speed_gbps: int = 0
    local: dict = field(default_factory=dict)
    remote: dict = field(default_factory=dict)
    eth_fw: str = ""

    @property
    def xy(self) -> tuple[int, int]:
        return ETH_TRANSLATED_X0 + self.channel, ETH_TRANSLATED_Y

    def read_static(self, noc: KmdNoc) -> None:
        w = noc.read_words(*self.xy, BOOT_RESULTS, 256)
        self.port_status = PORT_STATUS.get(w[1], str(w[1]))
        self.train_status = TRAIN_STATUS[w[2]] if w[2] < len(TRAIN_STATUS) else str(w[2])
        self.speed_gbps = w[3]
        self.eth_fw = f"{(w[239] >> 16) & 0xFF}.{(w[239] >> 8) & 0xFF}.{w[239] & 0xFF}"
        self.local = decode_chip_info(w[240:248])
        self.remote = decode_chip_info(w[248:256])

    @staticmethod
    def decode_live(w: list[int]) -> dict[str, int]:
        live = {name: w[i] for i, name in enumerate(LIVE_U32)}
        for i, name in enumerate(LIVE_U64):
            live[name] = w[8 + 2 * i] | (w[9 + 2 * i] << 32)
        return live

    def read_live(self, noc: KmdNoc) -> tuple[dict[str, int], int, int]:
        """(eth_live_status_t, base-firmware heartbeat word, link-status-check timestamp)."""

        live = self.decode_live(noc.read_words(*self.xy, BOOT_RESULTS + 4 * LIVE_WORD, LIVE_WORDS))
        heartbeat = noc.read(*self.xy, BOOT_RESULTS + 0x70, 4)
        stamp = noc.read(*self.xy, DEBUG_BUF + 4 * 466, 8)
        return live, heartbeat, stamp
