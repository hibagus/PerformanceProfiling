from __future__ import annotations

import csv
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT_DIR))

import tt_arc_telemetry_monitor as arcmon  # noqa: E402
import tt_card_monitor as cardmon  # noqa: E402
import tt_common as tt  # noqa: E402
import tt_eth_bw_monitor as ethmon  # noqa: E402


def make_card(root: Path, index: int, bus: str, *, aiclk: int = 800, counters: dict | None = None) -> Path:
    pci = root / "pci" / f"0000:{bus}:00.0"
    (pci / "hwmon" / f"hwmon{index}").mkdir(parents=True)
    card = root / "class" / f"tenstorrent!{index}"
    (card / "pcie_perf_counters").mkdir(parents=True)
    os.symlink(pci, card / "device")
    for name, value in {"tt_aiclk": aiclk, "tt_axiclk": 960, "tt_arcclk": 800, "tt_heartbeat": 7,
                        "tt_therm_trip_count": 0, "tt_card_type": "p150b", "tt_serial": "000004113190B008",
                        "tt_fw_bundle_ver": "19.17.1.0"}.items():
        (card / name).write_text(f"{value}\n")
    for name, value in {"power1_input": 17_000_000, "power1_max": 150_000_000, "curr1_input": 25_000,
                        "curr1_max": 200_000, "in0_input": 706, "in0_max": 900, "temp1_input": 33_000,
                        "temp1_max": 90_000}.items():
        (pci / "hwmon" / f"hwmon{index}" / name).write_text(f"{value}\n")
    for name in cardmon.PCIE_RAW:
        (card / "pcie_perf_counters" / name).write_text(f"{(counters or {}).get(name, 0)}\n")
    return card


class FakeDevice:
    """A sparse 32-bit word memory per (x, y) answering TENSTORRENT_IOCTL_NOC_READ."""

    def __init__(self) -> None:
        self.words: dict[tuple[int, int, int], int] = {}
        self.calls = 0

    def put(self, x: int, y: int, addr: int, values: list[int]) -> None:
        for i, v in enumerate(values):
            self.words[(x, y, addr + 4 * i)] = v & 0xFFFFFFFF

    def put64(self, x: int, y: int, addr: int, value: int) -> None:
        self.put(x, y, addr, [value & 0xFFFFFFFF, value >> 32])

    def ioctl(self, _fd: int, request: int, buf: bytearray, _mutate: bool) -> int:
        assert request == tt.TENSTORRENT_IOCTL_NOC_READ
        argsz, flags, x, y, noc, width, addr, _ = tt.NOC_IO.unpack(buf)
        assert argsz == 32 and flags == 0 and noc in (0, 1) and addr % width == 0
        lo = self.words.get((x, y, addr - addr % 4), 0)
        value = lo if width == 4 else lo | (self.words.get((x, y, addr + 4), 0) << 32)
        buf[:] = tt.NOC_IO.pack(argsz, flags, x, y, noc, width, addr, value)
        self.calls += 1
        return 0


def fake_arc(dev: FakeDevice, tags: dict[int, int]) -> None:
    table, data = tt.CSM_BASE + 0x1000, tt.CSM_BASE + 0x2004  # data deliberately not 8-byte aligned
    dev.put(tt.ARC_X, tt.ARC_Y, tt.ARC_TELEMETRY_PTR, [table])
    dev.put(tt.ARC_X, tt.ARC_Y, tt.ARC_TELEMETRY_DATA, [data])
    entries = [(tag | (i << 16)) for i, tag in enumerate(tags)]
    dev.put(tt.ARC_X, tt.ARC_Y, table, [0x100, len(entries), *entries])
    dev.put(tt.ARC_X, tt.ARC_Y, data, list(tags.values()))


def open_fake(dev: FakeDevice, directory: str) -> tt.KmdNoc:
    Path(directory, "0").write_text("")
    return tt.KmdNoc(0, Path(directory), ioctl=dev.ioctl)


class SysfsTests(unittest.TestCase):
    def test_discovery_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_card(root, 0, "3f")
            make_card(root, 3, "da")
            cards = tt.discover_cards(root / "class")
            self.assertEqual([c.label for c in cards], ["0:3f", "3:da"])
            self.assertEqual(cards[1].board_id, 0x4113190B008)
            self.assertEqual([c.label for c in tt.discover_cards(root / "class", ["da"])], ["3:da"])
            self.assertEqual([c.label for c in tt.discover_cards(root / "class", ["0000:3f:00.0"])], ["0:3f"])
            with self.assertRaises(ValueError):
                tt.discover_cards(root / "class", ["c3"])

    def test_sample_scales_hwmon_units(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            make_card(Path(d), 0, "3f")
            row = cardmon.sample_card(tt.discover_cards(Path(d) / "class")[0])
            self.assertEqual((row["power_w"], row["current_a"], row["asic_temp_c"]), (17.0, 25.0, 33.0))
            self.assertEqual((row["power_limit_w"], row["current_limit_a"], row["alive"]), (150.0, 200.0, 1))

    def test_dead_card(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            make_card(Path(d), 0, "3f", aiclk=tt.DEAD)
            card = tt.discover_cards(Path(d) / "class")[0]
            self.assertFalse(tt.card_alive(card))
            row = cardmon.sample_card(card)
            self.assertEqual(cardmon.pcie_rates(row, row, 0.1)["pcie_status"], "card_dead")


class PcieRateTests(unittest.TestCase):
    def test_flit_units_sum_over_nocs_and_wrap(self) -> None:
        prev = {n: 0 for n in cardmon.PCIE_RAW}
        cur = dict(prev, alive=1)
        prev["slv_rd_data_word_sent0"] = cardmon.WRAP - 1000          # wraps: +2000 flits
        cur["slv_rd_data_word_sent0"] = 1000
        cur["slv_rd_data_word_sent1"] = 3000
        cur["mst_posted_wr_data_word_sent0"] = 500
        cur["mst_nonposted_wr_data_word_sent1"] = 500
        out = cardmon.pcie_rates(prev, cur, 1.0)
        self.assertEqual(out["pcie_status"], "ok")
        self.assertAlmostEqual(float(out["slv_rd_gb_s"]), 5000 * 64 / 1e9)
        self.assertAlmostEqual(float(out["mst_wr_gb_s"]), 1000 * 32 / 1e9)
        self.assertAlmostEqual(float(out["h2d_gb_s"]), (5000 * 64 + 1000 * 32) / 1e9)
        self.assertAlmostEqual(float(cardmon.pcie_rates(prev, cur, 1.0, 64)["mst_wr_gb_s"]), 1000 * 64 / 1e9)
        self.assertAlmostEqual(float(out["d2h_gb_s"]), 0.0)


class NocAndArcTests(unittest.TestCase):
    def test_ioctl_layout(self) -> None:
        self.assertEqual(tt.NOC_IO.size, 32)
        self.assertEqual(tt.TENSTORRENT_IOCTL_NOC_READ, 0xFA12)

    def test_arc_table_walk_and_decode(self) -> None:
        dev = FakeDevice()
        fixed = int(46.625 * 65536)
        fake_arc(dev, {1: 0x411, 2: 0x3190B00B, 11: fixed, 14: 1350, 21: 0x3ED00000, 35: 0x3EDF,
                       42: 0x28282628, 66: 1350 | (3 << 16), 69: 1 | (1 << 16), 54: 163})
        with tempfile.TemporaryDirectory() as d, open_fake(dev, d) as noc:
            arc = tt.ArcTelemetry(noc)
            raw = arc.read_raw()
            decoded = arc.decode(raw)
            derived = arcmon.derive(raw, decoded)
        self.assertEqual(decoded["asic_temp_c"], 46.625)
        self.assertEqual(decoded["aiclk_arb_max"], "1350:tdc")
        self.assertEqual(decoded["aiclk_ppm_info"], "max_arb:tdp")
        self.assertEqual(decoded["input_power_w"], 163)
        self.assertEqual((derived["eth_heartbeat_mask"], derived["eth_link_mask"], derived["eth_links_up"]),
                         ("0x0", "0x3ed0", 8))
        self.assertEqual((derived["gddr0_temp_c"], derived["gddr1_temp_c"]), (40, 40))

    def test_arc_rejects_table_outside_csm(self) -> None:
        dev = FakeDevice()
        dev.put(tt.ARC_X, tt.ARC_Y, tt.ARC_TELEMETRY_PTR, [0xFFFFFFFF])
        with tempfile.TemporaryDirectory() as d, open_fake(dev, d) as noc:
            with self.assertRaises(tt.KmdNocError):
                tt.ArcTelemetry(noc)


class EthTests(unittest.TestCase):
    def test_channels_follow_enabled_tiles(self) -> None:
        self.assertEqual(tt.enabled_eth_channels(0x3EDF), [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13])

    def test_static_and_live_decode(self) -> None:
        dev = FakeDevice()
        core = tt.EthCore(channel=5, tile=6)
        x, y = core.xy
        self.assertEqual((x, y), (25, 25))
        boot = [0] * 256
        boot[1], boot[2], boot[3] = 1, 2, 400
        boot[239] = 0x10D01
        boot[240] = struct.unpack("<I", bytes([0, 0, 6, 5]))[0]
        boot[241], boot[242] = 0x411, 0x3190B008
        boot[248] = struct.unpack("<I", bytes([0, 0, 6, 5]))[0]
        boot[249], boot[250] = 0x411, 0x3190B007
        dev.put(x, y, tt.BOOT_RESULTS, boot)
        dev.put64(x, y, tt.BOOT_RESULTS + 4 * (128 + 8 + 2 * 3), 123_456)   # bytes_txd
        dev.put64(x, y, tt.BOOT_RESULTS + 4 * (128 + 8 + 2 * 10), 654_321)  # bytes_rxd
        dev.put64(x, y, tt.BOOT_RESULTS + 0x70, (0xDCBA0001 << 32) | 0xABCD0002)
        with tempfile.TemporaryDirectory() as d, open_fake(dev, d) as noc:
            core.read_static(noc)
            live, hb, _ = core.read_live(noc)
        self.assertEqual((core.port_status, core.train_status, core.speed_gbps), ("UP", "PASS", 400))
        self.assertEqual((core.local["eth_id"], core.local["board_id"]), (6, 0x4113190B008))
        self.assertEqual(core.remote["board_id"], 0x4113190B007)
        self.assertEqual((live["bytes_txd"], live["bytes_rxd"]), (123_456, 654_321))
        self.assertEqual(core.eth_fw, "1.13.1")

    def test_owner_needs_an_advancing_heartbeat(self) -> None:
        stale_metal = (0xDCBA0005 << 32) | 0xABCD0010
        self.assertEqual(ethmon.owner_of(None, stale_metal), "")
        self.assertEqual(ethmon.owner_of(stale_metal, stale_metal + 1), "base")
        self.assertEqual(ethmon.owner_of(stale_metal, stale_metal + (1 << 32)), "metal")
        self.assertEqual(ethmon.owner_of(stale_metal, stale_metal), "stalled")

    def test_rates_and_utilization(self) -> None:
        card = tt.SysfsCard(0, Path("/x"), "0000:3f:00.0", "p150b", "4113190B007", "", None)
        link = ethmon.Link(card, tt.EthCore(4, 4, speed_gbps=400))
        zero = {k: 0 for k in tt.LIVE_U32 + tt.LIVE_U64}
        link.last = zero
        now = dict(zero, bytes_txd=10_000_000_000, bytes_rxd=5_000_000_000, frames_txd=2_000_000, corr_cw=1000)
        row = ethmon.rates(link, now, 2.0)
        self.assertEqual(row["counter_status"], "ok")
        self.assertAlmostEqual(float(row["tx_gb_s"]), 5.0)
        self.assertAlmostEqual(float(row["tx_util_pct"]), 10.0)   # 5 GB/s = 40 Gb/s of 400
        self.assertEqual(ethmon.rates(link, dict(zero, bytes_txd=-1), 2.0)["counter_status"], "reset")


class EndToEndTests(unittest.TestCase):
    def test_card_monitor_on_fake_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            make_card(Path(d), 0, "3f")
            out = subprocess.run([sys.executable, str(TOOLKIT_DIR / "tt_card_monitor.py"), "--sysfs-root",
                                  str(Path(d) / "class"), "--stdout", "-w", "0.05", "-W", "0.2",
                                  "--holders-interval", "0"], capture_output=True, text=True, check=True)
            rows = list(csv.DictReader(io.StringIO(out.stdout)))
            self.assertGreaterEqual(len(rows), 3)
            self.assertEqual(rows[-1]["pcie_status"], "ok")
            self.assertEqual(rows[-1]["power_w"], "17.0")


if __name__ == "__main__":
    unittest.main()
