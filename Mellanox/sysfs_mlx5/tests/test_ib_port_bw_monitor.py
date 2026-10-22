from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT_DIR))

import ib_port_bw_monitor as ib  # noqa: E402


def make_port(root: Path, device: str, port: int, *, rx: int, tx: int,
              rate: str = "400 Gb/sec (4X NDR)", state: str = "4: ACTIVE",
              link_layer: str = "InfiniBand") -> Path:
    port_dir = root / device / "ports" / str(port)
    (port_dir / "counters").mkdir(parents=True)
    (port_dir / "rate").write_text(rate + "\n")
    (port_dir / "state").write_text(state + "\n")
    (port_dir / "link_layer").write_text(link_layer + "\n")
    (port_dir / "counters" / "port_rcv_data").write_text(f"{rx}\n")
    (port_dir / "counters" / "port_xmit_data").write_text(f"{tx}\n")
    return port_dir


class DiscoveryTests(unittest.TestCase):
    def test_rate_is_parsed(self) -> None:
        self.assertEqual(ib.parse_rate("200 Gb/sec (4X HDR)"), 200.0)
        self.assertEqual(ib.parse_rate("53.125 Gb/sec (1X NDR)"), 53.125)
        self.assertIsNone(ib.parse_rate("garbage"))

    def test_inactive_and_ethernet_ports_are_skipped_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_port(root, "mlx5_0", 1, rx=0, tx=0)
            make_port(root, "mlx5_1", 1, rx=0, tx=0, state="1: DOWN")
            make_port(root, "mlx5_2", 1, rx=0, tx=0, link_layer="Ethernet")
            self.assertEqual([p.key for p in ib.discover_ports(root, None, False, False)], [("mlx5_0", 1)])
            self.assertEqual(len(ib.discover_ports(root, None, True, True)), 3)
            self.assertEqual(ib.discover_ports(root, ["mlx5_0"], False, False)[0].capacity_gb_s, 50.0)

    def test_missing_root_gives_no_ports(self) -> None:
        self.assertEqual(ib.discover_ports(Path("/nonexistent/infiniband"), None, False, False), [])


class RateTests(unittest.TestCase):
    PORT = ib.Port("mlx5_0", 1, Path("."), "InfiniBand", "4: ACTIVE", 400.0)

    def test_counter_units_are_four_bytes(self) -> None:
        before = ib.Sample(10.0, 100.0, {("mlx5_0", 1): (0, 0)})
        after = ib.Sample(12.0, 102.0, {("mlx5_0", 1): (5_000_000_000, 2_500_000_000)})
        row = ib.bandwidth_rows(before, after, [self.PORT], None)[0]
        self.assertEqual(row["rx_delta_bytes"], 20_000_000_000)
        self.assertEqual(row["rx_gb_s"], "10.000000")
        self.assertEqual(row["tx_gb_s"], "5.000000")
        self.assertEqual(row["rx_utilization_pct"], "20.000000")
        self.assertEqual(row["bidirectional_utilization_pct"], "15.000000")

    def test_capacity_override(self) -> None:
        before = ib.Sample(0.0, 0.0, {("mlx5_0", 1): (0, 0)})
        after = ib.Sample(1.0, 1.0, {("mlx5_0", 1): (2_500_000_000, 0)})
        row = ib.bandwidth_rows(before, after, [self.PORT], 25.0)[0]
        self.assertEqual(row["rx_utilization_pct"], "40.000000")

    def test_counter_reset_has_blank_rates(self) -> None:
        before = ib.Sample(1, 1, {("mlx5_0", 1): (10, 20)})
        after = ib.Sample(2, 2, {("mlx5_0", 1): (5, 30)})
        row = ib.bandwidth_rows(before, after, [self.PORT], None)[0]
        self.assertEqual(row["counter_status"], "reset_or_wrap")
        self.assertNotIn("rx_gb_s", row)


class CommandLineTests(unittest.TestCase):
    def test_fake_sysfs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sysfs"
            make_port(root, "mlx5_0", 1, rx=100, tx=200)
            output = Path(directory) / "ib.csv"
            result = subprocess.run(
                [sys.executable, str(TOOLKIT_DIR / "ib_port_bw_monitor.py"), "--sysfs-root", str(root),
                 "-w", "0.05", "-W", "0.3", "-o", str(output)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = output.read_text().splitlines()
            self.assertTrue(lines[0].startswith("timestamp_epoch,"))
            self.assertGreaterEqual(len(lines), 3)
            self.assertIn(",mlx5_0,1,InfiniBand,", lines[1])

    def test_no_ports_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(TOOLKIT_DIR / "ib_port_bw_monitor.py"), "--sysfs-root", directory, "--stdout"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertIn("no InfiniBand ports", result.stderr)


if __name__ == "__main__":
    unittest.main()
