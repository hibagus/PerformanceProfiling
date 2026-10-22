from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT_DIR))

import run_marker  # noqa: E402
import window_summary as ws  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MarkerTests(unittest.TestCase):
    def test_markers_pair_into_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.csv"
            run_marker.append_marker(path, "start", "a", {"gpus": "8"}, epoch=10.0)
            run_marker.append_marker(path, "start", "b", {}, epoch=11.0)
            run_marker.append_marker(path, "end", "a", {}, epoch=20.0)
            run_marker.append_marker(path, "end", "b", {}, epoch=15.0)
            windows = ws.read_windows(path)
            self.assertEqual([(w.label, w.start, w.end) for w in windows], [("a", 10.0, 20.0), ("b", 11.0, 15.0)])
            self.assertEqual(path.read_text().splitlines()[0], ",".join(run_marker.FIELDS))

    def test_wrap_records_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.csv"
            status = subprocess.call([sys.executable, str(TOOLKIT_DIR / "run_marker.py"), str(path), "wrap", "run1",
                                      "--meta", "phase=test", "--", sys.executable, "-c", "raise SystemExit(3)"])
            self.assertEqual(status, 3)
            rows = list(csv.DictReader(path.open()))
            self.assertEqual([r["event"] for r in rows], ["start", "end"])
            self.assertIn('"exit_status": "3"', rows[1]["meta"])

    def test_meta_must_be_key_value(self) -> None:
        with self.assertRaises(ValueError):
            run_marker.parse_meta(["novalue"])


class SummaryTests(unittest.TestCase):
    def test_overlap_weighting(self) -> None:
        window = ws.Window("w", 10.0, 12.0, "")
        # spans (9,10] outside, (10,11] inside at 4 GB/s, (11,13] half inside at 2 GB/s
        points = [ws.Point(10.0, 1.0, 100.0, 50.0), ws.Point(11.0, 1.0, 4.0, 50.0), ws.Point(13.0, 2.0, 2.0, 50.0)]
        summary = ws.summarize(window, points)
        self.assertEqual(summary["gbytes"], "6.000000")      # 4*1 + 2*1
        self.assertEqual(summary["mean_gb_s"], "3.000000")   # 6 GB over 2 s
        self.assertEqual(summary["peak_gb_s"], "4.000000")
        self.assertEqual(summary["mean_utilization_pct"], "6.000")
        self.assertEqual(summary["coverage_pct"], "100.0")
        self.assertEqual(summary["samples"], 2)

    def test_end_to_end_with_all_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            run_marker.append_marker(data / "markers.csv", "start", "run", {}, epoch=100.0)
            run_marker.append_marker(data / "markers.csv", "end", "run", {}, epoch=101.0)
            write_csv(data / "node1_ib.csv", ["timestamp_epoch", "interval_seconds", "device", "port", "rx_gb_s",
                                              "tx_gb_s", "per_direction_capacity_gb_s", "counter_status"],
                      [{"timestamp_epoch": 100.5, "interval_seconds": 0.5, "device": "mlx5_0", "port": 1,
                        "rx_gb_s": 10, "tx_gb_s": 20, "per_direction_capacity_gb_s": 50, "counter_status": "ok"},
                       {"timestamp_epoch": 101.0, "interval_seconds": 0.5, "device": "mlx5_0", "port": 1,
                        "rx_gb_s": 30, "tx_gb_s": 20, "per_direction_capacity_gb_s": 50, "counter_status": "ok"}])
            write_csv(data / "node1_nvlink.csv", ["timestamp_epoch", "interval_seconds", "gpu", "link", "rx_gb_s",
                                                  "tx_gb_s", "per_direction_capacity_gb_s", "counter_status"],
                      [{"timestamp_epoch": 101.0, "interval_seconds": 1.0, "gpu": 0, "link": link, "rx_gb_s": 5,
                        "tx_gb_s": 1, "per_direction_capacity_gb_s": 50, "counter_status": "ok"} for link in (0, 1)])
            write_csv(data / "node1_dmon.csv", ["timestamp_epoch", "gpu", "pcie_rx_mb_s", "pcie_tx_mb_s"],
                      [{"timestamp_epoch": 100.0, "gpu": 0, "pcie_rx_mb_s": 0, "pcie_tx_mb_s": 0},
                       {"timestamp_epoch": 101.0, "gpu": 0, "pcie_rx_mb_s": 3000, "pcie_tx_mb_s": 1000}])
            out = data / "summary.csv"
            result = subprocess.run([sys.executable, str(TOOLKIT_DIR / "window_summary.py"), "--markers",
                                     str(data / "markers.csv"), "--data-dir", str(data), "-o", str(out)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = {(r["source"], r["entity"], r["direction"]): r for r in csv.DictReader(out.open())}
            self.assertEqual(rows[("ib", "mlx5_0/1", "rx")]["mean_gb_s"], "20.000000")
            self.assertEqual(rows[("nvlink", "gpu0", "rx")]["mean_gb_s"], "10.000000")
            self.assertEqual(rows[("nvlink", "gpu0", "rx")]["capacity_gb_s"], "100.000")
            self.assertEqual(rows[("pcie", "gpu0", "rx")]["mean_gb_s"], "3.000000")


class NodeMonitorTests(unittest.TestCase):
    def test_dry_run_lists_all_monitors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(TOOLKIT_DIR / "node_link_monitor.py"), "--output-dir",
                                     directory, "--host", "nodeA", "--dry-run"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("nodeA_ib.csv", "nodeA_nvlink.csv", "nodeA_dmon.csv"):
                self.assertIn(name, result.stderr)


if __name__ == "__main__":
    unittest.main()
