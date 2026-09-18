from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT_DIR))

import nvsmi_combined_monitor as combined  # noqa: E402
import nvsmi_gpu_monitor as gpu_monitor  # noqa: E402
import nvsmi_nvlink_bw_monitor as nvlink  # noqa: E402


class DmonTests(unittest.TestCase):
    def test_header_is_normalized(self) -> None:
        self.assertEqual(
            gpu_monitor.normalize_header(["# gpu", " pwr", " gtemp", " rxpci"]),
            ["gpu", "power_w", "gpu_temperature_c", "pcie_rx_mb_s"],
        )

    def test_command_selects_gpus_and_sample_count(self) -> None:
        args = gpu_monitor.build_parser().parse_args(["-g", "0,2", "-w", "2", "-W", "5"])
        command = gpu_monitor.build_command(args, [0, 2])
        self.assertEqual(command[command.index("-i") + 1], "0,2")
        self.assertEqual(command[command.index("-c") + 1], "3")


class NvlinkTests(unittest.TestCase):
    SAMPLE = """GPU 0: NVIDIA H200 (UUID: GPU-a)\n\n Link 0: Tx0: 1000000 KiB\n Link 0: Rx0: 2000000 KiB\n Link 1: Tx0: 3000000 KiB\n Link 1: Rx0: 4000000 KiB\nGPU 2: NVIDIA H200 (UUID: GPU-b)\n Link 0: Tx0: 50 KiB\n Link 0: Rx0: 70 KiB\n"""

    DRIVER_610_SAMPLE = """GPU 0: NVIDIA H200 (UUID: GPU-a)\n\t Link 0: Data Tx: 14984231343 KiB\n\t Link 0: Data Rx: 15105998777 KiB\n"""

    def test_parser_uses_gpu_and_link_ids(self) -> None:
        links = nvlink.parse_nvlink_counters(self.SAMPLE)
        self.assertEqual(set(links), {(0, 0), (0, 1), (2, 0)})
        self.assertEqual(links[(0, 1)].rx_kib, 4_000_000)

    def test_parser_accepts_driver_610_data_labels(self) -> None:
        link = nvlink.parse_nvlink_counters(self.DRIVER_610_SAMPLE)[(0, 0)]
        self.assertEqual(link.tx_kib, 14_984_231_343)
        self.assertEqual(link.rx_kib, 15_105_998_777)

    def test_rates_use_measured_elapsed_time(self) -> None:
        before = nvlink.Sample(10.0, 100.0, {(0, 0): nvlink.LinkCounter(0, 0, 0, 0)})
        after = nvlink.Sample(12.0, 102.0, {(0, 0): nvlink.LinkCounter(0, 0, 2_000_000, 1_000_000)})
        row = nvlink.bandwidth_rows(before, after, 50.0)[0]
        self.assertEqual(row["rx_gb_s"], "1.024000")
        self.assertEqual(row["tx_gb_s"], "0.512000")
        self.assertEqual(row["bidirectional_utilization_pct"], "1.536000")

    def test_counter_reset_has_blank_rates(self) -> None:
        before = nvlink.Sample(1, 1, {(0, 0): nvlink.LinkCounter(0, 0, 10, 20)})
        after = nvlink.Sample(2, 2, {(0, 0): nvlink.LinkCounter(0, 0, 5, 30)})
        row = nvlink.bandwidth_rows(before, after, 50)[0]
        self.assertEqual(row["counter_status"], "reset_or_wrap")
        self.assertEqual(row["rx_gb_s"], "")


class CombinedTests(unittest.TestCase):
    def test_physical_links_are_aggregated_per_gpu(self) -> None:
        fields = nvlink.CSV_FIELDS
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "links.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for link in (0, 1):
                    writer.writerow({
                        "timestamp_epoch": "10", "gpu": "0", "link": str(link),
                        "rx_gb_s": "10", "tx_gb_s": "5",
                        "per_direction_capacity_gb_s": "50", "counter_status": "ok",
                    })
            sample = combined.read_nvlink_samples(path)[0][0][1]
            self.assertEqual(sample["nvlink_rx_gb_s"], "20.000000")
            self.assertEqual(sample["nvlink_bidirectional_utilization_pct"], "15.000000")


if __name__ == "__main__":
    unittest.main()
