from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


TOOLKIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT_DIR))

import amdsmi_combined_monitor as combined  # noqa: E402
import amdsmi_gpu_monitor as gpu_monitor  # noqa: E402


class GpuMonitorTests(unittest.TestCase):
    def test_rocm72_header_is_normalized(self) -> None:
        header = "timestamp,gpu,xcp,power_usage,pcie_bw"
        self.assertEqual(
            gpu_monitor.normalize_csv_header(header),
            "timestamp,gpu,xcp,power_usage,pcie_bw_bidirectional_mbps",
        )

    def test_unexpected_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            gpu_monitor.normalize_csv_header("timestamp,gpu,power_usage")


class CombinedMonitorTests(unittest.TestCase):
    def test_combined_monitor_passes_mapping_profile(self) -> None:
        args = combined.build_parser().parse_args([])
        _gpu_command, xgmi_command = combined.monitor_commands(
            args, [], Path("gpu.csv"), Path("xgmi.csv")
        )
        position = xgmi_command.index("--peer-map")
        self.assertEqual(xgmi_command[position + 1], "rocm72-mi300x")

    def test_duplicate_logical_counter_is_rejected(self) -> None:
        fields = [
            "timestamp_epoch",
            "source_gpu",
            "peer_gpu",
            "bidirectional_utilization_pct",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xgmi.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp_epoch": "1.0",
                        "source_gpu": "0",
                        "peer_gpu": "1",
                        "bidirectional_utilization_pct": "10",
                    }
                )
                writer.writerow(
                    {
                        "timestamp_epoch": "1.0",
                        "source_gpu": "0",
                        "peer_gpu": "1",
                        "bidirectional_utilization_pct": "20",
                    }
                )
            with self.assertRaisesRegex(ValueError, "duplicate logical counter"):
                combined.read_xgmi_samples(path)


if __name__ == "__main__":
    unittest.main()
