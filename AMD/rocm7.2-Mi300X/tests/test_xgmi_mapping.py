from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLKIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT_DIR))

import amdsmi_xgmi_bw_monitor as monitor  # noqa: E402


def synthetic_payload() -> dict[str, object]:
    metrics = []
    for source_gpu in range(8):
        links = []
        for raw_peer_gpu in range(8):
            if raw_peer_gpu == source_gpu:
                continue
            links.append(
                {
                    "gpu": raw_peer_gpu,
                    "bdf": f"0000:{raw_peer_gpu:02x}:00.0",
                    "read": {"value": source_gpu * 100 + raw_peer_gpu, "unit": "KB"},
                    "write": {"value": source_gpu * 1000 + raw_peer_gpu, "unit": "KB"},
                }
            )
        metrics.append(
            {
                "gpu": source_gpu,
                "bdf": f"0000:{source_gpu:02x}:00.0",
                "link_metrics": {"links": links},
            }
        )
    return {"xgmi_metric": metrics}


class PeerMappingTests(unittest.TestCase):
    def test_validated_profile_is_bijective_for_every_source(self) -> None:
        monitor.validate_peer_map(monitor.ROCM72_MI300X_RAW_TO_LOGICAL_PEER)
        for source_gpu, source_map in monitor.ROCM72_MI300X_RAW_TO_LOGICAL_PEER.items():
            expected = set(range(8)) - {source_gpu}
            self.assertEqual(set(source_map), expected)
            self.assertEqual(set(source_map.values()), expected)

    def test_all_raw_rows_remain_distinct_after_remapping(self) -> None:
        raw = monitor.parse_xgmi_links(synthetic_payload())
        remapped = monitor.remap_peer_counters(raw, "rocm72-mi300x")

        self.assertEqual(len(raw), 56)
        self.assertEqual(len(remapped), 56)
        self.assertEqual(len({counter.key for counter in remapped.values()}), 56)
        self.assertEqual(remapped[(0, 1)].raw_peer_gpu, 7)
        self.assertEqual(remapped[(0, 1)].peer_bdf, "0000:01:00.0")
        self.assertEqual(remapped[(1, 0)].raw_peer_gpu, 6)
        self.assertEqual(remapped[(1, 0)].peer_bdf, "0000:00:00.0")

    def test_bad_profile_that_would_collide_is_rejected(self) -> None:
        bad_map = {
            source: dict(source_map)
            for source, source_map in monitor.ROCM72_MI300X_RAW_TO_LOGICAL_PEER.items()
        }
        bad_map[0][7] = bad_map[0][1]
        with self.assertRaisesRegex(RuntimeError, "not a one-to-one"):
            monitor.validate_peer_map(bad_map)

    def test_none_profile_preserves_raw_identity(self) -> None:
        raw = monitor.parse_xgmi_links(synthetic_payload())
        unchanged = monitor.remap_peer_counters(raw, "none")
        self.assertIs(unchanged, raw)
        self.assertEqual(unchanged[(0, 1)].peer_mapping, "none")


if __name__ == "__main__":
    unittest.main()
