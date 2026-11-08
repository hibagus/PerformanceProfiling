#!/usr/bin/env python3
# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Check the PCIe NIU counters (tt_card_monitor.py) against transfers of known size.

Run with tt-metal's Python environment (needs ``ttnn`` and ``torch``) on an otherwise idle card:

  TT_VISIBLE_DEVICES=3 python validate_pcie_counters.py --card 3 --mib 256 --reps 3
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_VISIBLE_DEVICES=3 python validate_pcie_counters.py --card 3 --mib 256

For each repetition it writes a row-major BF16 tensor of --mib MiB to the card's DRAM (``ttnn.from_torch``) and
reads it back (``ttnn.to_torch``), reading the card's 12 counters from sysfs just before and after each step.
It prints the bytes each counter family moved divided by the tensor's bytes. With fast dispatch the device pulls
host->device data (expected: slv_rd ~ 1.0) and pushes device->host data (slv_wr ~ 1.0); with slow dispatch the
host does both through TLB windows (mst_wr ~ 1.0, mst_rd ~ 1.0). Ratios slightly above 1 are command and
telemetry traffic. The ratios use tt_card_monitor's units (64 B per slv count, --mst-flit-bytes per mst
count), so a ratio near 0.5 or 2 means a unit is wrong for this host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tt_card_monitor import FAMILIES, MST_FLIT_BYTES, PCIE_RAW, SLV_FLIT_BYTES, WRAP  # noqa: E402


def counters(card: int) -> dict[str, int]:
    base = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/pcie_perf_counters")
    return {n: int((base / n).read_text()) for n in PCIE_RAW}


def family_bytes(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {fam.replace("_gb_s", ""): sum(((after[f"{n}{noc}"] - before[f"{n}{noc}"]) % WRAP)
                                         * (MST_FLIT_BYTES if fam.startswith("mst") else SLV_FLIT_BYTES)
                                         for n in names for noc in (0, 1))
            for fam, names in FAMILIES.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--card", type=int, required=True, help="sysfs index of the card (tenstorrent!N)")
    ap.add_argument("--device-id", type=int, default=0, help="ttnn device id (0 under TT_VISIBLE_DEVICES=N)")
    ap.add_argument("--mib", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    import torch  # noqa: PLC0415 - only needed here
    import ttnn  # noqa: PLC0415

    mode = "slow" if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") else "fast"
    n = a.mib * (1 << 20) // 2
    rows = n // 1024
    host = torch.randn(rows, 1024, dtype=torch.float32).to(torch.bfloat16)
    nbytes = rows * 1024 * 2
    dev = ttnn.open_device(device_id=a.device_id)
    results = []
    try:
        for rep in range(a.reps):
            c0 = counters(a.card)
            t0 = time.perf_counter()
            dt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            c1 = counters(a.card)
            back = ttnn.to_torch(dt)
            t2 = time.perf_counter()
            c2 = counters(a.card)
            ok = bool(torch.equal(back.reshape(host.shape), host))
            ttnn.deallocate(dt)
            for step, before, after, secs in (("h2d", c0, c1, t1 - t0), ("d2h", c1, c2, t2 - t1)):
                fam = family_bytes(before, after)
                r = {"mode": mode, "rep": rep, "step": step, "bytes": nbytes, "seconds": round(secs, 4),
                     "app_gb_s": round(nbytes / secs / 1e9, 3), "data_ok": ok,
                     **{f"{k}_ratio": round(v / nbytes, 4) for k, v in fam.items()}}
                results.append(r)
                print("PCIEVAL " + json.dumps(r), flush=True)
    finally:
        ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
