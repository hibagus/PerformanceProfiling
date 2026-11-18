# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""NVSwitch and per-GPU fabric telemetry with DCGM (dcgmi dmon).

Samples, at a fixed interval:
  NVSwitch (every switch DCGM discovers): cumulative TX/RX bytes (fields 1695/1696), turned
      into bytes/s per interval; temperature (858); rail power (705-707).
  GPU (every GPU): power (155), SM clock (100), NVLink throughput total (449) and, with
      --gpu-prof, DCGM profiling metrics: PCIe TX/RX bytes/s (1009/1010), NVLink TX/RX
      bytes/s (1011/1012), and GR-engine/SM/DRAM activity (1001/1002/1005).

Writes <output>.csv (one row per sample x entity, one column per field, plus *_Bps rates for
the cumulative byte counters) and <output>_raw/ (native dmon text).

Profiling fields (--gpu-prof) use the same GPU performance-monitor hardware as Nsight Systems
GPU metrics and Nsight Compute; on a shared host enable them only for your own runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

SWITCH_FIELDS = [1695, 1696, 858, 705, 706, 707]
GPU_FIELDS = [155, 100, 449]
GPU_PROF_FIELDS = [1009, 1010, 1011, 1012, 1001, 1002, 1005]
CUMULATIVE = {1695: "nvswitch_tx", 1696: "nvswitch_rx"}     # bytes since boot -> rate
LINE = re.compile(r"^(GPU|Switch)\s+(\d+)\s+(.*)$")
PROCS: list = []          # running dmon children, stopped on SIGINT/SIGTERM


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def field_names(dcgmi: str) -> dict[int, str]:
    out = subprocess.run([dcgmi, "dmon", "-l"], capture_output=True, text=True, timeout=60).stdout
    names = {}
    for line in out.splitlines():
        m = re.match(r"^(\S+)\s+.*?(\d+)\s*$", line)
        if m:
            names[int(m.group(2))] = m.group(1)
    return names


def stream(cmd: list[str], raw_path: Path, on_line) -> int:
    """Run cmd on a pty so each sample is stamped when it is printed."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave)
    PROCS.append(proc)
    os.close(slave)
    buf = b""
    with open(raw_path, "wb") as raw:
        while True:
            r, _, _ = select.select([master], [], [], 1.0)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                raw.write(chunk)
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    on_line(ln.decode(errors="replace").rstrip("\r"))
            elif proc.poll() is not None:
                break
    if buf:
        on_line(buf.decode(errors="replace"))
    os.close(master)
    return proc.wait()


def sample_group(dcgmi: str, kind: str, ids: str, fields: list[int], args, raw_dir: Path, rows: list, lock):
    entity = "nvswitch" if kind == "Switch" else "gpu"
    spec = ",".join(f"{entity}:{i}" for i in ids.split(",")) if ids != "all" else None
    cmd = [dcgmi, "dmon", "-e", ",".join(map(str, fields)), "-d", str(args.interval_ms), "-c", str(args.samples)]
    if spec:
        cmd += ["-i", spec]
    elif kind == "Switch":
        cmd += ["-i", ",".join(f"nvswitch:{i}" for i in range(args.max_switches))]
    else:
        cmd += ["-i", ",".join(f"gpu:{i}" for i in range(args.max_gpus))]
    seen, sample = set(), 0

    def on_line(line: str):
        nonlocal sample, seen
        m = LINE.match(line)
        if not m or m.group(1) != kind:
            return
        eid = int(m.group(2))
        if eid in seen or not seen:          # a repeated entity id starts the next sample
            if eid in seen:
                seen = set()
            if not seen:
                sample += 1
        seen.add(eid)
        vals = m.group(3).split()
        row = dict(utc_time=utcnow().isoformat(timespec="milliseconds"), sample=sample, entity=entity, id=eid)
        for f, v in zip(fields, vals):
            row[f] = None if v in ("N/A", "N/S") else float(v)
        with lock:
            rows.append(row)

    rc = stream(cmd, raw_dir / f"dmon_{entity}.txt", on_line)
    if rc != 0:
        print(f"warning: dcgmi dmon ({entity}) exited {rc}; see {raw_dir}", file=sys.stderr)


def add_rates(rows: list[dict]):
    last: dict[tuple, tuple] = {}
    for r in sorted(rows, key=lambda r: (r["entity"], r["id"], r["sample"])):
        t = dt.datetime.fromisoformat(r["utc_time"])
        for f, name in CUMULATIVE.items():
            if r.get(f) is None:
                continue
            key = (r["entity"], r["id"], f)
            if key in last:
                t0, v0 = last[key]
                secs = (t - t0).total_seconds()
                r[f"{name}_Bps"] = round((r[f] - v0) / secs) if secs > 0 else None
            last[key] = (t, r[f])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NVSwitch and GPU fabric telemetry with DCGM dmon.")
    p.add_argument("-w", "--interval-ms", type=int, default=1000, help="sample interval in ms (default 1000)")
    p.add_argument("-n", "--samples", type=int, default=10,
                   help="number of samples (default 10); SIGINT/SIGTERM stops early and still writes the CSV")
    p.add_argument("-o", "--output", help="CSV path (default: <UTC timestamp>_dcgm_fabric.csv)")
    p.add_argument("--switches", default="all", help="NVSwitch ids, comma-separated (default: all)")
    p.add_argument("--gpus", default="all", help="GPU ids, comma-separated (default: all); 'none' to skip")
    p.add_argument("--gpu-prof", action="store_true",
                   help="add DCGM profiling fields (PCIe/NVLink bytes/s, engine activity) for GPUs")
    p.add_argument("--max-switches", type=int, default=4, help="ids probed when --switches all (default 4)")
    p.add_argument("--max-gpus", type=int, default=8, help="ids probed when --gpus all (default 8)")
    p.add_argument("--dcgmi", default=shutil.which("dcgmi") or "/usr/bin/dcgmi", help="dcgmi executable")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not (1 <= args.samples <= 100000 and 100 <= args.interval_ms <= 60000):
        sys.exit("samples 1-100000, interval 100-60000 ms")
    out = Path(args.output or f"{utcnow():%Y%m%dT%H%M%SZ}_dcgm_fabric.csv").resolve()
    raw_dir = out.with_name(out.stem + "_raw"); raw_dir.mkdir(parents=True, exist_ok=True)
    names = field_names(args.dcgmi)
    rows: list[dict] = []
    lock = threading.Lock()
    jobs = [("Switch", args.switches, SWITCH_FIELDS)]
    if args.gpus != "none":
        jobs.append(("GPU", args.gpus, GPU_FIELDS + (GPU_PROF_FIELDS if args.gpu_prof else [])))
    def stop(signum, _frame):        # end sampling early but still write the CSV
        for pr in PROCS:
            if pr.poll() is None:
                pr.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    threads = [threading.Thread(target=sample_group, args=(args.dcgmi, k, ids, f, args, raw_dir, rows, lock))
               for k, ids, f in jobs]
    print(f"sampling {args.samples} x {args.interval_ms} ms -> {out}")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    add_rates(rows)
    fields = sorted({f for r in rows for f in r if isinstance(f, int)})
    rate_cols = [f"{n}_Bps" for n in CUMULATIVE.values()]
    header = ["utc_time", "sample", "entity", "id"] + [f"{names.get(f, 'field')}_{f}" for f in fields] + rate_cols
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in sorted(rows, key=lambda r: (r["sample"], r["entity"], r["id"])):
            w.writerow([r["utc_time"], r["sample"], r["entity"], r["id"]] + [r.get(f) for f in fields] +
                       [r.get(c) for c in rate_cols])
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
