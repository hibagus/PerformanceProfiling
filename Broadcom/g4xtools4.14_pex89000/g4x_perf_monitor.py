# (C) 2026 Bagus Hanindhito, Dell Technologies Inc.
"""Per-port PCIe switch traffic for Broadcom PEX88xxx/PEX89xxx switches with g4Xdiagnostics.

Runs the read-only `port` and `perf` commands of Broadcom g4Xdiagnostics (G4XTOOLS 4.14)
on every switch it lists, and writes:

  <output>.csv            one row per sample x switch x port x direction (Ing/Eg): TLP
                          counters, link utilization, total and payload rates, average
                          payload per TLP
  <output>_topology.csv   switch port -> role, link, host port, attached Linux endpoints
                          (GPU/NIC/NVMe BDFs) and the CPU root port above them
  <output>_raw/           the native text output, for audit

Only `--list`, `port` and `perf` are ever issued. The tool never runs `cli`, `mem`,
`counters -clear`, flash, erase or reset commands (see README.md).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COUNTER_FIELDS = ("PH", "PDW", "NPH", "NPDW", "CplH", "CplW", "DLLP")
ROW = re.compile(r"^\s*(\d+)\s+(Ing|Eg)\s+(.*)$")
UNIT = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def base_command(args) -> list[str]:
    cmd = [args.binary]
    if args.sudo_mode == "always" or (args.sudo_mode == "auto" and os.geteuid() != 0):
        cmd = ["sudo", "-n"] + cmd
    return cmd


def run_text(cmd: list[str], timeout: float) -> str:
    out = subprocess.run(cmd, cwd="/tmp", capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {out.returncode}: {out.stderr.strip()[-400:]}")
    return out.stdout


def list_switches(args) -> list[dict]:
    text = run_text(base_command(args) + ["--list"], 120)
    sw = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+([0-9A-Fa-f:]+)\s+(\S+)", line)
        if m and m.group(7) == "Switch":
            sw.append(dict(index=int(m.group(1)), device=m.group(2), rev=m.group(3), wwid=m.group(4),
                           fw=m.group(5), mgmt_pci=m.group(6).lower()))
    return sw


# ---------- topology ----------

def linux_slots() -> dict[str, str]:
    """Linux physical slot number -> bridge BDF (the switch downstream port), from sysfs."""
    slots = {}
    for d in glob.glob("/sys/bus/pci/slots/*"):
        try:
            addr = open(os.path.join(d, "address")).read().strip()
        except OSError:
            continue
        slots[os.path.basename(d)] = addr + ".0" if addr.count(".") == 0 else addr
    return slots


def linux_endpoints() -> list[dict]:
    eps = []
    for d in glob.glob("/sys/bus/pci/devices/*"):
        try:
            cls = open(os.path.join(d, "class")).read().strip()
        except OSError:
            continue
        if cls.startswith("0x0604"):                    # bridges are not endpoints
            continue
        path = [p for p in os.path.realpath(d).split("/")
                if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d", p)]
        eps.append(dict(bdf=os.path.basename(d), cls=cls, path=path, root_port=path[0] if path else ""))
    return eps


def parse_port(text: str, sw: dict, eps: list[dict], slots: dict[str, str]) -> list[dict]:
    ports, hosts, ds = {}, {}, {}
    section = "summary"
    for line in text.splitlines():
        if line.startswith("Host Ports"):
            section = "host"
        elif line.startswith("Downstream Ports"):
            section = "down"
        parts = line.split()
        if section == "summary" and len(parts) >= 11 and parts[0].isdigit() and parts[1].isdigit():
            # Stn Port Type MRR MPS MPSS Speed/ Gen Width/ x ClkMode PBus Stat
            m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*/\s*(\S+)\s+(\S+)\s*/\s*(\S+)"
                         r"\s+(\S+)\s+(\S+)\s+(\S+)", line)
            if m:
                ports[int(m.group(2))] = dict(station=int(m.group(1)), role=m.group(3), mps=int(m.group(5)),
                                              speed=f"{m.group(7)}/{m.group(8)}", width=f"{m.group(9)}/{m.group(10)}",
                                              status=m.group(13))
        elif section == "host" and len(parts) >= 4 and parts[0].isdigit():
            hosts[int(parts[0])] = [int(x) for x in re.findall(r"\d+", " ".join(parts[4:]))]
        elif section == "down" and len(parts) >= 4 and parts[0].isdigit():
            m = re.match(r"^\s*(\d+)\s+([0-9A-Fa-f]+)\s*->\s*([0-9A-Fa-f]+)\s+(\S+)\s+(\d+)", line)
            if m:
                ds[int(m.group(1))] = dict(lo=int(m.group(2), 16), hi=int(m.group(3), 16),
                                           slot=m.group(4), host=int(m.group(5)))
    rows = []
    for p, info in sorted(ports.items()):
        r = dict(switch=sw["index"], switch_mgmt_pci=sw["mgmt_pci"], port=p, **info,
                 host_port="", ds_buses="", phys_slot="", linux_bridge="", endpoints="", root_port="")
        if p in ds:
            d = ds[p]
            # PhysSlot -> Linux slot address = this downstream port's bridge; endpoints below it
            bridge = slots.get(str(d["slot"]), "")
            att = [e for e in eps if bridge and bridge in e["path"][:-1]]
            r.update(host_port=d["host"], ds_buses=f"{d['lo']:02x}-{d['hi']:02x}", phys_slot=d["slot"],
                     linux_bridge=bridge,
                     endpoints=" ".join(f"{e['bdf']}({e['cls']})" for e in att),
                     root_port=" ".join(sorted({e["root_port"] for e in att})))
        elif p in hosts:
            r.update(host_port=p)
        rows.append(r)
    # a host port's CPU root port = the root port above its downstream endpoints
    for r in rows:
        if r["role"] == "Host":
            r["root_port"] = " ".join(sorted({x for q in rows if q["host_port"] == r["port"] and q is not r
                                              for x in q["root_port"].split()}))
            r["endpoints"] = "downstream ports " + " ".join(str(q["port"]) for q in rows
                                                            if q["host_port"] == r["port"] and q is not r)
    return rows


# ---------- perf ----------

def to_bytes(tok: str) -> float:
    m = re.fullmatch(r"([0-9.]+)\s*([KMGT]?i?B)(/S)?", tok)
    return float(m.group(1)) * UNIT[m.group(2)] if m else float("nan")


def read_stream(cmd: list[str], raw_path: Path, on_line) -> int:
    """Run cmd on a pty (line-buffered output, so samples are stamped as they arrive)."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(cmd, cwd="/tmp", stdin=subprocess.DEVNULL, stdout=slave, stderr=slave)
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


def perf_switch(args, sw: dict, raw_dir: Path) -> list[dict]:
    cmd = base_command(args) + ["-i", str(sw["index"]), "perf", "-loop", str(args.samples),
                                "-delay", str(args.delay_ms), "-stat", str(args.interval_ms)]
    rows: dict[tuple, dict] = {}
    state = dict(loop=0, section=None, stat_seen=0)

    def on_line(line: str):
        s = state
        if line.startswith("Pre-counter"):
            s.update(loop=s["loop"] + 1, section="pre", stat_seen=0, stamp=utcnow())
            return
        if line.startswith("Current counter"):
            s["section"] = "cur"; return
        if line.startswith("Statistic Friendly"):
            s["section"] = "friendly"; return
        if line.startswith("Statistic"):
            s["stat_seen"] += 1
            s["section"] = "util" if s["stat_seen"] == 1 else "payload"; return
        m = ROW.match(line)
        if not m or s["section"] is None or s["loop"] == 0:
            return
        key = (s["loop"], int(m.group(1)), m.group(2))
        r = rows.setdefault(key, dict(utc_time=s["stamp"], sample=s["loop"], switch=sw["index"],
                                      switch_mgmt_pci=sw["mgmt_pci"], port=key[1], direction=key[2]))
        vals = m.group(3).split()
        sec = s["section"]
        if sec in ("pre", "cur") and len(vals) >= 7:
            for f, v in zip(COUNTER_FIELDS, vals):
                r[f"{sec}_{f}"] = int(v, 16)
        elif sec == "util" and len(vals) >= 3:
            r["total_data_B"], r["total_rate_Bps"] = int(vals[1]), int(vals[2])
        elif sec == "payload" and len(vals) >= 3:
            r["payload_B"], r["payload_rate_Bps"], r["payload_avg_tlp_B"] = (int(v) for v in vals[:3])
        elif sec == "friendly":
            mu = re.match(r"\s*([0-9.]+)%", m.group(3))
            if mu:
                r["link_util_pct"] = float(mu.group(1))

    rc = read_stream(cmd, raw_dir / f"perf_switch{sw['index']}.txt", on_line)
    if rc != 0:
        print(f"warning: switch {sw['index']} perf exited {rc} (see {raw_dir})", file=sys.stderr)
    for r in rows.values():
        if "cur_PDW" in r:
            r["cur_posted_payload_B"] = r["cur_PDW"] * 4
            r["cur_posted_avg_B"] = round(r["cur_PDW"] * 4 / r["cur_PH"], 1) if r["cur_PH"] else 0
            r["cur_cpl_payload_B"] = r["cur_CplW"] * 4
    return list(rows.values())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-port PCIe switch bandwidth, utilization and TLP counters "
                                            "for Broadcom PEX88xxx/PEX89xxx switches via g4Xdiagnostics.")
    p.add_argument("-w", "--interval-ms", type=int, default=1000, help="sample interval in ms (default 1000)")
    p.add_argument("-n", "--samples", type=int, default=10, help="number of samples per switch (default 10)")
    p.add_argument("--delay-ms", type=int, default=0, help="delay before the first sample in ms")
    p.add_argument("-s", "--switch", type=int, action="append",
                   help="switch index from --list (repeatable; default: all switches)")
    p.add_argument("-o", "--output", help="CSV path (default: <UTC timestamp>_g4x_perf.csv)")
    p.add_argument("--binary", default=os.environ.get("G4X_BIN") or shutil.which("g4Xdiagnostics")
                   or "/usr/local/sbin/g4Xdiagnostics",
                   help="g4Xdiagnostics executable or a validating wrapper with the same syntax "
                        "(default: $G4X_BIN, PATH, /usr/local/sbin/g4Xdiagnostics)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sudo", dest="sudo_mode", action="store_const", const="always", help="always use sudo -n")
    g.add_argument("--no-sudo", dest="sudo_mode", action="store_const", const="never", help="never use sudo")
    p.set_defaults(sudo_mode="auto")
    p.add_argument("--serial", action="store_true",
                   help="sample switches one after another instead of concurrently")
    p.add_argument("--topology-only", action="store_true", help="write the topology CSV and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not (1 <= args.samples <= 600 and 100 <= args.interval_ms <= 10000 and 0 <= args.delay_ms <= 10000):
        sys.exit("samples 1-600, interval 100-10000 ms, delay 0-10000 ms")
    out = Path(args.output or f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_g4x_perf.csv").resolve()
    raw_dir = out.with_name(out.stem + "_raw"); raw_dir.mkdir(parents=True, exist_ok=True)
    switches = list_switches(args)
    if args.switch:
        switches = [s for s in switches if s["index"] in args.switch]
    if not switches:
        sys.exit("no switches found (check --binary / privileges)")
    eps, slots = linux_endpoints(), linux_slots()
    topo = []
    for sw in switches:
        text = run_text(base_command(args) + ["-i", str(sw["index"]), "port"], 120)
        (raw_dir / f"port_switch{sw['index']}.txt").write_text(text)
        topo += parse_port(text, sw, eps, slots)
    tpath = out.with_name(out.stem + "_topology.csv")
    with open(tpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(topo[0].keys())); w.writeheader(); w.writerows(topo)
    print(f"topology: {tpath} ({len(topo)} ports on {len(switches)} switches)")
    if args.topology_only:
        return 0
    print(f"sampling {len(switches)} switch(es): {args.samples} x {args.interval_ms} ms -> {out}")
    t0 = time.time()
    if args.serial:
        results = [perf_switch(args, sw, raw_dir) for sw in switches]
    else:
        with ThreadPoolExecutor(len(switches)) as ex:
            results = list(ex.map(lambda sw: perf_switch(args, sw, raw_dir), switches))
    rows = [r for res in results for r in res]
    meta = {(t["switch"], t["port"]): t for t in topo}
    for r in rows:
        t = meta.get((r["switch"], r["port"]), {})
        r["role"], r["endpoints"], r["root_port"] = t.get("role", ""), t.get("endpoints", ""), t.get("root_port", "")
    fields = ["utc_time", "sample", "switch", "switch_mgmt_pci", "port", "direction", "role", "link_util_pct",
              "total_data_B", "total_rate_Bps", "payload_B", "payload_rate_Bps", "payload_avg_tlp_B",
              "cur_posted_payload_B", "cur_posted_avg_B", "cur_cpl_payload_B"] + \
             [f"{s}_{f}" for s in ("pre", "cur") for f in COUNTER_FIELDS] + ["endpoints", "root_port"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["sample"], r["switch"], r["port"], r["direction"])))
    print(f"wrote {len(rows)} rows in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
