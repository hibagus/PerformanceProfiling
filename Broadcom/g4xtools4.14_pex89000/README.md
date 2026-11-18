# Broadcom PEX89000 switch monitoring (G4XTOOLS 4.14)

### Per-port PCIe switch bandwidth, utilization and TLP counters with g4Xdiagnostics

`g4x_perf_monitor.py` samples every Broadcom PEX88xxx/PEX89xxx switch that `g4Xdiagnostics --list` reports. It records each switch port's traffic in both directions (`Ing` = into the switch, `Eg` = out of it):

- link utilization;
- total and payload bytes per second;
- average payload per TLP;
- raw TLP counters: posted headers and dwords (writes), non-posted headers and dwords (read requests), completion headers and dwords (read data), and DLLPs.

It also maps every switch port to its Linux endpoints (GPU, NIC, NVMe) and to the CPU root port above them.

Use it alongside `Intel/pcm202604_emeraldrapids` (the CPU side of the same links) and NVIDIA's NVLink counters. It sees what neither of them can:
- **Virtual-partition traffic:** with the switch in virtual (partitioned) mode, it shows which partition and uplink carries each GPU's traffic.
- **Payload size per TLP:** a 256 B maximum-payload GPU-to-GPU write looks different from small stores.
- **Congestion location:** whether a downstream (GPU) port or its host uplink saturates.

## Safety

`g4Xdiagnostics` is a full switch-firmware diagnostics suite. This tool issues only three commands:
- `--list`;
- `-i N port`;
- `-i N perf -loop L -delay D -stat S`.

Every one of them is read-only. The Broadcom README warns that SPI flash operations stop the switch ARM core and can drop the host link or halt the system, and that resets through the CLI are unsafe. **Do not run** `cli` (trace masks persist across reboots), `mem`, `counters -clear`, flash, erase or reset commands as part of monitoring. Diagnostic commands cannot run in parallel on one switch. Different switches are sampled concurrently by default; use `--serial` to sample them one after another.

## Prerequisites

- **The binary:** G4XTOOLS 4.14, `Linux/g4Xdiagnostics.x86_64`. Verify it with the bundled key:
  `gpg --import pubKey.asc && gpg --verify g4Xdiagnostics.x86_64.sig g4Xdiagnostics.x86_64`.
  Install it root-owned, for example as `/usr/local/sbin/g4Xdiagnostics`.
- **The interface:** in-band SCSI through the switch management endpoint (`1000:00b2`, "PCIe Switch management endpoint"), bound to `mpt3sas`. The `SliffDriver` kernel module in the package is only for `g4Xrecovery`; it is not needed here.
- **Root:** the binary needs root. The tool calls it with `sudo -n` when it is not already root (`--sudo` / `--no-sudo` override this). For least privilege, give users a validating wrapper instead of the binary, and point `--binary` at it. The wrapper should accept only `--list`, `-i N port [-p P]` and `-i N perf -loop L -delay D -stat S`. One example is `tools/pcienvlink-g4x` in the PCIe-vs-NVLink repository (`hibagus/PCIe_vs_NVLink`).

## Usage

```bash
# switch-port -> GPU/NIC/root-port map only
python3 g4x_perf_monitor.py --topology-only -o runs/xe9680l_switches.csv

# 30 one-second samples of all switches, through a sudo-enabled wrapper
python3 g4x_perf_monitor.py --binary /usr/local/sbin/pcienvlink-g4x -n 30 -w 1000 -o runs/exp1/g4x.csv

# only switch 1, 200 ms samples
python3 g4x_perf_monitor.py -s 1 -n 50 -w 200
```

Outputs:
- `<output>.csv`: one row per sample × switch × port × direction.
- `<output>_topology.csv`: the port map.
- `<output>_raw/`: the native text, for audit.

Samples are timestamped as their output arrives (the tool runs on a pty, so its output is line-buffered).

## Dell PowerEdge XE9680L (8× H200 SXM) validation topology

Four PEX89144 B0 switches (firmware 04.160.03.10), each in virtual mode with **two partitions**. Each partition has its own x16 Gen5 host port to a separate CPU root port, and one H200 on an x16 Gen5 downstream port. GPU-to-GPU PCIe traffic therefore always goes up to the CPU, even between the two GPUs on one physical switch. The two GPUs on a switch are always on the same CPU socket. Live capture, 2026-11-18:

| Switch (mgmt PCI) | Host port → CPU root port | GPU downstream port (slot) → GPU (nvidia-smi index) |
| --- | --- | --- |
| 1 (`1c:00`) | 128 → `15:01.0` | 112 (slot 28) → `1b:00.0` (GPU0) |
| 1 (`1c:00`) | 80 → `37:01.0` | 96 (slot 24) → `3c:00.0` (GPU1) |
| 2 (`60:00`) | 128 → `48:01.0` | 48 (slot 27) → `4b:00.0` (GPU2) |
| 2 (`60:00`) | 16 → `59:01.0` | 32 (slot 23) → `5c:00.0` (GPU3) |
| 3 (`9d:00`) | 16 → `97:01.0` | 32 (slot 21) → `9a:00.0` (GPU4) |
| 3 (`9d:00`) | 128 → `b7:01.0` | 48 (slot 25) → `bb:00.0` (GPU5) |
| 4 (`ce:00`) | 96 → `c7:01.0` | 64 (slot 26) → `cd:00.0` (GPU6) |
| 4 (`ce:00`) | 128 → `d7:01.0` | 80 (slot 22) → `dc:00.0` (GPU7) |

Port numbers differ between switches, so always take them from the topology CSV. **Do not map ports by bus number.** The switch reports partition-local bus ranges, which only sometimes coincide with Linux bus numbers. The tool maps by physical slot (`PhysSlot` → `/sys/bus/pci/slots/<n>/address`).

Max payload size on the GPU ports is 256 B (MPSS 2048), so a GPU-to-GPU write at full efficiency shows `payload_avg_tlp_B` ≈ 256.

## Output columns (perf CSV)

| Column | Meaning |
| --- | --- |
| `link_util_pct` | Link utilization in the sample (from the friendly display) |
| `total_data_B`, `total_rate_Bps` | All TLP bytes in the sample, and per second |
| `payload_B`, `payload_rate_Bps`, `payload_avg_tlp_B` | Payload bytes, payload per second, and average payload per TLP |
| `cur_PH`, `cur_PDW` | Posted (write) TLP headers and payload dwords |
| `cur_NPH`, `cur_NPDW` | Non-posted (read request) headers and dwords |
| `cur_CplH`, `cur_CplW` | Completion (read data) headers and dwords |
| `cur_DLLP` | DLLP count |
| `cur_posted_payload_B`, `cur_posted_avg_B` | `cur_PDW` × 4 bytes, and that divided by `cur_PH`: the average write payload |
| `pre_*` | The tool's "Pre-counter" snapshot for the same sample |
| `endpoints`, `root_port` | The topology join |
