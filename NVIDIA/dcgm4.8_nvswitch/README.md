# DCGM NVSwitch and GPU fabric monitor (DCGM 4.8)

### Per-NVSwitch traffic, temperature and power, plus per-GPU NVLink/PCIe bytes, with `dcgmi dmon`

`dcgm_fabric_monitor.py` samples, at a fixed interval:
- **every NVSwitch DCGM discovers:** cumulative TX/RX bytes, turned into bytes/s per interval, plus temperature and rail power;
- **every GPU:** power, SM clock and total NVLink throughput.

With `--gpu-prof` it adds DCGM profiling metrics for each GPU: PCIe and NVLink TX/RX bytes/s, and graphics-engine, SM and DRAM activity. It complements:
- `Broadcom/g4xtools4.14_pex89000` (PCIe switch ports);
- `Intel/pcm202604_emeraldrapids` (CPU IIO root ports and UPI).

Together the three cover NVLink, PCIe switch and CPU.

## Prerequisites

DCGM 4.8 and the NVSwitch Configuration and Query library (NSCQ). **The NSCQ version must match the driver exactly.** On the XE9680L (driver 610.57.04, Ubuntu 22.04, CUDA apt repository):

```bash
sudo apt-get install -y datacenter-gpu-manager-4-cuda13 libnvidia-nscq=610.57.04-1ubuntu1
sudo apt-mark hold libnvidia-nscq          # release it when the driver is upgraded
sudo systemctl enable --now nvidia-dcgm
dcgmi discovery -l                          # should list 8 GPUs and 4 NvSwitches
```

A dry run (`apt-get -s`) on the XE9680L showed no removals and no driver or Fabric Manager changes. `dcgmi dmon` needs no root once `nv-hostengine` is running.

## Usage

```bash
# all NVSwitches + all GPUs, 1 s for 60 s
python3 dcgm_fabric_monitor.py -n 60 -w 1000 -o runs/exp1/dcgm.csv

# with GPU profiling metrics, only GPUs 1 and 2
python3 dcgm_fabric_monitor.py --gpus 1,2 --gpu-prof -n 30

# NVSwitches only
python3 dcgm_fabric_monitor.py --gpus none -n 120 -w 500
```

Outputs:
- `<output>.csv`: one row per sample × entity (`nvswitch` or `gpu`), with columns named `<dcgm field name>_<id>`, plus `nvswitch_tx_Bps` and `nvswitch_rx_Bps`.
- `<output>_raw/`: the native `dcgmi dmon` text.

Samples are timestamped as they arrive (the tool runs on a pty). Rates are computed from consecutive cumulative counters and those arrival times.

## Fields

| Entity | Field id | Name | Notes |
| --- | --- | --- | --- |
| NVSwitch | 1695 / 1696 | `nvswitch_tx_bytes_total` / `nvswitch_rx_bytes_total` | Cumulative bytes; the tool derives `nvswitch_tx_Bps` and `nvswitch_rx_Bps` |
| NVSwitch | 858 | `nvswitch_current_temperature` | °C |
| NVSwitch | 705–707 | `nvswitch_power_vdd/dvdd/hvdd` | Rail power (N/A on some switches) |
| GPU | 155, 100 | `power_usage`, `sm_clock` | W, MHz |
| GPU | 449 | `nvlink_throughput_total` | All NVLink links, as reported by NVML |
| GPU (`--gpu-prof`) | 1009 / 1010 | `pcie_tx/rx_bytes_per_second` | Profiling metrics |
| GPU (`--gpu-prof`) | 1011 / 1012 | `prof_nvlink_tx/rx_bytes_per_second` | Profiling metrics |
| GPU (`--gpu-prof`) | 1001, 1002, 1005 | `gr_engine_active`, `sm_active`, `dram_active` | Fractions 0–1 |

Per-link NVSwitch fields (`nvswitch_link_tx/rx_bytes_total` 1655/1656, link errors 782–788, latency histograms 789–) also exist. They need link entities and are not sampled by default.

**Shared hosts.** The profiling fields use the same GPU performance-monitor hardware as Nsight Systems GPU metrics and Nsight Compute. Enable `--gpu-prof` only during your own runs.

## Validation on the Dell PowerEdge XE9680L (8× H200 SXM, 4 NVSwitch), 2026-11-18

A single GPU-to-GPU NVLink flow (GPU1 → GPU2, SM stores, 353 GB/s measured by the workload):

- **GPU1 `prof_nvlink_tx_bytes_per_second` = 352.1 GB/s** and **GPU2 `prof_nvlink_rx` = 352.1 GB/s**. These match the workload.
- **NVSwitch traffic splits 5:4:4:5.** Switches 1 and 2 each carry about 111 GB/s, and switches 0 and 3 about 89 GB/s, a ratio of 1.25. That matches each GPU's 18 NVLink links being spread 4/5/5/4 across the four switches. Lifetime counters show the same ratio (71.3 against 56.8 TB).
- **The four switches together carry about 400 GB/s for the 352 GB/s payload,** about 13% more than the payload. Treat switch bytes as wire traffic, not payload.
