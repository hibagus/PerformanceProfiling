<div align="center">

# NVIDIA H200 Monitoring Toolkit

### GPU/PCIe telemetry and per-link NVLink bandwidth with NVIDIA-SMI

</div>

Dependency-free Python collectors for NVIDIA H200 systems. They wrap
`nvidia-smi dmon` and `nvidia-smi nvlink --getthroughput d`, write
analysis-ready CSV, and can run together to create a GPU-level consolidated
capture.

## Scripts

| Script | Purpose |
| --- | --- |
| `nvsmi_combined_monitor.py` | Run both collectors concurrently and merge aggregate NVLink data into each GPU telemetry row |
| `nvsmi_gpu_monitor.py` | Collect power, temperature, utilization, clocks, memory, ECC, PCIe replay, and PCIe throughput from `dmon` |
| `nvsmi_nvlink_bw_monitor.py` | Convert cumulative per-link NVLink Tx/Rx counters into bandwidth and utilization |
| `nvsmi_common.py` | Shared argument, GPU-selection, timestamp, and safe-file helpers |

The scripts use only the Python standard library and require no `pip`
packages.

## Requirements

- Linux with NVIDIA H200 GPUs and a working NVIDIA driver.
- `nvidia-smi` available on `PATH`.
- Python 3.10 or newer.
- Permission to query the GPUs.

This directory targets CUDA 13.3-era H200 nodes. NVIDIA notes that textual
NVSMI output is not guaranteed to remain backward compatible, so validate the
parsers when changing driver branches.

Check the node before collection:

```bash
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
nvidia-smi nvlink --status
python3 --version
```

## Quick start

Run both collectors for 60 seconds on every GPU:

```bash
python3 nvsmi_combined_monitor.py -w 1 -W 60
```

Select GPUs and an output directory:

```bash
python3 nvsmi_combined_monitor.py \
  -g 0 1 \
  -w 1 \
  -W 60 \
  --output-dir runs/combined
```

One invocation produces three files with the same UTC run identifier:

```text
<UTC>_nvsmi_dmon.csv
<UTC>_nvsmi_nvlink_bandwidth.csv
<UTC>_nvsmi_consolidated.csv
```

Use `--dry-run` to preview both native commands and all output paths. Existing
files are never replaced unless `--overwrite` is supplied. Ctrl+C or SIGTERM
stops both child collectors, flushes their partial CSVs, and attempts the final
merge.

## GPU telemetry (`nvidia-smi dmon`)

The default native command is equivalent to:

```bash
nvidia-smi dmon -s pucmet -d 1 --format csv,nounit
```

The selected groups collect power/temperature (`p`), utilization (`u`), clocks
(`c`), memory (`m`), ECC/PCIe replay counters (`e`), and PCIe Rx/Tx throughput
(`t`). The wrapper prepends epoch and ISO-8601 UTC timestamps and gives each
abbreviated column an explicit unit where NVIDIA-SMI defines one.

```bash
# All GPUs until interrupted
python3 nvsmi_gpu_monitor.py

# GPUs 0 and 1 for about 30 seconds, streamed and saved
python3 nvsmi_gpu_monitor.py -g 0,1 -w 1 -W 30 --both

# Choose native metric groups
python3 nvsmi_gpu_monitor.py --metric-groups pucmt -W 30
```

`dmon` accepts whole-second intervals. For a finite duration, the wrapper uses
`ceil(duration / interval)` samples; this matches the requested capture length
as closely as `dmon` permits.

## NVLink bandwidth

The NVLink collector repeatedly invokes:

```bash
nvidia-smi nvlink --getthroughput d
```

The `d` counter type is Tx and Rx data payload in KiB. For every physical link,
the collector calculates:

```text
bandwidth = (current counter - previous counter) * 1024 / elapsed seconds
```

Rates are emitted in decimal GB/s. Elapsed time is measured around each query
rather than assuming the requested sleep interval. If a counter decreases, the
sample is marked `reset_or_wrap` and its derived fields are blank to prevent a
false spike.

```bash
# All physical links on every GPU
python3 nvsmi_nvlink_bw_monitor.py -w 1 -W 60

# Links on GPUs 0 and 1, with CSV also written to stdout
python3 nvsmi_nvlink_bw_monitor.py -g 0 1 -w 0.5 -W 30 --both
```

The default capacity is 50 GB/s per direction for each Hopper NVLink. Each
physical-link utilization is calculated against 50 GB/s for Rx and Tx, or 100
GB/s full duplex for the combined percentage. Override it if the installed
platform reports a different effective capacity:

```bash
python3 nvsmi_nvlink_bw_monitor.py --link-capacity-gb-s 50 -W 60
```

The consolidated CSV sums all valid physical links for a GPU at each NVLink
sample, then attaches the nearest aggregate sample to each `dmon` row. Its
aggregate utilization denominator is the sum of the capacities of the links
present in that sample. The default timestamp tolerance is half the collection
interval and can be changed with `--match-tolerance`.

## Output controls

Each standalone collector supports:

- `--output-mode file|stdout|both` (plus `--stdout` and `--both` shortcuts)
- `-o/--output` for a complete path
- `--output-dir` and `--filename` for generated paths
- `--overwrite` or `--append` for explicit collision behavior
- `-g/--gpus` with space-separated, comma-separated, or mixed indexes

Diagnostics go to standard error, so `--stdout` remains valid CSV.

## Validation

Run the parser and calculation tests without requiring a GPU:

```bash
python3 -m unittest discover -s tests -v
```

On an H200 node, also perform short live captures:

```bash
python3 nvsmi_gpu_monitor.py -g 0 -w 1 -W 5 --stdout
python3 nvsmi_nvlink_bw_monitor.py -g 0 -w 1 -W 5 --stdout
```

The first NVLink query establishes a baseline, so bandwidth rows begin with the
second successful query. NVIDIA-SMI exposes counters by physical NVLink ID, not
by remote GPU; consequently the raw NVLink CSV is per physical link rather than
a peer-GPU matrix.

## References

- [NVIDIA-SMI documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
- [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA Hopper NVLink topology and per-link bandwidth](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-throughput-by-another-1-5x-on-nvidia-h200-tensor-core-gpus-and-nvlink-switch/)
