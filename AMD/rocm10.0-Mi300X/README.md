# AMD MI300X Monitoring with AMD-SMI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ROCm](https://img.shields.io/badge/ROCm-10.0-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Python utilities for collecting GPU telemetry, PCIe bandwidth, and per-peer
xGMI bandwidth from AMD Instinct MI300X GPUs. The scripts use the `amd-smi`
command-line tool supplied with ROCm and write analysis-friendly CSV output.

The tools have no third-party Python package dependencies.

## Table of contents

- [Overview](#overview)
- [Tested environment](#tested-environment)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Combined GPU and xGMI monitor](#combined-gpu-and-xgmi-monitor)
- [GPU telemetry monitor](#gpu-telemetry-monitor)
- [xGMI bandwidth monitor](#xgmi-bandwidth-monitor)
- [Output handling](#output-handling)
- [Units and conversions](#units-and-conversions)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The directory contains four Python modules:

```text
rocm10.0-Mi300X/
├── amdsmi_combined_monitor.py
├── amdsmi_common.py
├── amdsmi_gpu_monitor.py
├── amdsmi_xgmi_bw_monitor.py
└── README.md
```

| File | Purpose |
| --- | --- |
| `amdsmi_combined_monitor.py` | Runs both monitors concurrently and merges their timestamped output. |
| `amdsmi_gpu_monitor.py` | Collects temperature, power, clocks, utilization, VRAM usage, and instantaneous PCIe bandwidth. |
| `amdsmi_xgmi_bw_monitor.py` | Samples cumulative xGMI counters and calculates per-peer bandwidth and utilization. |
| `amdsmi_common.py` | Contains shared argument, GPU-selection, and safe output-file helpers. It is not intended to be run directly. |

The GPU monitor streams CSV produced by `amd-smi monitor`. The xGMI monitor
requires additional processing because `amd-smi xgmi --metric` exposes
cumulative read and write counters rather than rates.

## Tested environment

These scripts were developed and tested with:

| Component | Version or configuration |
| --- | --- |
| GPU | 8 &times; AMD Instinct MI300X |
| Platform | Linux bare metal |
| Python | 3.10.12 |
| ROCm | 10.0.0 |
| AMD-SMI CLI | 27.0.0+6b0e43f3 |
| AMD-SMI library | 27.0.0 |
| AMDGPU driver | 7.1.3.31500000 |
| MI300X xGMI link capacity | 64 GB/s per direction |

Other AMD-SMI, ROCm, GPU, and partition configurations may expose different
fields or JSON layouts.

## Requirements

- Linux with supported AMD GPUs and a working AMDGPU driver
- ROCm with `amd-smi` available on `PATH`
- Python 3.10 or newer
- Permission to query the installed GPUs

Confirm the required tools:

```bash
python3 --version
amd-smi version
amd-smi list
```

No `pip install` step is required.

## Getting started

Clone the repository and enter this directory:

```bash
git clone https://github.com/hibagus/PerformanceProfiling.git
cd PerformanceProfiling/AMD/rocm10.0-Mi300X
```

Display the available options:

```bash
python3 amdsmi_gpu_monitor.py --help
python3 amdsmi_xgmi_bw_monitor.py --help
python3 amdsmi_combined_monitor.py --help
```

Run short smoke tests:

```bash
python3 amdsmi_gpu_monitor.py -g 0 -w 1 -W 5 --stdout
python3 amdsmi_xgmi_bw_monitor.py -g 0 -w 1 -W 5 --stdout
```

Diagnostic messages are written to standard error, so standard output remains
valid CSV when `--stdout` is selected.

## Combined GPU and xGMI monitor

`amdsmi_combined_monitor.py` launches both standalone monitors concurrently
with the same GPU selection, interval, duration, and UTC run ID. When collection
ends, it merges the nearest xGMI sample for each source GPU into every GPU
telemetry row.

Monitor all GPUs for 60 seconds and create all outputs in the current directory:

```bash
python3 amdsmi_combined_monitor.py -w 1 -W 60
```

Select GPUs and an output directory:

```bash
python3 amdsmi_combined_monitor.py \
  -g 0 1 \
  -w 1 \
  -W 60 \
  --output-dir runs/combined
```

One invocation creates three files with a shared run ID:

```text
20260910T161023Z_amdsmi_monitor.csv
20260910T161023Z_amdsmi_xgmi_bandwidth.csv
20260910T161023Z_amdsmi_consolidated.csv
```

Use `--run-id`, `--gpu-output`, `--xgmi-output`, or `--output` to override
these names. Existing files are protected unless `--overwrite` is supplied.
`--dry-run` prints both child commands and all output paths without collecting.

The two AMD-SMI commands timestamp their observations independently, so their
sample timestamps cannot be made exactly identical. The combined monitor
preserves both raw files and performs a nearest-timestamp join for the same
source GPU. The default maximum timestamp difference is half the requested
interval; it can be changed with `--match-tolerance`.

The consolidated file retains every GPU-monitor column and appends fields named
`xgmi_to_gpu_N_bidirectional_utilization_pct`. On an eight-GPU MI300X system,
each GPU row has values for its seven peers and a blank self-link column. A row
is left blank in all appended fields when there is no xGMI sample within the
tolerance. This is normally expected for the initial GPU row because the first
xGMI query establishes the counter baseline.

Pressing Ctrl+C stops both collectors cleanly and merges the partial raw files.

## GPU telemetry monitor

`amdsmi_gpu_monitor.py` executes the equivalent of:

```bash
amd-smi monitor \
  --power-usage --temperature --gfx --mem --vram-usage --pcie \
  --gpu all --csv --watch 1
```

### Basic usage

Monitor every GPU once per second until interrupted:

```bash
python3 amdsmi_gpu_monitor.py
```

Monitor GPUs 0 and 1 for 60 seconds:

```bash
python3 amdsmi_gpu_monitor.py -g 0 1 -w 1 -W 60
```

Comma-separated and mixed GPU lists are accepted:

```bash
python3 amdsmi_gpu_monitor.py -g 0,1,4 -W 60
```

Write to a specific file and replace it if it already exists:

```bash
python3 amdsmi_gpu_monitor.py \
  -g 0 \
  -w 1 \
  -W 60 \
  -o gpu0_telemetry.csv \
  --overwrite
```

Preview the underlying AMD-SMI command without starting collection:

```bash
python3 amdsmi_gpu_monitor.py -g 0,1 -w 2 -W 30 --dry-run
```

### Optional metrics

Add ECC and PCIe replay counters:

```bash
python3 amdsmi_gpu_monitor.py --ecc -W 60
```

Add MI300 power and thermal violation fields:

```bash
python3 amdsmi_gpu_monitor.py --violation -W 60
```

Both flags can be used together.

### Base CSV fields

AMD-SMI determines the exact field set. With the tested software stack, the
base output contains:

| Field | Meaning | Unit |
| --- | --- | --- |
| `timestamp` | AMD-SMI sample timestamp | Unix epoch seconds |
| `gpu` | AMD-SMI GPU index | — |
| `xcp` | Compute-partition identifier | — |
| `power_usage` | Current socket power | W |
| `max_power` | Maximum configured power | W |
| `hotspot_temperature` | GPU hotspot temperature | degrees C |
| `memory_temperature` | HBM temperature | degrees C |
| `gfx_clk` | Graphics clock | MHz |
| `gfx` | Graphics-engine utilization | % |
| `mem` | Memory activity | % |
| `mem_clock` | Memory clock | MHz |
| `vram_used` | Used VRAM | AMD-SMI-labeled MB |
| `vram_free` | Free VRAM | AMD-SMI-labeled MB |
| `vram_total` | Total VRAM | AMD-SMI-labeled MB |
| `vram_percent` | Used VRAM | % |
| `pcie_bw_bidirectional_mbps` | Instantaneous aggregate PCIe bandwidth (transmit + receive) | Mb/s |

`--ecc` adds `single_bit_ecc`, `double_bit_ecc`, and `pcie_replay`.
`--violation` adds the violation fields supplied by the installed AMD-SMI
version.

The script renames AMD-SMI's ambiguous `pcie_bw` CSV header to
`pcie_bw_bidirectional_mbps`. This is one combined value for traffic in both
directions, not separate transmit and receive measurements. The script does
not modify the numeric value.

## xGMI bandwidth monitor

`amdsmi_xgmi_bw_monitor.py` repeatedly runs:

```bash
amd-smi xgmi --metric --gpu all --json
```

AMD-SMI 27.0 only returns the complete per-peer matrix when all GPUs are
queried. The script therefore queries all GPUs internally and applies `--gpus`
as a source-GPU filter after parsing the response.

For each directed source-to-peer link, bandwidth is calculated as:

```text
rate = (current cumulative counter - previous cumulative counter)
       / measured elapsed time
```

Query midpoint timestamps and monotonic elapsed time are used to reduce timing
error. The first query establishes a baseline and does not produce rate rows.

### Basic usage

Monitor all directed GPU-to-GPU links once per second:

```bash
python3 amdsmi_xgmi_bw_monitor.py
```

Monitor links originating from GPU 0 for 60 seconds:

```bash
python3 amdsmi_xgmi_bw_monitor.py -g 0 -w 1 -W 60
```

Monitor source GPUs 0 and 1 at 500 ms intervals and show CSV on screen:

```bash
python3 amdsmi_xgmi_bw_monitor.py -g 0 1 -w 0.5 -W 30 --stdout
```

Override the per-direction capacity when the target system differs from the
tested MI300X platform:

```bash
python3 amdsmi_xgmi_bw_monitor.py \
  --link-capacity-gb-s 50 \
  -W 60
```

Control query failure handling:

```bash
python3 amdsmi_xgmi_bw_monitor.py \
  --query-timeout 15 \
  --max-errors 5 \
  -W 60
```

The default query timeout is 10 seconds. Collection stops after three
consecutive query failures by default. Successful queries reset the error
count.

### xGMI CSV fields

Each row represents one directed source-to-peer link during one measured
interval.

| Field | Meaning | Unit |
| --- | --- | --- |
| `timestamp_epoch` | Midpoint timestamp of the current query | Unix epoch seconds |
| `timestamp_utc` | The same timestamp in ISO 8601 UTC form | UTC |
| `interval_seconds` | Measured time between query midpoints | s |
| `source_gpu` | Source GPU index | — |
| `source_bdf` | Source PCI BDF | — |
| `peer_gpu` | Peer GPU index | — |
| `peer_bdf` | Peer PCI BDF | — |
| `read_counter_kb` | Current cumulative read counter | decimal KB |
| `write_counter_kb` | Current cumulative write counter | decimal KB |
| `read_delta_kb` | Read-counter change during the interval | decimal KB |
| `write_delta_kb` | Write-counter change during the interval | decimal KB |
| `read_gb_s` | Calculated read rate | decimal GB/s |
| `write_gb_s` | Calculated write rate | decimal GB/s |
| `total_gb_s` | `read_gb_s + write_gb_s` | decimal GB/s |
| `unidirectional_capacity_gb_s` | Configured capacity per direction | decimal GB/s |
| `read_utilization_pct` | Read rate divided by per-direction capacity | % |
| `write_utilization_pct` | Write rate divided by per-direction capacity | % |
| `bidirectional_utilization_pct` | Total rate divided by twice the per-direction capacity | % |
| `counter_status` | `ok` or `reset_or_wrap` | — |

If either counter decreases, the interval is marked `reset_or_wrap`. Rate,
delta, and utilization fields are left blank rather than guessing a counter
width and producing a false spike. The new values become the baseline for the
next interval.

On the tested MI300X system, AMD-SMI reports a maximum xGMI bandwidth of
512 Gb/s, which is 64 GB/s in each direction. Full-duplex capacity is therefore
128 GB/s. The default utilization calculations use those values.

## Output handling

Both standalone monitors support the same output modes:

| Option | Behavior |
| --- | --- |
| `--output-mode file` | Write CSV to a file. This is the default. |
| `--output-mode stdout` or `--stdout` | Write CSV to standard output. |
| `--output-mode both` or `--both` | Write CSV to both a file and standard output. |

When `--output` is omitted, the file is created in `--output-dir`, which
defaults to the current directory. UTC timestamps are included in generated
filenames.

Examples:

```bash
# Choose an output directory and generated filename.
python3 amdsmi_gpu_monitor.py --output-dir runs/telemetry -W 60

# Choose a complete output path.
python3 amdsmi_xgmi_bw_monitor.py -o runs/xgmi.csv -W 60

# Send CSV into another process.
python3 amdsmi_gpu_monitor.py --stdout -W 60 | gzip > gpu.csv.gz
```

Output files are protected against accidental replacement:

- `--overwrite` replaces an existing file.
- `--append` adds rows to an existing file without duplicating its header.
- With neither option, an existing destination is rejected.
- `--overwrite` and `--append` are mutually exclusive.

Parent output directories are created automatically after collision checks.

## Units and conversions

The scripts use decimal SI units for bandwidth:

```text
1 KB = 1,000 bytes
1 MB = 1,000,000 bytes
1 GB = 1,000,000,000 bytes
1 Mb = 1,000,000 bits
8 bits = 1 byte
```

This convention applies to the PCIe and calculated xGMI bandwidth fields.
AMD-SMI calculates its VRAM fields by dividing bytes by `1024**2` but labels
the result `MB`; numerically, those VRAM fields are MiB even though their
upstream label says MB.

PCIe bandwidth is reported by AMD-SMI in megabits per second. Convert a
`pcie_bw_bidirectional_mbps` value as follows:

```python
mb_per_second = pcie_bw_bidirectional_mbps / 8
gb_per_second = pcie_bw_bidirectional_mbps / 8_000
```

For example:

```text
524,000 Mb/s = 65,500 MB/s = 65.5 GB/s
```

Do not divide `pcie_bw_bidirectional_mbps` by 1,024 when converting between SI bandwidth
units. If binary byte units are specifically required:

```python
mib_per_second = pcie_bw_bidirectional_mbps * 1_000_000 / 8 / (1024**2)
gib_per_second = pcie_bw_bidirectional_mbps * 1_000_000 / 8 / (1024**3)
```

## Validation

The scripts were validated with syntax, argument, CSV structure, append,
collision, counter-reset, live telemetry, and live traffic tests.

TransferBench was used to generate sustained CPU-to-GPU, GPU-to-CPU, and
GPU-to-GPU traffic. Representative GPU 0 to GPU 1 results were:

| Measurement | Result |
| --- | ---: |
| TransferBench application payload | 49.14 GB/s |
| xGMI monitor median write traffic | 58.47 GB/s |
| xGMI monitor peak write traffic | 61.57 GB/s |
| Configured per-direction capacity | 64 GB/s |
| Peak reported write utilization | 96.2% |

AMD-SMI bandwidth counters represent link-accounted traffic and should not be
expected to equal application payload throughput exactly. Protocol overhead,
counter semantics, sampling windows, and other traffic can produce different
values.

TransferBench is useful for validation but is not required to use these
monitoring scripts.

## Troubleshooting

### `amd-smi was not found on PATH`

Confirm ROCm and AMD-SMI are installed, then locate the executable:

```bash
command -v amd-smi
amd-smi version
```

### Permission or device-access errors

Confirm that the current account can access the GPU device nodes and is in the
site-required GPU access groups, commonly `video` and `render`:

```bash
id
ls -l /dev/kfd /dev/dri/renderD*
```

Group configuration is system-specific; follow the security policy for the
target environment.

### Output file already exists

Select a different filename or explicitly choose the desired behavior:

```bash
python3 amdsmi_gpu_monitor.py -o telemetry.csv --overwrite -W 30
python3 amdsmi_gpu_monitor.py -o telemetry.csv --append -W 30
```

### No usable per-peer xGMI counters

Inspect AMD-SMI directly:

```bash
amd-smi xgmi --metric --gpu all --json
```

The xGMI monitor requires numeric `read` and `write` values for peer links.
Self-links and links reported as `N/A` are intentionally omitted.

### The requested interval is not exact

`--interval` controls the target spacing between query starts. AMD-SMI query
time is nonzero, so the xGMI monitor stores the actual elapsed time in
`interval_seconds` and uses that value for every rate calculation. If a query
takes longer than the requested interval, the next query starts immediately.

## Known limitations

- The code is tied to the AMD-SMI CLI output available in the tested ROCm 10.0
  environment. Future output-schema changes may require parser updates.
- `pcie_bw_bidirectional_mbps` is an instantaneous aggregate link metric that
  combines transmit and receive traffic. It does not expose the two directions
  separately on the tested platform.
- AMD-SMI PCIe and xGMI values measure link-accounted traffic, not application
  payload throughput.
- xGMI source selection does not reduce AMD-SMI query cost because the complete
  matrix must be queried before filtering.
- Collection duration is approximate. An in-progress AMD-SMI query is allowed
  to finish, so wall-clock runtime can slightly exceed `--duration`.
- Board-temperature groups and the encoder metric are intentionally excluded
  from the GPU monitor because they are unsupported or absent on the tested
  MI300X host.

## Contributing

Issues and pull requests are welcome at the
[PerformanceProfiling repository](https://github.com/hibagus/PerformanceProfiling).

When changing AMD-SMI parsing or bandwidth calculations, please include:

- The GPU model and topology
- ROCm, AMD-SMI, and AMDGPU driver versions
- A representative raw AMD-SMI response
- The command used to generate validation traffic
- Expected and observed CSV output

## License

Distributed under the MIT License. See the repository [LICENSE](../../LICENSE)
file for details.

## Acknowledgments

- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [AMD-SMI documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [TransferBench](https://github.com/ROCm/TransferBench)
- README organization adapted from
  [Best README Template](https://github.com/othneildrew/Best-README-Template)
