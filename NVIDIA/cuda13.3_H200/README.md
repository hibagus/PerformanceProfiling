<a id="readme-top"></a>

<div align="center">

# NVIDIA H200 Monitoring Toolkit

### GPU, PCIe, and NVLink telemetry with NVIDIA-SMI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/NVIDIA-H200-76B900?logo=nvidia&logoColor=white)](#tested-environment)

Dependency-free Python tools for collecting NVIDIA GPU telemetry, PCIe
throughput, and per-link NVLink bandwidth in analysis-ready CSV format.

[Get started](#getting-started) · [Usage](#usage) · [CSV reference](#csv-reference) · [Validation](#validation)

</div>

<details>
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#output-handling">Output handling</a></li>
    <li><a href="#csv-reference">CSV reference</a></li>
    <li><a href="#units-and-conversions">Units and conversions</a></li>
    <li><a href="#validation">Validation</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#known-limitations">Known limitations</a></li>
    <li><a href="#project-layout">Project layout</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the project

This toolkit wraps the `nvidia-smi` command-line utility supplied with the
NVIDIA driver. It supports finite or continuous collection, GPU selection,
safe file handling, standard-output streaming, and concurrent GPU/NVLink
monitoring.

| Script | Purpose |
| --- | --- |
| `nvsmi_combined_monitor.py` | Run both monitors concurrently and merge their timestamped output |
| `nvsmi_gpu_monitor.py` | Collect power, temperature, utilization, clocks, memory, ECC, and PCIe telemetry with `dmon` |
| `nvsmi_nvlink_bw_monitor.py` | Convert cumulative physical-link NVLink counters into bandwidth and utilization |
| `nvsmi_common.py` | Shared argument, GPU-selection, timestamp, and safe output-file helpers |

The GPU monitor normalizes NVIDIA-SMI's abbreviated headers and adds epoch and
ISO 8601 UTC timestamps. The NVLink monitor derives rates from cumulative Tx/Rx
payload counters using the measured elapsed time between queries.

### Tested environment

| Component | Version or configuration |
| --- | --- |
| GPU | 8 × NVIDIA H200 |
| Platform | Linux bare metal |
| Python | 3.10.12 |
| CUDA toolkit | 13.3, NVCC 13.3.73 |
| NVIDIA-SMI / NVML | 610.57.04 / 610.57 |
| NVIDIA kernel driver | 610.57.04 |
| CUDA user-mode driver | 13.3 |
| NVLink topology | 18 active physical links per GPU |
| Published Hopper link capacity | 50 GB/s per direction |
| `nvidia-smi nvlink --status` value | 26.562 GB/s per active link |

Other NVIDIA-SMI, driver, GPU, MIG, and NVLink configurations may expose
different fields or textual layouts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

### Prerequisites

- Linux with supported NVIDIA GPUs and a working NVIDIA kernel driver.
- NVIDIA-SMI available from `PATH`.
- Python 3.10 or newer.
- Permission to query the installed GPU device nodes.

The scripts use only the Python standard library; no `pip install` step is
required. The CUDA toolkit is not required for monitoring, although it is
needed to build CUDA workloads such as TransferBench.

Confirm the required commands:

```bash
python3 --version
nvidia-smi --version
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
nvidia-smi nvlink --status
```

Clone the repository and enter the toolkit directory:

```bash
git clone https://github.com/hibagus/PerformanceProfiling.git
cd PerformanceProfiling/NVIDIA/cuda13.3_H200
```

Display the available options:

```bash
python3 nvsmi_combined_monitor.py --help
python3 nvsmi_gpu_monitor.py --help
python3 nvsmi_nvlink_bw_monitor.py --help
```

Run short smoke tests on GPU 0:

```bash
python3 nvsmi_gpu_monitor.py -g 0 -w 1 -W 5 --stdout
python3 nvsmi_nvlink_bw_monitor.py -g 0 -w 0.5 -W 5 --stdout
```

Diagnostic messages go to standard error, leaving standard output as valid CSV
when `--stdout` is selected.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Combined monitor

`nvsmi_combined_monitor.py` launches both standalone monitors with the same GPU
selection, interval, duration, and UTC run ID. It aggregates all valid physical
NVLinks for each GPU and attaches the nearest aggregate sample to every GPU
telemetry row.

Monitor every GPU for 60 seconds:

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

One invocation creates three files with a shared run ID:

```text
<UTC>_nvsmi_dmon.csv
<UTC>_nvsmi_nvlink_bandwidth.csv
<UTC>_nvsmi_consolidated.csv
```

Override these names with `--run-id`, `--gpu-output`, `--nvlink-output`, or
`--output`. Use `--dry-run` to inspect both child commands and output paths.

The streams are timestamped independently. The default merge tolerance is half
the requested interval and can be changed with `--match-tolerance`. Initial
merged NVLink cells may be blank because the first counter query establishes a
baseline. Pressing Ctrl+C stops both collectors cleanly and merges their partial
raw files.

### GPU telemetry monitor

The GPU monitor uses NVIDIA-SMI device monitoring groups for power and
temperature (`p`), utilization (`u`), clocks (`c`), memory (`m`), ECC and PCIe
replay counters (`e`), and PCIe throughput (`t`).

<details>
  <summary>Show the equivalent native command</summary>

```bash
nvidia-smi dmon -s pucmet -d 1 --format csv,nounit
```

</details>

```bash
# Monitor every GPU until interrupted.
python3 nvsmi_gpu_monitor.py

# Monitor GPUs 0 and 1 for approximately 60 seconds.
python3 nvsmi_gpu_monitor.py -g 0 1 -w 1 -W 60

# Comma-separated and mixed GPU lists are accepted.
python3 nvsmi_gpu_monitor.py -g 0,1,4 -W 60
```

Choose a file and explicitly replace it when present:

```bash
python3 nvsmi_gpu_monitor.py \
  -g 0 \
  -W 60 \
  -o gpu0_telemetry.csv \
  --overwrite
```

Preview the native command or choose different metric groups:

```bash
python3 nvsmi_gpu_monitor.py -g 0,1 -w 2 -W 30 --dry-run
python3 nvsmi_gpu_monitor.py --metric-groups pucmt -W 30
```

NVIDIA-SMI `dmon` accepts whole-second intervals. For finite collection, the
wrapper requests `ceil(duration / interval)` samples, so duration is
approximate when it is not an exact interval multiple.

### NVLink bandwidth monitor

The NVLink monitor repeatedly invokes:

```bash
nvidia-smi nvlink --getthroughput d
```

The `d` counter type is Tx and Rx data payload in KiB. For every physical link,
the rate is calculated from the counter delta and measured query-midpoint
interval:

```text
rate = (current counter - previous counter) * 1,024
       / measured elapsed time
```

```bash
# Monitor every physical link until interrupted.
python3 nvsmi_nvlink_bw_monitor.py

# Monitor links on GPUs 0 and 1 for 60 seconds.
python3 nvsmi_nvlink_bw_monitor.py -g 0 1 -w 1 -W 60

# Use a 250 ms target interval and stream the CSV.
python3 nvsmi_nvlink_bw_monitor.py -g 0 -w 0.25 -W 30 --stdout
```

Override capacity or failure handling when required:

```bash
python3 nvsmi_nvlink_bw_monitor.py \
  --link-capacity-gb-s 26.562 \
  --query-timeout 15 \
  --max-errors 5 \
  -W 60
```

The default utilization basis is 50 GB/s per direction, matching NVIDIA's
published Hopper link figure. The validated driver reports 26.562 GB/s per
active link through `nvidia-smi nvlink --status`; use the override above when
utilization should be relative to that driver-reported value. Raw rates and
counters are unaffected by the configured capacity.

The default query timeout is 10 seconds and collection stops after three
consecutive failures. A successful query resets the error count.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Output handling

| Option | Behavior |
| --- | --- |
| `--output-mode file` | Write CSV to a file; this is the default |
| `--output-mode stdout` or `--stdout` | Write CSV to standard output |
| `--output-mode both` or `--both` | Write CSV to a file and standard output |

When `--output` is omitted, a UTC-prefixed filename is created in
`--output-dir`, which defaults to the current directory.

```bash
# Choose an output directory and generated filename.
python3 nvsmi_gpu_monitor.py --output-dir runs/telemetry -W 60

# Choose a complete output path.
python3 nvsmi_nvlink_bw_monitor.py -o runs/nvlink.csv -W 60

# Stream CSV into another process.
python3 nvsmi_gpu_monitor.py --stdout -W 60 | gzip > gpu.csv.gz
```

Output files are protected against accidental replacement:

- `--overwrite` replaces an existing file.
- `--append` adds rows without duplicating the CSV header.
- Without either option, an existing destination is rejected.
- `--overwrite` and `--append` are mutually exclusive.

Parent output directories are created automatically after collision checks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## CSV reference

### GPU telemetry fields

The script normalizes the abbreviated NVIDIA-SMI fields selected by the
default `pucmet` metric groups. Unsupported metrics contain NVIDIA-SMI's `-`
marker. Selecting other metric groups may add fields whose sanitized upstream
names are retained.

<details>
  <summary><strong>Show GPU telemetry fields</strong></summary>

| Field | Meaning | Unit |
| --- | --- | --- |
| `timestamp_epoch` | Time when the row reached the wrapper | Unix epoch seconds |
| `timestamp_utc` | The same timestamp in ISO 8601 form | UTC |
| `gpu` | NVIDIA-SMI GPU index | — |
| `power_w` | Board power usage | W |
| `gpu_temperature_c` / `memory_temperature_c` | GPU and memory temperatures | °C |
| `sm_utilization_pct` / `memory_utilization_pct` | SM and memory utilization | % |
| `encoder_utilization_pct` / `decoder_utilization_pct` | Video engine utilization | % |
| `jpeg_utilization_pct` / `ofa_utilization_pct` | JPEG and optical-flow utilization | % |
| `memory_clock_mhz` / `processor_clock_mhz` | Memory and processor clocks | MHz |
| `framebuffer_memory_used_mb` | Frame-buffer memory usage | NVIDIA-SMI-labeled MB |
| `bar1_memory_used_mb` | BAR1 memory usage | NVIDIA-SMI-labeled MB |
| `confidential_compute_memory_used_mb` | Protected-memory usage | NVIDIA-SMI-labeled MB |
| `ecc_single_bit_errors` / `ecc_double_bit_errors` | Aggregated ECC errors | count |
| `pcie_replay_errors` | PCIe replay errors | count |
| `pcie_rx_mb_s` / `pcie_tx_mb_s` | PCIe receive and transmit throughput | MB/s |

</details>

### NVLink bandwidth fields

Each row represents one physical NVLink on one GPU over one measured interval.
Query midpoint timestamps and monotonic elapsed time reduce timing error. The
first successful query establishes the baseline and produces no rate rows.

<details>
  <summary><strong>Show NVLink bandwidth fields</strong></summary>

| Field | Meaning | Unit |
| --- | --- | --- |
| `timestamp_epoch` | Midpoint timestamp of the current query | Unix epoch seconds |
| `timestamp_utc` | The same timestamp in ISO 8601 form | UTC |
| `interval_seconds` | Measured time between query midpoints | s |
| `gpu` / `link` | GPU index and physical NVLink ID | — |
| `rx_counter_kib` / `tx_counter_kib` | Current cumulative data-payload counters | KiB |
| `rx_delta_kib` / `tx_delta_kib` | Counter changes during the interval | KiB |
| `rx_gb_s` / `tx_gb_s` | Calculated directional rates | decimal GB/s |
| `total_gb_s` | Sum of Rx and Tx rates | decimal GB/s |
| `per_direction_capacity_gb_s` | Configured capacity per direction | decimal GB/s |
| `rx_utilization_pct` / `tx_utilization_pct` | Directional rate divided by configured capacity | % |
| `bidirectional_utilization_pct` | Total rate divided by twice the configured capacity | % |
| `counter_status` | `ok` or `reset_or_wrap` | — |

</details>

If either counter decreases, the interval is marked `reset_or_wrap`; delta,
rate, and utilization cells remain blank rather than assuming a counter width
and reporting a false spike. The new values become the following baseline.

The consolidated CSV keeps every GPU-monitor column and adds:

| Field | Meaning |
| --- | --- |
| `nvlink_rx_gb_s` | Sum of valid Rx rates across the GPU's physical links |
| `nvlink_tx_gb_s` | Sum of valid Tx rates across the GPU's physical links |
| `nvlink_total_gb_s` | Sum of aggregate Rx and Tx rates |
| `nvlink_bidirectional_utilization_pct` | Aggregate total divided by the full-duplex capacity of links present in the sample |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Units and conversions

NVLink counters use binary kibibytes, while emitted bandwidth uses decimal
gigabytes per second:

```text
1 KiB = 1,024 bytes
1 GB  = 1,000,000,000 bytes

GB/s = delta_KiB * 1,024 / elapsed_seconds / 1,000,000,000
```

NVIDIA-SMI labels `dmon` PCIe throughput as MB/s. Those values are preserved
without numeric conversion. To convert a decimal MB/s value to decimal GB/s:

```python
gb_per_second = mb_per_second / 1_000
```

For the default 50 GB/s directional NVLink capacity, full-duplex capacity is
100 GB/s per physical link. An H200 with 18 active links therefore uses a
configured aggregate denominator of 900 GB/s per direction or 1,800 GB/s full
duplex. Capacity affects utilization percentages only, not calculated rates.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Validation

Validation covered syntax, arguments, CSV structure, GPU selection, append and
collision behavior, counter resets, NVIDIA driver 610 counter labels, live
telemetry, and controlled NVLink traffic.

TransferBench 1.70.01 was built with CUDA 13.3 for `sm_90`. A complete 8-GPU
peer-to-peer matrix used 256 MiB per direction, five timed iterations per pair,
GPU kernel executors, and both unidirectional and bidirectional modes.

| Measurement | Result |
| --- | ---: |
| Unidirectional GPU-to-GPU average | 316.48 GB/s |
| Unidirectional off-diagonal range | 313.16–319.00 GB/s |
| Bidirectional average per direction | 311.47 GB/s |
| Combined bidirectional range | 616.84–627.91 GB/s |
| NVLink counter sampling epochs | 573 |
| Valid physical-link rows | 82,512 |
| Median measured sampling interval | 0.250156 s |
| Counter resets or query failures | 0 |

TransferBench uses CUDA event timing around short transfer bursts, whereas the
NVLink monitor divides counter changes by the complete interval between
NVIDIA-SMI queries. The largest sampled aggregate was 17.936 GB/s because the
sub-millisecond transfers occupied only a small fraction of a roughly 250 ms
counter interval. This does not conflict with TransferBench's event-timed peak;
use a sustained workload when validating utilization percentages.

Run the software tests without requiring a GPU:

```bash
python3 -m unittest discover -s tests -v
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Troubleshooting

### `nvidia-smi was not found on PATH`

```bash
command -v nvidia-smi
nvidia-smi --version
```

Confirm that the NVIDIA driver utilities are installed and visible from the
invoking shell.

### NVIDIA-SMI cannot communicate with the driver

```bash
ls -l /dev/nvidia*
cat /proc/driver/nvidia/version
nvidia-smi
```

The kernel module, user-space libraries, and device-node access must all be
available. Containers and restricted execution sandboxes must explicitly pass
through the NVIDIA devices, commonly through NVIDIA Container Toolkit or the
platform's GPU-access controls.

### Output file already exists

```bash
python3 nvsmi_gpu_monitor.py -o telemetry.csv --overwrite -W 30
python3 nvsmi_gpu_monitor.py -o telemetry.csv --append -W 30
```

### No complete NVLink Tx/Rx counters

```bash
nvidia-smi nvlink --getthroughput d
```

The monitor requires a GPU heading and numeric Tx/Rx data counters for each
usable link. Driver 610's `Data Tx`/`Data Rx` labels and the older `Tx0`/`Rx0`
labels are supported. Inactive, unsupported, or incomplete links are omitted.

### The requested interval is not exact

`--interval` controls target spacing between query starts. NVIDIA-SMI queries
take nonzero time, so the monitor records `interval_seconds` and uses the
measured value. If a query exceeds the interval, the next begins immediately.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Known limitations

- NVIDIA states that NVSMI textual output is not guaranteed to be backward
  compatible; future driver layouts may require parser updates.
- NVIDIA-SMI exposes NVLink counters by physical link ID, not remote GPU, so
  the raw CSV cannot identify a peer-GPU matrix on an NVSwitch system.
- `dmon` supports integer-second intervals and up to 16 selected devices.
- `dmon` timestamps are added when rows reach the wrapper rather than at the
  device's internal sampling instant.
- NVIDIA-SMI and TransferBench use different sampling windows and traffic
  semantics; their instantaneous values should not be expected to match.
- Utilization depends on the configured capacity. Confirm whether the
  published link figure or `nvidia-smi nvlink --status` is appropriate for the
  analysis and set `--link-capacity-gb-s` accordingly.
- Duration is approximate because an in-progress query is allowed to finish.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project layout

```text
cuda13.3_H200/
├── nvsmi_combined_monitor.py
├── nvsmi_common.py
├── nvsmi_gpu_monitor.py
├── nvsmi_nvlink_bw_monitor.py
├── tests/
│   └── test_nvsmi_monitors.py
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Issues and pull requests are welcome in the
[PerformanceProfiling repository](https://github.com/hibagus/PerformanceProfiling).
For parser or bandwidth changes, include:

- The GPU model and NVLink topology.
- CUDA, NVIDIA-SMI, NVML, and driver versions.
- Sanitized representative `dmon` and NVLink output.
- The command used to generate validation traffic.
- Expected and observed CSV output.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [LICENSE](../../LICENSE) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [NVIDIA-SMI documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
- [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA Hopper NVLink documentation](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-throughput-by-another-1-5x-on-nvidia-h200-tensor-core-gpus-and-nvlink-switch/)
- [ROCm TransferBench](https://github.com/ROCm/TransferBench)
- README structure inspired by
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
