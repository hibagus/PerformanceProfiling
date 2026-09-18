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

### TransferBench 10 GB peer-to-peer matrix

TransferBench 1.70.01 was built with CUDA 13.3 for `sm_90`. The test performed
a complete 8-GPU peer-to-peer matrix with exactly 10,000,000,000 bytes per
direction, 10 timed iterations per pair, GPU kernel executors, and both
unidirectional and bidirectional modes:

```bash
NUM_CPU_DEVICES=0 NUM_GPU_DEVICES=8 P2P_MODE=0 NUM_ITERATIONS=10 \
  /home/bagus/TransferBench/TransferBenchCuda p2p 10000000000
```

The test ran on 2026-09-18 while both monitors collected data at a one-second
target interval. NVLink utilization used the 26.562 GB/s per-direction,
per-link capacity reported by the installed driver.

| Measurement | Result |
| --- | ---: |
| Unidirectional GPU-to-GPU average | 369.94 GB/s |
| Unidirectional off-diagonal range | 367.87–371.41 GB/s |
| Bidirectional average per direction | 364.24 GB/s |
| Combined bidirectional average | 728.49 GB/s |
| Combined bidirectional range | 727.39–729.33 GB/s |

#### Unidirectional bandwidth matrix

Values are GB/s. Diagonal entries are local GPU copies; off-diagonal entries
are peer-to-peer copies.

| Source \ Destination | GPU 0 | GPU 1 | GPU 2 | GPU 3 | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU 0 | 1733.77 | 371.09 | 371.36 | 371.32 | 371.10 | 371.35 | 371.15 | 371.13 |
| GPU 1 | 371.32 | 1736.03 | 371.30 | 371.27 | 371.08 | 371.29 | 371.33 | 371.13 |
| GPU 2 | 371.05 | 370.81 | 1738.49 | 371.06 | 371.28 | 371.10 | 371.10 | 371.31 |
| GPU 3 | 371.33 | 371.37 | 371.34 | 1736.03 | 371.04 | 371.01 | 371.03 | 371.34 |
| GPU 4 | 371.36 | 371.41 | 371.31 | 371.19 | 1662.75 | 368.33 | 368.24 | 368.03 |
| GPU 5 | 368.06 | 368.37 | 368.09 | 368.07 | 368.47 | 1666.21 | 368.22 | 368.45 |
| GPU 6 | 368.37 | 368.02 | 368.39 | 367.87 | 368.04 | 368.00 | 1663.96 | 368.39 |
| GPU 7 | 368.36 | 368.36 | 368.38 | 368.43 | 368.10 | 368.35 | 368.37 | 1665.65 |

#### Combined bidirectional bandwidth matrix

Each value is the sum of the two simultaneously measured directions in GB/s.

| Source \ Destination | GPU 0 | GPU 1 | GPU 2 | GPU 3 | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU 0 | — | 728.70 | 728.15 | 728.34 | 728.63 | 728.62 | 728.27 | 728.47 |
| GPU 1 | 728.56 | — | 728.40 | 728.34 | 728.16 | 728.21 | 729.21 | 729.04 |
| GPU 2 | 728.42 | 728.41 | — | 728.49 | 728.83 | 728.63 | 727.56 | 727.39 |
| GPU 3 | 728.58 | 727.69 | 728.48 | — | 729.09 | 728.24 | 727.63 | 728.62 |
| GPU 4 | 728.88 | 728.60 | 727.60 | 728.27 | — | 729.24 | 727.95 | 727.80 |
| GPU 5 | 728.42 | 728.85 | 728.90 | 728.96 | 728.87 | — | 728.92 | 729.33 |
| GPU 6 | 728.34 | 728.52 | 728.59 | 728.67 | 728.53 | 727.78 | — | 729.15 |
| GPU 7 | 728.28 | 728.32 | 728.37 | 728.50 | 728.74 | 729.04 | 728.62 | — |

#### Monitor results during the matrix test

| Measurement | Result |
| --- | ---: |
| Monitoring duration | 5,258 seconds |
| Valid physical-link rows | 756,432 |
| Median measured NVLink sampling interval | 1.000696 s |
| Counter resets or query failures | 0 |
| Consolidated GPU rows matched to NVLink samples | 31,165 / 31,264 (99.68%) |
| Peak sampled aggregate NVLink traffic | 262.501 GB/s on GPU 4 |
| Peak sampled Rx / Tx traffic | 131.251 / 131.251 GB/s |
| Peak aggregate utilization | 27.45% |
| Peak GPU power | 165 W |
| Maximum GPU / memory temperature | 30 / 29 °C |
| Maximum SM utilization | 100% |
| ECC errors | 0 |

TransferBench uses CUDA event timing around each transfer, whereas the NVLink
monitor divides physical-link counter changes by the complete interval between
NVIDIA-SMI queries. The application payload and sampled physical-link values
therefore use different measurement windows and should not be expected to
match directly.

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
