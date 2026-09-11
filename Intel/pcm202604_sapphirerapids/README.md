<a id="readme-top"></a>

<div align="center">

# Intel PCM Monitoring Toolkit

### Portable CPU, PCIe, IIO, and UPI telemetry for Intel Sapphire Rapids

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Intel PCM](https://img.shields.io/badge/Intel_PCM-202604-0071C5)](https://github.com/intel/pcm/releases/tag/202604)
[![Platform](https://img.shields.io/badge/platform-Linux-FCC624?logo=linux&logoColor=black)](#prerequisites)

Dependency-free Python wrappers for collecting native Intel® Performance
Counter Monitor CSV data, plus optional TransferBench validation workflows.

[Getting started](#getting-started) · [Usage](#usage) · [Choosing a monitor](#choosing-a-monitor) · [Validation](#validation-results)

</div>

<details>
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#choosing-a-monitor">Choosing a monitor</a></li>
    <li><a href="#permissions-and-troubleshooting">Permissions and troubleshooting</a></li>
    <li><a href="#validation-results">Validation results</a></li>
    <li><a href="#project-layout">Project layout</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the project

This toolkit adds a consistent Python interface around Intel PCM utilities. It
handles UTC-based output names, collection limits, diagnostics, safe overwrite
behavior, signal forwarding, and privilege escalation for IIO topology access.
PCM output remains in its native schema because available counters vary by CPU,
kernel, access mode, and PCM release.

| Script | Native utility | Purpose |
| --- | --- | --- |
| `pcm_cpu_monitor.py` | `pcm` | CPU, cache, memory, power, and per-link UPI telemetry |
| `pcm_pcie_monitor.py` | `pcm-pcie` | Approximate socket-level PCIe transaction activity |
| `pcm_iio_monitor.py` | `pcm-iio` | Per-socket, IIO-stack, root-port, and device PCIe bandwidth |
| `pcm_cpu_iio_combined_monitor.py` | `pcm` + `pcm-iio` | Concurrent CPU/UPI and IIO collection with a timestamp-aligned consolidated CSV |
| `validate_pcm_transferbench.py` | Multiple | Optional focused PCIe, IIO, and UPI validation cases |
| `validate_pcm_cpu_gpu_matrix.py` | Multiple | Optional full CPU-NUMA × GPU × direction validation matrix |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

### Prerequisites

Required for the monitoring scripts:

- Linux on a supported Intel server platform; validation was performed on
  Sapphire Rapids.
- Python 3.10 or newer. The wrappers use only the standard library.
- [Intel PCM release `202604`](https://github.com/intel/pcm/releases/tag/202604)
  with `pcm`, `pcm-pcie`, and `pcm-iio` compiled or installed.
- Permission to use the required core and uncore performance counters.

TransferBench is **not required** to run `pcm_cpu_monitor.py`,
`pcm_pcie_monitor.py`, or `pcm_iio_monitor.py`. It and its GPU runtime
dependencies are optional and used only by the two validation scripts.

### Supported Intel PCM release

This toolkit was implemented and validated against one pinned Intel PCM
revision:

| Component | Pinned value |
| --- | --- |
| Release/tag | [`202604`](https://github.com/intel/pcm/releases/tag/202604) |
| Commit | [`abb6bce87cc2ed23d6677541ebcfc47ca769d1ed`](https://github.com/intel/pcm/commit/abb6bce87cc2ed23d6677541ebcfc47ca769d1ed) |

Other Intel PCM revisions may change command options, CSV schemas, counter
availability, or behavior. Use release `202604` when reproducing the documented
validation results.

### Binary discovery

PCM binaries are resolved from `PCM_BIN`, `PCM_PCIE_BIN`, or `PCM_IIO_BIN`, then
from `PATH`. Use `--binary` to override either source.

If you choose to run validation, the validators resolve TransferBench from
`TRANSFERBENCH_BIN`, then `PATH`, or from `--transferbench`. If its GPU runtime
libraries are not already discoverable, set `ROCM_LIB_DIR` or pass
`--rocm-lib`.

Confirm that the required commands are available:

```bash
pcm --version
pcm-pcie --version
pcm-iio --version
```

TransferBench only needs to be checked before an optional validation run:

```bash
TransferBench
```

Every script provides command-line help:

```bash
python3 pcm_iio_monitor.py --help
python3 validate_pcm_transferbench.py --help
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Basic monitoring

Run these commands from the toolkit directory:

```bash
# CPU, memory, cache, power, and UPI metrics
python3 pcm_cpu_monitor.py --duration 60 --no-cores

# Approximate socket-level PCIe activity
python3 pcm_pcie_monitor.py --duration 60

# Per-root-port and per-device PCIe bandwidth
python3 pcm_iio_monitor.py --duration 60
```

The standalone monitors default to a one-second interval. Without `--duration`
or `--iterations`, collection continues until Ctrl+C. Each run creates a
UTC-prefixed CSV and a sibling `_stderr.log` file by default.

Select where the native CSV is written with the same options on all three
standalone monitors:

| Option | Result |
| --- | --- |
| `--output-mode file` | Write CSV to `--output`; this is the default |
| `--output-mode stdout` or `--stdout` | Stream CSV to standard output only |
| `--output-mode both` or `--both` | Stream CSV and write the same rows to `--output` |

Diagnostics remain on standard error and in the sibling diagnostic log, so
redirected standard output contains CSV only.

```bash
python3 pcm_cpu_monitor.py --duration 60 --stdout
python3 pcm_pcie_monitor.py --duration 60 --both --output pcm_pcie.csv
python3 pcm_iio_monitor.py --duration 60 --output-mode file
```

Choose output files explicitly when collecting beside a workload:

```bash
python3 pcm_cpu_monitor.py \
  --interval 1 \
  --duration 300 \
  --no-cores \
  --output runs/experiment/pcm_cpu.csv \
  --stderr-log runs/experiment/pcm_cpu_stderr.log
```

Preview a native command without starting collection:

```bash
python3 pcm_iio_monitor.py --duration 60 --dry-run
```

Pass native PCM options with repeatable `--pcm-arg` arguments. Use the equals
form when a native argument begins with a dash:

```bash
python3 pcm_cpu_monitor.py --duration 10 --pcm-arg=-m=1
```

### Combined CPU and IIO monitor

`pcm_cpu_iio_combined_monitor.py` runs `pcm_cpu_monitor.py` and
`pcm_iio_monitor.py` under one run ID and sampling interval. It initializes PCM
CPU first and waits for a structurally valid sample before starting IIO. This
ordering prevents IIO initialization from changing PCM's native CPU schema on
platforms where the tools discover overlapping uncore resources. A finite
`--duration` begins after both monitors have produced their first sample, so
initialization time is excluded from the overlapping collection window. Both
raw native CSV files are preserved.
At the end, the script pivots every IIO identity into four columns—`IB write`,
`IB read`, `OB read`, and `OB write`—and appends the nearest IIO sample to each
native PCM CPU row. It also sums Part0 through Part7 for every socket/IIO stack
and bandwidth direction, so both per-root-port detail and stack totals are
available in the same row.

The combined monitor omits per-core CPU metrics and disables CPU RDT metrics by
default. System- and socket-level CPU, memory, and UPI metrics remain enabled.
Disabling RDT prevents a mounted but inaccessible or stale Linux `resctrl`
hierarchy from making `pcm` repeatedly emit headers without a numeric sample.
Use `--cpu-rdt` only when L3 occupancy and local/remote memory-bandwidth RDT
metrics are required and the collector can create and read its resctrl monitor
groups.

```bash
python3 pcm_cpu_iio_combined_monitor.py \
  --interval 1 \
  --duration 60 \
  --output-dir runs/combined
```

Use `--with-cores` only when per-core columns are required. This mode is more
platform-dependent and may produce an inconsistent native PCM header when run
concurrently with `pcm-iio`; the merger detects and reports that condition
instead of writing a misaligned consolidated CSV.

The consolidated CSV retains PCM's two-row CPU header. Added columns are under
the `PCM IIO` category and are named with socket, stack/device, part, root-port
BDF, metric, and unit. Two metadata columns record the matched IIO timestamp
and signed time delta, making every alignment auditable.

On Sapphire Rapids, `Part0` through `Part7` are the eight channel selections
inside one IIO PMON stack, not CPU cores or eight pieces of one measurement.
PCM programs channel masks `0x01` through `0x80`; for a PCIe stack, PCM maps
Part0 through Part7 to enabled root-port device numbers 1 through 8 on that
stack's root bus. A part with `no_root_port` has no enabled/discovered root port
in PCM's topology, although its filtered counter is still emitted. IDX and DMI
stacks use the same part IDs for their internal accelerator or DMI channel
mapping, so their topology is not necessarily the PCIe device-number mapping.

Stack aggregate columns omit the root-port component and use this form:

```text
pcm_iio__Socket0__IIO_Stack_2_PCIe0__Part0_to_Part7_total__IB_read_bytes_per_second
```

The aggregate is the exact numeric sum of all available Part0-Part7 values for
that socket, stack, timestamp, and direction. It represents all traffic on the
IIO stack; use the individual part/root-port column when attributing traffic to
one GPU or PCIe port.

The default match tolerance is half the sampling interval. Override it when
the two native samplers have a larger stable phase difference:

```bash
python3 pcm_cpu_iio_combined_monitor.py \
  --duration 60 \
  --match-tolerance 0.75
```

Each monitor must produce its first valid sample within 30 seconds by default.
Slow platforms can increase this per-monitor limit with `--startup-timeout`.

The consolidated result also supports `--stdout`, `--both`, and
`--output-mode`. In stdout-only mode the consolidated CSV is emitted after the
run, while the two raw files are still retained. If IIO access requires sudo,
the combined launcher authenticates before starting either monitor so the
password prompt does not offset their launch times.

The combined mode intentionally uses the core and IIO PMUs concurrently. The
launcher keeps IIO active throughout the CPU sampling window and coordinates
shutdown afterward. PCM utilities can still perform broad uncore cleanup when
exiting, so use isolated passes when strict experiment reproducibility is more
important than a single aligned file.

### Controlled validation

This section is optional. The monitoring scripts operate independently and do
not invoke or depend on TransferBench.

The focused validator runs six sequential cases: H2D and D2H under `pcm-pcie`,
H2D and D2H under `pcm-iio`, and local and cross-socket CPU copies under `pcm`.

```bash
python3 validate_pcm_transferbench.py
```

Useful variations:

```bash
python3 validate_pcm_transferbench.py --dry-run
python3 validate_pcm_transferbench.py --cases pcie_h2d iio_h2d upi_cross
python3 validate_pcm_transferbench.py --cases combined_h2d combined_d2h
python3 validate_pcm_transferbench.py --cpu-node 1 --gpu 4
```

### Full CPU-to-GPU matrix

The matrix validator tests H2D and D2H between CPU NUMA nodes 0 and 1 and GPUs
0 through 7. By default, each of the 32 directional paths is measured once by
the combined monitor, producing timestamp-aligned CPU/UPI and IIO data while
the same TransferBench workload is active.

```bash
python3 validate_pcm_cpu_gpu_matrix.py
```

Preview the matrix or run a shorter subset with:

```bash
python3 validate_pcm_cpu_gpu_matrix.py --dry-run
python3 validate_pcm_cpu_gpu_matrix.py --duration 3 --gpus 0 4
python3 validate_pcm_cpu_gpu_matrix.py --monitor-mode isolated
```

`--monitor-mode isolated` retains the previous behavior: four sequential
captures per CPU/GPU pair (`iio_h2d`, `iio_d2h`, `upi_h2d`, and `upi_d2h`).

Results are organized beneath `runs/` with manifests, TransferBench topology,
benchmark logs, native PCM CSVs, and diagnostic logs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Choosing a monitor

| Measurement goal | Recommended tool | Interpretation |
| --- | --- | --- |
| Per-GPU or per-root-port PCIe bandwidth | `pcm_iio_monitor.py` | Primary quantitative tool; H2D is normally `IB read`, D2H is normally `IB write` |
| UPI bandwidth and link utilization | `pcm_cpu_monitor.py --no-cores` | `dataIn` is incoming payload; `trafficOut` includes data and protocol traffic |
| Socket-level PCIe activity | `pcm_pcie_monitor.py` | Directional/debug cross-check, not the primary bandwidth value |
| Timestamp-aligned PCIe and UPI for one workload | `pcm_cpu_iio_combined_monitor.py` | Runs distinct core/IIO PMUs together and produces one wide CSV |
| Strictly isolated PCIe and UPI validation | Separate `pcm-iio` and `pcm` passes | Each capture owns its PMU lifecycle |

### Why `pcm-iio` is preferred for GPU traffic

`pcm-iio` reports timestamped bytes per second by socket, IIO stack, root port,
bus, and device. For bulk DMA traffic:

- `IB read`: the PCIe device reads host memory, typically H2D payload.
- `IB write`: the PCIe device writes host memory, typically D2H payload.
- `OB read/write`: CPU MMIO traffic, usually control rather than bulk payload.

Capture topology when validating a new platform:

```bash
python3 pcm_iio_monitor.py --list-topology --output pcm_iio_topology.csv
lspci -Dnn | grep -iE 'vga|display|3d'
```

### Limits of `pcm-pcie`

`pcm-pcie` estimates bytes by multiplying counted transactions by 64. It cannot
attribute activity to a GPU or root port, and its estimate can diverge from the
payload rate when transactions are not full cache lines. With a non-default
interval, divide its byte count by the interval to obtain bytes per second. Its
CSV repeats the header for every sample and has no timestamp.

Do not run `pcm` and `pcm-pcie` together on Sapphire Rapids: both program
CHA/C-box counters. `pcm-iio` and `pcm` target distinct PMUs, but independent PCM
processes perform broad uncore cleanup when they exit. Isolated repeated passes
remain the most reproducible approach.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Permissions and troubleshooting

The wrappers request Linux `perf_event` access and preserve the NMI watchdog:

```text
PCM_NO_MSR=1
PCM_KEEP_NMI_WATCHDOG=1
```

The system's `perf_event_paranoid` policy must permit the required counters.
Use `--direct-msr` only when direct MSR and PCI configuration access has been
deliberately provided.

### `pcm-iio` requests sudo

IIO topology discovery may require the ACPI MCFG table. If it is unreadable,
the wrapper interactively elevates only the native `pcm-iio` command. The CSV
is pre-created by the calling user so output ownership is preserved.

```bash
python3 pcm_iio_monitor.py --sudo     # Force elevation
python3 pcm_iio_monitor.py --no-sudo  # Never elevate
```

### Repetitive topology warnings

Known non-fatal warnings about unmapped CPU buses and absent Sapphire Rapids
IIO stacks 10/11 are collapsed into one summary. Restore every line with:

```bash
python3 pcm_iio_monitor.py --show-topology-warnings
```

### Empty output or busy counters

- Inspect the sibling `_stderr.log` first.
- Ensure no other profiler owns the same PMU event set.
- Confirm that the native binary supports the processor generation.
- Compare idle data with a controlled one-direction workload.
- Allow several seconds for `pcm-iio` initialization and its first sample.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Validation results

A complete dual-socket Xeon Platinum 8460Y+ validation ran 64 isolated
10-second captures across eight AMD Instinct MI300X GPUs. GPUs 0--3 were local
to CPU NUMA node 0; GPUs 4--7 were local to CPU NUMA node 1. Every capture
completed successfully.

The theoretical local-path peak is 63.0 GB/s per direction for PCIe 5.0 x16
after 128b/130b encoding. For remote paths, the configured three-link UPI
connection provided an approximately 54 GB/s path limit. All values are decimal
GB/s.

`TB` is the mean of two independent TransferBench repetitions. IIO and UPI
cells contain `plateau median (observed peak)`. IIO uses root-port `IB read` for
H2D and `IB write` for D2H. UPI uses system `TotalUPIin`; it measures one
incoming payload direction and is not expected to equal TransferBench
one-to-one. `Peak UPI in` is the largest incoming-link utilization.

<details>
  <summary><strong>Show the complete 16-path validation table</strong></summary>

| Path | Route | Path peak | TB H2D | IIO H2D | UPI H2D | TB D2H | IIO D2H | UPI D2H | Peak UPI in |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU0–GPU0 | local | 63.0 | 54.95 | 56.03 (57.70) | 0.03 (0.11) | 56.37 | 56.44 (56.46) | 0.04 (0.10) | 0% |
| CPU0–GPU1 | local | 63.0 | 54.89 | 57.23 (58.03) | 0.04 (0.08) | 56.36 | 54.63 (56.44) | 0.03 (0.08) | 0% |
| CPU0–GPU2 | local | 63.0 | 54.94 | 56.28 (58.07) | 0.03 (0.09) | 56.36 | 56.38 (56.44) | 0.03 (0.08) | 0% |
| CPU0–GPU3 | local | 63.0 | 54.91 | 55.60 (57.85) | 0.04 (0.08) | 56.38 | 56.44 (56.48) | 0.03 (0.06) | 0% |
| CPU0–GPU4 | remote | ~54 | 54.28 | 52.11 (52.82) | 25.06 (25.35) | 52.68 | 51.36 (52.38) | 25.04 (26.30) | 52% |
| CPU0–GPU5 | remote | ~54 | 54.52 | 52.25 (55.47) | 25.06 (25.26) | 52.66 | 51.91 (53.56) | 24.75 (25.02) | 48% |
| CPU0–GPU6 | remote | ~54 | 54.48 | 52.38 (57.37) | 25.06 (25.31) | 52.60 | 51.21 (52.05) | 24.20 (24.77) | 49% |
| CPU0–GPU7 | remote | ~54 | 54.54 | 54.60 (57.43) | 25.01 (25.34) | 52.25 | 52.34 (52.64) | 23.60 (23.98) | 49% |
| CPU1–GPU0 | remote | ~54 | 54.26 | 54.65 (57.01) | 26.80 (27.00) | 53.14 | 53.14 (53.24) | 26.32 (26.66) | 51% |
| CPU1–GPU1 | remote | ~54 | 54.36 | 54.69 (57.57) | 26.84 (27.14) | 53.75 | 53.45 (53.75) | 26.72 (26.94) | 52% |
| CPU1–GPU2 | remote | ~54 | 54.56 | 55.12 (57.37) | 26.99 (27.11) | 54.22 | 54.04 (56.44) | 26.87 (27.04) | 53% |
| CPU1–GPU3 | remote | ~54 | 54.25 | 55.84 (57.47) | 26.85 (27.09) | 54.21 | 53.78 (55.21) | 26.88 (27.21) | 51% |
| CPU1–GPU4 | local | 63.0 | 55.00 | 51.98 (54.77) | 0.07 (0.64) | 56.43 | 56.26 (57.63) | 0.06 (0.67) | 0% |
| CPU1–GPU5 | local | 63.0 | 55.00 | 52.59 (55.70) | 0.05 (0.63) | 56.43 | 54.21 (54.96) | 0.06 (0.08) | 0% |
| CPU1–GPU6 | local | 63.0 | 54.97 | 52.02 (53.04) | 0.04 (0.62) | 56.39 | 54.45 (55.22) | 0.05 (0.61) | 0% |
| CPU1–GPU7 | local | 63.0 | 54.99 | 56.05 (58.12) | 0.07 (0.60) | 56.42 | 56.25 (56.33) | 0.04 (0.59) | 0% |

</details>

### Key findings

- Local TransferBench averages: **54.96 GB/s H2D** and **56.39 GB/s D2H**.
- Remote TransferBench averages: **54.41 GB/s H2D** and **53.19 GB/s D2H**.
- Local paths produced only background incoming UPI traffic.
- Remote paths produced **23.6–27.2 GB/s** incoming UPI payload and **48–53%**
  peak incoming-link utilization.
- `pcm-iio` tracked TransferBench closely enough for quantitative root-port
  measurement.
- `pcm-pcie` preserved direction but reported about 19.2 GB/s—roughly 35% of
  the payload rate—so it should remain a qualitative cross-check on this host.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project layout

```text
.
├── pcm_common.py
├── pcm_cpu_monitor.py
├── pcm_cpu_iio_combined_monitor.py
├── pcm_iio_monitor.py
├── pcm_pcie_monitor.py
├── validate_pcm_transferbench.py
└── validate_pcm_cpu_gpu_matrix.py
```

Generated CSV and log artifacts beneath `runs/` are intentionally ignored by
version control.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the repository license. See [`LICENSE`](../../LICENSE) for
details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [Intel Performance Counter Monitor](https://github.com/intel/pcm)
- [ROCm TransferBench](https://github.com/ROCm/TransferBench)
- README structure inspired by
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
