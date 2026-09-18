<a id="readme-top"></a>

<div align="center">

# Intel PCM Monitoring Toolkit

### Portable CPU, PCIe, IIO, and UPI telemetry for Intel Emerald Rapids

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
  Emerald Rapids.
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

### Tested environment

| Component | Version or configuration |
| --- | --- |
| CPU | 2 × Intel Xeon Platinum 8570 |
| Microarchitecture | Emerald Rapids-SP, family 6 model 207 stepping 2 |
| CPU topology | 2 sockets, 56 cores per socket, 112 online CPUs, SMT disabled |
| NUMA | Node 0 uses even-numbered CPUs; node 1 uses odd-numbered CPUs |
| GPU | 8 × NVIDIA H200; GPUs 0–3 on NUMA 0 and GPUs 4–7 on NUMA 1 |
| Operating system | Ubuntu 22.04.5 LTS, Linux 5.15 |
| Intel PCM | 202604 (`abb6bce`), Release build |
| PCM access mode | Linux `perf_event` (`PCM_NO_MSR=1`) |
| TransferBench | 1.70.01 CUDA build for `sm_90` |

The matching PCM IIO event definition is `opCode-6-207.txt`; it must remain
beside the `pcm-iio` executable.

### Build PCM 202604

The validated binaries were built from `/home/bagus/pcm`:

```bash
sudo apt-get update
sudo apt-get install -y cmake

cd /home/bagus/pcm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Point the wrappers at that build without installing it system-wide:

```bash
export PCM_BIN=/home/bagus/pcm/build/bin/pcm
export PCM_PCIE_BIN=/home/bagus/pcm/build/bin/pcm-pcie
export PCM_IIO_BIN=/home/bagus/pcm/build/bin/pcm-iio
export TRANSFERBENCH_BIN=/home/bagus/TransferBench/TransferBenchCuda
```

### Binary discovery

PCM binaries are resolved from `PCM_BIN`, `PCM_PCIE_BIN`, or `PCM_IIO_BIN`, then
from `PATH`. Use `--binary` to override either source.

If you choose to run validation, the validators resolve TransferBench from
`TRANSFERBENCH_BIN`, then `PATH`, or from `--transferbench`. The CUDA build used here links only to the NVIDIA driver library. If another
CUDA build needs an additional runtime-library path, set `CUDA_LIB_DIR` or pass
`--cuda-lib`.

Confirm that the required commands are available:

```bash
pcm --version
pcm-pcie --version
pcm-iio --version
```

TransferBench only needs to be checked before an optional validation run:

```bash
TransferBenchCuda
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

On Linux, prepare RDT on the host before using `--cpu-rdt`:

```bash
sudo mount -t resctrl resctrl /sys/fs/resctrl
findmnt /sys/fs/resctrl
```

When `resctrl` is mounted and `--cpu-rdt` is selected, automatic privilege
detection elevates the native `pcm` binary so it can manage resctrl monitoring
groups. In a container, mount `resctrl` on the host and expose that mount to the
container; a read-only `/sys` or missing `CAP_SYS_ADMIN` prevents mounting it
from inside the container.

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

On Emerald Rapids, `Part0` through `Part7` are the eight channel selections
inside one IIO PMON stack, not CPU cores or eight pieces of one measurement.
PCM programs channel masks `0x01` through `0x80`; for a PCIe stack, PCM maps
Part0 through Part7 to enabled root-port device numbers 1 through 8 on that
stack's root bus. A part with `no_root_port` has no enabled/discovered root port
in PCM's topology, although its filtered counter is still emitted. IDX and DMI
stacks use the same part IDs for their internal accelerator or DMI channel
mapping, so their topology is not necessarily the PCIe device-number mapping.

#### IIO traffic terminology

PCM names IIO traffic from the perspective of the transaction initiator and
the operation it requests. `IB` means a PCIe device initiated a DMA request
into the host; `OB` means the CPU initiated an MMIO request to a PCIe device.
`read` and `write` describe the requested operation against the destination
address space, not simply the direction in which payload data moves.

| PCM metric | Initiator and requested operation |
| --- | --- |
| `IB read` | PCIe device reads host memory through DMA |
| `IB write` | PCIe device writes host memory through DMA |
| `OB read` | CPU reads the PCIe device through MMIO |
| `OB write` | CPU writes the PCIe device through MMIO |

Consequently, TransferBench H2D copies normally appear as `IB read`: the GPU's
DMA engine reads host memory, although the returned payload travels from host
to GPU. TransferBench D2H copies normally appear as `IB write`: the GPU writes
the returned data into host memory. `OB read` and `OB write` generally capture
CPU-issued register or mapped-BAR MMIO rather than the bulk DMA payload.

#### Dell PowerEdge XE9680L validation topology

The validated system is a Dell PowerEdge XE9680L with two Intel Xeon Platinum
8570 processors and eight NVIDIA H200 GPUs. Each socket has 56 online physical
cores; SMT is disabled. GPUs 0–3 attach to socket/NUMA node 0, while GPUs 4–7
attach to socket/NUMA node 1. NVIDIA reports an `NV18` connection between every
GPU pair.

The mapping below was derived from `nvidia-smi`, Linux PCI sysfs, `lspci`, and
the PCM IIO topology emitted during the validated TransferBench captures. It is
specific to this XE9680L configuration and firmware. The Excel letters and
1-based indices refer to the 862-column consolidated CSV produced with the
documented default combined-monitor options.

| GPU | GPU endpoint BDF | NUMA/socket | PCM IIO route and root port | H2D: `IB read` Excel/index | D2H: `IB write` Excel/index |
| --- | --- | --- | --- | ---: | ---: |
| GPU0 | `0000:1b:00.0` | 0 | `Socket0 / Stack 1 PCIe3 / Part0 / 15:01.0` | `FT` / 176 | `FS` / 175 |
| GPU1 | `0000:3c:00.0` | 0 | `Socket0 / Stack 3 IDX1 / Part0 / 37:01.0` | `IF` / 240 | `IE` / 239 |
| GPU2 | `0000:4b:00.0` | 0 | `Socket0 / Stack 8 IDX3 / Part0 / 48:01.0` | `OJ` / 400 | `OI` / 399 |
| GPU3 | `0000:5c:00.0` | 0 | `Socket0 / Stack 6 PCIe2 / Part0 / 59:01.0` | `LX` / 336 | `LW` / 335 |
| GPU4 | `0000:9a:00.0` | 1 | `Socket1 / Stack 1 PCIe3 / Part0 / 97:01.0` | `SB` / 496 | `SA` / 495 |
| GPU5 | `0000:bb:00.0` | 1 | `Socket1 / Stack 3 IDX1 / Part0 / b7:01.0` | `UN` / 560 | `UM` / 559 |
| GPU6 | `0000:cd:00.0` | 1 | `Socket1 / Stack 8 IDX3 / Part0 / c7:01.0` | `AAR` / 720 | `AAQ` / 719 |
| GPU7 | `0000:dc:00.0` | 1 | `Socket1 / Stack 6 PCIe2 / Part0 / d7:01.0` | `YF` / 656 | `YE` / 655 |

For example, GPU0 H2D traffic is reported in:

```text
pcm_iio__Socket0__IIO_Stack_1_PCIe3__Part0__15_01_0__IB_read_bytes_per_second
```

Replace `IB_read` with `IB_write` for GPU0 D2H traffic. Use the complete header
name as the stable lookup key because Excel letters and numeric positions can
change when PCM options, the PCM version, or discovered topology changes.

Discover the PCM IIO stack and root-port mapping on the running firmware with:

```bash
python3 pcm_iio_monitor.py --list-topology --output pcm_iio_topology.csv
```

Do not copy IIO stack or root-port mappings from another platform: PCIe BDFs,
stack numbering, and enabled parts are firmware- and wiring-specific. The
combined CSV encodes the discovered socket, stack, part, and root-port BDF in
each IIO column name. Stack aggregate columns sum Part0 through Part7 for the
same stack and direction.

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

Do not run `pcm` and `pcm-pcie` together on Emerald Rapids: both program
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

The tested host used the following setting for the current boot:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```

To make it persistent, create a site-approved sysctl configuration containing
`kernel.perf_event_paranoid=-1`. Loading the `msr` kernel module is not required
for the default `perf_event` mode.

### PCM requests sudo

CPU TPMI and IIO topology discovery may require the ACPI MCFG table. If it is
unreadable, the wrappers interactively elevate only the native `pcm` or
`pcm-iio` binary. CPU RDT collection also requests elevation. CSV files are
pre-created by the calling user so output ownership is preserved. This gives
CPU PCM access to TPMI-derived uncore-frequency data without weakening ACPI
sysfs permissions.

```bash
python3 pcm_cpu_monitor.py --sudo     # Force CPU PCM elevation
python3 pcm_cpu_monitor.py --no-sudo  # Never elevate CPU PCM
python3 pcm_iio_monitor.py --sudo     # Force elevation
python3 pcm_iio_monitor.py --no-sudo  # Never elevate
```

The combined monitor applies its `--sudo`, `--no-sudo`, or automatic decision
to both native collectors and authenticates once before starting either child.
The child collectors stay in the same controlling-terminal session and use
non-interactive `sudo -n`, so they reuse that ticket and can never pause a
redirected run for another password prompt.

### Repetitive topology warnings

Known non-fatal warnings about unmapped CPU buses and absent Emerald Rapids
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

PCM 202604 built successfully from tag `202604`. A three-sample CPU smoke test
identified both sockets as Emerald Rapids-SP and produced the expected system,
socket, memory, power, and four-link UPI columns through Linux `perf_event`.

CUDA TransferBench validation used a 256 MiB transfer size and three-second
sustained cases. GPU0 is local to NUMA node 0, so CPU node 1 exercises the
remote UPI path.

| Validation case | TransferBench peak | PCM observation |
| --- | ---: | --- |
| CPU1 → GPU0 H2D | 55.282 GB/s | Peak system `TotalUPIin` 54.646 GB/s |
| GPU0 → CPU1 D2H | 55.004 GB/s | Directional UPI response present |
| CPU1 local copy | 180.062 GB/s | Only background UPI traffic |
| CPU1 → CPU0 cross-socket copy | 113.500 GB/s | Peak system `TotalUPIin` 110.778 GB/s |
| CPU1 → GPU0 under `pcm-pcie` | 55.287 GB/s | PCM PCIe CSV produced successfully |
| GPU0 → CPU1 under `pcm-pcie` | 54.968 GB/s | PCM PCIe CSV produced successfully |
| CPU1 → GPU0 under `pcm-iio` | 55.264 GB/s | GPU0 root-port `IB read` peaked at 55.228 GB/s |
| GPU0 → CPU1 under `pcm-iio` | 55.038 GB/s | GPU0 root-port `IB write` peaked at 54.985 GB/s |
| CPU1 → GPU0 with combined monitor | 55.265 GB/s | 9/19 CPU samples matched; 720 IIO columns added |
| GPU0 → CPU1 with combined monitor | 55.009 GB/s | 9/19 CPU samples matched; 720 IIO columns added |

The IIO measurements were within 0.1% of TransferBench for both directions.
The `pcm-iio` path requires privileged MCFG access on this host, so its
validation was launched interactively with sudo using:

```bash
PCM_BIN=/home/bagus/pcm/build/bin/pcm \
PCM_PCIE_BIN=/home/bagus/pcm/build/bin/pcm-pcie \
PCM_IIO_BIN=/home/bagus/pcm/build/bin/pcm-iio \
python3 validate_pcm_transferbench.py \
  --transferbench /home/bagus/TransferBench/TransferBenchCuda \
  --cpu-node 1 --remote-cpu-node 0 --gpu 0 \
  --size 256M --duration 3 --cases iio_h2d iio_d2h
```

TransferBench measures application payload with CUDA timing. PCM measures
link-accounted traffic over its sampling interval, so close agreement is useful
validation but exact equality is not required.

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
- [TransferBench](https://github.com/ROCm/TransferBench)
- README structure inspired by
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
