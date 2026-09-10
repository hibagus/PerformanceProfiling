# Performance Profiling Utilities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/Platforms-AMD%20%7C%20NVIDIA%20%7C%20Mellanox%20%7C%20Dell-blue)](#repository-layout)
[![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)](#contributing)

A collection of lightweight scripts, parsers, and configuration files for
collecting and processing performance telemetry from accelerator servers.

The repository currently covers AMD GPU telemetry and interconnect bandwidth,
AMD CPU memory and socket-interconnect counters, NVIDIA PCIe and NVLink
throughput, Mellanox InfiniBand port throughput, and Dell iDRAC telemetry
processing.

## Table of contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Current tools](#current-tools)
- [Legacy utilities](#legacy-utilities)
- [Getting started](#getting-started)
- [Choosing a tool](#choosing-a-tool)
- [Output and data handling](#output-and-data-handling)
- [Compatibility and support](#compatibility-and-support)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The project provides small, inspectable utilities for situations where a full
monitoring stack is unnecessary or unavailable. Typical uses include:

- Recording GPU power, temperature, clocks, utilization, and memory usage
- Measuring PCIe, xGMI, NVLink, or InfiniBand traffic
- Collecting counters alongside benchmark or application runs
- Converting vendor telemetry into CSV suitable for later analysis
- Flattening or summarizing previously collected CSV data

Tools are organized first by hardware vendor and then, where applicable, by
software or hardware generation. Documentation and requirements can differ
between directories, so consult the README nearest to the selected tool.

## Repository layout

```text
PerformanceProfiling/
├── AMD/
│   ├── rocm10.0-Mi300X/
│   │   ├── README.md
│   │   ├── amdsmi_common.py
│   │   ├── amdsmi_gpu_monitor.py
│   │   └── amdsmi_xgmi_bw_monitor.py
│   └── legacy/
│       ├── AMD_EPYC_Milan_PCIe_xGMI_MEM_BW_Monitor/
│       └── AMD_ROCM_SMI/
├── Dell/
│   └── legacy/IDRAC_Fan_Speed/
├── Mellanox/
│   └── legacy/Mellanox_Infiniband_Throughput_Counter/
├── NVIDIA/
│   └── legacy/
│       ├── NVIDIA_PCIe_Throughput_Counter/
│       └── NVLink_Throughput_Counter/
├── .gitignore
├── LICENSE
└── README.md
```

Directories named `legacy` contain older, environment-specific utilities.
They are retained for reference and reproducibility, but should be reviewed
and validated on the target system before production use.

## Current tools

### AMD ROCm 10.0 and MI300X

The actively documented toolset is located in
[`AMD/rocm10.0-Mi300X`](AMD/rocm10.0-Mi300X/README.md).

| Tool | Purpose | Output |
| --- | --- | --- |
| `amdsmi_gpu_monitor.py` | Collects GPU temperature, power, clocks, utilization, VRAM usage, and instantaneous PCIe bandwidth. | CSV |
| `amdsmi_xgmi_bw_monitor.py` | Converts cumulative AMD-SMI xGMI counters into per-peer bandwidth and utilization. | CSV |
| `amdsmi_common.py` | Provides shared argument, GPU-selection, and safe output-file helpers. | Internal module |

These scripts were tested on an eight-GPU AMD Instinct MI300X bare-metal
system with Python 3.10, ROCm 10.0, AMD-SMI 27.0.0, and AMDGPU 7.1.3.
They depend only on the Python standard library and the `amd-smi` executable.

Important characteristics include:

- Selection of all GPUs or specific GPU indices
- Finite-duration or continuous monitoring
- CSV output to a file, standard output, or both
- Safe overwrite and append behavior
- Explicit PCIe directionality and units in the
  `pcie_bw_bidirectional_mbps` column (transmit + receive)
- Actual elapsed-time accounting for calculated xGMI rates
- Detection of xGMI counter resets or wraps
- Configurable xGMI link capacity, timeout, and error threshold

See the [MI300X monitoring guide](AMD/rocm10.0-Mi300X/README.md) for complete
installation instructions, CLI examples, CSV schemas, units, validation
results, and troubleshooting.

## Legacy utilities

The following tools predate the current MI300X monitors or were created for a
specific experiment. Their assumptions may be tightly coupled to device count,
command output, input filename, CSV column positions, or system topology.

### AMD

| Path | Description | Primary dependency |
| --- | --- | --- |
| [`AMD/legacy/AMD_EPYC_Milan_PCIe_xGMI_MEM_BW_Monitor`](AMD/legacy/AMD_EPYC_Milan_PCIe_xGMI_MEM_BW_Monitor/README.md) | AMD uProf PCM configuration for PCIe, memory-channel, and intersocket xGMI counters on AMD EPYC Milan family `0x19`, model `0x01`. | AMD uProf PCM |
| `AMD/legacy/AMD_ROCM_SMI/rocm-smi-parser.py` | Flattens multi-row ROCm-SMI CSV samples by timestamp. | Python, pandas, NumPy |
| `AMD/legacy/AMD_ROCM_SMI/amd-smi-parser-1gpu.py` | Extracts and summarizes one-GPU benchmark telemetry using fixed input layouts and filename metadata. | Python, pandas, NumPy, numpyencoder |
| `AMD/legacy/AMD_ROCM_SMI/amd-smi-parser-8gpus.py` | Extracts and summarizes eight-GPU LLM-serving telemetry using fixed input layouts and filename metadata. | Python, pandas, NumPy, numpyencoder |

### NVIDIA

| Path | Description | Primary dependency |
| --- | --- | --- |
| `NVIDIA/legacy/NVIDIA_PCIe_Throughput_Counter/nvidia_pcie_throughput.sh` | Polls NVIDIA-SMI PCIe transmit and receive throughput and prints per-GPU and aggregate CSV-like rows. | Bash, `nvidia-smi` |
| [`NVIDIA/legacy/NVLink_Throughput_Counter`](NVIDIA/legacy/NVLink_Throughput_Counter/README.md) | Calculates per-GPU NVLink transmit and receive rates from cumulative NVIDIA-SMI counters. | Bash, `nvidia-smi` |

### Mellanox

| Path | Description | Primary dependency |
| --- | --- | --- |
| [`Mellanox/legacy/Mellanox_Infiniband_Throughput_Counter`](Mellanox/legacy/Mellanox_Infiniband_Throughput_Counter/README.md) | Reads InfiniBand transmit and receive counters from sysfs and calculates per-port and aggregate throughput. | Bash, Linux InfiniBand sysfs |

### Dell

| Path | Description | Primary dependency |
| --- | --- | --- |
| `Dell/legacy/IDRAC_Fan_Speed/idrac-fan-speed-parser.py` | Pivots multi-row iDRAC fan-speed CSV samples into a wider timestamp-indexed table. | Python, pandas, NumPy |

## Getting started

Clone the repository:

```bash
git clone https://github.com/hibagus/PerformanceProfiling.git
cd PerformanceProfiling
```

For the current AMD MI300X tools:

```bash
cd AMD/rocm10.0-Mi300X

python3 --version
amd-smi version

python3 amdsmi_gpu_monitor.py --help
python3 amdsmi_xgmi_bw_monitor.py --help
```

Run five-second smoke tests on GPU 0:

```bash
python3 amdsmi_gpu_monitor.py -g 0 -w 1 -W 5 --stdout
python3 amdsmi_xgmi_bw_monitor.py -g 0 -w 1 -W 5 --stdout
```

For a legacy utility, inspect its source and local README before running it.
Install only the dependencies required by that utility and confirm that its
device-count, filename, counter-unit, and column-layout assumptions match the
target environment.

## Choosing a tool

| Monitoring goal | Recommended starting point |
| --- | --- |
| MI300X power, temperature, clocks, utilization, VRAM, or PCIe bandwidth | [`amdsmi_gpu_monitor.py`](AMD/rocm10.0-Mi300X/README.md#gpu-telemetry-monitor) |
| MI300X GPU-to-GPU xGMI bandwidth and utilization | [`amdsmi_xgmi_bw_monitor.py`](AMD/rocm10.0-Mi300X/README.md#xgmi-bandwidth-monitor) |
| AMD EPYC Milan PCIe, DRAM, or intersocket xGMI counters | [`0x19_0x01.conf`](AMD/legacy/AMD_EPYC_Milan_PCIe_xGMI_MEM_BW_Monitor/0x19_0x01.conf) |
| NVIDIA PCIe throughput | `NVIDIA/legacy/NVIDIA_PCIe_Throughput_Counter/` |
| NVIDIA NVLink throughput | `NVIDIA/legacy/NVLink_Throughput_Counter/` |
| Mellanox InfiniBand port throughput | `Mellanox/legacy/Mellanox_Infiniband_Throughput_Counter/` |
| Flatten an older ROCm-SMI or iDRAC CSV | The appropriate parser under `AMD/legacy/` or `Dell/legacy/` |

## Output and data handling

Generated telemetry can be large and machine-specific. The repository
`.gitignore` excludes common generated data and development artifacts,
including:

- `*.csv`
- `runs/`
- `artifacts/`
- Python bytecode and cache directories
- Virtual environments
- Logs, coverage output, and editor metadata

Use a dedicated output directory for longer captures:

```bash
mkdir -p runs/telemetry
cd AMD/rocm10.0-Mi300X
python3 amdsmi_gpu_monitor.py --output-dir ../../runs/telemetry -W 300
```

Before sharing telemetry, check it for hostnames, device identifiers, workload
names, or other environment-specific information.

## Compatibility and support

Hardware-management interfaces and command output can change between driver,
firmware, and vendor-tool versions. Treat each directory's documented tested
environment as the compatibility baseline rather than a guarantee for every
system.

When moving a tool to a different environment:

1. Record the hardware topology and relevant software versions.
2. Inspect the raw vendor-tool output before relying on a parser.
3. Run an idle control capture.
4. Generate a known workload and confirm that the expected counters respond.
5. Verify counter units, direction semantics, sampling intervals, and link
   capacities.

The legacy utilities are provided as-is and may require modification for newer
tool output or different hardware layouts.

## Contributing

Issues and pull requests are welcome. When reporting a problem or adding a new
tool, include as much of the following as practical:

- Hardware model and topology
- Operating system and kernel version
- Driver, firmware, and vendor-tool versions
- Exact command used
- Sanitized raw input or a minimal representative sample
- Expected and observed behavior
- Units and conversion assumptions

Keep generated telemetry and local environments out of commits. Prefer a new
version- or platform-specific directory when supporting a substantially
different schema instead of silently changing the behavior of an established
legacy tool.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [AMD ROCm](https://rocm.docs.amd.com/)
- [AMD-SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [NVIDIA System Management Interface](https://developer.nvidia.com/system-management-interface)
- [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)
- [NVIDIA Networking](https://www.nvidia.com/en-us/networking/)
- [Dell Technologies iDRAC](https://www.dell.com/en-us/lp/dt/open-manage-idrac)
- README organization inspired by
  [Best README Template](https://github.com/othneildrew/Best-README-Template)
