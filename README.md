<a id="readme-top"></a>

<div align="center">

# Performance Profiling Utilities

### Lightweight telemetry for accelerator servers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/AMD%20%7C%20Intel%20%7C%20NVIDIA%20%7C%20Mellanox%20%7C%20Dell-555?logo=linux&logoColor=white)](#tool-catalog)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](#contributing)

Small, inspectable tools for collecting CPU, GPU, interconnect, network, and
platform telemetry without deploying a full monitoring stack.

[Get started](#getting-started) · [Browse tools](#tool-catalog) · [Choose a monitor](#choosing-a-monitor) · [Contribute](#contributing)

</div>

<details>
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#tool-catalog">Tool catalog</a></li>
    <li><a href="#choosing-a-monitor">Choosing a monitor</a></li>
    <li><a href="#output-and-data-handling">Output and data handling</a></li>
    <li><a href="#compatibility-and-support">Compatibility and support</a></li>
    <li><a href="#project-layout">Project layout</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the project

Performance Profiling Utilities is a collection of focused monitoring scripts,
parsers, and counter configurations for accelerator servers. The tools record
vendor telemetry alongside benchmarks or applications and produce CSV suitable
for later analysis.

The actively maintained toolkits cover:

- AMD Instinct GPU telemetry, PCIe bandwidth, and peer-to-peer xGMI traffic.
- Intel CPU, cache, memory, power, PCIe IIO, and UPI counters.

Older utilities for AMD EPYC, NVIDIA, Mellanox InfiniBand, and Dell iDRAC are
retained under `legacy/` for reference and reproducibility. Always consult the
README nearest to a tool because requirements and counter semantics vary by
platform.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

Clone the repository:

```bash
git clone https://github.com/hibagus/PerformanceProfiling.git
cd PerformanceProfiling
```

### AMD ROCm and MI300X

```bash
cd AMD/rocm10.0-Mi300X
python3 amdsmi_combined_monitor.py --help
python3 amdsmi_combined_monitor.py -g 0 -w 1 -W 5
```

See the [AMD MI300X monitoring guide](AMD/rocm10.0-Mi300X/README.md) for
installation, CSV schemas, units, and validation results.

### Intel PCM and Sapphire Rapids

```bash
cd Intel/pcm202604_sapphirerapids
python3 pcm_iio_monitor.py --help
python3 pcm_iio_monitor.py --duration 5
```

See the [Intel PCM monitoring guide](Intel/pcm202604_sapphirerapids/README.md)
for binary discovery, permissions, monitor selection, and TransferBench
validation workflows.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Tool catalog

### Current toolkits

| Platform | Tool | Purpose |
| --- | --- | --- |
| AMD ROCm 10.0 / MI300X | [`amdsmi_combined_monitor.py`](AMD/rocm10.0-Mi300X/README.md#combined-monitor) | Collect and merge GPU telemetry with peer xGMI utilization |
| AMD ROCm 10.0 / MI300X | [`amdsmi_gpu_monitor.py`](AMD/rocm10.0-Mi300X/README.md#gpu-telemetry-monitor) | Temperature, power, clocks, utilization, VRAM, and aggregate PCIe bandwidth |
| AMD ROCm 10.0 / MI300X | [`amdsmi_xgmi_bw_monitor.py`](AMD/rocm10.0-Mi300X/README.md#xgmi-bandwidth-monitor) | Per-peer xGMI bandwidth and utilization |
| Intel PCM 202604 / Sapphire Rapids | [`pcm_cpu_monitor.py`](Intel/pcm202604_sapphirerapids/README.md#choosing-a-monitor) | CPU, cache, memory, power, and UPI telemetry |
| Intel PCM 202604 / Sapphire Rapids | [`pcm_cpu_iio_combined_monitor.py`](Intel/pcm202604_sapphirerapids/README.md#combined-cpu-and-iio-monitor) | Concurrent CPU/UPI and IIO telemetry in one timestamp-aligned CSV |
| Intel PCM 202604 / Sapphire Rapids | [`pcm_pcie_monitor.py`](Intel/pcm202604_sapphirerapids/README.md#limits-of-pcm-pcie) | Approximate socket-level PCIe transaction activity |
| Intel PCM 202604 / Sapphire Rapids | [`pcm_iio_monitor.py`](Intel/pcm202604_sapphirerapids/README.md#why-pcm-iio-is-preferred-for-gpu-traffic) | PCIe bandwidth by socket, IIO stack, root port, and device |
| Intel PCM 202604 / Sapphire Rapids | [`validate_pcm_transferbench.py`](Intel/pcm202604_sapphirerapids/README.md#controlled-validation) | Focused PCIe, IIO, and UPI validation cases |
| Intel PCM 202604 / Sapphire Rapids | [`validate_pcm_cpu_gpu_matrix.py`](Intel/pcm202604_sapphirerapids/README.md#full-cpu-to-gpu-matrix) | Complete CPU-NUMA × GPU × direction validation matrix |

### Legacy utilities

These tools may assume a fixed device count, filename convention, counter
layout, or topology. Review and validate them on the target system before use.

| Platform | Utility | Primary dependency |
| --- | --- | --- |
| AMD EPYC Milan | [`AMD_EPYC_Milan_PCIe_xGMI_MEM_BW_Monitor`](AMD/legacy/AMD_EPYC_Milan_PCIe_xGMI_MEM_BW_Monitor/README.md) | AMD uProf PCM |
| AMD ROCm-SMI | Parsers under `AMD/legacy/AMD_ROCM_SMI/` | Python, pandas, NumPy |
| NVIDIA PCIe | `NVIDIA/legacy/NVIDIA_PCIe_Throughput_Counter/` | Bash, `nvidia-smi` |
| NVIDIA NVLink | [`NVLink_Throughput_Counter`](NVIDIA/legacy/NVLink_Throughput_Counter/README.md) | Bash, `nvidia-smi` |
| Mellanox InfiniBand | [`Mellanox_Infiniband_Throughput_Counter`](Mellanox/legacy/Mellanox_Infiniband_Throughput_Counter/README.md) | Bash, Linux InfiniBand sysfs |
| Dell iDRAC | `Dell/legacy/IDRAC_Fan_Speed/idrac-fan-speed-parser.py` | Python, pandas, NumPy |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Choosing a monitor

| Measurement goal | Recommended starting point |
| --- | --- |
| AMD GPU power, temperature, clocks, utilization, VRAM, and PCIe | [`amdsmi_gpu_monitor.py`](AMD/rocm10.0-Mi300X/README.md#gpu-telemetry-monitor) |
| AMD GPU-to-GPU xGMI bandwidth | [`amdsmi_xgmi_bw_monitor.py`](AMD/rocm10.0-Mi300X/README.md#xgmi-bandwidth-monitor) |
| Combined AMD GPU and xGMI telemetry | [`amdsmi_combined_monitor.py`](AMD/rocm10.0-Mi300X/README.md#combined-monitor) |
| Intel CPU metrics or per-link UPI utilization | [`pcm_cpu_monitor.py`](Intel/pcm202604_sapphirerapids/README.md#choosing-a-monitor) |
| CPU-to-GPU PCIe bandwidth by root port or device | [`pcm_iio_monitor.py`](Intel/pcm202604_sapphirerapids/README.md#why-pcm-iio-is-preferred-for-gpu-traffic) |
| Approximate aggregate Intel PCIe traffic by socket | [`pcm_pcie_monitor.py`](Intel/pcm202604_sapphirerapids/README.md#limits-of-pcm-pcie) |
| AMD EPYC Milan PCIe, DRAM, or intersocket xGMI counters | [`0x19_0x01.conf`](AMD/legacy/AMD_EPYC_Milan_PCIe_xGMI_MEM_BW_Monitor/0x19_0x01.conf) |
| NVIDIA PCIe or NVLink throughput | The corresponding utility under `NVIDIA/legacy/` |
| Mellanox InfiniBand throughput | `Mellanox/legacy/Mellanox_Infiniband_Throughput_Counter/` |
| Flatten an older ROCm-SMI or iDRAC CSV | The corresponding parser under `AMD/legacy/` or `Dell/legacy/` |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Output and data handling

Generated telemetry can be large and machine-specific. The repository ignores
common generated artifacts, including CSV files, logs, `runs/`, `artifacts/`,
Python caches, virtual environments, and editor metadata.

Use a dedicated output directory for longer captures:

```bash
python3 AMD/rocm10.0-Mi300X/amdsmi_gpu_monitor.py \
  --output-dir runs/telemetry \
  --duration 300
```

Before sharing telemetry, inspect it for hostnames, device identifiers,
workload names, and other environment-specific information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Compatibility and support

Hardware counters and command output can change across drivers, firmware, and
vendor-tool releases. Treat each toolkit's tested environment as a compatibility
baseline, not a guarantee for every platform.

When bringing a tool to another environment:

1. Record the hardware topology and relevant software versions.
2. Inspect raw vendor-tool output before relying on a parser.
3. Capture an idle baseline.
4. Generate a controlled workload and confirm the expected counters respond.
5. Verify units, direction semantics, sampling intervals, and link capacities.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project layout

```text
PerformanceProfiling/
├── AMD/
│   ├── rocm10.0-Mi300X/
│   └── legacy/
├── Intel/
│   └── pcm202604_sapphirerapids/
├── NVIDIA/
│   └── legacy/
├── Mellanox/
│   └── legacy/
├── Dell/
│   └── legacy/
├── LICENSE
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Issues and pull requests are welcome. Include as much of the following as
practical when reporting a problem or adding a tool:

- Hardware model and topology.
- Operating system and kernel version.
- Driver, firmware, and vendor-tool versions.
- Exact command used and sanitized representative input.
- Expected and observed behavior.
- Units and conversion assumptions.

Keep generated telemetry and local environments out of commits. Prefer a new
version- or platform-specific directory when a substantially different schema
is required.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [AMD ROCm](https://rocm.docs.amd.com/)
- [AMD-SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [Intel Performance Counter Monitor](https://github.com/intel/pcm)
- [NVIDIA System Management Interface](https://developer.nvidia.com/system-management-interface)
- [NVIDIA Networking](https://www.nvidia.com/en-us/networking/)
- [Dell Technologies iDRAC](https://www.dell.com/en-us/lp/dt/open-manage-idrac)
- README structure inspired by
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
