<a id="readme-top"></a>

<div align="center">

# AMD MI300X Monitoring Toolkit

### GPU, PCIe, and xGMI telemetry with AMD-SMI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ROCm](https://img.shields.io/badge/ROCm-7.2.0-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![Platform](https://img.shields.io/badge/AMD_Instinct-MI300X-ED1C24?logo=amd&logoColor=white)](#tested-environment)

Dependency-free Python tools for collecting AMD GPU telemetry, aggregate PCIe
bandwidth, and per-peer xGMI bandwidth in analysis-ready CSV format.

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

This toolkit wraps the `amd-smi` command-line utility supplied with ROCm. It
supports finite or continuous collection, GPU selection, safe file handling,
standard-output streaming, and concurrent GPU/xGMI monitoring.

| Script | Purpose |
| --- | --- |
| `amdsmi_combined_monitor.py` | Run both monitors concurrently and merge their timestamped output |
| `amdsmi_gpu_monitor.py` | Collect temperature, power, clocks, utilization, VRAM, and instantaneous PCIe bandwidth |
| `amdsmi_xgmi_bw_monitor.py` | Convert cumulative xGMI counters into per-peer bandwidth and utilization |
| `amdsmi_common.py` | Shared argument, GPU-selection, and safe output-file helpers |

The GPU monitor preserves AMD-SMI CSV data while clarifying the ambiguous PCIe
field name. The xGMI monitor derives rates from cumulative counters using the
actual elapsed time between queries.

### Tested environment

| Component | Version or configuration |
| --- | --- |
| GPU | 8 × AMD Instinct MI300X |
| Platform | Linux bare metal |
| Python | 3.10.12 |
| ROCm | 7.2.0 |
| AMD-SMI CLI | 26.2.1+fc0010cf6a |
| AMD-SMI library | 26.2.1 |
| Linux kernel | 5.15.0-191-generic |
| MI300X xGMI capacity | 64 GB/s per direction |

Other AMD-SMI, ROCm, GPU, and partition configurations may expose different
fields or JSON layouts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

### Prerequisites

- Linux with supported AMD GPUs and a working AMDGPU driver.
- ROCm with `amd-smi` available from `PATH`.
- Python 3.10 or newer.
- Permission to query the installed GPUs.

The scripts use only the Python standard library; no `pip install` step is
required.

Confirm the required commands:

```bash
python3 --version
amd-smi version
amd-smi list
```

Clone the repository and enter the toolkit directory:

```bash
git clone https://github.com/hibagus/PerformanceProfiling.git
cd PerformanceProfiling/AMD/rocm7.2-Mi300X
```

Display the available options:

```bash
python3 amdsmi_combined_monitor.py --help
python3 amdsmi_gpu_monitor.py --help
python3 amdsmi_xgmi_bw_monitor.py --help
```

Run five-second smoke tests on GPU 0:

```bash
python3 amdsmi_gpu_monitor.py -g 0 -w 1 -W 5 --stdout
python3 amdsmi_xgmi_bw_monitor.py -g 0 -w 1 -W 5 --stdout
```

Diagnostic messages go to standard error, leaving standard output as valid CSV
when `--stdout` is selected.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Combined monitor

`amdsmi_combined_monitor.py` launches both standalone monitors with the same
GPU selection, interval, duration, and UTC run ID. It then attaches the nearest
xGMI sample for each source GPU to every GPU telemetry row.

Monitor every GPU for 60 seconds:

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
<UTC>_amdsmi_monitor.csv
<UTC>_amdsmi_xgmi_bandwidth.csv
<UTC>_amdsmi_consolidated.csv
```

Override these names with `--run-id`, `--gpu-output`, `--xgmi-output`, or
`--output`. Use `--dry-run` to inspect both child commands and output paths.

AMD-SMI timestamps the two data streams independently. The combined monitor
preserves both raw files and performs a nearest-timestamp join for the same
source GPU. The default tolerance is half the requested interval and can be
changed with `--match-tolerance`. Initial merged xGMI cells may be blank because
the first counter query establishes a baseline.

Pressing Ctrl+C stops both collectors cleanly and merges the partial raw files.

### GPU telemetry monitor

The GPU monitor requests power, temperatures, clocks, engine utilization, VRAM,
and PCIe bandwidth from `amd-smi monitor`.

<details>
  <summary>Show the equivalent native command</summary>

```bash
amd-smi monitor \
  --power-usage --temperature --gfx --mem --vram-usage --pcie \
  --gpu all --csv --watch 1
```

</details>

```bash
# Monitor every GPU until interrupted.
python3 amdsmi_gpu_monitor.py

# Monitor GPUs 0 and 1 for 60 seconds.
python3 amdsmi_gpu_monitor.py -g 0 1 -w 1 -W 60

# Comma-separated and mixed GPU lists are accepted.
python3 amdsmi_gpu_monitor.py -g 0,1,4 -W 60
```

Choose a file and explicitly replace it when present:

```bash
python3 amdsmi_gpu_monitor.py \
  -g 0 \
  -W 60 \
  -o gpu0_telemetry.csv \
  --overwrite
```

Preview the native command:

```bash
python3 amdsmi_gpu_monitor.py -g 0,1 -w 2 -W 30 --dry-run
```

Add optional counters:

```bash
python3 amdsmi_gpu_monitor.py --ecc --violation -W 60
```

`--ecc` adds ECC and PCIe replay counters. `--violation` adds the MI300 power
and thermal violation fields supported by the installed AMD-SMI version.

### xGMI bandwidth monitor

The xGMI monitor repeatedly invokes:

```bash
amd-smi xgmi --metric --gpu all --json
```

AMD-SMI 26.2.1 returns the complete peer matrix only when all GPUs are queried.
The script therefore queries all GPUs and applies `--gpus` afterward as a
source-GPU filter.

#### ROCm 7.2 peer-label correction

On the tested ROCm 7.2.0/AMD-SMI 26.2.1 host, the `gpu` and `bdf` attached to
each AMD-SMI peer counter identify an internal counter slot, not the logical
TransferBench destination. Isolated tests of all 56 directed pairs established
the following map. Each cell is the **raw AMD-SMI peer GPU label** used for the
logical destination in its column; `—` is the self-link.

| Source \ logical destination | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU0 | — | 7 | 1 | 6 | 3 | 2 | 4 | 5 |
| GPU1 | 6 | — | 7 | 4 | 0 | 2 | 3 | 5 |
| GPU2 | 3 | 5 | — | 6 | 0 | 7 | 4 | 1 |
| GPU3 | 4 | 7 | 1 | — | 6 | 2 | 0 | 5 |
| GPU4 | 6 | 5 | 7 | 0 | — | 2 | 3 | 1 |
| GPU5 | 6 | 7 | 1 | 0 | 3 | — | 4 | 2 |
| GPU6 | 3 | 7 | 1 | 4 | 0 | 2 | — | 5 |
| GPU7 | 4 | 2 | 5 | 0 | 6 | 1 | 3 | — |

The default `--peer-map rocm72-mi300x` corrects `peer_gpu` and `peer_bdf` while
retaining the original labels in `raw_peer_gpu` and `raw_peer_bdf`. Use
`--peer-map none` only for raw-counter investigation. The mapping is specific
to this validated eight-GPU MI300X topology; the monitor stops with an error if
the observed topology cannot be mapped.

#### Which counters overlap or “collide”

No two logical peers of one source map to the same raw counter: every row in
the mapping above is a seven-element bijection. Consequently, simultaneous
GPU0→GPU1 and GPU0→GPU7 traffic remains separated as raw `GPU0→GPU7` and
`GPU0→GPU5`; the monitor does not sum them. Concurrent GPU0→GPU1 and GPU2→GPU3
also uses four distinct endpoint rows.

There is intentional **endpoint mirroring**, however. A physical link is
reported once at each endpoint. These are the two raw rows that observe each
logical GPU pair:

| Logical link | Raw endpoint counters | Logical link | Raw endpoint counters |
| --- | --- | --- | --- |
| GPU0↔GPU1 | GPU0→7, GPU1→6 | GPU0↔GPU2 | GPU0→1, GPU2→3 |
| GPU0↔GPU3 | GPU0→6, GPU3→4 | GPU0↔GPU4 | GPU0→3, GPU4→6 |
| GPU0↔GPU5 | GPU0→2, GPU5→6 | GPU0↔GPU6 | GPU0→4, GPU6→3 |
| GPU0↔GPU7 | GPU0→5, GPU7→4 | GPU1↔GPU2 | GPU1→7, GPU2→5 |
| GPU1↔GPU3 | GPU1→4, GPU3→7 | GPU1↔GPU4 | GPU1→0, GPU4→5 |
| GPU1↔GPU5 | GPU1→2, GPU5→7 | GPU1↔GPU6 | GPU1→3, GPU6→7 |
| GPU1↔GPU7 | GPU1→5, GPU7→2 | GPU2↔GPU3 | GPU2→6, GPU3→1 |
| GPU2↔GPU4 | GPU2→0, GPU4→7 | GPU2↔GPU5 | GPU2→7, GPU5→1 |
| GPU2↔GPU6 | GPU2→4, GPU6→1 | GPU2↔GPU7 | GPU2→1, GPU7→5 |
| GPU3↔GPU4 | GPU3→6, GPU4→0 | GPU3↔GPU5 | GPU3→2, GPU5→0 |
| GPU3↔GPU6 | GPU3→0, GPU6→4 | GPU3↔GPU7 | GPU3→5, GPU7→0 |
| GPU4↔GPU5 | GPU4→2, GPU5→3 | GPU4↔GPU6 | GPU4→3, GPU6→0 |
| GPU4↔GPU7 | GPU4→1, GPU7→6 | GPU5↔GPU6 | GPU5→4, GPU6→2 |
| GPU5↔GPU7 | GPU5→2, GPU7→1 | GPU6↔GPU7 | GPU6→5, GPU7→3 |

For traffic A→B, endpoint A's `write` and endpoint B's `read` measure the same
transfer from opposite endpoints. Summing those two rows double-counts it.
Reverse B→A traffic uses the other direction fields in the same endpoint-row
pair. Simultaneous bidirectional traffic therefore shares the two endpoint
rows, but remains separated into `read_gb_s` and `write_gb_s`. `total_gb_s` and
`bidirectional_utilization_pct` intentionally combine both directions; use the
directional fields when attributing a one-way workload.

The code applies each raw row independently and rejects duplicate logical keys;
there is no monitor-side aggregation of different GPU pairs.

For every directed peer link, the rate is calculated from the counter delta and
measured elapsed time:

```text
rate = (current cumulative counter - previous cumulative counter)
       / measured elapsed time
```

```bash
# Monitor every directed GPU-to-GPU link until interrupted.
python3 amdsmi_xgmi_bw_monitor.py

# Monitor links originating from GPU 0 for 60 seconds.
python3 amdsmi_xgmi_bw_monitor.py -g 0 -w 1 -W 60

# Use a 500 ms interval and stream the CSV.
python3 amdsmi_xgmi_bw_monitor.py -g 0 1 -w 0.5 -W 30 --stdout

# Inspect the uncorrected AMD-SMI peer slots (diagnostic use only).
python3 amdsmi_xgmi_bw_monitor.py -g 0 -W 30 --peer-map none --stdout
```

Override capacity or failure handling when required:

```bash
python3 amdsmi_xgmi_bw_monitor.py \
  --link-capacity-gb-s 50 \
  --query-timeout 15 \
  --max-errors 5 \
  -W 60
```

The default timeout is 10 seconds and collection stops after three consecutive
query failures. A successful query resets the error count.

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
python3 amdsmi_gpu_monitor.py --output-dir runs/telemetry -W 60

# Choose a complete output path.
python3 amdsmi_xgmi_bw_monitor.py -o runs/xgmi.csv -W 60

# Stream CSV into another process.
python3 amdsmi_gpu_monitor.py --stdout -W 60 | gzip > gpu.csv.gz
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

AMD-SMI controls the exact base schema. The script renames AMD-SMI's ambiguous
`pcie_bw` header to `pcie_bw_bidirectional_mbps`. This is one instantaneous
transmit-plus-receive value; the numeric value is not modified.

<details>
  <summary><strong>Show GPU telemetry fields</strong></summary>

| Field | Meaning | Unit |
| --- | --- | --- |
| `timestamp` | AMD-SMI sample timestamp | Unix epoch seconds |
| `gpu` | AMD-SMI GPU index | — |
| `xcp` | Compute-partition identifier | — |
| `power_usage` | Current socket power | W |
| `max_power` | Maximum configured power | W |
| `hotspot_temperature` | GPU hotspot temperature | °C |
| `memory_temperature` | HBM temperature | °C |
| `gfx_clk` | Graphics clock | MHz |
| `gfx` | Graphics-engine utilization | % |
| `mem` | Memory activity | % |
| `mem_clock` | Memory clock | MHz |
| `vram_used` | Used VRAM | AMD-SMI-labeled MB |
| `vram_free` | Free VRAM | AMD-SMI-labeled MB |
| `vram_total` | Total VRAM | AMD-SMI-labeled MB |
| `vram_percent` | Used VRAM | % |
| `pcie_bw_bidirectional_mbps` | Aggregate PCIe transmit + receive bandwidth | Mb/s |

`--ecc` adds `single_bit_ecc`, `double_bit_ecc`, and `pcie_replay`.
`--violation` adds fields supplied by the installed AMD-SMI version.

</details>

### xGMI bandwidth fields

Each row represents one directed source-to-peer link over one measured
interval. Query midpoint timestamps and monotonic elapsed time reduce timing
error. The first query establishes the baseline and produces no rate rows.

<details>
  <summary><strong>Show xGMI bandwidth fields</strong></summary>

| Field | Meaning | Unit |
| --- | --- | --- |
| `timestamp_epoch` | Midpoint timestamp of the current query | Unix epoch seconds |
| `timestamp_utc` | The same timestamp in ISO 8601 form | UTC |
| `interval_seconds` | Measured time between query midpoints | s |
| `source_gpu` / `peer_gpu` | Directed link endpoints | GPU index |
| `source_bdf` / `peer_bdf` | PCI addresses for the endpoints | BDF |
| `raw_peer_gpu` / `raw_peer_bdf` | Uncorrected AMD-SMI counter-slot identity | GPU index / BDF |
| `peer_mapping` | Mapping profile used (`rocm72-mi300x` or `none`) | — |
| `read_counter_kb` / `write_counter_kb` | Current cumulative counters | decimal KB |
| `read_delta_kb` / `write_delta_kb` | Counter changes during the interval | decimal KB |
| `read_gb_s` / `write_gb_s` | Calculated directional rates | decimal GB/s |
| `total_gb_s` | Sum of read and write rates | decimal GB/s |
| `unidirectional_capacity_gb_s` | Configured capacity per direction | decimal GB/s |
| `read_utilization_pct` / `write_utilization_pct` | Directional rate divided by capacity | % |
| `bidirectional_utilization_pct` | Total rate divided by twice the directional capacity | % |
| `counter_status` | `ok` or `reset_or_wrap` | — |

</details>

If a counter decreases, the interval is marked `reset_or_wrap`; delta, rate,
and utilization cells remain blank instead of assuming a counter width and
reporting a false spike. The new values become the following baseline.

The consolidated CSV keeps every GPU-monitor column and adds fields named
`xgmi_to_gpu_N_bidirectional_utilization_pct`. A source GPU has values for its
peers and a blank self-link cell.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Units and conversions

Bandwidth uses decimal SI units:

```text
1 KB = 1,000 bytes
1 MB = 1,000,000 bytes
1 GB = 1,000,000,000 bytes
1 Mb = 1,000,000 bits
8 bits = 1 byte
```

AMD-SMI reports PCIe bandwidth in megabits per second:

```python
mb_per_second = pcie_bw_bidirectional_mbps / 8
gb_per_second = pcie_bw_bidirectional_mbps / 8_000
```

For example, `524,000 Mb/s` is `65,500 MB/s`, or `65.5 GB/s`. Do not divide by
1,024 when converting between SI bandwidth units.

If binary byte units are specifically required:

```python
mib_per_second = pcie_bw_bidirectional_mbps * 1_000_000 / 8 / (1024**2)
gib_per_second = pcie_bw_bidirectional_mbps * 1_000_000 / 8 / (1024**3)
```

AMD-SMI calculates VRAM fields using `1024**2` but labels them MB; numerically,
those values are MiB despite the upstream label.

On the tested system AMD-SMI reports an xGMI maximum of 512 Gb/s, or 64 GB/s,
in each direction. The resulting full-duplex capacity is 128 GB/s and is the
default basis for utilization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Validation

Validation covered syntax, arguments, CSV structure, append and collision
behavior, counter resets, live telemetry, and controlled traffic. TransferBench
was built against ROCm 7.2.0 and exercised all 56 directed GPU pairs.

The isolated directed-pair matrix produced:

| Measurement | Result |
| --- | ---: |
| TransferBench median application payload | 48.272 GB/s |
| TransferBench range | 47.365–48.606 GB/s |
| xGMI source-write median | 57.144 GB/s |
| xGMI source-write range | 56.353–57.709 GB/s |
| Source-write vs destination-read maximum difference | 0.123 GB/s |
| Counter status | `ok` for every tested pair |

Controlled concurrent tests confirmed the ownership rules:

- GPU0→GPU1 used raw rows GPU0→7 and GPU1→6.
- GPU2→GPU3 simultaneously used GPU2→6 and GPU3→1, with no shared row.
- GPU0→GPU1 plus GPU0→GPU7 used distinct GPU0 source rows 7 and 5.
- Bidirectional GPU0↔GPU1 shared its endpoint-row pair as expected, with the
  two directions separated into read and write counters.

AMD-SMI counts link-accounted traffic, so it need not equal application payload
throughput. Protocol overhead, counter semantics, sampling windows, and
unrelated traffic can all contribute to the difference. TransferBench is useful
for validation but is not required for normal monitoring.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Troubleshooting

### `amd-smi was not found on PATH`

```bash
command -v amd-smi
amd-smi version
```

Confirm that ROCm and AMD-SMI are installed and that the executable is visible
from the invoking shell.

### Permission or device-access errors

```bash
id
ls -l /dev/kfd /dev/dri/renderD*
```

The account must have site-approved access to GPU device nodes, commonly via
the `video` and `render` groups. Follow the target environment's security policy.

### Output file already exists

```bash
python3 amdsmi_gpu_monitor.py -o telemetry.csv --overwrite -W 30
python3 amdsmi_gpu_monitor.py -o telemetry.csv --append -W 30
```

### No usable per-peer xGMI counters

```bash
amd-smi xgmi --metric --gpu all --json
```

The monitor requires numeric peer `read` and `write` counters. Self-links and
links reported as `N/A` are intentionally omitted.

### The requested interval is not exact

`--interval` controls target spacing between query starts. AMD-SMI queries take
nonzero time, so the monitor records `interval_seconds` and uses the measured
value for each rate. If a query exceeds the interval, the next begins
immediately.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Known limitations

- Parsing follows the AMD-SMI CLI schema observed with ROCm 7.2.0; future schema
  changes may require updates.
- `pcie_bw_bidirectional_mbps` combines transmit and receive traffic and does
  not expose the directions separately on the tested platform.
- AMD-SMI PCIe and xGMI values represent link-accounted traffic, not application
  payload throughput.
- Filtering xGMI source GPUs does not reduce query cost because AMD-SMI must
  return the complete matrix first.
- Duration is approximate because an in-progress query is allowed to finish.
- Unsupported or absent board-temperature and encoder metrics are intentionally
  excluded from the GPU monitor.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project layout

```text
rocm7.2-Mi300X/
├── amdsmi_combined_monitor.py
├── amdsmi_common.py
├── amdsmi_gpu_monitor.py
├── amdsmi_xgmi_bw_monitor.py
├── tests/
│   └── test_xgmi_mapping.py
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Issues and pull requests are welcome in the
[PerformanceProfiling repository](https://github.com/hibagus/PerformanceProfiling).
For parser or bandwidth changes, include:

- The GPU model and topology.
- ROCm, AMD-SMI, and AMDGPU driver versions.
- A sanitized representative AMD-SMI response.
- The command used to generate validation traffic.
- Expected and observed CSV output.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [LICENSE](../../LICENSE) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [AMD-SMI documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [ROCm TransferBench](https://github.com/ROCm/TransferBench)
- README structure inspired by
  [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
