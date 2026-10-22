<a id="readme-top"></a>

<div align="center">

# Slurm Job Link Monitor

### InfiniBand, NVLink and PCIe throughput per node, cut to each run of a job

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Slurm](https://img.shields.io/badge/Slurm-srun%20--overlap-555)](#usage)

Three small, dependency-free scripts that run the per-node monitors of this repository inside a
Slurm job, mark when each run starts and ends, and report the bytes and rates every link carried
during each run.

</div>

## About the tools

| Script | Purpose |
| --- | --- |
| `node_link_monitor.py` | Run on every node (one task per node); starts the InfiniBand, NVLink and `nvidia-smi dmon` monitors that apply to the node, writes `<host>_ib.csv`, `<host>_nvlink.csv`, `<host>_dmon.csv` and a `<host>_node_info.txt`, and stops on a stop file, a duration or SIGTERM |
| `run_marker.py` | Append `start` / `end` markers for a run label to one CSV; `wrap` runs a command between the two and records its exit status |
| `window_summary.py` | Cut every node's samples to each marked run and summarize per host, link and direction |
| `example_job.sbatch` | A complete two-node example |

The monitors it starts:

| Link | Monitor | Default interval |
| --- | --- | --- |
| InfiniBand ports | [`Mellanox/sysfs_mlx5/ib_port_bw_monitor.py`](../../Mellanox/sysfs_mlx5/README.md) | 0.1 s |
| NVLink (GPU to GPU) | [`NVIDIA/cuda13.3_H200/nvsmi_nvlink_bw_monitor.py`](../../NVIDIA/cuda13.3_H200/README.md#nvlink-bandwidth-monitor) | 0.5 s |
| PCIe (GPU to host), power, clocks, utilization | [`NVIDIA/cuda13.3_H200/nvsmi_gpu_monitor.py`](../../NVIDIA/cuda13.3_H200/README.md#gpu-telemetry-monitor) | 1 s |

A monitor whose link is absent is not started (no HCA, no NVLink counters, no `nvidia-smi`).
`--nvidia-toolkit` and `--ib-toolkit` point at other toolkit directories.

### Tested environment

| Component | Version or configuration |
| --- | --- |
| Python | 3.9.25 and 3.10.12 |
| Node | 8 × NVIDIA H200 (NVSwitch), driver 610.57.04, no HCA; without Slurm |
| Check | A 2-GPU run of a stencil solver on GPUs 2 and 3: the NVLink bytes counted in its window, 4.2926 GB per GPU and direction, equal the bytes the application sends (8,192 ghost slots × 1,048 B × 500 stages) |
| Slurm | TACC Vista `gh-dev`, 8 GH200 nodes, NDR 400 (2026-10-22): the InfiniBand bytes of rank 0's node in an 8-node run's window, 23.24 GB, are the application's 23.07 GB of ghost data plus 0.7% of setup; the monitors cost nothing measurable (175.36 vs 175.24 ms/step) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Inside a job (see `example_job.sbatch`):

```bash
M=PerformanceProfiling/Slurm/job_link_monitor
srun --ntasks-per-node=1 --overlap python3 $M/node_link_monitor.py \
     --output-dir "$TEL" --stop-file "$TEL/STOP" &

python3 $M/run_marker.py "$TEL/markers.csv" wrap run-8gpu --meta gpus=8 -- srun -n 8 ./app input.json

touch "$TEL/STOP"; wait
python3 $M/window_summary.py --markers "$TEL/markers.csv" --data-dir "$TEL" -o link_summary.csv
```

- **One monitor per node.** Start it with `--ntasks-per-node=1 --overlap` so it shares the nodes with
  the application steps. Where a cluster restricts GPU devices per job step, give the monitor step the
  node's GPUs too (e.g. `--gpus-per-node=4 --overlap`); the sysfs InfiniBand counters need nothing.
- **Mark the runs.** `run_marker.py ... start LABEL` / `... end LABEL`, or `wrap LABEL -- COMMAND`. A
  window covers everything between the two markers (setup included); to isolate a phase, mark the
  phase boundaries, or compare with the application's own phase timers.
- **Packed runs** (several runs side by side on disjoint nodes) can share one markers file; each
  marker is one atomic append. The summary then reports every node for every window; select the
  nodes of each run afterwards.
- On a node without Slurm, run `node_link_monitor.py` directly in the background (`--duration` or
  `--stop-file`).

## Summary CSV

`window_summary.py` reads every `<host>_ib.csv`, `<host>_nvlink.csv` and `<host>_dmon.csv` in
`--data-dir`. A sample stamped `t` with interval `dt` covers `(t - dt, t]`; its share of a window is
the overlap of that span with `[start, end]`.

| Field | Meaning |
| --- | --- |
| `label`, `start_epoch`, `end_epoch`, `window_seconds` | The run window |
| `host`, `source`, `entity`, `direction` | `ib` (`mlx5_0/1`), `nvlink` (`gpu3`: all links of the GPU summed), `pcie` (`gpu3`); `rx` / `tx` |
| `gbytes` | Bytes moved during the window (decimal GB) |
| `mean_gb_s` | `gbytes` / window length |
| `peak_gb_s` | Highest sample rate in the window (an average over one interval) |
| `capacity_gb_s`, `mean_utilization_pct`, `peak_utilization_pct` | Against the per-direction capacity (InfiniBand: the port rate / 8; NVLink: links × `--nvlink-capacity-gb-s`; PCIe: `--pcie-capacity-gb-s` if given) |
| `coverage_pct` | Sampled share of the window; below 100 means a monitor started late or stopped early |
| `samples` | Samples that overlap the window |

## Known limitations

- The counters are per port and per GPU, not per process: other jobs on a shared node are counted.
- The interval averages bursts. A ghost exchange of a few hundred microseconds per stage shows as its
  average over 0.1-0.5 s; use a profiler (Nsight Systems `--gpu-metrics-devices`, `--nic-metrics`)
  for burst rates, and these counters for the bytes and averages.
- `nvidia-smi dmon` reports PCIe in whole MB/s once per second; on Grace Hopper and Grace Blackwell
  the CPU-GPU path is NVLink-C2C, which dmon's PCIe columns do not cover.
- The NVIDIA monitors were validated on H200 (driver 610). On other GPUs and drivers check that
  `nvidia-smi nvlink --getthroughput d` prints `Link N: Data Tx/Rx: ... KiB` lines.

## Tests

```bash
python3 -m unittest discover -s Slurm/job_link_monitor/tests -v
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
